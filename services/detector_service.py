"""Watermark auto-detection service.

Pipeline (v2):
  1. Whole-image binary classifier (boomb0om convnext-tiny) — gate.
     If P(watermarked) < 0.3 → return None, image is clean.
  2. Grad-CAM localizer — produces a continuous attention heatmap
     showing which pixels the classifier attends to.
  3. Refinement — combine the heatmap with the original heuristic
     (low-saturation grey + high-pass edge response). Pixels marked
     iff both signals fire, dilated to cover anti-aliasing.

Falls back to None when coverage is implausibly low or high.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from services.cuda_policy import get_device
from services.watermark_classifier import load_model as load_classifier, predict_batch
from services.watermark_localizer import localize as gradcam_localize

logger = logging.getLogger(__name__)

TMP_DIR = Path(__file__).parent.parent / "tmp"

_classifier_cache: dict[str, torch.nn.Module] = {}


def _get_classifier(device: torch.device) -> torch.nn.Module:
    key = str(device)
    if key not in _classifier_cache:
        _classifier_cache[key] = load_classifier(device)
    return _classifier_cache[key]


def _heuristic_pixel_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Pixel-level texture mask: low-saturation grey + high-pass edges. uint8."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Loosened: capture more semi-transparent watermark colors, not just neutral grey.
    low_sat = hsv[..., 1] < 90
    mid_val = (hsv[..., 2] > 50) & (hsv[..., 2] < 235)
    grey = (low_sat & mid_val).astype(np.uint8) * 255

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=15)
    highpass = cv2.absdiff(gray, blurred)
    _, hp_mask = cv2.threshold(highpass, 6, 255, cv2.THRESH_BINARY)

    candidate = cv2.bitwise_and(grey, hp_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    candidate = cv2.dilate(candidate, kernel, iterations=1)
    return candidate


def _strip_text_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Detect text-like content in the bottom strip and right column.

    Most stock photos place the URL/ID/copyright text in a bottom band
    or bottom-right corner over a solid background. We catch the text
    by looking for high-pass strokes (per-letter), then keep only the
    components small enough to be glyphs (height < ~3% of image) and
    located in the corner regions.

    Crucially we keep individual letter shapes — never collapse them
    into a contiguous bar. LaMa needs the white inter-letter pixels
    intact to reconstruct the underlying background; a solid bar mask
    triggers its "banner" prior and produces a coloured rectangle.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    sharp = cv2.absdiff(gray, blurred)
    _, edges = cv2.threshold(sharp, 6, 255, cv2.THRESH_BINARY)

    # Restrict to bottom strip + right column where stock-photo text lives
    band_h = max(20, h // 12)
    band_w = max(40, w // 4)
    region = np.zeros_like(edges)
    region[h - band_h:h, :] = 255
    region[:, w - band_w:w] = 255
    edges = cv2.bitwise_and(edges, region)

    # Drop noise specks but keep glyph strokes
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )

    # Keep only components that look like text glyphs:
    #   - height roughly 6..3% of image height (typical text-line size)
    #   - width <= 4× height (full words filtered later by close-and-keep)
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    glyph_mask = np.zeros_like(edges)
    min_h, max_h = max(4, h // 200), max(20, h // 18)
    for i in range(1, n_lbl):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        ca = stats[i, cv2.CC_STAT_AREA]
        if min_h <= ch <= max_h and ca >= 6 and cw <= ch * 6:
            glyph_mask[labels == i] = 255

    # Modest dilation around each letter
    glyph_mask = cv2.dilate(
        glyph_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    # Solid-bar extension: stock photos often place text inside a
    # solid-colour bar (Dreamstime's blue strip, Shutterstock's white
    # band, etc.). When we find glyphs near the bottom edge, look at
    # their bounding rows — if those rows are dominated by a single
    # non-white colour, mark every pixel in those rows as "watermark"
    # so TELEA fills the full bar from the content above. We do NOT
    # do this for body/inner watermarks where adjacent pixels are
    # legitimate subject content.
    bottom_glyph_rows = np.where(glyph_mask[h - band_h:h].any(axis=1))[0]
    if bottom_glyph_rows.size:
        first_row = h - band_h + bottom_glyph_rows[0]
        # Probe the brightness of the rows around the glyphs. If they
        # form a contiguous non-white bar, extend the mask all the way
        # to the bottom edge so TELEA only sources fill pixels from
        # above (the legitimate content) — leaving even one row of
        # bar pixels at the boundary causes TELEA to propagate the
        # bar colour into the mask.
        bar_top = max(0, first_row - 4)
        bar_slice = img_bgr[bar_top:h, :]
        if bar_slice.size:
            median_v = float(np.median(cv2.cvtColor(bar_slice, cv2.COLOR_BGR2GRAY)))
            if median_v < 215:
                glyph_mask[bar_top:h, :] = 255

    return glyph_mask


def detect(
    image_path: str,
    cls_threshold: float = 0.30,
    cam_threshold: float = 0.20,
    strip_text_confidence: float = 0.70,
) -> Optional[str]:
    """Single-mask compatibility shim.

    Returns the combined mask path (body + strip text), or None if
    nothing detected. Useful for callers that take one mask only
    (the public /api/detect route). For best-quality results, prefer
    :func:`detect_split` and use a two-pass inpaint pipeline.
    """
    result = detect_split(image_path, cls_threshold=cls_threshold,
                          cam_threshold=cam_threshold,
                          strip_text_confidence=strip_text_confidence)
    if result is None:
        return None
    body_path, strip_path, _ = result

    body = cv2.imread(body_path, cv2.IMREAD_GRAYSCALE)
    if strip_path is not None:
        strip = cv2.imread(strip_path, cv2.IMREAD_GRAYSCALE)
        body = cv2.bitwise_or(body, strip)

    out_path = tempfile.mktemp(suffix=".png", dir=str(TMP_DIR))
    cv2.imwrite(out_path, body)
    return out_path


def detect_split(
    image_path: str,
    cls_threshold: float = 0.30,
    cam_threshold: float = 0.20,
    strip_text_confidence: float = 0.70,
) -> Optional[tuple[str, Optional[str], float]]:
    """Detect watermark regions, returning (body_mask, strip_mask, p_full).

    Body mask covers larger watermark regions and goes to LaMa. Strip
    mask covers thin text glyphs at image borders and should be
    pre-inpainted with cv2.inpaint(TELEA) before LaMa — LaMa's banner
    prior hallucinates colored bars when given thin horizontal masks.

    Returns None when the image is classified clean.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        logger.warning("Cannot read image: %s", image_path)
        return None

    h, w = img_bgr.shape[:2]
    total_px = h * w

    device = get_device()
    classifier = _get_classifier(device)

    pil = Image.open(image_path)
    p_full = float(predict_batch(classifier, [pil], device)[0])
    logger.info("classifier P(watermarked) full = %.3f", p_full)
    if p_full < cls_threshold:
        return None

    # Grad-CAM heatmap — body-area watermarks
    for p in classifier.parameters():
        p.requires_grad_(True)
    classifier.train(False)
    heatmap = gradcam_localize(classifier, pil, device)
    for p in classifier.parameters():
        p.requires_grad_(False)

    cam_mask = (heatmap > cam_threshold).astype(np.uint8) * 255
    pixel_mask = _heuristic_pixel_mask(img_bgr)
    body = cv2.bitwise_and(cam_mask, pixel_mask)
    body = cv2.morphologyEx(
        body, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    )
    body = cv2.dilate(
        body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    )

    body_coverage = body.sum() / 255 / total_px
    logger.info("detector: body coverage=%.4f cam>%.2f", body_coverage, cam_threshold)

    strip_path: Optional[str] = None
    if p_full >= strip_text_confidence:
        strip = _strip_text_mask(img_bgr)
        if strip.any():
            strip_path = tempfile.mktemp(suffix="_strip.png", dir=str(TMP_DIR))
            TMP_DIR.mkdir(exist_ok=True)
            cv2.imwrite(strip_path, strip)
            logger.info("detector: strip text glyph count > 0")

    if body_coverage > 0.50:
        logger.warning("detector: body coverage > 50%%, returning None")
        return None

    TMP_DIR.mkdir(exist_ok=True)
    body_path = tempfile.mktemp(suffix="_body.png", dir=str(TMP_DIR))
    cv2.imwrite(body_path, body)
    return body_path, strip_path, p_full


def prefill_strip(image_path: str, strip_mask_path: str, radius: int = 3) -> str:
    """Use cv2.inpaint TELEA to fill the strip-text mask in-place.

    Returns path to a new image with the strip text pixel-propagated
    out. TELEA is a fast marching method that fills small regions by
    propagating boundary pixels — perfect for thin text on a uniform
    background, with no learned banner prior.
    """
    img = cv2.imread(image_path)
    mask = cv2.imread(strip_mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return image_path
    filled = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    out = tempfile.mktemp(suffix="_prefilled.png", dir=str(TMP_DIR))
    cv2.imwrite(out, filled)
    return out
