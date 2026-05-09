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

    # Modest dilation around each letter — keeps inter-letter white pixels visible
    glyph_mask = cv2.dilate(
        glyph_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return glyph_mask


def detect(
    image_path: str,
    cls_threshold: float = 0.30,
    cam_threshold: float = 0.20,
    strip_text_confidence: float = 0.70,
) -> Optional[str]:
    """Detect watermark region in an image.

    Returns:
        Path to binary mask PNG (255=watermark, 0=keep), or None when
        the classifier says the image is clean or coverage is out of
        plausible range.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        logger.warning("Cannot read image: %s", image_path)
        return None

    h, w = img_bgr.shape[:2]
    total_px = h * w

    device = get_device()
    classifier = _get_classifier(device)

    # 1. Gate: is this image watermarked at all?
    pil = Image.open(image_path)
    p_full = float(predict_batch(classifier, [pil], device)[0])
    logger.info("classifier P(watermarked) full = %.3f", p_full)
    if p_full < cls_threshold:
        return None

    # 2. Grad-CAM heatmap — needs gradients on the leaves
    for p in classifier.parameters():
        p.requires_grad_(True)
    classifier.train(False)
    heatmap = gradcam_localize(classifier, pil, device)
    for p in classifier.parameters():
        p.requires_grad_(False)

    # 3. Threshold the heatmap → coarse spatial prior
    cam_mask = (heatmap > cam_threshold).astype(np.uint8) * 255

    # 4. Pixel-level refinement: AND with low-sat / high-pass mask
    pixel_mask = _heuristic_pixel_mask(img_bgr)
    refined = cv2.bitwise_and(cam_mask, pixel_mask)

    # 5. Morphology on the body-area mask: close gaps inside text
    #    strokes, dilate to cover anti-aliasing.
    refined = cv2.morphologyEx(
        refined, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    )
    refined = cv2.dilate(
        refined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    )

    # 6. After the body-area morphology, OR-in the strip text glyph
    #    mask. We add it AFTER global morphology so glyphs aren't
    #    re-collapsed into a contiguous bar — LaMa needs the
    #    inter-letter pixels untouched to reconstruct the background.
    if p_full >= strip_text_confidence:
        strip_mask = _strip_text_mask(img_bgr)
        refined = cv2.bitwise_or(refined, strip_mask)

    coverage = refined.sum() / 255 / total_px
    logger.info("detector: coverage=%.4f cam>%.2f", coverage, cam_threshold)

    if coverage < 0.0005 or coverage > 0.50:
        logger.warning("detector: coverage out of plausible range, returning None")
        return None

    TMP_DIR.mkdir(exist_ok=True)
    out_path = tempfile.mktemp(suffix=".png", dir=str(TMP_DIR))
    cv2.imwrite(out_path, refined)
    return out_path
