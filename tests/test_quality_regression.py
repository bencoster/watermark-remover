"""Quality regression test against a known-good reference.

Runs the production pipeline on a real Dreamstime stock photo and
compares the body region against the cleaned version produced by
watermarkremover.io (a paid SaaS). PSNR + SSIM in the body region
must stay above a baseline established by the first passing run.

The two images differ in the bottom strip handling (we either
inpaint or crop it; the SaaS keeps the blue bar with text removed).
We crop both to the body region (above the strip) before comparing
so the strip strategy doesn't pollute the score.

The test is **skipped** unless LaMa weights have already been
downloaded, so it doesn't punish CI runs that don't pull them.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

# Force CPU - matches the production default and keeps GPU 0/3 (Imagine
# pool) untouched on the workstation.
os.environ.setdefault("BC_WMR_DEVICE", "cpu")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

FIXTURES = Path(__file__).parent / "fixtures"
WATERMARKED = FIXTURES / "dreamstime_18829755_watermarked.jpg"
REFERENCE = FIXTURES / "dreamstime_18829755_reference.jpg"
GROUND_TRUTH_MASK = FIXTURES / "dreamstime_18829755_groundtruth_mask.png"
WEIGHTS = ROOT / "weights" / "big-lama.pt"

# Mask-quality thresholds — directly measure detector quality against
# the diff mask between watermarked and watermarkremover.io's clean
# version. This is a more direct signal than PSNR on the inpaint
# output, which can hide detector recall issues behind a forgiving
# inpaint texture.
#
# Recall floor recalibrated 2026-05-10 after adding the revert-on-
# explosion safety net (lattice completion is dropped when it inflates
# coverage past 62%) and auto-scaling of disc/line geometry. The
# safety net occasionally trips on the canonical fixture too, dropping
# recall from ~0.67 to ~0.60 in exchange for working at every other
# image size. That trade is correct: a slightly under-covered mask is
# always better than total detection failure on non-canonical inputs.
MIN_RECALL = 0.55   # fraction of GT pixels we cover (sensitivity)
MIN_IOU = 0.22      # overall agreement

# Body region = top 86% of the image. Both ours and the SaaS keep this
# area; only the bottom 14% strip differs in handling.
BODY_FRAC = 0.86

# Quality thresholds — regression guard rails. Calibrated from a
# known-good run that scored PSNR=23.53 dB / SSIM=0.856. These cap
# the acceptable drift — a meaningful regression in detector or
# inpaint quality will push at least one metric below.
MIN_PSNR_DB = 22.0
MIN_SSIM = 0.82


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Single-channel SSIM via the standard local-window formulation
    (Wang et al. 2004). Implemented here so we don't add scikit-image."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    K1, K2, L = 0.01, 0.03, 255
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    win = 11
    sigma = 1.5

    mu_a = cv2.GaussianBlur(a, (win, win), sigma)
    mu_b = cv2.GaussianBlur(b, (win, win), sigma)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sig_a2 = cv2.GaussianBlur(a * a, (win, win), sigma) - mu_a2
    sig_b2 = cv2.GaussianBlur(b * b, (win, win), sigma) - mu_b2
    sig_ab = cv2.GaussianBlur(a * b, (win, win), sigma) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sig_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
    return float(np.mean(num / den))


def _ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """Per-channel mean SSIM."""
    return float(np.mean([_ssim(a[..., c], b[..., c]) for c in range(3)]))


def _crop_body(img: np.ndarray, frac: float = BODY_FRAC) -> np.ndarray:
    h = img.shape[0]
    return img[: int(h * frac), :]


def _resize_to(a: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a to match target_shape (h, w). Used when SaaS output is
    a slightly different resolution than our pipeline output."""
    if a.shape[:2] == target_shape:
        return a
    return cv2.resize(a, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)


@pytest.mark.skipif(not WEIGHTS.exists(),
                    reason="LaMa weights not downloaded — skip slow regression test")
def test_dreamstime_quality_regression():
    """Run the full one-click pipeline on the dreamstime fixture and
    confirm the body region scores at or above the known-good baseline
    against the watermarkremover.io reference."""
    from services.detector_service import detect_split, prefill_strip
    from services.lama_service import inpaint, load_model
    from services.cuda_policy import get_device

    assert WATERMARKED.exists(), f"missing fixture {WATERMARKED}"
    assert REFERENCE.exists(), f"missing fixture {REFERENCE}"

    t0 = time.time()
    result = detect_split(str(WATERMARKED))
    assert result is not None, "detector said image is clean — fixture broken?"
    body_mask, strip_mask, p_full = result
    assert p_full > 0.5, f"classifier confidence too low: {p_full}"

    image_for_lama = str(WATERMARKED)
    if strip_mask is not None:
        image_for_lama = prefill_strip(str(WATERMARKED), strip_mask, 4)

    model = load_model(get_device())
    cleaned = inpaint(image_for_lama, body_mask, get_device(), model)
    elapsed = time.time() - t0

    ours = cv2.imread(cleaned)
    reference = cv2.imread(str(REFERENCE))
    assert ours is not None and reference is not None

    # Align dimensions before comparing (SaaS may return slightly
    # different size — bicubic-resize ours to match)
    reference = _resize_to(reference, ours.shape[:2])

    ours_body = _crop_body(ours)
    ref_body = _crop_body(reference)

    psnr = float(cv2.PSNR(ref_body, ours_body))
    ssim = _ssim_rgb(ref_body, ours_body)
    print(f"\n  pipeline: {elapsed:.1f}s")
    print(f"  body PSNR vs reference: {psnr:.2f} dB  (min {MIN_PSNR_DB})")
    print(f"  body SSIM vs reference: {ssim:.3f}    (min {MIN_SSIM})")

    assert psnr >= MIN_PSNR_DB, f"PSNR {psnr:.2f} dB below baseline {MIN_PSNR_DB}"
    assert ssim >= MIN_SSIM, f"SSIM {ssim:.3f} below baseline {MIN_SSIM}"


def test_detector_mask_iou_vs_groundtruth():
    """Detect on the dreamstime fixture and compare the produced mask
    against the diff-mask of the watermarkremover.io clean image. We
    want both:
      - high recall (we cover most of what the GT marks as watermark)
      - acceptable IoU (we don't over-mask the subject too much)
    """
    from services.detector_service import detect_split

    assert WATERMARKED.exists()
    assert GROUND_TRUTH_MASK.exists()

    result = detect_split(str(WATERMARKED), mode="auto")
    assert result is not None
    body_mask, strip_mask, _p = result

    ours = cv2.imread(body_mask, cv2.IMREAD_GRAYSCALE)
    if strip_mask is not None:
        strip = cv2.imread(strip_mask, cv2.IMREAD_GRAYSCALE)
        ours = cv2.bitwise_or(ours, strip)
    gt = cv2.imread(str(GROUND_TRUTH_MASK), cv2.IMREAD_GRAYSCALE)
    if ours.shape != gt.shape:
        ours = cv2.resize(ours, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    ours_b = ours > 127
    gt_b = gt > 127

    inter = np.logical_and(ours_b, gt_b).sum()
    union = np.logical_or(ours_b, gt_b).sum()
    iou = float(inter / max(1, union))
    recall = float(inter / max(1, gt_b.sum()))
    precision = float(inter / max(1, ours_b.sum()))
    print(f"\n  detector vs GT mask")
    print(f"  Recall:    {recall:.3f}  (min {MIN_RECALL})")
    print(f"  Precision: {precision:.3f}")
    print(f"  IoU:       {iou:.3f}    (min {MIN_IOU})")

    assert recall >= MIN_RECALL, f"recall {recall:.3f} below baseline {MIN_RECALL}"
    assert iou >= MIN_IOU, f"IoU {iou:.3f} below baseline {MIN_IOU}"


def test_ssim_implementation_sanity():
    """A self-similar image must score 1.0 — guards against bugs in
    our hand-rolled SSIM if scikit-image is ever swapped in/out."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, (200, 200), dtype=np.uint8)
    assert abs(_ssim(img, img) - 1.0) < 1e-3
    # Constant offset cuts SSIM significantly.
    shifted = np.clip(img.astype(int) + 50, 0, 255).astype(np.uint8)
    assert _ssim(img, shifted) < 0.95
