"""Robustness sweep: detection vs image size and watermark alpha.

The main regression test runs on one resolution + one watermark
strength. This module probes how detection holds up when those
inputs vary — exactly the question 'does it work at other sizes / on
fainter watermarks?' you'd want answered before shipping the tool to
users feeding in arbitrary stock photos.

The tests are slow and skipped without LaMa weights / classifier
weights, so they don't run in plain CI.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("BC_WMR_DEVICE", "cpu")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

FIXTURES = Path(__file__).parent / "fixtures"
WATERMARKED = FIXTURES / "dreamstime_18829755_watermarked.jpg"
REFERENCE = FIXTURES / "dreamstime_18829755_reference.jpg"
GT_MASK = FIXTURES / "dreamstime_18829755_groundtruth_mask.png"
WEIGHTS = ROOT / "weights" / "big-lama.pt"


def _resize(img: np.ndarray, scale: float, interp=cv2.INTER_AREA) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)


def _detect_metrics(image_bgr: np.ndarray, gt: np.ndarray) -> dict:
    """Run the detector and return mask-quality stats vs the resized GT."""
    from services.detector_service import detect_split

    tmp = ROOT / "tmp" / "robustness_in.png"
    tmp.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(tmp), image_bgr)

    t0 = time.time()
    result = detect_split(str(tmp), mode="auto")
    elapsed = time.time() - t0

    if result is None:
        return {"detected": False, "elapsed_s": elapsed}
    body, strip, p_full = result
    ours = cv2.imread(body, cv2.IMREAD_GRAYSCALE)
    if strip is not None:
        ours = cv2.bitwise_or(ours, cv2.imread(strip, cv2.IMREAD_GRAYSCALE))
    if ours.shape != gt.shape:
        gt = cv2.resize(gt, (ours.shape[1], ours.shape[0]), interpolation=cv2.INTER_NEAREST)
    ours_b = ours > 127
    gt_b = gt > 127
    inter = np.logical_and(ours_b, gt_b).sum()
    union = np.logical_or(ours_b, gt_b).sum()
    return {
        "detected": True,
        "elapsed_s": elapsed,
        "p_full": float(p_full),
        "coverage": float(ours_b.mean()),
        "iou": float(inter / max(1, union)),
        "recall": float(inter / max(1, gt_b.sum())),
        "precision": float(inter / max(1, ours_b.sum())),
    }


def _synthetic_alpha_blend(clean: np.ndarray, watermarked: np.ndarray, alpha: float) -> np.ndarray:
    """Reconstruct a watermark layer from (watermarked - clean) and re-blend
    onto clean at the given alpha. alpha=1.0 -> identical to original
    watermarked image. alpha=0.3 -> faint overlay."""
    diff = watermarked.astype(np.int16) - clean.astype(np.int16)
    blended = clean.astype(np.float32) + alpha * diff.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ─── Size sweep ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(not WEIGHTS.exists(), reason="LaMa weights not downloaded")
@pytest.mark.parametrize("scale", [0.25, 0.50, 0.75, 1.0, 1.5])
def test_detector_size_sweep(scale: float):
    """Detection must succeed and IoU stay reasonable at 0.25x..1.5x scale."""
    img = cv2.imread(str(WATERMARKED))
    gt = cv2.imread(str(GT_MASK), cv2.IMREAD_GRAYSCALE)
    img_s = _resize(img, scale)
    gt_s = _resize(gt, scale, cv2.INTER_NEAREST)

    m = _detect_metrics(img_s, gt_s)
    print(
        f"\n  scale={scale:.2f}  "
        f"size={img_s.shape[1]}x{img_s.shape[0]}  "
        f"{'detected' if m.get('detected') else 'NO-DETECT'}"
        f"  p={m.get('p_full', 0):.2f}  "
        f"recall={m.get('recall', 0):.2f}  "
        f"iou={m.get('iou', 0):.2f}  "
        f"in {m['elapsed_s']:.1f}s"
    )

    assert m["detected"], f"detector failed at scale {scale}"
    # Loose floors that allow degradation at extreme sizes but flag total failure.
    if 0.5 <= scale <= 1.5:
        assert m["recall"] >= 0.45, f"recall {m['recall']:.2f} too low at scale {scale}"


# ─── Alpha sweep ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(not WEIGHTS.exists() or not REFERENCE.exists(),
                    reason="missing fixtures")
@pytest.mark.parametrize("alpha", [0.30, 0.50, 0.70, 1.00])
def test_detector_alpha_sweep(alpha: float):
    """Synthetic transparency sweep — same watermark layout, varying opacity.

    alpha=1.0 reproduces the original; lower values fade the watermark
    proportionally. Below ~0.3 most pixel-statistic detectors fail; we
    want to know exactly where ours starts to drop off.
    """
    watermarked = cv2.imread(str(WATERMARKED))
    clean = cv2.imread(str(REFERENCE))
    gt = cv2.imread(str(GT_MASK), cv2.IMREAD_GRAYSCALE)

    # SaaS reference is occasionally a slightly different resolution
    if clean.shape[:2] != watermarked.shape[:2]:
        clean = cv2.resize(clean, (watermarked.shape[1], watermarked.shape[0]))

    blended = _synthetic_alpha_blend(clean, watermarked, alpha)
    m = _detect_metrics(blended, gt)
    print(
        f"\n  alpha={alpha:.2f}  "
        f"{'detected' if m.get('detected') else 'NO-DETECT'}"
        f"  p={m.get('p_full', 0):.2f}  "
        f"recall={m.get('recall', 0):.2f}  "
        f"iou={m.get('iou', 0):.2f}  "
        f"in {m['elapsed_s']:.1f}s"
    )

    if alpha >= 0.5:
        assert m["detected"], f"detector failed at alpha {alpha}"
        assert m["recall"] >= 0.40, f"recall {m['recall']:.2f} below floor at alpha {alpha}"
    # alpha < 0.5: don't enforce — we want the data point in the printout, not a hard fail.
