"""Iteration harness: produce a result + crops for visual inspection.

Usage:
    python tools/iterate.py [iter_label]

Outputs everything under tmp/iter/<label>/:
    metrics.txt        - PSNR/SSIM/IoU/Recall/Precision
    result.png         - full pipeline output
    overlay.png        - GT vs ours (green/red/blue)
    crop_chest.png     - centred on the chest swirl region
    crop_hands.png     - left hand area
    crop_topright.png  - top-right corner where text often survives
    crop_botright.png  - bottom-right copyright/ID stamp
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("BC_WMR_DEVICE", "cpu")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

FIXTURES = ROOT / "tests" / "fixtures"
WATERMARKED = FIXTURES / "dreamstime_18829755_watermarked.jpg"
REFERENCE = FIXTURES / "dreamstime_18829755_reference.jpg"
GT_MASK = FIXTURES / "dreamstime_18829755_groundtruth_mask.png"


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    K1, K2, L = 0.01, 0.03, 255
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    win, sigma = 11, 1.5
    mu_a = cv2.GaussianBlur(a, (win, win), sigma)
    mu_b = cv2.GaussianBlur(b, (win, win), sigma)
    sa2 = cv2.GaussianBlur(a*a, (win,win), sigma) - mu_a*mu_a
    sb2 = cv2.GaussianBlur(b*b, (win,win), sigma) - mu_b*mu_b
    sab = cv2.GaussianBlur(a*b, (win,win), sigma) - mu_a*mu_b
    num = (2*mu_a*mu_b + C1) * (2*sab + C2)
    den = (mu_a*mu_a + mu_b*mu_b + C1) * (sa2 + sb2 + C2)
    return float(np.mean(num / den))


def _ssim_rgb(a, b):
    return float(np.mean([_ssim(a[..., c], b[..., c]) for c in range(3)]))


def _crop(img, cx, cy, size):
    h, w = img.shape[:2]
    half = size // 2
    x0 = max(0, cx - half); y0 = max(0, cy - half)
    x1 = min(w, x0 + size); y1 = min(h, y0 + size)
    return img[y0:y1, x0:x1]


def main(label: str = "current"):
    out_dir = ROOT / "tmp" / "iter" / label
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    from services.detector_service import detect_split, prefill_strip
    from services.lama_service import inpaint, load_model
    from services.cuda_policy import get_device

    src = cv2.imread(str(WATERMARKED))
    ref = cv2.imread(str(REFERENCE))
    gt = cv2.imread(str(GT_MASK), cv2.IMREAD_GRAYSCALE)
    if ref.shape[:2] != src.shape[:2]:
        ref = cv2.resize(ref, (src.shape[1], src.shape[0]))

    t0 = time.time()
    result = detect_split(str(WATERMARKED), mode="auto")
    if result is None:
        print("DETECT FAILED")
        return 1
    body, strip, p_full = result

    image_for_lama = str(WATERMARKED)
    if strip is not None:
        image_for_lama = prefill_strip(str(WATERMARKED), strip, 4)

    model = load_model(get_device())
    final = inpaint(image_for_lama, body, get_device(), model)
    elapsed = time.time() - t0
    shutil.copy(final, out_dir / "result.png")

    ours = cv2.imread(final)
    ours_body = ours[: int(ours.shape[0] * 0.86)]
    ref_body = ref[: int(ref.shape[0] * 0.86)]
    psnr = float(cv2.PSNR(ref_body, ours_body))
    ssim = _ssim_rgb(ref_body, ours_body)

    body_m = cv2.imread(body, cv2.IMREAD_GRAYSCALE)
    if strip:
        body_m = cv2.bitwise_or(body_m, cv2.imread(strip, cv2.IMREAD_GRAYSCALE))
    if body_m.shape != gt.shape:
        body_m = cv2.resize(body_m, (gt.shape[1], gt.shape[0]), cv2.INTER_NEAREST)
    om = body_m > 127; gm = gt > 127
    inter = np.logical_and(om, gm).sum()
    union = np.logical_or(om, gm).sum()
    iou = inter / max(1, union)
    recall = inter / max(1, gm.sum())
    precision = inter / max(1, om.sum())

    metrics = (
        f"label: {label}\n"
        f"elapsed: {elapsed:.1f}s\n"
        f"PSNR vs SaaS: {psnr:.2f} dB\n"
        f"SSIM vs SaaS: {ssim:.3f}\n"
        f"Recall vs GT mask: {recall:.3f}\n"
        f"Precision vs GT mask: {precision:.3f}\n"
        f"IoU vs GT mask: {iou:.3f}\n"
        f"Coverage of our mask: {om.mean():.3f}\n"
        f"Coverage of GT mask: {gm.mean():.3f}\n"
    )
    (out_dir / "metrics.txt").write_text(metrics)
    print(metrics)

    # Overlay (green=on-target, red=missed, blue=over-mask)
    overlay = (cv2.resize(src, (gt.shape[1], gt.shape[0])) * 0.4).astype(np.uint8)
    overlay[gm & ~om] = (0, 0, 200)
    overlay[~gm & om] = (200, 100, 0)
    overlay[gm & om] = (0, 200, 0)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)

    # Crops centred on known failure regions of the canonical 1600x1157
    crops = {
        "chest":    (790, 510, 380),
        "hands":    (1200, 870, 380),
        "topright": (1300, 220, 380),
        "botright": (1500, 1080, 200),
    }
    for name, (cx, cy, size) in crops.items():
        side_by_side = np.hstack([
            _crop(src,   cx, cy, size),
            _crop(ours,  cx, cy, size),
            _crop(ref,   cx, cy, size),
        ])
        cv2.imwrite(str(out_dir / f"crop_{name}.png"), side_by_side)

    return 0


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "current"
    sys.exit(main(label))
