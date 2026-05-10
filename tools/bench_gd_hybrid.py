"""Test the GD-as-spatial-prior + pixel-heuristic hybrid.

GD has near-perfect recall (~97%) on tiled stock-photo watermarks but
loose boxes give it 25% precision. Our existing low-sat/high-pass
pixel heuristic has high pixel precision but only fires where edges
exist. ANDing them should give high recall AND high precision.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("BC_WMR_DEVICE", "cpu")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from services.detector_service import _heuristic_pixel_mask

IMG = ROOT / "tests" / "fixtures" / "dreamstime_18829755_watermarked.jpg"
GT = ROOT / "tests" / "fixtures" / "dreamstime_18829755_groundtruth_mask.png"
OUT = ROOT / "tmp" / "gd_bench"


def metrics(o: np.ndarray, g: np.ndarray) -> tuple[float, float, float, float]:
    if o.shape != g.shape:
        o = cv2.resize(o, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    ob = o > 127
    gb = g > 127
    inter = np.logical_and(ob, gb).sum()
    union = np.logical_or(ob, gb).sum()
    return (
        float(inter / max(1, gb.sum())),
        float(inter / max(1, ob.sum())),
        float(inter / max(1, union)),
        float(ob.mean()),
    )


def overlay(src, mask, gt, out_path):
    if mask.shape != gt.shape:
        mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    if src.shape[:2] != gt.shape:
        src = cv2.resize(src, (gt.shape[1], gt.shape[0]))
    bg = (src * 0.4).astype(np.uint8)
    ob = mask > 127
    gb = gt > 127
    bg[gb & ~ob] = (0, 0, 200)
    bg[~gb & ob] = (200, 100, 0)
    bg[gb & ob] = (0, 200, 0)
    cv2.imwrite(str(out_path), bg)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = cv2.imread(str(IMG))
    gt = cv2.imread(str(GT), cv2.IMREAD_GRAYSCALE)
    pil = Image.open(str(IMG)).convert("RGB")
    H, W = src.shape[:2]

    print("Loading Grounding DINO...")
    device = torch.device("cpu")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    ).to(device).eval()
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")

    inputs = proc(images=pil, text="watermark. logo. text.", return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, inputs["input_ids"], threshold=0.25, text_threshold=0.20,
        target_sizes=torch.tensor([[pil.height, pil.width]]),
    )[0]

    gd_box_mask = np.zeros((H, W), dtype=np.uint8)
    for box in res["boxes"].cpu().numpy():
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        cv2.rectangle(
            gd_box_mask,
            (max(0, x1), max(0, y1)),
            (min(W - 1, x2), min(H - 1, y2)),
            255, -1,
        )

    pixel = _heuristic_pixel_mask(src)
    hybrid = cv2.bitwise_and(gd_box_mask, pixel)
    hybrid_dilated = cv2.dilate(
        hybrid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    hybrid_dilated_more = cv2.dilate(
        hybrid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )

    fmt = "{:<32} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f}"
    print()
    print(f"{'name':<32} {'recall':>6} {'prec':>6} {'iou':>6} {'cov':>6}")
    print("-" * 70)
    for name, mask in [
        ("GD boxes raw", gd_box_mask),
        ("pixel heuristic only", pixel),
        ("GD AND pixel", hybrid),
        ("GD AND pixel + 5x5 dilate", hybrid_dilated),
        ("GD AND pixel + 7x7 dilate x2", hybrid_dilated_more),
    ]:
        print(fmt.format(name, *metrics(mask, gt)))

    cv2.imwrite(str(OUT / "hybrid_mask.png"), hybrid_dilated)
    overlay(src, hybrid_dilated, gt, OUT / "hybrid_overlay.png")
    print()
    print(f"Saved tmp/gd_bench/hybrid_mask.png + hybrid_overlay.png")


if __name__ == "__main__":
    main()
