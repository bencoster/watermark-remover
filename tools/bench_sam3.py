"""Benchmark SAM 3.1 vs current and Grounding-DINO+heuristic.

Run this from the .venv-sam3 venv:

    .venv-sam3\\Scripts\\python.exe tools/bench_sam3.py

Requires HF auth (`hf auth login`) with a token that has access to
facebook/sam3.1 (gated=manual — visit huggingface.co/facebook/sam3.1
and click 'Agree and access repository' first).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from PIL import Image

FIXTURES = ROOT / "tests" / "fixtures"
WATERMARKED = FIXTURES / "dreamstime_18829755_watermarked.jpg"
GT_MASK = FIXTURES / "dreamstime_18829755_groundtruth_mask.png"
OUT = ROOT / "tmp" / "sam3_bench"

PROMPTS = [
    "watermark.",
    "logo.",
    "text.",
    "watermark. logo. text.",
    "watermark logo text overlay.",
    "diagonal watermark text.",
    "tiled stock photo watermark.",
]


def metrics(o, g):
    if o.shape != g.shape:
        o = cv2.resize(o, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    ob = o > 127; gb = g > 127
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
    ob = mask > 127; gb = gt > 127
    bg[gb & ~ob] = (0, 0, 200)
    bg[~gb & ob] = (200, 100, 0)
    bg[gb & ob] = (0, 200, 0)
    cv2.imwrite(str(out_path), bg)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = cv2.imread(str(WATERMARKED))
    gt = cv2.imread(str(GT_MASK), cv2.IMREAD_GRAYSCALE)
    pil = Image.open(str(WATERMARKED)).convert("RGB")

    print("Loading SAM 3.1 (facebook/sam3.1) ...")
    from transformers import Sam3Model, Sam3Processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"  device={device}, dtype={dtype}")
    t0 = time.time()
    processor = Sam3Processor.from_pretrained("facebook/sam3.1")
    model = Sam3Model.from_pretrained("facebook/sam3.1", torch_dtype=dtype).to(device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    rows = []
    for prompt in PROMPTS:
        print(f"\n  prompt: {prompt!r}")
        try:
            inputs = processor(images=pil, text=prompt, return_tensors="pt").to(device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            t0 = time.time()
            with torch.inference_mode():
                outputs = model(**inputs)
            elapsed = time.time() - t0
            results = processor.post_process_instance_segmentation(
                outputs, threshold=0.30, mask_threshold=0.50,
                target_sizes=torch.tensor([[pil.height, pil.width]], device=device),
            )[0]
            masks_t = results.get("masks")
            n = 0 if masks_t is None else len(masks_t)
            mask = np.zeros((pil.height, pil.width), dtype=np.uint8)
            for m in masks_t:
                if hasattr(m, "cpu"):
                    m = m.cpu().numpy()
                mask |= (m > 0).astype(np.uint8) * 255
            r = metrics(mask, gt)
            print(f"    detections: {n}, elapsed: {elapsed:.2f}s")
            print(f"    recall {r[0]:.3f}  precision {r[1]:.3f}  iou {r[2]:.3f}  cov {r[3]:.3f}")
            rows.append({"prompt": prompt, "n": n, "t": elapsed,
                         "recall": r[0], "precision": r[1], "iou": r[2], "cov": r[3]})
            slug = prompt.replace(" ", "_").replace(".", "").replace(",", "")[:40]
            cv2.imwrite(str(OUT / f"sam3_{slug}_mask.png"), mask)
            overlay(src, mask, gt, OUT / f"sam3_{slug}_overlay.png")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print("\n=== Summary ===")
    print(f"{'prompt':<40} {'n':>4} {'recall':>6} {'precis':>6} {'iou':>6} {'cov':>6} {'t':>5}")
    for r in rows:
        print(f"{r['prompt']:<40} {r['n']:>4d} {r['recall']:>6.3f} "
              f"{r['precision']:>6.3f} {r['iou']:>6.3f} {r['cov']:>6.3f} {r['t']:>5.1f}s")

    print("\nReference baselines on this fixture:")
    print("  Current  (ConvNeXt + Grad-CAM)    IoU 0.248  recall 0.637")
    print("  GD ∩ pixel + dilate (live)        IoU 0.310  recall 0.549")


if __name__ == "__main__":
    main()
