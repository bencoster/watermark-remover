"""Benchmark Grounding DINO vs the current detector.

Grounding DINO is text-prompted *and* multi-instance — that's the
architectural property that defeated SAM 2 (one mask per point prompt)
and Florence-2 (one box per concept). On a tiled-watermark image we
care whether GD can find ~50 instances from one text prompt.

Usage:
    python tools/bench_grounding_dino.py [box_threshold] [text_threshold]

Tries several prompt phrasings so we can see which works best on
this watermark style. Saves overlays under tmp/gd_bench/.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("BC_WMR_DEVICE", "cpu")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from PIL import Image

FIXTURES = ROOT / "tests" / "fixtures"
WATERMARKED = FIXTURES / "dreamstime_18829755_watermarked.jpg"
GT_MASK = FIXTURES / "dreamstime_18829755_groundtruth_mask.png"
OUT = ROOT / "tmp" / "gd_bench"

# Prompts to try. Grounding DINO expects period-separated phrases for
# multiple concepts. Lowercase + ending in '.' is the format the
# canonical examples use.
PROMPTS = [
    "watermark.",
    "logo.",
    "text.",
    "watermark. logo. text.",
    "diagonal watermark text.",
    "transparent watermark logo.",
    "small circular logo.",
]

# Default thresholds — Grounding DINO uses two:
#   box_threshold    — confidence the detected region IS the concept
#   text_threshold   — confidence the text query matches the region
DEFAULT_BOX_T = 0.25
DEFAULT_TEXT_T = 0.20


def metrics(ours: np.ndarray, gt: np.ndarray) -> dict:
    if ours.shape != gt.shape:
        ours = cv2.resize(ours, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    ob = ours > 127; gb = gt > 127
    inter = np.logical_and(ob, gb).sum()
    union = np.logical_or(ob, gb).sum()
    return {
        "recall": float(inter / max(1, gb.sum())),
        "precision": float(inter / max(1, ob.sum())),
        "iou": float(inter / max(1, union)),
        "coverage": float(ob.mean()),
    }


def overlay(image_bgr: np.ndarray, ours: np.ndarray, gt: np.ndarray, out_path: Path):
    if ours.shape != gt.shape:
        ours = cv2.resize(ours, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    src = cv2.resize(image_bgr, (gt.shape[1], gt.shape[0]))
    bg = (src * 0.4).astype(np.uint8)
    ob = ours > 127; gb = gt > 127
    bg[gb & ~ob] = (0, 0, 200)
    bg[~gb & ob] = (200, 100, 0)
    bg[gb & ob] = (0, 200, 0)
    cv2.imwrite(str(out_path), bg)


def boxes_to_mask(boxes, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        cv2.rectangle(mask, (max(0, x1), max(0, y1)),
                      (min(img_w - 1, x2), min(img_h - 1, y2)), 255, -1)
    return mask


def bench_grounding_dino(prompt: str, box_t: float, text_t: float, model, processor,
                         device, dtype, pil) -> dict:
    inputs = processor(images=pil, text=prompt, return_tensors="pt").to(device)
    # Cast pixel tensor only — text inputs stay as their native dtype
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    t0 = time.time()
    with torch.inference_mode():
        outputs = model(**inputs)
    elapsed = time.time() - t0

    target_size = torch.tensor([[pil.height, pil.width]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_t,
        text_threshold=text_t,
        target_sizes=target_size,
    )[0]

    boxes = results.get("boxes")
    scores = results.get("scores")
    labels = results.get("labels", [])
    nbox = 0 if boxes is None else len(boxes)
    return {
        "prompt": prompt,
        "elapsed": elapsed,
        "n_detections": nbox,
        "boxes": boxes.cpu().numpy() if boxes is not None and nbox else np.zeros((0, 4)),
        "scores": scores.cpu().numpy() if scores is not None and nbox else np.zeros((0,)),
        "labels": labels,
    }


def main(box_t: float = DEFAULT_BOX_T, text_t: float = DEFAULT_TEXT_T):
    OUT.mkdir(parents=True, exist_ok=True)
    src = cv2.imread(str(WATERMARKED))
    gt = cv2.imread(str(GT_MASK), cv2.IMREAD_GRAYSCALE)

    print("\n=== Current detector (ConvNeXt + Grad-CAM) ===")
    from services.detector_service import detect_split
    t0 = time.time()
    result = detect_split(str(WATERMARKED), mode="auto")
    cur_elapsed = time.time() - t0
    if result is not None:
        body, strip, _ = result
        cur = cv2.imread(body, cv2.IMREAD_GRAYSCALE)
        if strip is not None:
            cur = cv2.bitwise_or(cur, cv2.imread(strip, cv2.IMREAD_GRAYSCALE))
        m = metrics(cur, gt)
        print(f"  elapsed: {cur_elapsed:.1f}s")
        print(f"  recall {m['recall']:.3f}  precision {m['precision']:.3f}  "
              f"iou {m['iou']:.3f}  coverage {m['coverage']:.3f}")
        cv2.imwrite(str(OUT / "current_mask.png"), cur)
        overlay(src, cur, gt, OUT / "current_overlay.png")

    print("\n=== Grounding DINO base ===")
    print(f"  box_threshold={box_t}, text_threshold={text_t}")
    print("  loading IDEA-Research/grounding-dino-base ...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    t0 = time.time()
    pref = os.environ.get("BC_WMR_DEVICE", "auto").lower()
    if pref == "cpu":
        device = torch.device("cpu")
    elif pref.startswith("cuda"):
        device = torch.device(pref)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base", torch_dtype=dtype,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    print(f"  loaded in {time.time() - t0:.1f}s")

    pil = Image.open(str(WATERMARKED)).convert("RGB")

    rows = []
    for prompt in PROMPTS:
        print(f"\n  prompt: {prompt!r}")
        try:
            r = bench_grounding_dino(prompt, box_t, text_t, model, processor,
                                     device, dtype, pil)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            continue

        mask = boxes_to_mask(r["boxes"], pil.height, pil.width)
        m = metrics(mask, gt)
        print(f"    detections: {r['n_detections']}, elapsed: {r['elapsed']:.2f}s")
        print(f"    recall {m['recall']:.3f}  precision {m['precision']:.3f}  "
              f"iou {m['iou']:.3f}  coverage {m['coverage']:.3f}")
        rows.append({"prompt": prompt, **m, "n": r["n_detections"], "t": r["elapsed"]})
        slug = prompt.replace(" ", "_").replace(".", "").replace(",", "")[:40]
        cv2.imwrite(str(OUT / f"gd_{slug}_mask.png"), mask)
        overlay(src, mask, gt, OUT / f"gd_{slug}_overlay.png")

    print("\n=== Summary ===")
    print(f"{'prompt':<32} {'detect':>6} {'recall':>6} {'precis':>6} {'iou':>6} {'cov':>6} {'time':>6}")
    for r in rows:
        print(f"{r['prompt']:<32} {r['n']:>6d} {r['recall']:>6.3f} "
              f"{r['precision']:>6.3f} {r['iou']:>6.3f} {r['coverage']:>6.3f} "
              f"{r['t']:>6.1f}")


if __name__ == "__main__":
    box_t = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BOX_T
    text_t = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TEXT_T
    main(box_t, text_t)
