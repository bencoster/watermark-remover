"""Benchmark Florence-2 vs the current ConvNeXt+Grad-CAM detector.

Usage:
    python tools/bench_florence2.py

Prints recall / precision / IoU vs the GT mask for both detectors,
plus elapsed times. Saves debug overlays under tmp/florence_bench/.
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
OUT = ROOT / "tmp" / "florence_bench"


def metrics(ours: np.ndarray, gt: np.ndarray) -> dict:
    if ours.shape != gt.shape:
        ours = cv2.resize(ours, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    ob = ours > 127
    gb = gt > 127
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
    ob = ours > 127
    gb = gt > 127
    bg[gb & ~ob] = (0, 0, 200)
    bg[~gb & ob] = (200, 100, 0)
    bg[gb & ob] = (0, 200, 0)
    cv2.imwrite(str(out_path), bg)


# ─── 1. Current detector (ConvNeXt + Grad-CAM) ────────────────────────────────

def bench_current() -> tuple[np.ndarray, dict, float]:
    from services.detector_service import detect_split

    t0 = time.time()
    result = detect_split(str(WATERMARKED), mode="auto")
    elapsed = time.time() - t0
    if result is None:
        return None, {"error": "no detection"}, elapsed
    body, strip, _ = result
    mask = cv2.imread(body, cv2.IMREAD_GRAYSCALE)
    if strip is not None:
        mask = cv2.bitwise_or(mask, cv2.imread(strip, cv2.IMREAD_GRAYSCALE))
    return mask, metrics(mask, cv2.imread(str(GT_MASK), cv2.IMREAD_GRAYSCALE)), elapsed


# ─── 2. Florence-2 detector ───────────────────────────────────────────────────

def bench_florence(model_size: str = "base", task_prompt: str = "<OPEN_VOCABULARY_DETECTION>",
                   text_prompt: str = "watermark") -> tuple[np.ndarray, dict, float]:
    """Florence-2 with open-vocabulary detection. Returns boxes for
    each watermark instance; we paint each box into a binary mask."""
    from transformers import AutoProcessor, AutoModelForCausalLM
    # Florence-2's custom modeling predates _supports_sdpa; patch the
    # base class so transformers 4.5x doesn't trip during generate.
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "_supports_sdpa"):
        PreTrainedModel._supports_sdpa = False

    # Florence-2's prepare_inputs_for_generation also assumes
    # past_key_values[0][0] is always a tensor. transformers 4.57+ passes
    # None on the first decode call. Wrap the method to coerce None into
    # an empty placeholder so .shape[2] returns 0.
    def _patch_prepare(model_obj):
        lm = getattr(model_obj, "language_model", model_obj)
        original = lm.prepare_inputs_for_generation

        def patched(*args, **kwargs):
            pkv = kwargs.get("past_key_values")
            if pkv is not None:
                # Ensure pkv[0][0] has .shape; if it doesn't, fake one
                try:
                    _ = pkv[0][0].shape[2]
                except Exception:
                    kwargs["past_key_values"] = None
            return original(*args, **kwargs)

        lm.prepare_inputs_for_generation = patched

    model_id = f"microsoft/Florence-2-{model_size}"
    print(f"  loading {model_id} ...")
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager",  # avoid sdpa code path entirely
    ).to(device).eval()
    # Belt-and-braces in case the instance lacks the attribute too.
    if not hasattr(model, "_supports_sdpa"):
        model._supports_sdpa = False
    _patch_prepare(model)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    load_elapsed = time.time() - t0
    print(f"  loaded in {load_elapsed:.1f}s")

    pil = Image.open(str(WATERMARKED)).convert("RGB")
    full_prompt = task_prompt + text_prompt
    print(f"  prompt: {full_prompt!r}")

    inputs = processor(text=full_prompt, images=pil, return_tensors="pt")
    # Move tensors to device with appropriate dtype
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device, dtype)
    t0 = time.time()
    with torch.inference_mode():
        gen = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=1024,
            num_beams=1,           # transformers 4.57 + Florence-2 beam search incompat
            do_sample=False,
        )
    text = processor.batch_decode(gen, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=task_prompt, image_size=(pil.width, pil.height),
    )
    elapsed = time.time() - t0

    # Pull bboxes from whichever key Florence used
    print(f"  raw response keys: {list(parsed.get(task_prompt, {}).keys())}")
    bboxes = parsed.get(task_prompt, {}).get("bboxes") \
             or parsed.get(task_prompt, {}).get("quad_boxes") \
             or []
    print(f"  detections: {len(bboxes)}")

    mask = np.zeros((pil.height, pil.width), dtype=np.uint8)
    for box in bboxes:
        x1, y1, x2, y2 = [int(round(c)) for c in box[:4]]
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask, metrics(mask, cv2.imread(str(GT_MASK), cv2.IMREAD_GRAYSCALE)), elapsed


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = cv2.imread(str(WATERMARKED))
    gt = cv2.imread(str(GT_MASK), cv2.IMREAD_GRAYSCALE)

    print("\n=== Current detector (ConvNeXt + Grad-CAM) ===")
    mask, m, t = bench_current()
    print(f"  elapsed: {t:.1f}s")
    print(f"  recall {m['recall']:.3f}  precision {m['precision']:.3f}  "
          f"iou {m['iou']:.3f}  coverage {m['coverage']:.3f}")
    cv2.imwrite(str(OUT / "current_mask.png"), mask)
    overlay(src, mask, gt, OUT / "current_overlay.png")

    for size in ["base", "large"]:
        for task in ["<OPEN_VOCABULARY_DETECTION>", "<REFERRING_EXPRESSION_SEGMENTATION>"]:
            label = f"florence2-{size}-{task.strip('<>')}"
            print(f"\n=== Florence-2 {size}, task={task} ===")
            try:
                mask, m, t = bench_florence(model_size=size, task_prompt=task,
                                            text_prompt="watermark")
                print(f"  elapsed: {t:.1f}s")
                print(f"  recall {m['recall']:.3f}  precision {m['precision']:.3f}  "
                      f"iou {m['iou']:.3f}  coverage {m['coverage']:.3f}")
                cv2.imwrite(str(OUT / f"{label}_mask.png"), mask)
                overlay(src, mask, gt, OUT / f"{label}_overlay.png")
            except Exception as e:
                import traceback
                print(f"  FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
