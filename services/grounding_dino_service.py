"""Grounding DINO box-prompt detector for tiled watermarks.

Empirically (see tools/bench_gd_hybrid.py) this is the only zero-shot
text-prompted detector tested so far that handles multi-instance
tiled stock-photo watermarks. SAM 2 and Florence-2 both treat
"watermark" as a single object and return one detection.

GD on its own returns large bounding boxes covering the watermarked
region with ~97% recall but only ~25% precision (boxes include
plenty of subject content). The high-quality mask comes from
intersecting GD's bounding-box union with the existing
low-saturation/high-pass pixel heuristic — that combination delivers
IoU 0.31 vs 0.25 from the current ConvNeXt + Grad-CAM path.

Public function:
    detect_box_mask(image_bgr, device, prompt) -> np.ndarray (H, W) uint8

Returns the union of detected boxes as a binary mask. Caller is
responsible for AND-ing with a pixel-level filter — that's what makes
the result useful, and we don't want to bake that decision in here so
detect_split.py can choose its own refinement strategy.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

MODEL_ID = "IDEA-Research/grounding-dino-base"

# Empirically the strongest tiled-watermark prompt on the canonical
# fixture: hits all 14 instance-clusters at threshold 0.25 / 0.20.
DEFAULT_PROMPT = "watermark. logo. text."
DEFAULT_BOX_T = 0.25
DEFAULT_TEXT_T = 0.20

_pipe_cache: dict = {}


def is_available() -> tuple[bool, str]:
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection  # noqa: F401
    except ImportError as e:
        return False, f"transformers Grounding DINO classes unavailable: {e}"
    return True, "ok"


def _load(device: torch.device):
    key = str(device)
    if key in _pipe_cache:
        return _pipe_cache[key]
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID, torch_dtype=dtype, cache_dir=str(WEIGHTS_DIR),
    ).to(device).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=str(WEIGHTS_DIR))
    _pipe_cache[key] = (model, proc, dtype)
    logger.info("grounding-dino loaded on %s", device)
    return _pipe_cache[key]


@torch.inference_mode()
def detect_box_mask(
    image_bgr: np.ndarray,
    device: torch.device,
    prompt: str = DEFAULT_PROMPT,
    box_threshold: float = DEFAULT_BOX_T,
    text_threshold: float = DEFAULT_TEXT_T,
) -> np.ndarray:
    """Return a binary mask (uint8, H×W) covering the union of every
    detected box. Empty if no detection."""
    h, w = image_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    model, proc, dtype = _load(device)

    inputs = proc(images=pil, text=prompt, return_tensors="pt").to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=torch.tensor([[h, w]], device=device),
    )[0]

    boxes = res.get("boxes")
    n = 0 if boxes is None else len(boxes)
    mask = np.zeros((h, w), dtype=np.uint8)
    if n == 0:
        logger.info("grounding-dino: 0 detections for prompt %r", prompt)
        return mask
    for box in boxes.cpu().numpy():
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        cv2.rectangle(
            mask,
            (max(0, x1), max(0, y1)),
            (min(w - 1, x2), min(h - 1, y2)),
            255, -1,
        )
    logger.info("grounding-dino: %d detections, coverage %.3f",
                n, float((mask > 0).mean()))
    return mask
