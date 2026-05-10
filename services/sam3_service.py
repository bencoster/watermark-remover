"""SAM 3 multi-instance text-prompted detector for tiled watermarks.

Empirically (tools/bench_sam3.py) SAM 3 with two complementary prompts
beats every other detector tested on the canonical Dreamstime fixture:

                                    Recall  Prec   IoU    Cov
  ConvNeXt + Grad-CAM (baseline)    0.637   0.289  0.248  0.533
  GD ∩ pixel + dilate               0.549   0.416  0.310  0.321
  SAM 3 hybrid (this module)        0.507   0.502  0.337  0.245

Two prompts run together:
  * `"watermark."` → ~97 tight per-instance masks (precision 0.81)
  * `"tiled stock photo watermark."` → ~38 loose boxes (recall 0.99)

Combined as `precise OR (loose ∩ pixel_heuristic) + 3×3 dilate ×1`:
  - Precise per-instance masks land tight pixel boundaries
  - Loose ∩ pixel adds coverage SAM missed without the over-mask of
    the loose boxes alone (the AND with low-saturation/edge masks
    drops huge chunks of subject content the loose box swallowed)
  - 3×3 dilate ×1 catches anti-aliased mask edges

Requires:
  - facebook/sam3 from HF (Apache-2.0-ish; gated=manual but the user
    has cleared it). Note: facebook/sam3.1 ships only a custom .pt
    that needs the github sam3 package; the safetensors checkpoint
    in facebook/sam3 is what transformers.Sam3Model loads natively.
  - transformers >= 5.0 (for native Sam3Model + Sam3Processor)

Lazy-loaded; swappable per device.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
DEFAULT_REPO = "facebook/sam3"
PRECISE_PROMPT = "watermark."
LOOSE_PROMPT = "tiled stock photo watermark."
SCORE_THRESHOLD = 0.05
MASK_THRESHOLD = 0.50

_pipe_cache: dict = {}


def is_available() -> tuple[bool, str]:
    try:
        from transformers import Sam3Model, Sam3Processor  # noqa: F401
    except ImportError as e:
        return False, f"transformers Sam3Model unavailable (need transformers>=5.0): {e}"
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(DEFAULT_REPO, token=True)
    except Exception as e:
        return False, (
            f"can't reach {DEFAULT_REPO} ({type(e).__name__}); "
            "set HF_TOKEN env var or `hf auth login` with a token "
            "that has read access to the repo."
        )
    return True, "ok"


def _load(device: torch.device):
    key = str(device)
    if key in _pipe_cache:
        return _pipe_cache[key]
    from transformers import Sam3Model, Sam3Processor
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    repo = os.environ.get("SAM3_REPO", DEFAULT_REPO)
    proc = Sam3Processor.from_pretrained(repo)
    model = Sam3Model.from_pretrained(repo, torch_dtype=dtype).to(device).eval()
    _pipe_cache[key] = (model, proc, dtype)
    logger.info("sam3 loaded on %s from %s", device, repo)
    return _pipe_cache[key]


@torch.inference_mode()
def _union_masks_for_prompt(
    pil: Image.Image, prompt: str, device: torch.device,
    score_threshold: float = SCORE_THRESHOLD,
    mask_threshold: float = MASK_THRESHOLD,
) -> tuple[np.ndarray, int]:
    model, proc, dtype = _load(device)
    inputs = proc(images=pil, text=prompt, return_tensors="pt").to(device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    outputs = model(**inputs)
    # target_sizes must be a list of (h, w) tuples — passing a tensor
    # triggers a torch upsample_bilinear2d type error in tx 5.x.
    res = proc.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold, mask_threshold=mask_threshold,
        target_sizes=[(pil.height, pil.width)],
    )[0]
    masks = res.get("masks")
    union = np.zeros((pil.height, pil.width), dtype=np.uint8)
    n = 0
    if masks is not None:
        for m in masks:
            if hasattr(m, "cpu"):
                m = m.cpu().numpy()
            union |= (m > 0).astype(np.uint8) * 255
            n += 1
    return union, n


def detect_hybrid_mask(
    image_bgr: np.ndarray,
    device: torch.device,
    pixel_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run the bench-best SAM 3 hybrid: precise OR (loose AND pixel) + dilate.

    The pixel_mask is the existing low-saturation/high-pass heuristic.
    Without it we still combine precise OR loose, but the result
    over-masks subject content (loose alone is 0.957 coverage). We
    fall back to a quick local heuristic if the caller doesn't pass one.
    """
    if pixel_mask is None:
        from services.detector_service import _heuristic_pixel_mask
        pixel_mask = _heuristic_pixel_mask(image_bgr)

    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    precise, n_p = _union_masks_for_prompt(pil, PRECISE_PROMPT, device)
    loose, n_l = _union_masks_for_prompt(pil, LOOSE_PROMPT, device)

    refined_loose = cv2.bitwise_and(loose, pixel_mask)
    combined = cv2.bitwise_or(precise, refined_loose)
    dilated = cv2.dilate(
        combined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    logger.info(
        "sam3 hybrid: %d precise + %d loose detections, coverage %.3f",
        n_p, n_l, float((dilated > 0).mean()),
    )
    return dilated
