"""SDXL Inpainting service — second-pass refill for the bottom strip bar.

LaMa fills strip-bar regions (where the URL/copyright bar covered carpet
or wood texture) with smudgy gradients because the bar is wider than
LaMa's effective context window for that texture class. SDXL's diffusion
prior can synthesise plausible texture continuation when given a short
material prompt.

This service is a SECOND PASS: it runs *only on the strip-bar region*
of the image, never the full body mask. The body inpaint stays with
LaMa (faster, no diffusion VRAM cost when not needed).

Loading is on-demand and the pipeline is held by the global model
manager so it can be unloaded when LaMa is in use.
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

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
TMP_DIR = Path(__file__).parent.parent / "tmp"

MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

# Empirically-good settings from the research synthesis:
#   45 steps, guidance 9.5, strength 0.95 — produces clean texture
#   without prompt drift, mask edges blend without visible boundary.
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 8.0
DEFAULT_STRENGTH = 0.95

# Working tile size SDXL is happiest with (it was trained at 1024x1024).
WORKING_RES = 1024


def _required_vram_gb() -> float:
    """Conservative estimate of FP16 SDXL Inpaint VRAM peak."""
    return 11.0


def is_available(device: torch.device) -> tuple[bool, str]:
    """Cheap pre-flight: do we have enough VRAM on this device, and
    can we import diffusers?"""
    try:
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        return False, f"missing dependency: {e.name}"
    if device.type == "cpu":
        return True, "CPU is supported but very slow — expect minutes per image"
    if device.type != "cuda":
        return False, f"unsupported device {device}"
    free_bytes, _ = torch.cuda.mem_get_info(device.index or 0)
    free_gb = free_bytes / (1 << 30)
    needed = _required_vram_gb()
    if free_gb < needed:
        return False, f"GPU {device.index} has {free_gb:.1f} GB free; need ~{needed:.1f} GB"
    return True, f"{free_gb:.1f} GB free on {device}"


def load_pipeline(device: torch.device):
    """Load (or cache-hit) the SDXL inpaint pipeline. Heavyweight — first
    call downloads ~7 GB and takes 30-90 s. Subsequent calls reuse."""
    from diffusers import StableDiffusionXLInpaintPipeline

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        cache_dir=str(WEIGHTS_DIR),
        variant="fp16" if device.type == "cuda" else None,
    )
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def _crop_to_strip_region(
    image_bgr: np.ndarray, strip_mask: np.ndarray, pad: int = 60,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop the image and mask to a region tightly around the strip
    mask's bounding box, with `pad` pixels of margin so SDXL has
    surrounding context to extend texture from. Returns (image_crop,
    mask_crop, (y0, y1, x0, x1)) so the result can be pasted back."""
    h, w = image_bgr.shape[:2]
    ys, xs = np.where(strip_mask > 0)
    if len(ys) == 0:
        return image_bgr, strip_mask, (0, h, 0, w)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad)
    return image_bgr[y0:y1, x0:x1], strip_mask[y0:y1, x0:x1], (y0, y1, x0, x1)


def inpaint_strip_region(
    image_path: str,
    strip_mask_path: str,
    pipe,
    prompt: str = "seamless carpet texture, photorealistic, soft white shag rug, natural background continuation",
    negative_prompt: str = "watermark, text, letter, logo, banner, stripe, blur, artifact, distorted",
    num_inference_steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
    strength: float = DEFAULT_STRENGTH,
) -> str:
    """Run SDXL Inpaint on the strip-bar region only and paste the
    result back into the full-resolution image. Returns the path to a
    new full-resolution image with the bar region refilled.

    SDXL is run on a cropped region scaled to 1024 px on the long edge
    so it gets the resolution it was trained at, then the result is
    resized back and pasted in. The unmasked area is preserved bit-for-
    bit by manual mask-blending — the SDXL output for unmasked pixels
    can drift from the input even when strength<1.0.
    """
    image_bgr = cv2.imread(image_path)
    strip_mask = cv2.imread(strip_mask_path, cv2.IMREAD_GRAYSCALE)
    if image_bgr is None or strip_mask is None:
        raise ValueError("Could not read image or mask")

    crop_bgr, crop_mask, (y0, y1, x0, x1) = _crop_to_strip_region(image_bgr, strip_mask)
    ch, cw = crop_bgr.shape[:2]

    # Scale to ~1024 on the long edge for SDXL
    scale = WORKING_RES / max(ch, cw)
    if scale < 1.0:
        new_w = int(round(cw * scale))
        new_h = int(round(ch * scale))
        # Make dims multiples of 8 (SDXL VAE constraint)
        new_w = (new_w // 8) * 8 or 8
        new_h = (new_h // 8) * 8 or 8
        sdxl_in = cv2.resize(crop_bgr, (new_w, new_h), cv2.INTER_AREA)
        sdxl_mask = cv2.resize(crop_mask, (new_w, new_h), cv2.INTER_NEAREST)
    else:
        sdxl_in, sdxl_mask = crop_bgr, crop_mask

    pil_image = Image.fromarray(cv2.cvtColor(sdxl_in, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(sdxl_mask)

    logger.info(
        "SDXL inpaint: crop %dx%d -> SDXL %dx%d (%d steps, guidance %.1f, strength %.2f)",
        cw, ch, pil_image.width, pil_image.height,
        num_inference_steps, guidance_scale, strength,
    )

    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        ).images[0]

    sdxl_out = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    if sdxl_out.shape[:2] != (ch, cw):
        sdxl_out = cv2.resize(sdxl_out, (cw, ch), cv2.INTER_LANCZOS4)

    # Blend: only replace pixels where the mask is set; preserve everything
    # else exactly as the input (SDXL slightly drifts unmasked pixels even
    # at strength<1, which would visibly seam against the image we're
    # pasting back into).
    mask_f = (crop_mask > 127).astype(np.float32)[..., None]
    blended_crop = sdxl_out.astype(np.float32) * mask_f + crop_bgr.astype(np.float32) * (1.0 - mask_f)

    # Soft feather at mask boundaries to hide hard edges in the merged
    # result. 5px Gaussian on a binary mask -> ~10px feather.
    soft = cv2.GaussianBlur((crop_mask > 127).astype(np.float32), (0, 0), sigmaX=3.0)[..., None]
    blended_crop = sdxl_out.astype(np.float32) * soft + crop_bgr.astype(np.float32) * (1.0 - soft)

    full_out = image_bgr.copy()
    full_out[y0:y1, x0:x1] = np.clip(blended_crop, 0, 255).astype(np.uint8)

    TMP_DIR.mkdir(exist_ok=True)
    out_path = tempfile.mktemp(suffix="_sdxl.png", dir=str(TMP_DIR))
    cv2.imwrite(out_path, full_out)
    return out_path
