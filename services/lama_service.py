"""LaMa inpainting service - extracted from IOPaint, no dependency."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from download_models import download

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"


def pad_to_mod(
    image: np.ndarray, mask: np.ndarray, mod: int = 8
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    pad_h = (mod - h % mod) % mod
    pad_w = (mod - w % mod) % mod
    if pad_h == 0 and pad_w == 0:
        return image, mask, (0, 0, 0, 0)
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    if mask.ndim == 3:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
    else:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return image, mask, (0, pad_h, 0, pad_w)


def unpad(image: np.ndarray, pad: tuple[int, int, int, int]) -> np.ndarray:
    _, pad_h, _, pad_w = pad
    if pad_h == 0 and pad_w == 0:
        return image
    h, w = image.shape[:2]
    return image[: h - pad_h, : w - pad_w]


def load_model(device: torch.device) -> torch.jit.ScriptModule:
    path = download("big-lama.pt")
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model


def inpaint(image_path: str, mask_path: str, device: torch.device, model: torch.jit.ScriptModule) -> str:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    mask = mask[:, :, np.newaxis]

    image, mask, pad = pad_to_mod(image, mask, mod=8)

    img_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mask_t = torch.from_numpy((mask > 0).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)

    img_t = img_t.to(device)
    mask_t = mask_t.to(device)

    with torch.inference_mode():
        result = model(img_t, mask_t)

    result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    result = unpad(result, pad)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    tmp_dir = Path(__file__).parent.parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    out_path = tempfile.mktemp(suffix=".png", dir=str(tmp_dir))
    cv2.imwrite(out_path, result)
    return out_path
