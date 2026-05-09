"""ConvNeXt-tiny watermark binary classifier.

Loads the checkpoint published at HF Hub:
  boomb0om/watermark-detectors :: convnext-tiny_watermarks_classifier_v1.pt

Architecture: timm convnext_tiny with a 2-class classification head.
Input: 256x256 RGB normalised with ImageNet mean/std.
Output: int 0 (clean) or 1 (watermarked).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
HF_REPO_ID = "boomb0om/watermark-detectors"
HF_FILENAME = "convnext-tiny_watermarks_detector.pth"

_IM_MEAN = (0.485, 0.456, 0.406)
_IM_STD = (0.229, 0.224, 0.225)
_IM_SIZE = 256

_transform = T.Compose([
    T.Resize((_IM_SIZE, _IM_SIZE)),
    T.ToTensor(),
    T.Normalize(_IM_MEAN, _IM_STD),
])


def _load_checkpoint() -> Path:
    dest = WEIGHTS_DIR / HF_FILENAME
    if dest.exists():
        return dest
    from huggingface_hub import hf_hub_download
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from %s", HF_FILENAME, HF_REPO_ID)
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, local_dir=str(WEIGHTS_DIR))
    return Path(path)


def _build_model() -> torch.nn.Module:
    """Build ConvNeXt-tiny + boomb0om's custom 5-layer classification head."""
    from models.convnext.convnext import convnext_tiny
    model = convnext_tiny(num_classes=2)
    model.head = torch.nn.Sequential(
        torch.nn.Linear(768, 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 2),
    )
    return model


def load_model(device: torch.device) -> torch.nn.Module:
    ckpt_path = _load_checkpoint()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model = _build_model()
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        logger.warning("classifier load: missing=%d unexpected=%d", len(missing), len(unexpected))
    model.eval().to(device)
    return model


@torch.inference_mode()
def predict_batch(model: torch.nn.Module, pil_images: Sequence[Image.Image], device: torch.device) -> np.ndarray:
    """Returns array of P(watermarked) per image, shape (N,)."""
    if not pil_images:
        return np.zeros((0,), dtype=np.float32)
    batch = torch.stack([_transform(img.convert("RGB")) for img in pil_images]).to(device)
    logits = model(batch)
    probs = F.softmax(logits, dim=1)[:, 1]  # P(class=1) = watermarked
    return probs.cpu().numpy()
