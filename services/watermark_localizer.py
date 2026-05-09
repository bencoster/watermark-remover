"""Grad-CAM watermark localizer.

The boomb0om classifier knows what watermarks look like — Grad-CAM
extracts that knowledge as a per-pixel attention map. For each input
image we:

  1. Forward through the network, capturing the last-stage feature
     map (B, 768, H/32, W/32).
  2. Backprop ∂P(watermarked)/∂features.
  3. Weight feature maps by global-average-pooled gradients, sum,
     ReLU. This is the standard Grad-CAM recipe (Selvaraju et al.).
  4. Upsample to image resolution and threshold.

To handle high-resolution stock photos (where downsampling to 256x256
loses fine watermark text), we run Grad-CAM on overlapping crops at
the network's native input size, then stitch.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

logger = logging.getLogger(__name__)

_IM_MEAN = (0.485, 0.456, 0.406)
_IM_STD = (0.229, 0.224, 0.225)
_TILE = 256

_normalize = T.Compose([
    T.ToTensor(),
    T.Normalize(_IM_MEAN, _IM_STD),
])


class _GradCAM:
    """Last-stage Grad-CAM hook for ConvNeXt-tiny."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._features: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        # Hook the last block of the last stage (output before global avg pool).
        last_stage = model.stages[-1]
        last_block = last_stage[-1]
        last_block.register_forward_hook(self._save_features)
        last_block.register_full_backward_hook(self._save_grads)

    def _save_features(self, _module, _inp, out):
        self._features = out

    def _save_grads(self, _module, _grad_in, grad_out):
        self._grads = grad_out[0]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return Grad-CAM map for class=1 (watermarked). Shape: (B, h, w)."""
        self.model.zero_grad()
        logits = self.model(x)
        target = logits[:, 1].sum()
        target.backward(retain_graph=False)

        feats = self._features  # (B, 768, h, w)
        grads = self._grads     # (B, 768, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, 768, 1, 1)
        cam = F.relu((feats * weights).sum(dim=1))      # (B, h, w)
        # Per-image normalisation
        b = cam.shape[0]
        flat = cam.view(b, -1)
        max_v = flat.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        cam = (flat / max_v).view_as(cam)
        return cam.detach()


def _tile_coords(h: int, w: int, tile: int = _TILE, overlap: float = 0.5):
    """Yield (y0, x0) tile origins covering the image with the given overlap."""
    stride = max(1, int(tile * (1.0 - overlap)))
    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if not ys or ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    if not xs or xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    seen = set()
    for y in ys:
        for x in xs:
            if (y, x) in seen:
                continue
            seen.add((y, x))
            yield y, x


def _smoothstep(arr: np.ndarray) -> np.ndarray:
    """Hann-like 2D taper for blending tile contributions."""
    h, w = arr.shape
    yy = np.linspace(-1, 1, h, dtype=np.float32)
    xx = np.linspace(-1, 1, w, dtype=np.float32)
    wy = 0.5 * (1 + np.cos(np.pi * yy))
    wx = 0.5 * (1 + np.cos(np.pi * xx))
    return np.outer(wy, wx)


def localize(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    overlap: float = 0.5,
    batch_size: int = 8,
) -> np.ndarray:
    """Compute a watermark heatmap (H,W) in [0,1] over the original image.

    Pads to a multiple of TILE, tiles with overlap, runs Grad-CAM on
    each tile, blends with a smooth window. Returns array shape (H,W).
    """
    cam_extractor = _GradCAM(model)
    img = image.convert("RGB")
    w, h = img.size

    # Pad-up so the rightmost/bottom tiles fully fit the network input.
    pad_h = max(0, _TILE - h) + (_TILE - h % _TILE) % _TILE
    pad_w = max(0, _TILE - w) + (_TILE - w % _TILE) % _TILE
    padded_h, padded_w = h + pad_h, w + pad_w

    np_img = np.array(img)  # (H, W, 3)
    if pad_h or pad_w:
        np_img = np.pad(np_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    coords = list(_tile_coords(padded_h, padded_w, _TILE, overlap))
    tile_window = _smoothstep(np.ones((_TILE, _TILE), dtype=np.float32))

    heatmap = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight = np.zeros_like(heatmap)

    # Mini-batched inference for speed.
    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        batch_tensors = []
        for y0, x0 in chunk:
            tile = np_img[y0:y0 + _TILE, x0:x0 + _TILE]
            batch_tensors.append(_normalize(Image.fromarray(tile)))
        batch = torch.stack(batch_tensors).to(device)
        cams = cam_extractor(batch).cpu().numpy()  # (B, h, w) at network resolution
        for (y0, x0), cam_small in zip(chunk, cams):
            cam_full = np.array(
                Image.fromarray((cam_small * 255).astype(np.uint8)).resize((_TILE, _TILE), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            heatmap[y0:y0 + _TILE, x0:x0 + _TILE] += cam_full * tile_window
            weight[y0:y0 + _TILE, x0:x0 + _TILE] += tile_window

    weight = np.maximum(weight, 1e-6)
    heatmap /= weight
    heatmap = heatmap[:h, :w]  # crop padding
    # Per-image renormalisation
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap
