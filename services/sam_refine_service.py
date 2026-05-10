"""SAM-based mask refinement.

The Grad-CAM body mask gives us blob centroids at every watermark logo
position the classifier attends to, but the blob *shape* is a coarse
heatmap-derived approximation. SAM produces a precise per-instance
mask given a point or box prompt, so feeding each Grad-CAM centroid
to SAM and unioning the results gives us pixel-precise per-watermark
masks.

Two backends supported behind a small interface:

  * **sam2** — `pip install sam2` (transformers 4.x compatible). Uses
    point prompts (one centroid per logo). Available today on the
    project's Python 3.10 install.
  * **sam3.1** — Meta's text-prompt-capable SAM. Released March 2026
    but requires Python 3.12, a checkpoint-access request, and a
    project-local venv. Plug-in slot kept here for the day that
    happens; until then is_available() reports the gap.

Both paths emit the same `refine(image, points_or_box)` interface so
the route layer doesn't care which is loaded.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

# SAM 2 — fits today's environment (transformers 4.57 + Python 3.10).
SAM2_HF_ID = "facebook/sam2-hiera-base-plus"

# SAM 3.1 — Released March 2026, requires Python 3.12 + checkpoint
# access request + the official `sam3` package from GitHub.
SAM3_HF_ID = "facebook/sam3.1"


def is_available(engine: str = "sam2") -> tuple[bool, str]:
    """Pre-flight check for either SAM engine. Returns (ok, message)."""
    if engine == "sam2":
        try:
            import sam2  # noqa: F401
        except ImportError:
            return False, "pip install sam2  (~30 MB, no checkpoint access needed)"
        if not torch.cuda.is_available():
            return True, "CPU is supported but very slow"
        return True, "sam2 installed"
    if engine == "sam3.1":
        if sys.version_info < (3, 12):
            return False, (
                f"SAM 3.1 requires Python 3.12+ (you have "
                f"{sys.version_info.major}.{sys.version_info.minor}). "
                f"Create a project-local venv and install per "
                f"https://github.com/facebookresearch/sam3"
            )
        try:
            from sam3 import build_sam3  # noqa: F401
        except ImportError:
            return False, "pip install -e git+https://github.com/facebookresearch/sam3.git#egg=sam3"
        return True, "sam3.1 installed"
    return False, f"unknown engine: {engine}"


# ─── SAM 2 implementation ─────────────────────────────────────────────────────

_sam2_cache: dict = {}


def _load_sam2(device: torch.device):
    """Lazy-load and cache the SAM 2 predictor."""
    key = (str(device), SAM2_HF_ID)
    if key in _sam2_cache:
        return _sam2_cache[key]
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor.from_pretrained(SAM2_HF_ID, device=str(device))
    _sam2_cache[key] = predictor
    logger.info("sam2 loaded on %s", device)
    return predictor


def _centroids_from_mask(mask: np.ndarray, min_area: int = 30) -> np.ndarray:
    """(N, 2) array of [x, y] centroids for blobs in the mask."""
    n_lbl, _, stats, cents = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    keep = []
    for i in range(1, n_lbl):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep.append(cents[i])
    return np.array(keep) if keep else np.zeros((0, 2))


def refine_with_sam2(
    image_bgr: np.ndarray,
    rough_mask: np.ndarray,
    device: torch.device,
    score_threshold: float = 0.5,
    max_points: int = 80,
) -> np.ndarray:
    """Refine a Grad-CAM rough mask into a precise per-instance mask.

    For each centroid in `rough_mask`, prompt SAM 2 with that point,
    take the highest-scoring mask, union all of them.

    Returns a uint8 binary mask the same size as image_bgr.
    """
    predictor = _load_sam2(device)
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    h, w = image_bgr.shape[:2]

    centroids = _centroids_from_mask(rough_mask)
    if len(centroids) == 0:
        logger.info("sam2 refine: no centroids — returning input mask unchanged")
        return rough_mask
    if len(centroids) > max_points:
        # Sort by distance to mask centroid and keep the closest — those
        # are most likely on actual watermarks rather than spurious noise.
        order = np.argsort(np.linalg.norm(centroids - centroids.mean(axis=0), axis=1))
        centroids = centroids[order[:max_points]]

    predictor.set_image(np.array(pil))
    union = np.zeros((h, w), dtype=np.uint8)

    for cx, cy in centroids:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[cx, cy]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            continue
        # Pick highest-scoring mask, but prefer smaller masks since a
        # watermark logo is small relative to the full subject; SAM
        # sometimes returns the whole subject area as the "best" mask.
        # Filter to masks below 5% of image area, then take best score.
        max_area = h * w * 0.05
        candidates = [(s, m) for s, m in zip(scores, masks)
                      if m.sum() < max_area and s >= score_threshold]
        if not candidates:
            continue
        candidates.sort(key=lambda sm: -sm[0])
        chosen = candidates[0][1]
        union |= (chosen > 0).astype(np.uint8) * 255

    logger.info("sam2 refine: %d centroids -> mask coverage %.3f",
                len(centroids), float((union > 0).mean()))
    return union


# ─── SAM 3.1 implementation (placeholder, future-proof) ───────────────────────

def refine_with_sam3(
    image_bgr: np.ndarray,
    text_prompt: str = "watermark logo text overlay",
    device: torch.device | None = None,
) -> np.ndarray:
    """Text-prompt-driven refinement via SAM 3.1.

    Not yet active — gated behind `is_available('sam3.1')`. The
    expected call would build the predictor, run with `text=...`, and
    union all returned instance masks.
    """
    raise NotImplementedError(
        "SAM 3.1 path stubbed. Set up Python 3.12 venv, install per "
        "https://github.com/facebookresearch/sam3 , then implement "
        "the body of this function. The interface signature is "
        "intentionally similar to refine_with_sam2 so the route layer "
        "stays the same."
    )


# ─── Unified entry point ──────────────────────────────────────────────────────

def refine(
    image_bgr: np.ndarray,
    rough_mask: np.ndarray,
    device: torch.device,
    engine: str = "sam2",
) -> np.ndarray:
    """Refine `rough_mask` to a precise per-instance mask. Returns the
    rough mask unchanged on any failure, so the caller can always trust
    the return value as 'at least as good as the input mask'."""
    ok, why = is_available(engine)
    if not ok:
        logger.warning("sam refine skipped (%s): %s", engine, why)
        return rough_mask
    try:
        if engine == "sam2":
            return refine_with_sam2(image_bgr, rough_mask, device)
        if engine == "sam3.1":
            return refine_with_sam3(image_bgr, device=device)
    except Exception:
        logger.exception("sam refine failed; returning rough mask")
        return rough_mask
    return rough_mask
