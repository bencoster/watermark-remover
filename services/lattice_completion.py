"""Mask post-processing utilities.

:func:`complete_lines` — Hough-based partial-line extension.
   Given a binary mask containing fragmented line segments (e.g. a
   diff mask that captured most of a tiled pattern but skipped weak
   strokes), use HoughLinesP on the skeleton to recover each segment's
   slope and extend the mask along that slope to bridge gaps. The
   Diff Editor's "Complete Partial Lines" button calls this.

:func:`_component_centroids` and :func:`_vote_lattice_vectors` are
internal helpers used by detector_service to decide whether a Grad-CAM
mask looks like a 2D tile grid (auto-mode tiled vs single decision).

Earlier iterations included `grid_complete` and `connect_centroids`
that auto-augmented the detector mask with predicted lattice nodes
or centroid-to-centroid connecting strokes. Both over-masked subject
content and never delivered a clear quality gain — the library-mask
short-circuit closed the remaining gap on repeat watermarks instead.
Removed 2026-05-10.

Pure NumPy + OpenCV — no torch dependency.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─── Pass 1: line completion via Hough on the skeleton ────────────────────────

def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Single-pixel-wide skeleton via the morphological erode-and-subtract
    method (Lantuéjoul 1980). Cheap and dependency-free."""
    img = mask.copy()
    skel = np.zeros_like(img)
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, elem)
        opened = cv2.dilate(eroded, elem)
        skel |= cv2.subtract(img, opened)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def complete_lines(
    mask: np.ndarray,
    min_line_len: int = 30,
    gap: int = 50,
    extend_px: int = 35,
    stroke_thickness: int = 6,
) -> np.ndarray:
    """Fill gaps along detected linear segments in a binary mask.

    Args:
        mask: uint8 (H,W), 255=foreground, 0=background.
        min_line_len: minimum length for HoughLinesP to consider.
        gap: max gap between collinear segments to be merged.
        extend_px: how far to extend each segment past its endpoints.
        stroke_thickness: thickness of the painted completion lines.
    """
    if mask is None or mask.size == 0:
        return mask
    bin_ = (mask > 0).astype(np.uint8) * 255
    skel = _skeletonize(bin_)

    # HoughLinesP on the skeleton finds endpoints of each linear run.
    lines = cv2.HoughLinesP(
        skel, rho=1, theta=np.pi / 180,
        threshold=20, minLineLength=min_line_len, maxLineGap=gap,
    )
    if lines is None:
        return mask

    out = mask.copy()
    h, w = mask.shape[:2]
    for ln in lines[:, 0]:
        x1, y1, x2, y2 = ln
        dx, dy = x2 - x1, y2 - y1
        L = max(1.0, np.hypot(dx, dy))
        ux, uy = dx / L, dy / L
        # Extend both endpoints outward along the line direction
        ex1 = int(np.clip(x1 - ux * extend_px, 0, w - 1))
        ey1 = int(np.clip(y1 - uy * extend_px, 0, h - 1))
        ex2 = int(np.clip(x2 + ux * extend_px, 0, w - 1))
        ey2 = int(np.clip(y2 + uy * extend_px, 0, h - 1))
        cv2.line(out, (ex1, ey1), (ex2, ey2), 255, stroke_thickness)

    logger.info("complete_lines: %d segments extended", len(lines))
    return out


# ─── Pass 2: lattice completion for tiled stock-photo patterns ────────────────

def _component_centroids(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """Return (N, 2) array of (x, y) centroids for each foreground blob."""
    n_lbl, _, stats, cents = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    keep = []
    for i in range(1, n_lbl):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep.append(cents[i])
    return np.array(keep) if keep else np.zeros((0, 2))


def _vote_lattice_vectors(centroids: np.ndarray, n_neighbors: int = 6,
                          bin_size: float = 4.0) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Estimate the two generator vectors of a lattice from a set of points.

    For each point, look at its ``n_neighbors`` nearest neighbours and
    record the pairwise offset vectors. The two most common offsets
    (modulo sign) are the lattice generators.
    """
    if len(centroids) < 4:
        return None

    # All pairwise diffs to nearest neighbours
    diffs = []
    for c in centroids:
        d = centroids - c
        # Drop self
        dist = np.hypot(d[:, 0], d[:, 1])
        order = np.argsort(dist)[1:1 + n_neighbors]
        for j in order:
            diffs.append(d[j])
    diffs = np.array(diffs)
    # Canonical sign: dy >= 0, with dx flipped for dy == 0 negatives
    flip = (diffs[:, 1] < 0) | ((diffs[:, 1] == 0) & (diffs[:, 0] < 0))
    diffs[flip] = -diffs[flip]

    # Histogram in (dx, dy) space
    quant = np.round(diffs / bin_size).astype(int)
    keys, counts = np.unique(quant, axis=0, return_counts=True)
    if len(keys) < 2:
        return None
    order = np.argsort(-counts)

    # Pick the two strongest non-collinear bins
    v1 = keys[order[0]] * bin_size
    v2 = None
    for k_idx in order[1:]:
        cand = keys[k_idx] * bin_size
        # Reject if (nearly) collinear with v1 — cross product near 0
        if abs(v1[0] * cand[1] - v1[1] * cand[0]) > bin_size * 8:
            v2 = cand
            break
    if v2 is None:
        return None
    return v1.astype(float), v2.astype(float)


