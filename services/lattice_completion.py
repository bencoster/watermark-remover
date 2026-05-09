"""Lattice + line-completion utilities for periodic stock-photo watermarks.

Two related operations live here:

1. :func:`complete_lines` — given a binary mask containing partial line
   segments (e.g. a diff mask that captured *most* of a tiled pattern
   but skipped low-contrast strokes), use Hough line detection on the
   skeleton to recover each segment's slope, then extend the mask
   along that slope until adjacent in-line components are joined. No
   external image needed — works on the mask alone.

2. :func:`grid_complete` — given the *original* image and a partial
   mask, fit a 2D lattice to the centroids of mask blobs (autocorrelation
   of the centroid set is dominated by the lattice generator vectors).
   For every predicted lattice point, draw a disc; for every adjacent
   pair, draw a thick line. Closes the gaps in tiled-grid patterns
   that no per-pixel detector reliably catches.

Both functions are pure NumPy + OpenCV — no torch dependency.
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


def grid_complete(
    mask: np.ndarray,
    image_bgr: Optional[np.ndarray] = None,
    disc_radius: int = 18,
    line_thickness: int = 6,
    max_lattice_points: int = 400,
) -> np.ndarray:
    """Predict missing lattice points and connecting strokes from a partial mask.

    Args:
        mask: the partial binary mask (typically Grad-CAM body mask).
        image_bgr: original image (used only to bound the lattice — not
            required, but the lattice is clipped to the image extent).
        disc_radius: each predicted lattice point gets a disc of this
            radius painted into the mask.
        line_thickness: each lattice edge (between adjacent predicted
            points) is drawn with this thickness — covers the diagonal
            text strokes in stock-photo watermarks.
    """
    centroids = _component_centroids(mask)
    if len(centroids) < 6:
        logger.info("grid_complete: only %d centroids — skipping", len(centroids))
        return mask

    vec = _vote_lattice_vectors(centroids)
    if vec is None:
        logger.info("grid_complete: no clear lattice — skipping")
        return mask
    v1, v2 = vec

    h, w = mask.shape[:2]
    # Anchor the lattice at the median centroid to minimise misalignment.
    anchor = np.median(centroids, axis=0)
    out = mask.copy()

    # Sweep an integer (i, j) grid wide enough to cover the image.
    # Bound size = max image dim divided by min vector length, plus margin.
    vlen = max(1.0, min(np.linalg.norm(v1), np.linalg.norm(v2)))
    span = int(np.ceil(max(w, h) / vlen) + 4)
    points: list[tuple[int, int]] = []
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            x = anchor[0] + i * v1[0] + j * v2[0]
            y = anchor[1] + i * v1[1] + j * v2[1]
            if 0 <= x < w and 0 <= y < h:
                points.append((int(x), int(y)))
            if len(points) > max_lattice_points:
                break
        if len(points) > max_lattice_points:
            break

    logger.info("grid_complete: lattice v1=%s v2=%s -> %d points",
                v1.round(1).tolist(), v2.round(1).tolist(), len(points))

    # Draw discs at lattice nodes (capture each watermark logo)
    for (x, y) in points:
        cv2.circle(out, (x, y), disc_radius, 255, -1)

    # Connect adjacent lattice neighbours (covers the diagonal text strokes).
    pt_set = set(points)
    deltas = [(int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1]))]
    for (x, y) in points:
        for dx, dy in deltas:
            nb = (x + dx, y + dy)
            if nb in pt_set:
                cv2.line(out, (x, y), nb, 255, line_thickness)
    return out
