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


def connect_centroids(
    mask: np.ndarray,
    line_thickness: int | None = None,
    n_neighbors: int = 6,
    max_link_dist_factor: float = 1.6,
) -> np.ndarray:
    """Draw thick lines between each detected centroid and its nearest
    neighbours. No prediction of new positions — uses ONLY the centroids
    Grad-CAM already lit up, so we can't drift away from the real
    watermark grid.

    For tiled stock-photo watermarks the diagonal text strokes run
    between adjacent logo centres; connecting those centres directly
    captures those strokes without the linear drift that any anchored-
    lattice prediction suffers far from its anchor.
    """
    centroids = _component_centroids(mask)
    if len(centroids) < 4:
        return mask

    # Estimate a reasonable median nearest-neighbour distance — used
    # both for the auto line thickness and as a maximum link distance
    # so we don't accidentally connect across the whole image.
    nn_dists = []
    for c in centroids:
        d = np.linalg.norm(centroids - c, axis=1)
        d = d[d > 0]
        if d.size:
            nn_dists.append(np.min(d))
    median_nn = float(np.median(nn_dists)) if nn_dists else 60.0

    if line_thickness is None:
        # ~5% of nearest-neighbour pitch — matches GT text-stroke width.
        line_thickness = max(2, int(round(median_nn * 0.05)))

    out = mask.copy()
    max_link = median_nn * max_link_dist_factor
    seen = set()
    for c in centroids:
        d = np.linalg.norm(centroids - c, axis=1)
        order = np.argsort(d)[1:1 + n_neighbors]
        cx, cy = int(c[0]), int(c[1])
        for j in order:
            if d[j] > max_link:
                continue
            nx, ny = int(centroids[j][0]), int(centroids[j][1])
            key = (min(cx, nx), min(cy, ny), max(cx, nx), max(cy, ny))
            if key in seen:
                continue
            seen.add(key)
            cv2.line(out, (cx, cy), (nx, ny), 255, line_thickness)
    logger.info("connect_centroids: %d centroids, median NN=%.0f, "
                "thickness=%d, %d edges drawn",
                len(centroids), median_nn, line_thickness, len(seen))
    return out


def grid_complete(
    mask: np.ndarray,
    image_bgr: Optional[np.ndarray] = None,
    disc_radius: int = 18,
    line_thickness: int = 6,
    max_lattice_points: int = 400,
    auto_scale: bool = True,
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

    # Sanity-check the lattice. At small image scales the vote can latch
    # onto outlier offsets — e.g. v1 ≈ image_width, which would draw a
    # single huge line spanning the whole image. We require both
    # generator vectors to be:
    #   - shorter than half the smaller image dimension (real watermark
    #     tiles are much smaller than half the image)
    #   - longer than 16 px (no degenerate near-zero vectors)
    #   - not too close to parallel (cross product must be > 5% of |v1||v2|)
    h, w = mask.shape[:2]
    max_len = 0.5 * min(h, w)
    min_len = 16.0
    l1 = float(np.linalg.norm(v1))
    l2 = float(np.linalg.norm(v2))
    if not (min_len <= l1 <= max_len and min_len <= l2 <= max_len):
        logger.info("grid_complete: lattice vectors out of range "
                    "(|v1|=%.1f |v2|=%.1f, allowed [%.1f, %.1f]) — skipping",
                    l1, l2, min_len, max_len)
        return mask
    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
    if cross < 0.05 * l1 * l2:
        logger.info("grid_complete: lattice generators near-parallel — skipping")
        return mask

    # Scale disc/line dimensions to match the detected tile spacing.
    # Hardcoded values (14 / 4) were calibrated against the 1600x1157
    # canonical fixture; on smaller or larger inputs the same physical
    # tile occupies fewer/more pixels. Use a fraction of the shorter
    # generator vector — that's roughly the tile pitch.
    if auto_scale:
        tile_pitch = min(l1, l2)
        # 14 px disc / 90 px tile pitch -> 0.155 ratio at calibration.
        # Likewise 4/90 -> 0.044 for line thickness.
        disc_radius = max(4, int(round(tile_pitch * 0.16)))
        line_thickness = max(2, int(round(tile_pitch * 0.045)))
        logger.info("grid_complete: auto-scaled disc=%d line=%d (pitch=%.0f)",
                    disc_radius, line_thickness, tile_pitch)

    h, w = mask.shape[:2]
    # Anchor the lattice at the median centroid to minimise misalignment.
    anchor = np.median(centroids, axis=0)
    out = mask.copy()

    # Sweep an integer (i, j) grid wide enough to cover the image, plus
    # a one-cell margin in every direction. Including the off-image
    # margin matters: tiles that sit *just outside* the image still
    # contribute connecting strokes that cross the visible edge — that
    # is precisely the diagonal text we were missing in the top-right
    # and right-edge regions of the dreamstime fixture.
    vlen = max(1.0, min(np.linalg.norm(v1), np.linalg.norm(v2)))
    span = int(np.ceil(max(w, h) / vlen) + 4)
    margin_cells = 2
    visible: list[tuple[int, int]] = []
    all_pts: list[tuple[int, int]] = []
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            x = anchor[0] + i * v1[0] + j * v2[0]
            y = anchor[1] + i * v1[1] + j * v2[1]
            xi, yi = int(round(x)), int(round(y))
            # Visible = inside image, used for circle drawing.
            if 0 <= xi < w and 0 <= yi < h:
                visible.append((xi, yi))
            # all_pts = visible + a margin ring outside the image, used
            # for line drawing so edge cells get their connecting strokes.
            margin_pad = margin_cells * vlen
            if -margin_pad <= xi < w + margin_pad and -margin_pad <= yi < h + margin_pad:
                all_pts.append((xi, yi))
            if len(visible) > max_lattice_points:
                break
        if len(visible) > max_lattice_points:
            break

    logger.info("grid_complete: lattice v1=%s v2=%s -> %d visible (%d incl margin)",
                v1.round(1).tolist(), v2.round(1).tolist(),
                len(visible), len(all_pts))

    # Draw discs at every visible lattice node (each is a watermark logo).
    for (x, y) in visible:
        cv2.circle(out, (x, y), disc_radius, 255, -1)

    # Connect lattice neighbours along both generator vectors AND the
    # two diagonals (v1+v2 / v1-v2). Stock-photo watermarks like
    # Dreamstime render the URL text crossing each tile — adjacent
    # lattice nodes are linked along *both* diagonal directions, so
    # the GT mask shows X patterns at every node. Lines along just
    # v1 and v2 leave the diagonals empty.
    pt_set = set(all_pts)
    deltas = [
        (int(round(v1[0])), int(round(v1[1]))),
        (int(round(v2[0])), int(round(v2[1]))),
        (int(round(v1[0] + v2[0])), int(round(v1[1] + v2[1]))),
        (int(round(v1[0] - v2[0])), int(round(v1[1] - v2[1]))),
    ]
    for (x, y) in all_pts:
        for dx, dy in deltas:
            nb = (x + dx, y + dy)
            if nb in pt_set:
                cv2.line(out, (x, y), nb, 255, line_thickness)
    return out
