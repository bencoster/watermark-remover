"""Watermark auto-detection service.

Uses a lightweight watermark classifier to produce a binary mask.
Falls back to None (frontend shows manual mask editor) when
confidence is below threshold.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def detect(image_path: str) -> Optional[str]:
    """Detect watermark region in an image.

    Args:
        image_path: Path to input image.

    Returns:
        Path to binary mask PNG (255=watermark, 0=keep), or None if
        no watermark detected with sufficient confidence.
    """
    logger.warning("Watermark auto-detection not yet implemented - returning None")
    return None
