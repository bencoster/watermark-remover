"""ProPainter video inpainting service - extracted from sczhou/ProPainter.

Stub. Full implementation requires extracting RAFT, flow completion,
and ProPainter model code from the upstream repo into models/propainter/.
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


def process_video_job(
    job_id: str,
    payload: dict,
    on_progress: Callable[[float, str], None],
) -> str:
    """Process a video inpainting job. Returns path to result video.

    Called by jobs/worker.py on the worker thread.
    """
    on_progress(0.0, "starting")
    raise NotImplementedError(
        "ProPainter video inpainting not yet implemented. "
        "Requires extracting model code from sczhou/ProPainter into models/propainter/."
    )
