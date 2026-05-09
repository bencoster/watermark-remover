"""Background worker thread - polls job queue for video inpaint jobs."""
from __future__ import annotations

import json
import logging
import threading
import traceback

from jobs.store import connect, claim_next, update_status

logger = logging.getLogger(__name__)

_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _worker_loop():
    logger.info("Job worker started")
    while not _stop_event.is_set():
        conn = connect()
        try:
            row = claim_next(conn)
        finally:
            conn.close()

        if row is None:
            _stop_event.wait(timeout=2.0)
            continue

        job_id = row["id"]
        kind = row["kind"]
        payload = json.loads(row["payload"])
        logger.info("Processing job %s (kind=%s)", job_id, kind)

        try:
            if kind == "video_inpaint":
                from services.propainter_service import process_video_job
                result_path = process_video_job(job_id, payload, _progress_cb(job_id))
                conn = connect()
                update_status(conn, job_id, "succeeded", progress=1.0, stage="done", result_path=result_path)
                conn.close()
            else:
                raise ValueError(f"Unknown job kind: {kind}")
        except Exception:
            logger.exception("Job %s failed", job_id)
            conn = connect()
            update_status(conn, job_id, "failed", error=traceback.format_exc()[-500:])
            conn.close()


def _progress_cb(job_id: str):
    def cb(progress: float, stage: str):
        conn = connect()
        try:
            update_status(conn, job_id, "running", progress=progress, stage=stage)
        finally:
            conn.close()
    return cb


def start_worker():
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_worker_loop, daemon=True, name="job-worker")
    _thread.start()


def stop_worker():
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=10)
