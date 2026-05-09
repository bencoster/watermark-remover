"""POST /api/video - submit video watermark removal job."""
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

from jobs.store import connect, init_db, enqueue

router = APIRouter()

TMP_DIR = Path(__file__).parent.parent / "tmp"


@router.post("/api/video")
async def video_inpaint(
    file: UploadFile = File(...),
    mask: UploadFile = File(None),
    auto_detect: bool = Form(True),
):
    TMP_DIR.mkdir(exist_ok=True)
    video_dir = tempfile.mkdtemp(dir=str(TMP_DIR), prefix="video_")
    video_path = os.path.join(video_dir, file.filename or "input.mp4")

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    mask_path = None
    if mask is not None:
        mask_path = os.path.join(video_dir, "mask.png")
        with open(mask_path, "wb") as f:
            f.write(await mask.read())

    payload = {
        "video_path": video_path,
        "mask_path": mask_path,
        "auto_detect": auto_detect,
        "work_dir": video_dir,
    }

    conn = connect()
    try:
        init_db(conn)
        job_id = enqueue(conn, kind="video_inpaint", payload=payload)
    finally:
        conn.close()

    return JSONResponse({"job_id": job_id}, status_code=202)
