"""BC_WatermarkRemover — FastAPI entry point."""
import asyncio
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from routes import inpaint, video, detect, jobs, status

logger = logging.getLogger(__name__)

PROJ_DIR = Path(__file__).parent
WEB_DIR = PROJ_DIR / "web"
WEIGHTS_DIR = PROJ_DIR / "weights"
TMP_DIR = PROJ_DIR / "tmp"


async def _auto_download_models():
    """Download model weights on startup if missing. Runs off the event loop."""
    try:
        from download_models import download_all
        await asyncio.to_thread(download_all)
        logger.info("Model auto-download complete")
    except Exception:
        logger.exception("Model auto-download failed - first inference may stall")


@asynccontextmanager
async def lifespan(app: FastAPI):
    TMP_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)

    from jobs.store import connect, init_db
    conn = connect()
    init_db(conn)
    conn.close()

    from services.model_manager import manager
    from services.lama_service import load_model
    manager.register("lama", load_model)

    # Kick off model download in the background — server stays responsive
    # while big-lama.pt (~200MB) is fetched on first run.
    if os.environ.get("BC_WMR_AUTO_DOWNLOAD", "1") != "0":
        asyncio.create_task(_auto_download_models())

    from jobs.worker import start_worker
    start_worker()
    yield
    from jobs.worker import stop_worker
    stop_worker()


app = FastAPI(title="BC_WatermarkRemover", version="0.1.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=500)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

app.include_router(inpaint.router)
app.include_router(video.router)
app.include_router(detect.router)
app.include_router(jobs.router, prefix="/api/jobs")
app.include_router(status.router)


@app.get("/")
async def index():
    return FileResponse(str(WEB_DIR / "index.html"))
