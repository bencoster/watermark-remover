"""BC_WatermarkRemover — FastAPI entry point."""
import asyncio
import logging
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware

from routes import inpaint, video, detect, jobs, status, masks

logger = logging.getLogger(__name__)

PROJ_DIR = Path(__file__).parent
WEB_DIR = PROJ_DIR / "web"
WEIGHTS_DIR = PROJ_DIR / "weights"
TMP_DIR = PROJ_DIR / "tmp"

# Boot token — appended as ?v= to every static asset URL the server-rendered
# index.html serves. Every time uvicorn restarts (e.g. after a code update)
# the token changes, so the browser is forced to refetch JS/CSS instead of
# serving last session's cached copy. This is the same pattern BC_LocalLLM
# uses to prevent "I see the old UI" support loops.
_BOOT_TOKEN = str(int(time.time()))


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
    from jobs.masks_store import init_masks_table
    conn = connect()
    init_db(conn)
    init_masks_table(conn)
    conn.close()

    from services.model_manager import manager
    from services.lama_service import load_model
    manager.register("lama", load_model)

    if os.environ.get("BC_WMR_AUTO_DOWNLOAD", "1") != "0":
        asyncio.create_task(_auto_download_models())

    from jobs.worker import start_worker
    start_worker()
    yield
    from jobs.worker import stop_worker
    stop_worker()


class _StaticCacheHeaders(BaseHTTPMiddleware):
    """Make /static/* short-lived and / no-cache.

    Without this, the browser would happily serve the previous run's JS for
    minutes after a deploy — that's exactly how a stale UI ends up sending
    `detect_mode=recall` to a server that has since changed defaults.
    """
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/"):
            # 60s cache + must-revalidate -> the browser will check on every
            # navigation, but won't refetch on every click within a page.
            response.headers["Cache-Control"] = "public, max-age=60, must-revalidate"
        elif path in ("/", "/index.html"):
            # The HTML carries the ?v=<boot_token> cache-buster, so it MUST
            # NOT be cached itself or the buster never changes.
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response


app = FastAPI(title="BC_WatermarkRemover", version="0.1.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(_StaticCacheHeaders)
app.add_middleware(GZipMiddleware, minimum_size=500)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

app.include_router(inpaint.router)
app.include_router(video.router)
app.include_router(detect.router)
app.include_router(jobs.router, prefix="/api/jobs")
app.include_router(masks.router, prefix="/api/masks")
app.include_router(status.router)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve index.html with a per-restart cache-buster baked into asset URLs."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("/static/css/style.css", f"/static/css/style.css?v={_BOOT_TOKEN}")
    html = html.replace("/static/js/", f"/static/js/")  # idempotent base
    # Append ?v=<token> to every <script src="/static/js/...">
    import re
    html = re.sub(
        r'(<script\s+src="/static/js/[^"]+)(")',
        rf'\1?v={_BOOT_TOKEN}\2',
        html,
    )
    return HTMLResponse(html)
