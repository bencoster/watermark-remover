"""POST /api/detect - auto-detect watermark region and return mask."""
import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()

TMP_DIR = Path(__file__).parent.parent / "tmp"


@router.post("/api/detect")
async def detect_watermark(file: UploadFile = File(...)):
    TMP_DIR.mkdir(exist_ok=True)
    img_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "img.png").suffix, dir=str(TMP_DIR))

    try:
        img_tmp.write(await file.read())
        img_tmp.close()

        from services.detector_service import detect
        result = await asyncio.to_thread(detect, img_tmp.name)

        if result is None:
            return JSONResponse({"detected": False}, status_code=200)

        return FileResponse(result, media_type="image/png", filename="mask.png")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            os.unlink(img_tmp.name)
        except OSError:
            pass
