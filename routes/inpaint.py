"""POST /api/inpaint - still image watermark removal via LaMa."""
import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

from services.model_manager import manager
from services.lama_service import inpaint

router = APIRouter()

TMP_DIR = Path(__file__).parent.parent / "tmp"


@router.post("/api/inpaint")
async def inpaint_endpoint(
    file: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    TMP_DIR.mkdir(exist_ok=True)
    img_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "img.png").suffix, dir=str(TMP_DIR))
    mask_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=str(TMP_DIR))

    try:
        img_tmp.write(await file.read())
        img_tmp.close()
        mask_tmp.write(await mask.read())
        mask_tmp.close()

        model = manager.get("lama")
        result_path = await asyncio.to_thread(inpaint, img_tmp.name, mask_tmp.name, manager.device, model)
        return FileResponse(result_path, media_type="image/png", filename="inpainted.png")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        for p in (img_tmp.name, mask_tmp.name):
            try:
                os.unlink(p)
            except OSError:
                pass
