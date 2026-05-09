"""POST /api/inpaint - still image watermark removal via LaMa."""
import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
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


@router.post("/api/auto")
async def auto_pipeline_endpoint(
    file: UploadFile = File(...),
    strip_mode: str = Form("inpaint"),
):
    """One-click watermark removal.

    Pipeline: classifier-gated detector → strip handling → LaMa body inpaint.

    `strip_mode` controls how the bottom solid-colour footer (e.g.
    Dreamstime's blue bar with the URL/ID text) is handled:
      - "inpaint" (default) — TELEA pre-fill the strip from the rows
        above, then LaMa cleans body watermarks. Image dimensions
        preserved.
      - "crop" — physically crop the image just above the footer.
        No hallucination risk, but image gets shorter. Falls back to
        inpaint when the strip mask isn't a clean full-width bar.
    """
    if strip_mode not in ("inpaint", "crop"):
        return JSONResponse(
            {"error": f"Invalid strip_mode: {strip_mode!r}. Use 'inpaint' or 'crop'."},
            status_code=400,
        )

    from services.detector_service import (
        detect_split, prefill_strip, crop_strip, crop_strip_mask,
    )

    TMP_DIR.mkdir(exist_ok=True)
    img_tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename or "img.png").suffix, dir=str(TMP_DIR)
    )

    try:
        img_tmp.write(await file.read())
        img_tmp.close()

        result = await asyncio.to_thread(detect_split, img_tmp.name)
        if result is None:
            return JSONResponse(
                {"detected": False, "message": "No watermark detected"},
                status_code=422,
            )
        body_mask_path, strip_mask_path, p_full = result

        image_for_lama = img_tmp.name
        body_mask_for_lama = body_mask_path

        if strip_mask_path is not None:
            if strip_mode == "crop":
                cropped = await asyncio.to_thread(crop_strip, img_tmp.name, strip_mask_path)
                if cropped is not None:
                    image_for_lama = cropped
                    body_mask_for_lama = await asyncio.to_thread(
                        crop_strip_mask, body_mask_path, strip_mask_path
                    )
                else:
                    # No clean full-width bar -> fall back to TELEA.
                    image_for_lama = await asyncio.to_thread(
                        prefill_strip, img_tmp.name, strip_mask_path, 4
                    )
            else:
                image_for_lama = await asyncio.to_thread(
                    prefill_strip, img_tmp.name, strip_mask_path, 4
                )

        model = manager.get("lama")
        result_path = await asyncio.to_thread(
            inpaint, image_for_lama, body_mask_for_lama, manager.device, model
        )
        return FileResponse(result_path, media_type="image/png", filename="cleaned.png")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            os.unlink(img_tmp.name)
        except OSError:
            pass
