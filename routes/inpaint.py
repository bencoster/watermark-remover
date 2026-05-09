"""POST /api/inpaint - still image watermark removal via LaMa."""
import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import cv2
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

from services.model_manager import manager
from services.lama_service import inpaint
from jobs.store import connect
from jobs import masks_store

router = APIRouter()

TMP_DIR = Path(__file__).parent.parent / "tmp"
MASKS_DIR = masks_store.MASKS_DIR
THUMBS_DIR = masks_store.THUMBS_DIR


def _save_mask_to_library(
    mask_path: str,
    source_image_path: str,
    source_filename: str,
    p_full: float,
    body_coverage: float,
    has_strip: bool,
) -> str:
    """Copy a mask + render a thumbnail, persist the record, return the
    mask id. Failures are logged but never bubble — the inpaint request
    still succeeds even if library archiving doesn't."""
    try:
        MASKS_DIR.mkdir(parents=True, exist_ok=True)
        THUMBS_DIR.mkdir(parents=True, exist_ok=True)

        mask_id = os.urandom(6).hex()
        new_mask = MASKS_DIR / f"{mask_id}.png"
        shutil.copy(mask_path, new_mask)

        thumb_path = THUMBS_DIR / f"{mask_id}.jpg"
        img = cv2.imread(source_image_path)
        if img is not None:
            h, w = img.shape[:2]
            scale = 200 / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(thumb_path), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        else:
            thumb_path = None

        conn = connect()
        try:
            masks_store.init_masks_table(conn)
            saved_id, _ = masks_store.save_mask(
                conn,
                mask_path=str(new_mask),
                thumb_path=str(thumb_path) if thumb_path else None,
                source_filename=source_filename,
                p_full=p_full,
                body_coverage=body_coverage,
                has_strip=has_strip,
            )
            return saved_id
        finally:
            conn.close()
    except Exception:
        import logging
        logging.getLogger(__name__).exception("save_mask_to_library failed")
        return ""


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

        # Combine body + strip into a single library entry for reuse.
        combined_mask_path = body_mask_path
        if strip_mask_path is not None:
            combined = cv2.bitwise_or(
                cv2.imread(body_mask_path, cv2.IMREAD_GRAYSCALE),
                cv2.imread(strip_mask_path, cv2.IMREAD_GRAYSCALE),
            )
            combined_mask_path = tempfile.mktemp(suffix=".png", dir=str(TMP_DIR))
            cv2.imwrite(combined_mask_path, combined)

        body = cv2.imread(body_mask_path, cv2.IMREAD_GRAYSCALE)
        body_coverage = float((body > 0).mean()) if body is not None else 0.0
        await asyncio.to_thread(
            _save_mask_to_library,
            combined_mask_path,
            img_tmp.name,
            file.filename or "",
            float(p_full),
            body_coverage,
            strip_mask_path is not None,
        )

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
