"""POST /api/inpaint - still image watermark removal via LaMa."""
import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path

import cv2
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)

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


def _find_matching_library_mask(image_path: str, tolerance: float = 0.05) -> str | None:
    """Find a saved library mask whose dimensions are within `tolerance`
    of the input image. Returns the local mask file path if a candidate
    exists. We rank by recency — newest matches first.

    Why this matters: when a user has built a perfect diff mask for a
    Dreamstime layout once, the next Dreamstime image at that resolution
    can reach ~28 dB PSNR vs the SaaS reference (vs ~24 dB from auto
    detect). The simple dimension match is the cheapest way to get
    that without per-image manual work.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    target_h, target_w = img.shape[:2]

    conn = connect()
    try:
        masks_store.init_masks_table(conn)
        rows = masks_store.list_masks(conn, limit=200)
    finally:
        conn.close()

    for row in rows:
        path = row["mask_path"]
        if not path or not os.path.exists(path):
            continue
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        mh, mw = m.shape
        if mh == 0 or mw == 0:
            continue
        # Within tolerance both dimensions
        if abs(mh - target_h) / max(target_h, 1) <= tolerance and \
           abs(mw - target_w) / max(target_w, 1) <= tolerance:
            return path
    return None


@router.post("/api/auto")
async def auto_pipeline_endpoint(
    file: UploadFile = File(...),
    strip_mode: str = Form("inpaint"),
    detect_mode: str = Form("auto"),
    library_mask: str = Form("auto"),
    strip_engine: str = Form("telea"),
):
    """One-click watermark removal.

    Pipeline: classifier-gated detector → strip handling → LaMa body inpaint.

    `detect_mode`:
      - "auto" (default) — fit a 2D lattice to detected blobs. Tiled
        pattern → recall; single mark → precision. Picks correctly for
        stock photos and corner stamps without user input.
      - "recall" — use Grad-CAM heatmap directly. Force this for tiled
        watermarks (Dreamstime, Shutterstock) where auto isn't sure.
      - "precision" — AND with low-saturation heuristic. Force this
        for single-watermark images where over-mask is risky.

    `strip_mode` controls bottom solid-colour footer handling:
      - "inpaint" (default) — TELEA pre-fill, LaMa cleans rest.
      - "crop" — cut the image above the footer.
    """
    if strip_mode not in ("inpaint", "crop"):
        return JSONResponse(
            {"error": f"Invalid strip_mode: {strip_mode!r}. Use 'inpaint' or 'crop'."},
            status_code=400,
        )
    if detect_mode not in ("auto", "recall", "precision"):
        return JSONResponse(
            {"error": f"Invalid detect_mode: {detect_mode!r}. Use 'auto', 'recall', or 'precision'."},
            status_code=400,
        )
    if library_mask not in ("auto", "off"):
        return JSONResponse(
            {"error": f"Invalid library_mask: {library_mask!r}. Use 'auto' or 'off'."},
            status_code=400,
        )
    if strip_engine not in ("telea", "sdxl"):
        return JSONResponse(
            {"error": f"Invalid strip_engine: {strip_engine!r}. Use 'telea' or 'sdxl'."},
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

        # Library-mask short-circuit: if a previously-saved mask matches
        # the input dimensions, use it directly. This is the path that
        # delivers SaaS-equivalent quality on repeat watermarks (~28 dB
        # PSNR on the dreamstime fixture vs ~24 dB from auto detect).
        if library_mask == "auto":
            cached_mask = await asyncio.to_thread(
                _find_matching_library_mask, img_tmp.name
            )
            if cached_mask:
                logger.info("library mask hit: %s", cached_mask)
                model = manager.get("lama")
                result_path = await asyncio.to_thread(
                    inpaint, img_tmp.name, cached_mask, manager.device, model
                )
                resp = FileResponse(result_path, media_type="image/png", filename="cleaned.png")
                resp.headers["X-Mask-Source"] = "library"
                return resp

        result = await asyncio.to_thread(detect_split, img_tmp.name, 0.30, 0.10, 0.70, detect_mode)
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
            elif strip_engine == "sdxl":
                # SDXL second pass on the strip region only — gives
                # plausible texture continuation where TELEA produces
                # smudgy gradients. Falls back to TELEA on any failure
                # (model download issue, OOM, etc.) so a strip is
                # always handled even if SDXL is unavailable.
                from services import sdxl_inpaint_service
                ok, why = sdxl_inpaint_service.is_available(manager.device)
                if ok:
                    try:
                        manager.unload_all()  # free LaMa VRAM before SDXL
                        pipe = await asyncio.to_thread(
                            sdxl_inpaint_service.load_pipeline, manager.device
                        )
                        image_for_lama = await asyncio.to_thread(
                            sdxl_inpaint_service.inpaint_strip_region,
                            img_tmp.name, strip_mask_path, pipe,
                        )
                        # Free SDXL VRAM so LaMa body pass can reload.
                        del pipe
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        logger.exception("SDXL strip pass failed; falling back to TELEA")
                        image_for_lama = await asyncio.to_thread(
                            prefill_strip, img_tmp.name, strip_mask_path, 4
                        )
                else:
                    logger.warning("SDXL skipped: %s", why)
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
