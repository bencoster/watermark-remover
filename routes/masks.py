"""Mask library endpoints — list, fetch, rename, delete, import-from-diff."""
import os
import shutil
import tempfile
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from jobs.store import connect
from jobs import masks_store

router = APIRouter()

TMP_DIR = Path(__file__).parent.parent / "tmp"


class RenameRequest(BaseModel):
    name: str


@router.get("")
async def list_masks_route(limit: int = 100):
    conn = connect()
    try:
        masks_store.init_masks_table(conn)
        rows = masks_store.list_masks(conn, limit=limit)
        return JSONResponse({"masks": [dict(r) for r in rows]})
    finally:
        conn.close()


@router.get("/{mask_id}")
async def get_mask_route(mask_id: str):
    conn = connect()
    try:
        masks_store.init_masks_table(conn)
        row = masks_store.get_mask(conn, mask_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Mask not found")
        return JSONResponse(dict(row))
    finally:
        conn.close()


@router.get("/{mask_id}/file")
async def get_mask_file(mask_id: str):
    conn = connect()
    try:
        masks_store.init_masks_table(conn)
        row = masks_store.get_mask(conn, mask_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Mask not found")
        path = Path(row["mask_path"])
        if not path.exists():
            raise HTTPException(status_code=410, detail="Mask file gone")
        return FileResponse(str(path), media_type="image/png")
    finally:
        conn.close()


@router.get("/{mask_id}/thumb")
async def get_mask_thumb(mask_id: str):
    conn = connect()
    try:
        masks_store.init_masks_table(conn)
        row = masks_store.get_mask(conn, mask_id)
        if row is None or not row["thumb_path"]:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        path = Path(row["thumb_path"])
        if not path.exists():
            raise HTTPException(status_code=410, detail="Thumbnail file gone")
        return FileResponse(str(path), media_type="image/jpeg")
    finally:
        conn.close()


@router.patch("/{mask_id}")
async def rename_mask_route(mask_id: str, body: RenameRequest):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    conn = connect()
    try:
        masks_store.init_masks_table(conn)
        if not masks_store.rename_mask(conn, mask_id, name):
            raise HTTPException(status_code=404, detail="Mask not found")
        return JSONResponse({"id": mask_id, "name": name})
    finally:
        conn.close()


@router.post("/purge-auto")
async def purge_auto_saved_masks():
    """Delete every library entry that wasn't user-blessed.

    Diff Mask saves use p_full=1.0 (the user explicitly built and
    saved the mask). Auto-detection saves use the classifier score
    (typically 0.97 for watermarked images). This endpoint removes
    everything in the second category — the polluted-library issue
    that was causing the auto-match short-circuit to pull inferior
    masks.

    Also drops rows whose mask file is missing on disk.
    """
    conn = connect()
    try:
        masks_store.init_masks_table(conn)
        rows = masks_store.list_masks(conn, limit=1000)
        removed = 0
        kept = 0
        for r in rows:
            path = r["mask_path"]
            user_blessed = float(r["p_full"] or 0) >= 0.999
            file_missing = not (path and os.path.exists(path))
            if user_blessed and not file_missing:
                kept += 1
                continue
            if masks_store.delete_mask(conn, r["id"]):
                removed += 1
        return JSONResponse({"removed": removed, "kept": kept})
    finally:
        conn.close()


@router.post("/complete-lines")
async def complete_lines_route(
    mask: UploadFile = File(...),
    extend_px: int = 35,
    stroke_thickness: int = 6,
):
    """Extend partial line segments in a mask along their detected slope.

    Useful when a diff or auto-detect captured most of a tiled watermark
    pattern but skipped weak strokes between logos. Returns the completed
    mask as PNG.
    """
    from services.lattice_completion import complete_lines

    TMP_DIR.mkdir(exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=str(TMP_DIR))
    try:
        tmp.write(await mask.read())
        tmp.close()
        img = cv2.imread(tmp.name, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(status_code=400, detail="Cannot read mask image")
        completed = complete_lines(
            img, extend_px=extend_px, stroke_thickness=stroke_thickness
        )
        out_path = tempfile.mktemp(suffix=".png", dir=str(TMP_DIR))
        cv2.imwrite(out_path, completed)
        return FileResponse(out_path, media_type="image/png")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@router.post("/from-diff")
async def import_diff_mask(
    mask: UploadFile = File(...),
    source_image: UploadFile = File(...),
):
    """Persist a user-built diff mask to the library.

    Caller (the JS diff-mask tool) computes the binary mask client-side
    by differencing two aligned images. We just need to file it as a
    library entry alongside a thumbnail of the source image.
    """
    TMP_DIR.mkdir(exist_ok=True)
    masks_store.MASKS_DIR.mkdir(parents=True, exist_ok=True)
    masks_store.THUMBS_DIR.mkdir(parents=True, exist_ok=True)

    mask_id = os.urandom(6).hex()
    mask_path = masks_store.MASKS_DIR / f"{mask_id}.png"
    thumb_path = masks_store.THUMBS_DIR / f"{mask_id}.jpg"

    # Save the mask straight to the library folder
    with open(mask_path, "wb") as f:
        f.write(await mask.read())

    # Build a thumbnail from the source image
    src_tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(source_image.filename or "img.png").suffix, dir=str(TMP_DIR)
    )
    try:
        src_tmp.write(await source_image.read())
        src_tmp.close()
        img = cv2.imread(src_tmp.name)
        thumb_saved = None
        if img is not None:
            h, w = img.shape[:2]
            scale = 200 / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(thumb_path), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            thumb_saved = thumb_path

        # Mask coverage stat from the saved file
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        coverage = float((m > 0).mean()) if m is not None else 0.0

        conn = connect()
        try:
            masks_store.init_masks_table(conn)
            saved_id, _ = masks_store.save_mask(
                conn,
                mask_path=str(mask_path),
                thumb_path=str(thumb_saved) if thumb_saved else None,
                source_filename=source_image.filename or "diff",
                p_full=1.0,           # diff masks are user-supplied -> assume confident
                body_coverage=coverage,
                has_strip=False,
            )
            row = masks_store.get_mask(conn, saved_id)
            return JSONResponse({"id": saved_id, "name": row["name"], "coverage": coverage})
        finally:
            conn.close()
    finally:
        try:
            os.unlink(src_tmp.name)
        except OSError:
            pass


@router.delete("/{mask_id}")
async def delete_mask_route(mask_id: str):
    conn = connect()
    try:
        masks_store.init_masks_table(conn)
        if not masks_store.delete_mask(conn, mask_id):
            raise HTTPException(status_code=404, detail="Mask not found")
        return JSONResponse({"id": mask_id, "deleted": True})
    finally:
        conn.close()
