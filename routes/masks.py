"""Mask library endpoints — list, fetch, rename, delete."""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from jobs.store import connect
from jobs import masks_store

router = APIRouter()


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
