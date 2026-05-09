"""Job management endpoints - mounted at /api/jobs."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse

from jobs.store import connect, get, list_jobs

router = APIRouter()


@router.get("")
async def list_jobs_route(status: str | None = None, limit: int = 50):
    conn = connect()
    try:
        rows = list_jobs(conn, status=status, limit=limit)
        return JSONResponse({"jobs": [dict(r) for r in rows]})
    finally:
        conn.close()


@router.get("/{job_id}")
async def get_job(job_id: str):
    conn = connect()
    try:
        row = get(conn, job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JSONResponse(dict(row))
    finally:
        conn.close()


@router.get("/{job_id}/result")
async def get_job_result(job_id: str):
    conn = connect()
    try:
        row = get(conn, job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if row["status"] != "succeeded":
            raise HTTPException(status_code=409, detail=f"Job status: {row['status']}")
        path = row["result_path"]
        if not path:
            raise HTTPException(status_code=404, detail="No result file")
        return FileResponse(path)
    finally:
        conn.close()
