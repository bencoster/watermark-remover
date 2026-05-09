"""GET /api/status - GPU info and loaded model status."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from services.cuda_policy import gpu_status
from services.model_manager import manager

router = APIRouter()


@router.get("/api/status")
async def status():
    return JSONResponse({
        "gpus": gpu_status(),
        "loaded_model": manager.loaded_model_name,
        "device": str(manager.device),
    })
