"""GPU device selection via NVML ctypes — no pynvml dependency.

Picks the GPU with the most free VRAM. Skips GPU 1 (LLM card) on
Ben's workstation. Falls back to CPU when no CUDA is available.
"""
from __future__ import annotations

import ctypes
import logging
import os

import torch

logger = logging.getLogger(__name__)

_SKIP_GPUS = {int(x) for x in os.environ.get("BC_WMR_SKIP_GPUS", "1").split(",") if x.strip().isdigit()}


def _nvml_free_bytes() -> dict[int, int]:
    """Return {gpu_index: free_bytes} via NVML ctypes. Empty dict on failure."""
    try:
        nvml = ctypes.cdll.LoadLibrary("nvml.dll" if os.name == "nt" else "libnvidia-ml.so.1")
    except OSError:
        return {}
    if nvml.nvmlInit_v2() != 0:
        return {}
    count = ctypes.c_uint()
    if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
        return {}
    result = {}
    for i in range(count.value):
        handle = ctypes.c_void_p()
        if nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle)) != 0:
            continue

        class MemInfo(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

        info = MemInfo()
        if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(info)) != 0:
            continue
        result[i] = info.free
    nvml.nvmlShutdown()
    return result


def get_device() -> torch.device:
    """Pick the best available device."""
    if not torch.cuda.is_available():
        logger.info("CUDA not available - using CPU")
        return torch.device("cpu")

    n_gpus = torch.cuda.device_count()
    candidates = [i for i in range(n_gpus) if i not in _SKIP_GPUS]
    if not candidates:
        candidates = list(range(n_gpus))

    free = _nvml_free_bytes()
    if free:
        candidates_with_free = [i for i in candidates if i in free]
        if candidates_with_free:
            best = max(candidates_with_free, key=lambda i: free[i])
            logger.info("Selected GPU %d (%s free)", best, _fmt_bytes(free[best]))
            return torch.device(f"cuda:{best}")

    best = max(candidates, key=lambda i: torch.cuda.mem_get_info(i)[0])
    logger.info("Selected GPU %d via torch.cuda.mem_get_info", best)
    return torch.device(f"cuda:{best}")


def gpu_status() -> list[dict]:
    """Return per-GPU status dicts for the /api/status endpoint."""
    if not torch.cuda.is_available():
        return []
    free_map = _nvml_free_bytes()
    result = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free = free_map.get(i, 0)
        total = props.total_memory
        result.append({
            "index": i,
            "name": props.name,
            "total_mb": total // (1024 * 1024),
            "free_mb": free // (1024 * 1024),
            "used_mb": (total - free) // (1024 * 1024),
            "skipped": i in _SKIP_GPUS,
        })
    return result


def _fmt_bytes(b: int) -> str:
    if b >= 1 << 30:
        return f"{b / (1 << 30):.1f} GB"
    return f"{b / (1 << 20):.0f} MB"
