"""GPU device selection via NVML ctypes — no pynvml dependency.

Default is CPU — co-residency with the Imagine pool / LLM caused
desktop freezes (see learning_gpu_allocation policy). To opt into
GPU, set BC_WMR_DEVICE=auto (or cuda:N for a specific card).

When auto-picking, skips GPUs listed in BC_WMR_SKIP_GPUS (default
"0,1,3" — leaves only GPU 2 free for watermark-remover work, since
0/2/3 are the Imagine pool and 1 is the LLM card). Override with an
empty value to consider all GPUs.
"""
from __future__ import annotations

import ctypes
import logging
import os

import torch

logger = logging.getLogger(__name__)

_DEVICE_PREF = os.environ.get("BC_WMR_DEVICE", "cpu").strip().lower()
_SKIP_GPUS = {int(x) for x in os.environ.get("BC_WMR_SKIP_GPUS", "0,1,3").split(",") if x.strip().isdigit()}


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
    # Honour explicit override first.
    if _DEVICE_PREF == "cpu":
        logger.info("BC_WMR_DEVICE=cpu - using CPU")
        return torch.device("cpu")
    if _DEVICE_PREF.startswith("cuda:") and torch.cuda.is_available():
        logger.info("BC_WMR_DEVICE=%s - using explicit GPU", _DEVICE_PREF)
        return torch.device(_DEVICE_PREF)

    if not torch.cuda.is_available():
        logger.info("CUDA not available - using CPU")
        return torch.device("cpu")

    n_gpus = torch.cuda.device_count()
    candidates = [i for i in range(n_gpus) if i not in _SKIP_GPUS]
    if not candidates:
        logger.warning("All GPUs are in skip-list - falling back to CPU")
        return torch.device("cpu")

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
