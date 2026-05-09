"""Single-model-in-VRAM manager with on-demand loading and swapping."""
from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from services.cuda_policy import get_device

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self._loaded: str | None = None
        self._models: dict[str, Any] = {}
        self._loaders: dict[str, Callable[[torch.device], Any]] = {}
        self._device = get_device()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def loaded_model_name(self) -> str | None:
        return self._loaded

    def register(self, name: str, loader: Callable[[torch.device], Any]):
        self._loaders[name] = loader

    def get(self, name: str) -> Any:
        if self._loaded != name:
            self._unload_current()
            self._load(name)
        return self._models[name]

    def _load(self, name: str):
        if name not in self._loaders:
            raise ValueError(f"Unknown model: {name!r}. Registered: {sorted(self._loaders)}")
        logger.info("Loading model %r onto %s", name, self._device)
        model = self._loaders[name](self._device)
        self._models[name] = model
        self._loaded = name
        logger.info("Model %r loaded", name)

    def _unload_current(self):
        if self._loaded is None:
            return
        logger.info("Unloading model %r", self._loaded)
        model = self._models.pop(self._loaded, None)
        if model is not None and hasattr(model, "cpu"):
            model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = None

    def unload_all(self):
        self._unload_current()
        self._models.clear()


manager = ModelManager()
