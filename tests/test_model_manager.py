import pytest
from unittest.mock import MagicMock, patch
import torch


def test_manager_unload_clears_vram():
    """Unloading a model moves it to CPU and clears cache."""
    from services.model_manager import ModelManager
    mgr = ModelManager.__new__(ModelManager)
    mgr._loaded = "test"
    mock_model = MagicMock()
    mgr._models = {"test": mock_model}
    mgr._device = torch.device("cpu")
    mgr._loaders = {}

    with patch("torch.cuda.is_available", return_value=False):
        mgr._unload_current()
    mock_model.cpu.assert_called_once()
    assert mgr._loaded is None


def test_manager_get_loads_on_demand():
    """Requesting a model loads it if not already loaded."""
    from services.model_manager import ModelManager
    mgr = ModelManager.__new__(ModelManager)
    mgr._loaded = None
    mgr._models = {}
    mgr._device = torch.device("cpu")

    mock_model = MagicMock()
    mgr._loaders = {"lama": lambda dev: mock_model}

    result = mgr.get("lama")
    assert result is mock_model
    assert mgr._loaded == "lama"
