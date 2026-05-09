import pytest
from unittest.mock import patch


def test_get_device_no_cuda():
    """When CUDA is unavailable, falls back to CPU."""
    with patch("torch.cuda.is_available", return_value=False):
        from services.cuda_policy import get_device
        import torch
        device = get_device()
        assert device == torch.device("cpu")


def test_get_device_returns_torch_device():
    """Return type is always torch.device."""
    from services.cuda_policy import get_device
    import torch
    device = get_device()
    assert isinstance(device, torch.device)
