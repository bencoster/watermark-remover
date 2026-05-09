import numpy as np
import pytest


def test_pad_to_mod():
    from services.lama_service import pad_to_mod
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100, 1), dtype=np.uint8)
    img_p, mask_p, pad = pad_to_mod(img, mask, mod=8)
    assert img_p.shape[0] % 8 == 0
    assert img_p.shape[1] % 8 == 0
    assert mask_p.shape[:2] == img_p.shape[:2]


def test_pad_already_aligned():
    from services.lama_service import pad_to_mod
    img = np.zeros((104, 96, 3), dtype=np.uint8)
    mask = np.zeros((104, 96, 1), dtype=np.uint8)
    img_p, mask_p, pad = pad_to_mod(img, mask, mod=8)
    assert img_p.shape == (104, 96, 3)
    assert pad == (0, 0, 0, 0)


def test_unpad():
    from services.lama_service import unpad
    img = np.zeros((104, 104, 3), dtype=np.uint8)
    result = unpad(img, (0, 4, 0, 4))
    assert result.shape == (100, 100, 3)
