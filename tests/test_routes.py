import io
from pathlib import Path

import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from server import app
    with TestClient(app) as c:
        yield c


def _make_png(h=64, w=64, color=(255, 0, 0)):
    img = np.full((h, w, 3), color, dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return io.BytesIO(buf.tobytes())


def _make_mask(h=64, w=64):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    _, buf = cv2.imencode(".png", mask)
    return io.BytesIO(buf.tobytes())


def test_status_endpoint(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert "gpus" in data
    assert "loaded_model" in data


def test_jobs_list_empty(client):
    r = client.get("/api/jobs")
    assert r.status_code == 200
    data = r.json()
    assert "jobs" in data


def test_jobs_get_not_found(client):
    r = client.get("/api/jobs/nonexistent")
    assert r.status_code == 404


@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath("weights", "big-lama.pt").exists(),
    reason="LaMa weights not downloaded"
)
def test_inpaint_e2e(client):
    """End-to-end: upload image + mask, get inpainted result."""
    img_buf = _make_png(256, 256, color=(100, 150, 200))
    mask_buf = _make_mask(256, 256)

    r = client.post(
        "/api/inpaint",
        files=[
            ("file", ("test.png", img_buf, "image/png")),
            ("mask", ("mask.png", mask_buf, "image/png")),
        ],
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 100
