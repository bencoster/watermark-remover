import os
import sqlite3
import tempfile
import pytest

from jobs.masks_store import (
    init_masks_table, save_mask, list_masks, get_mask,
    rename_mask, delete_mask, auto_name, fingerprint_mask,
)


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_masks_table(conn)
    return conn


@pytest.fixture
def mask_file(tmp_path):
    """A small uniformly-grey PNG that we can re-save and fingerprint."""
    p = tmp_path / "m.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)  # not a real PNG, just bytes
    return str(p)


def test_auto_name_with_meaningful_filename():
    name = auto_name("dreamstime_1234.jpg")
    assert "dreamstime" in name


def test_auto_name_with_hash_filename():
    # Pure-hex filenames should be ignored - use generic auto-name
    name = auto_name("a1b2c3d4e5f60718.jpg")
    assert name.startswith("wm_")


def test_fingerprint_is_stable():
    assert fingerprint_mask(b"abc") == fingerprint_mask(b"abc")
    assert fingerprint_mask(b"abc") != fingerprint_mask(b"abd")


def test_save_and_fetch(db, mask_file):
    mid, created = save_mask(db, mask_file, None, "img.jpg", 0.95, 0.04, has_strip=True)
    assert created is True
    row = get_mask(db, mid)
    assert row["p_full"] == 0.95
    assert row["has_strip"] == 1


def test_dedup_returns_same_id(db, mask_file):
    mid1, c1 = save_mask(db, mask_file, None, "a.jpg", 0.9, 0.04, has_strip=False)
    mid2, c2 = save_mask(db, mask_file, None, "b.jpg", 0.9, 0.04, has_strip=False)
    assert mid1 == mid2
    assert c1 is True and c2 is False


def test_list_returns_newest_first(db, tmp_path):
    p1 = tmp_path / "a.png"; p1.write_bytes(b"first")
    p2 = tmp_path / "b.png"; p2.write_bytes(b"second")
    id1, _ = save_mask(db, str(p1), None, "a.jpg", 0.9, 0.0, False)
    id2, _ = save_mask(db, str(p2), None, "b.jpg", 0.9, 0.0, False)
    rows = list_masks(db)
    assert len(rows) == 2
    # Newest first - id2 was created after id1
    assert rows[0]["id"] == id2


def test_rename(db, mask_file):
    mid, _ = save_mask(db, mask_file, None, "a.jpg", 0.9, 0.0, False)
    assert rename_mask(db, mid, "my-watermark") is True
    assert get_mask(db, mid)["name"] == "my-watermark"


def test_delete(db, mask_file):
    mid, _ = save_mask(db, mask_file, None, "a.jpg", 0.9, 0.0, False)
    assert delete_mask(db, mid) is True
    assert get_mask(db, mid) is None
