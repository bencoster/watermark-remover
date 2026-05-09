"""Mask library — persisted detected watermark masks for reuse.

Lives in the same SQLite DB as the job queue (jobs.db) but in its
own table. A "mask" entry pairs:
  - the binary watermark mask (PNG on disk)
  - a small thumbnail of the source image (for the library UI)
  - metadata: auto-generated name, classifier confidence, coverage,
    whether a strip-text mask was detected, and a fingerprint hash
    used for de-duplication.
"""
from __future__ import annotations

import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from jobs.store import connect

MASKS_DIR = Path(__file__).parent.parent / "masks"
THUMBS_DIR = MASKS_DIR / "thumbs"


def init_masks_table(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS masks (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            mask_path TEXT NOT NULL,
            thumb_path TEXT,
            source_filename TEXT,
            p_full REAL DEFAULT 0.0,
            body_coverage REAL DEFAULT 0.0,
            has_strip INTEGER DEFAULT 0,
            fingerprint TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_masks_created ON masks(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_masks_fingerprint ON masks(fingerprint);
    """)


def auto_name(source_filename: Optional[str], when: Optional[datetime] = None) -> str:
    """Generate a friendly default mask name.

    Format: `wm_<YYYY-MM-DD>_<hh-mm>_<short>` — short enough to read,
    unique enough to avoid collisions in casual use. Source filename
    stem is included only when it looks like a meaningful name (not a
    pure-hash filename which we treat as noise).
    """
    when = when or datetime.now(timezone.utc)
    short = uuid.uuid4().hex[:4]
    base = f"wm_{when.strftime('%Y-%m-%d_%H-%M')}_{short}"
    if source_filename:
        stem = Path(source_filename).stem
        # Skip pure-hex filenames (looks like a content hash / GUID,
        # e.g. "a1b2c3d4e5f60718") and mostly-digit IDs.
        is_hex_only = stem and all(c in "0123456789abcdefABCDEF" for c in stem) and len(stem) >= 12
        digit_frac = sum(c.isdigit() for c in stem) / max(1, len(stem))
        if not is_hex_only and digit_frac < 0.7 and len(stem) <= 32:
            cleaned = "".join(c if c.isalnum() else "_" for c in stem)[:24]
            base = f"{cleaned}_{short}"
    return base


def fingerprint_mask(mask_bytes: bytes) -> str:
    """SHA-1 of the binary mask — used to detect "this exact mask
    has been saved before". Good enough for de-dup."""
    return hashlib.sha1(mask_bytes).hexdigest()[:16]


def save_mask(
    conn: sqlite3.Connection,
    mask_path: str,
    thumb_path: Optional[str],
    source_filename: Optional[str],
    p_full: float,
    body_coverage: float,
    has_strip: bool,
    name: Optional[str] = None,
) -> tuple[str, bool]:
    """Insert a new mask record. Returns (id, created).

    `created` is False when the same fingerprint already exists — in
    that case we return the pre-existing id and skip writing a duplicate.
    """
    with open(mask_path, "rb") as f:
        fp = fingerprint_mask(f.read())

    existing = conn.execute(
        "SELECT id FROM masks WHERE fingerprint = ? LIMIT 1", (fp,)
    ).fetchone()
    if existing is not None:
        return existing["id"], False

    mask_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO masks
           (id, name, mask_path, thumb_path, source_filename, p_full,
            body_coverage, has_strip, fingerprint, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (mask_id, name or auto_name(source_filename), mask_path,
         thumb_path, source_filename, p_full, body_coverage,
         1 if has_strip else 0, fp, now),
    )
    conn.commit()
    return mask_id, True


def list_masks(conn: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    # ROWID tiebreaker so saves within the same wall-clock second still
    # come out newest-first (the timestamp resolution is one second).
    return conn.execute(
        "SELECT * FROM masks ORDER BY created_at DESC, ROWID DESC LIMIT ?", (limit,)
    ).fetchall()


def get_mask(conn: sqlite3.Connection, mask_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM masks WHERE id = ?", (mask_id,)).fetchone()


def rename_mask(conn: sqlite3.Connection, mask_id: str, new_name: str) -> bool:
    cur = conn.execute("UPDATE masks SET name = ? WHERE id = ?", (new_name, mask_id))
    conn.commit()
    return cur.rowcount > 0


def delete_mask(conn: sqlite3.Connection, mask_id: str) -> bool:
    row = get_mask(conn, mask_id)
    if row is None:
        return False
    conn.execute("DELETE FROM masks WHERE id = ?", (mask_id,))
    conn.commit()
    # Best-effort cleanup of files; don't fail the request if a file is
    # already gone (mask folder was cleared, etc.)
    for col in ("mask_path", "thumb_path"):
        path = row[col]
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
    return True
