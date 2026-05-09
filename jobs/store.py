"""SQLite-backed job queue - simplified from BC_LocalLLM pattern."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "jobs.db"


def connect(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            payload TEXT NOT NULL DEFAULT '{}',
            progress REAL DEFAULT 0.0,
            stage TEXT DEFAULT '',
            result_path TEXT DEFAULT '',
            error TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    """)


def enqueue(conn: sqlite3.Connection, kind: str, payload: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO jobs (id, kind, payload, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, kind, json.dumps(payload), now, now),
    )
    conn.commit()
    return job_id


def get(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def update_status(
    conn: sqlite3.Connection,
    job_id: str,
    status: str,
    progress: float | None = None,
    stage: str | None = None,
    result_path: str | None = None,
    error: str | None = None,
):
    now = datetime.now(timezone.utc).isoformat()
    fields = ["status = ?", "updated_at = ?"]
    values: list = [status, now]
    if progress is not None:
        fields.append("progress = ?")
        values.append(progress)
    if stage is not None:
        fields.append("stage = ?")
        values.append(stage)
    if result_path is not None:
        fields.append("result_path = ?")
        values.append(result_path)
    if error is not None:
        fields.append("error = ?")
        values.append(error)
    values.append(job_id)
    conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()


def list_jobs(conn: sqlite3.Connection, status: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
    if status:
        return conn.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()


def claim_next(conn: sqlite3.Connection) -> sqlite3.Row | None:
    row = conn.execute(
        "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
    ).fetchone()
    if row:
        update_status(conn, row["id"], "running", progress=0.0, stage="starting")
    return row
