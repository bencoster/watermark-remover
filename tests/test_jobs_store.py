import sqlite3
import pytest
from jobs.store import init_db, enqueue, get, update_status, list_jobs


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def test_enqueue_creates_job(db):
    job_id = enqueue(db, kind="video_inpaint", payload={"file": "test.mp4"})
    assert job_id is not None
    row = get(db, job_id)
    assert row["status"] == "queued"
    assert row["kind"] == "video_inpaint"


def test_update_status(db):
    job_id = enqueue(db, kind="video_inpaint", payload={})
    update_status(db, job_id, "running", progress=0.5, stage="flow_completion")
    row = get(db, job_id)
    assert row["status"] == "running"
    assert row["progress"] == 0.5
    assert row["stage"] == "flow_completion"


def test_list_jobs_filters_by_status(db):
    id1 = enqueue(db, kind="video_inpaint", payload={})
    id2 = enqueue(db, kind="video_inpaint", payload={})
    update_status(db, id1, "succeeded")
    rows = list_jobs(db, status="queued")
    assert len(rows) == 1
    assert rows[0]["id"] == id2
