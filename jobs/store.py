"""Stub — full implementation in Task 4."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "jobs.db"


def connect(path=None):
    return sqlite3.connect(str(path or DB_PATH))


def init_db(conn):
    pass
