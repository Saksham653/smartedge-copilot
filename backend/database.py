"""
database.py — SmartEdge Copilot
Central database management module.
Handles SQLite connection and table initialisation for project_root/data/metrics.db.
"""
from __future__ import annotations

import os
import sqlite3

# Resolve project root reliably assuming this file lives in project_root/research_copilot/backend/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research_copilot
DB_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DB_DIR, "metrics.db")


def get_connection() -> sqlite3.Connection:
    """
    Return a new SQLite connection to the project database.
    Ensures the data directory exists before connecting.

    Caller is responsible for closing the connection when done.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """
    Initialise the database by creating all required tables (idempotent).
    Tables:
      - metrics
      - research_notes
      - meeting_notes
      - tasks
    Safe to call multiple times (call at app startup).
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name      TEXT,
                model             TEXT,
                prompt_tokens     INTEGER,
                completion_tokens INTEGER,
                total_tokens      INTEGER,
                latency_ms        REAL,
                cost              REAL,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_notes (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                query            TEXT,
                optimized_prompt TEXT,
                summary          TEXT,
                key_concepts     TEXT,
                applications     TEXT,
                references_text  TEXT,
                total_tokens     INTEGER,
                latency_ms       REAL,
                cost             REAL,
                model            TEXT,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meeting_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                transcript TEXT,
                summary TEXT,
                key_topics TEXT,
                action_items TEXT,
                deadlines TEXT,
                decisions TEXT,
                total_tokens INTEGER,
                latency_ms REAL,
                cost REAL,
                model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT,
                source_id INTEGER,
                assignee TEXT,
                task_description TEXT,
                deadline TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
    finally:
        conn.close()