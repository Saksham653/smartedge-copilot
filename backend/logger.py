# backend/logger.py
"""
SmartEdge Copilot – Metrics Logger
Handles SQLite initialization and structured metric logging for every LLM call.
"""

import sqlite3
import os
from pathlib import Path
from typing import Optional

from backend.utils import generate_request_id, utc_timestamp

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DEFAULT_DB_PATH = Path("data/metrics.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS llm_metrics (
    id              TEXT PRIMARY KEY,
    feature         TEXT    NOT NULL,
    tokens_used     INTEGER NOT NULL DEFAULT 0,
    latency_ms      REAL    NOT NULL DEFAULT 0.0,
    cost            REAL    NOT NULL DEFAULT 0.0,
    timestamp       TEXT    NOT NULL,
    response_text   TEXT,
    prompt_version  TEXT,
    model           TEXT,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    keyword_tags    TEXT
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_feature   ON llm_metrics (feature);",
    "CREATE INDEX IF NOT EXISTS idx_timestamp ON llm_metrics (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_model     ON llm_metrics (model);",
]


# ─────────────────────────────────────────────
# Database Init
# ─────────────────────────────────────────────

def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> str:
    """
    Initialize the SQLite database and create the metrics table if it
    does not already exist.  Safe to call on every app startup.

    Args:
        db_path: Path to the .db file. Parent directories are created
                 automatically.

    Returns:
        Resolved absolute path to the database file as a string.
    """
    db_path = Path(db_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with _get_connection(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)
        for index_sql in CREATE_INDEXES_SQL:
            conn.execute(index_sql)
        conn.commit()

    return str(db_path)


# ─────────────────────────────────────────────
# Connection Helper
# ─────────────────────────────────────────────

def _get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Return a SQLite connection with sensible defaults:
    - WAL mode for concurrent read/write
    - Row factory for dict-style access
    - Foreign keys enabled
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ─────────────────────────────────────────────
# Core Logging Function
# ─────────────────────────────────────────────

def log_metric(
    feature: str,
    tokens_used: int,
    latency_ms: float,
    cost: float,
    *,
    response_text: Optional[str] = None,
    prompt_version: Optional[str] = None,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    keyword_tags: Optional[list[str]] = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    request_id: Optional[str] = None,
) -> str:
    """
    Persist a single LLM call's performance metrics to SQLite.

    Args:
        feature:           Name of the product feature that triggered the call
                           e.g. "summarizer", "planner", "research".
        tokens_used:       Total tokens consumed (prompt + completion).
        latency_ms:        Wall-clock time for the API round-trip in milliseconds.
        cost:              Estimated USD cost for this call.
        response_text:     Optional – raw or truncated LLM response for audit/debug.
        prompt_version:    Optional – semantic version tag for the prompt template,
                           e.g. "v1.2".  Useful for A/B prompt experiments.
        model:             Optional – model identifier, e.g. "gpt-4o".
        prompt_tokens:     Optional – prompt-side token count (from API usage dict).
        completion_tokens: Optional – completion-side token count.
        keyword_tags:      Optional – list of searchable keywords, stored as
                           comma-separated string, e.g. ["python", "async"].
        db_path:           Path to the SQLite database file.
        request_id:        Optional – supply your own UUID; one is generated if omitted.

    Returns:
        The ``id`` (UUID string) of the newly inserted row.

    Raises:
        sqlite3.DatabaseError: On any database write failure.
    """
    row_id    = request_id or generate_request_id()
    timestamp = utc_timestamp()
    tags_str  = ",".join(keyword_tags) if keyword_tags else None

    insert_sql = """
        INSERT INTO llm_metrics (
            id, feature, tokens_used, latency_ms, cost, timestamp,
            response_text, prompt_version, model,
            prompt_tokens, completion_tokens, keyword_tags
        ) VALUES (
            :id, :feature, :tokens_used, :latency_ms, :cost, :timestamp,
            :response_text, :prompt_version, :model,
            :prompt_tokens, :completion_tokens, :keyword_tags
        );
    """

    params = {
        "id":                row_id,
        "feature":           feature,
        "tokens_used":       tokens_used,
        "latency_ms":        round(latency_ms, 3),
        "cost":              round(cost, 8),
        "timestamp":         timestamp,
        "response_text":     response_text,
        "prompt_version":    prompt_version,
        "model":             model,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "keyword_tags":      tags_str,
    }

    with _get_connection(db_path) as conn:
        conn.execute(insert_sql, params)
        conn.commit()

    return row_id


# ─────────────────────────────────────────────
# Convenience Query Helpers
# ─────────────────────────────────────────────

def fetch_metrics(
    *,
    feature: Optional[str] = None,
    model: Optional[str] = None,
    keyword: Optional[str] = None,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    limit: int = 500,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Query logged metrics with optional filters.

    Args:
        feature:  Filter by exact feature name.
        model:    Filter by exact model name.
        keyword:  Substring search across ``keyword_tags`` and ``feature``.
        start_ts: ISO-8601 lower bound for ``timestamp`` (inclusive).
        end_ts:   ISO-8601 upper bound for ``timestamp`` (inclusive).
        limit:    Maximum number of rows returned (default 500).
        db_path:  Path to the SQLite database file.

    Returns:
        List of dicts, newest records first.
    """
    conditions: list[str] = []
    params: dict = {}

    if feature:
        conditions.append("feature = :feature")
        params["feature"] = feature

    if model:
        conditions.append("model = :model")
        params["model"] = model

    if keyword:
        conditions.append(
            "(keyword_tags LIKE :kw OR feature LIKE :kw OR response_text LIKE :kw)"
        )
        params["kw"] = f"%{keyword}%"

    if start_ts:
        conditions.append("timestamp >= :start_ts")
        params["start_ts"] = start_ts

    if end_ts:
        conditions.append("timestamp <= :end_ts")
        params["end_ts"] = end_ts

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params["limit"] = limit

    sql = f"""
        SELECT *
        FROM   llm_metrics
        {where_clause}
        ORDER  BY timestamp DESC
        LIMIT  :limit;
    """

    with _get_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    return [dict(row) for row in rows]


def fetch_summary_stats(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict:
    """
    Return aggregate statistics across all logged calls.

    Returns a dict with keys:
        total_calls, total_tokens, total_cost_usd,
        avg_latency_ms, avg_cost_usd, unique_features.
    """
    sql = """
        SELECT
            COUNT(*)            AS total_calls,
            SUM(tokens_used)    AS total_tokens,
            SUM(cost)           AS total_cost_usd,
            AVG(latency_ms)     AS avg_latency_ms,
            AVG(cost)           AS avg_cost_usd,
            COUNT(DISTINCT feature) AS unique_features
        FROM llm_metrics;
    """
    with _get_connection(db_path) as conn:
        row = conn.execute(sql).fetchone()

    return dict(row) if row else {}


def delete_metric(
    request_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> bool:
    """
    Hard-delete a single metric row by its UUID.

    Returns True if a row was deleted, False if the ID was not found.
    """
    with _get_connection(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM llm_metrics WHERE id = ?;", (request_id,)
        )
        conn.commit()
    return cursor.rowcount > 0