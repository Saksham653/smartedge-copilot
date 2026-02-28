# backend/metrics.py
"""
SmartEdge Copilot – Metrics Query Layer
Fetches logged LLM metrics from SQLite and returns clean pandas DataFrames
ready for analytics, charting, and the Streamlit dashboard.
"""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from backend.logger import DEFAULT_DB_PATH, _get_connection

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

NUMERIC_COLS = ["tokens_used", "latency_ms", "cost",
                "prompt_tokens", "completion_tokens"]

DATETIME_COL = "timestamp"


# ─────────────────────────────────────────────
# Core Query
# ─────────────────────────────────────────────

def query_metrics(
    start_time: Optional[str] = None,
    end_time:   Optional[str] = None,
    feature:    Optional[str] = None,
    *,
    model:          Optional[str]       = None,
    keyword:        Optional[str]       = None,
    prompt_version: Optional[str]       = None,
    min_cost:       Optional[float]     = None,
    max_cost:       Optional[float]     = None,
    min_latency_ms: Optional[float]     = None,
    max_latency_ms: Optional[float]     = None,
    limit:          int                 = 5_000,
    db_path:        str | Path          = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Query the llm_metrics table and return a typed, analytics-ready DataFrame.

    Args:
        start_time:     ISO-8601 lower bound for timestamp (inclusive).
                        e.g. "2024-01-01T00:00:00+00:00"
        end_time:       ISO-8601 upper bound for timestamp (inclusive).
        feature:        Exact feature name filter, e.g. "summarizer".
        model:          Exact model name filter, e.g. "gpt-4o".
        keyword:        Substring search across feature, keyword_tags,
                        and response_text columns.
        prompt_version: Exact prompt version filter, e.g. "v2.1".
        min_cost:       Minimum cost (USD) filter.
        max_cost:       Maximum cost (USD) filter.
        min_latency_ms: Minimum latency filter.
        max_latency_ms: Maximum latency filter.
        limit:          Maximum rows returned (default 5 000).
        db_path:        Path to the SQLite database file.

    Returns:
        pandas DataFrame with columns:
            id, feature, tokens_used, latency_ms, cost, timestamp,
            response_text, prompt_version, model, prompt_tokens,
            completion_tokens, keyword_tags

        - ``timestamp`` is parsed to UTC-aware datetime.
        - Numeric columns are cast to float64 / Int64.
        - Missing numerics are filled with 0.
        - Returns an empty DataFrame (with correct columns) when no rows match.
    """
    conditions: list[str] = []
    params: dict           = {}

    if start_time:
        conditions.append("timestamp >= :start_time")
        params["start_time"] = start_time

    if end_time:
        conditions.append("timestamp <= :end_time")
        params["end_time"] = end_time

    if feature:
        conditions.append("feature = :feature")
        params["feature"] = feature

    if model:
        conditions.append("model = :model")
        params["model"] = model

    if prompt_version:
        conditions.append("prompt_version = :prompt_version")
        params["prompt_version"] = prompt_version

    if keyword:
        conditions.append(
            "(feature       LIKE :kw "
            " OR keyword_tags  LIKE :kw "
            " OR response_text LIKE :kw)"
        )
        params["kw"] = f"%{keyword}%"

    if min_cost is not None:
        conditions.append("cost >= :min_cost")
        params["min_cost"] = min_cost

    if max_cost is not None:
        conditions.append("cost <= :max_cost")
        params["max_cost"] = max_cost

    if min_latency_ms is not None:
        conditions.append("latency_ms >= :min_latency")
        params["min_latency"] = min_latency_ms

    if max_latency_ms is not None:
        conditions.append("latency_ms <= :max_latency")
        params["max_latency"] = max_latency_ms

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params["limit"] = limit

    sql = f"""
        SELECT
            id,
            feature,
            tokens_used,
            latency_ms,
            cost,
            timestamp,
            response_text,
            prompt_version,
            model,
            prompt_tokens,
            completion_tokens,
            keyword_tags
        FROM  llm_metrics
        {where}
        ORDER BY timestamp DESC
        LIMIT :limit;
    """

    try:
        with _get_connection(db_path) as conn:
            df = pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return _empty_dataframe()

    if df.empty:
        return _empty_dataframe()

    return _clean(df)


# ─────────────────────────────────────────────
# Aggregation Queries
# ─────────────────────────────────────────────

def query_feature_summary(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Return per-feature aggregate statistics.

    Columns:
        feature, call_count, total_tokens, total_cost,
        avg_latency_ms, avg_cost, avg_tokens,
        p95_latency_ms, p99_latency_ms
    """
    sql = """
        SELECT
            feature,
            COUNT(*)                    AS call_count,
            SUM(tokens_used)            AS total_tokens,
            SUM(cost)                   AS total_cost,
            AVG(latency_ms)             AS avg_latency_ms,
            AVG(cost)                   AS avg_cost,
            AVG(tokens_used)            AS avg_tokens
        FROM  llm_metrics
        GROUP BY feature
        ORDER BY call_count DESC;
    """
    try:
        with _get_connection(db_path) as conn:
            df = pd.read_sql_query(sql, conn)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    # Compute percentiles in pandas (SQLite lacks PERCENTILE_CONT)
    raw = query_metrics(limit=100_000, db_path=db_path)
    if not raw.empty:
        p_stats = (
            raw.groupby("feature")["latency_ms"]
            .quantile([0.95, 0.99])
            .unstack(level=-1)
            .rename(columns={0.95: "p95_latency_ms", 0.99: "p99_latency_ms"})
            .reset_index()
        )
        df = df.merge(p_stats, on="feature", how="left")

    return df.round(4)


def query_time_series(
    freq:    str              = "1h",
    feature: Optional[str]   = None,
    db_path: str | Path       = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Resample metrics into a time-series DataFrame for trend charts.

    Args:
        freq:    Pandas offset alias – "1h", "1D", "15min", etc.
        feature: Optional feature filter applied before resampling.
        db_path: SQLite path.

    Returns:
        DataFrame indexed by UTC timestamp with columns:
            call_count, total_tokens, total_cost, avg_latency_ms, avg_cost
    """
    df = query_metrics(feature=feature, limit=100_000, db_path=db_path)

    if df.empty:
        return pd.DataFrame()

    df = df.set_index("timestamp").sort_index()

    ts = df.resample(freq).agg(
        call_count    = ("tokens_used",  "count"),
        total_tokens  = ("tokens_used",  "sum"),
        total_cost    = ("cost",         "sum"),
        avg_latency_ms= ("latency_ms",   "mean"),
        avg_cost      = ("cost",         "mean"),
    ).fillna(0)

    return ts.round(6)


def query_cost_trend(
    freq:    str            = "1D",
    db_path: str | Path     = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Daily (or custom frequency) cumulative cost trend.

    Returns:
        DataFrame with columns: period, daily_cost, cumulative_cost
    """
    ts = query_time_series(freq=freq, db_path=db_path)

    if ts.empty:
        return pd.DataFrame()

    df = ts[["total_cost"]].rename(columns={"total_cost": "daily_cost"}).copy()
    df["cumulative_cost"] = df["daily_cost"].cumsum()
    df = df.reset_index().rename(columns={"timestamp": "period"})

    return df.round(8)


def query_latency_over_time(
    freq:    str            = "1h",
    feature: Optional[str] = None,
    db_path: str | Path     = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Average latency resampled over time for a latency-trend chart.

    Returns:
        DataFrame with columns: timestamp, avg_latency_ms, p95_latency_ms
    """
    df = query_metrics(feature=feature, limit=100_000, db_path=db_path)

    if df.empty:
        return pd.DataFrame()

    df = df.set_index("timestamp").sort_index()

    ts = df["latency_ms"].resample(freq).agg(
        avg_latency_ms=("mean"),
        p95_latency_ms=(lambda s: s.quantile(0.95) if len(s) else np.nan),
    ).reset_index()

    return ts.round(3)


def query_tokens_per_feature(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Total and average tokens broken down by feature – feeds a bar chart.

    Returns:
        DataFrame with columns:
            feature, total_tokens, avg_tokens, call_count
    """
    sql = """
        SELECT
            feature,
            SUM(tokens_used) AS total_tokens,
            AVG(tokens_used) AS avg_tokens,
            COUNT(*)         AS call_count
        FROM  llm_metrics
        GROUP BY feature
        ORDER BY total_tokens DESC;
    """
    try:
        with _get_connection(db_path) as conn:
            return pd.read_sql_query(sql, conn).round(2)
    except Exception:
        return pd.DataFrame()


def query_model_comparison(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Side-by-side model performance comparison.

    Returns:
        DataFrame with columns:
            model, call_count, avg_latency_ms, avg_cost,
            avg_tokens, total_cost
    """
    sql = """
        SELECT
            COALESCE(model, 'unknown') AS model,
            COUNT(*)         AS call_count,
            AVG(latency_ms)  AS avg_latency_ms,
            AVG(cost)        AS avg_cost,
            AVG(tokens_used) AS avg_tokens,
            SUM(cost)        AS total_cost
        FROM  llm_metrics
        GROUP BY model
        ORDER BY call_count DESC;
    """
    try:
        with _get_connection(db_path) as conn:
            return pd.read_sql_query(sql, conn).round(6)
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# KPI Snapshot
# ─────────────────────────────────────────────

def get_kpi_snapshot(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict:
    """
    Single-call KPI snapshot for the Streamlit dashboard header.

    Returns a dict with:
        total_calls, total_tokens, total_cost_usd, avg_latency_ms,
        avg_cost_per_call, unique_features, unique_models,
        p95_latency_ms, busiest_feature
    """
    df = query_metrics(limit=100_000, db_path=db_path)

    if df.empty:
        return {k: 0 for k in (
            "total_calls", "total_tokens", "total_cost_usd", "avg_latency_ms",
            "avg_cost_per_call", "unique_features", "unique_models",
            "p95_latency_ms", "busiest_feature",
        )}

    busiest = (
        df["feature"].value_counts().idxmax()
        if not df["feature"].empty else "—"
    )

    return {
        "total_calls":       len(df),
        "total_tokens":      int(df["tokens_used"].sum()),
        "total_cost_usd":    round(float(df["cost"].sum()), 6),
        "avg_latency_ms":    round(float(df["latency_ms"].mean()), 2),
        "avg_cost_per_call": round(float(df["cost"].mean()), 6),
        "unique_features":   int(df["feature"].nunique()),
        "unique_models":     int(df["model"].nunique()),
        "p95_latency_ms":    round(float(df["latency_ms"].quantile(0.95)), 2),
        "busiest_feature":   busiest,
    }


# ─────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────

def _empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical column schema."""
    return pd.DataFrame(columns=[
        "id", "feature", "tokens_used", "latency_ms", "cost",
        "timestamp", "response_text", "prompt_version", "model",
        "prompt_tokens", "completion_tokens", "keyword_tags",
    ])


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce correct dtypes and fill missing values on a raw query result.

    - timestamp  → UTC-aware datetime64
    - numeric    → float64 / Int64, NaN → 0
    - string     → object, NaN → ""
    """
    # Timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Numeric columns – coerce then fill
    float_cols = ["latency_ms", "cost"]
    int_cols   = ["tokens_used", "prompt_tokens", "completion_tokens"]

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in int_cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype("Int64")
            )

    # String columns
    for col in ["feature", "model", "prompt_version",
                "keyword_tags", "response_text", "id"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df