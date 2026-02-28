# backend/analytics_service.py

from typing import Dict, List, Any
import sqlite3
from backend.database import get_connection


# ---------------------------------------------------------------------------
# Feature Summary
# ---------------------------------------------------------------------------

def get_feature_summary() -> Dict[str, Dict[str, Any]]:
    """
    Returns aggregated metrics grouped by feature_name.
    """

    query = """
        SELECT 
            feature_name,
            COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms,
            COALESCE(SUM(total_tokens), 0) AS total_tokens,
            COALESCE(SUM(cost), 0.0) AS total_cost,
            COUNT(id) AS total_runs
        FROM metrics
        GROUP BY feature_name;
    """

    summary: Dict[str, Dict[str, Any]] = {}

    conn = get_connection()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)

        for row in cursor.fetchall():
            summary[row["feature_name"]] = {
                "avg_latency_ms": float(row["avg_latency_ms"]),
                "total_tokens": int(row["total_tokens"]),
                "total_cost": float(row["total_cost"]),
                "total_runs": int(row["total_runs"]),
            }

    finally:
        conn.close()

    return summary


# ---------------------------------------------------------------------------
# Overall Totals
# ---------------------------------------------------------------------------

def get_overall_totals() -> Dict[str, Any]:
    """
    Returns global totals across all features.
    """

    query = """
        SELECT 
            COALESCE(SUM(total_tokens), 0) AS total_tokens,
            COALESCE(SUM(cost), 0.0) AS total_cost,
            COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms,
            COUNT(id) AS total_runs
        FROM metrics;
    """

    conn = get_connection()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)

        row = cursor.fetchone()

        if not row or row["total_runs"] == 0:
            return {
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_latency_ms": 0.0,
                "total_runs": 0,
            }

        return {
            "total_tokens": int(row["total_tokens"]),
            "total_cost": float(row["total_cost"]),
            "avg_latency_ms": float(row["avg_latency_ms"]),
            "total_runs": int(row["total_runs"]),
        }

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Usage Over Time
# ---------------------------------------------------------------------------

def get_usage_over_time() -> List[Dict[str, Any]]:
    """
    Returns time-series data grouped by date (YYYY-MM-DD).
    """

    query = """
        SELECT 
            DATE(created_at) AS metric_date,
            COALESCE(SUM(total_tokens), 0) AS total_tokens,
            COALESCE(SUM(cost), 0.0) AS total_cost
        FROM metrics
        GROUP BY DATE(created_at)
        ORDER BY metric_date ASC;
    """

    time_series: List[Dict[str, Any]] = []

    conn = get_connection()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)

        for row in cursor.fetchall():
            if row["metric_date"] is None:
                continue

            time_series.append({
                "date": str(row["metric_date"]),
                "total_tokens": int(row["total_tokens"]),
                "total_cost": float(row["total_cost"]),
            })

    finally:
        conn.close()

    return time_series

def generate_performance_insights() -> dict:
    """
    Generates simple performance insights based on aggregated metrics.
    Uses get_feature_summary().
    """

    summary = get_feature_summary()

    if not summary:
        return {
            "most_expensive_feature": None,
            "slowest_feature": None,
            "feature_efficiency": {}
        }

    most_expensive = None
    slowest = None

    max_cost = -1
    max_latency = -1

    efficiency = {}

    for feature, data in summary.items():
        total_cost = data["total_cost"]
        total_tokens = data["total_tokens"]
        total_runs = data["total_runs"]
        avg_latency = data["avg_latency_ms"]

        # Most expensive feature
        if total_cost > max_cost:
            max_cost = total_cost
            most_expensive = feature

        # Slowest feature
        if avg_latency > max_latency:
            max_latency = avg_latency
            slowest = feature

        # Cost per 1K tokens
        if total_tokens > 0:
            cost_per_1k = (total_cost / total_tokens) * 1000
        else:
            cost_per_1k = 0.0

        # Average cost per run
        if total_runs > 0:
            avg_cost_per_run = total_cost / total_runs
        else:
            avg_cost_per_run = 0.0

        efficiency[feature] = {
            "cost_per_1k_tokens": round(cost_per_1k, 6),
            "avg_cost_per_run": round(avg_cost_per_run, 6),
        }

    return {
        "most_expensive_feature": most_expensive,
        "slowest_feature": slowest,
        "feature_efficiency": efficiency
    }