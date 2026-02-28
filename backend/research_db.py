# backend/research_db.py

"""
Database operations for the Research Copilot feature.
Handles saving and retrieving research notes from the research_notes table.
Also logs performance metrics into the metrics table.
"""

from typing import List, Dict, Any
from backend.database import get_connection


# ---------------------------------------------------------------------------
# Write Operations
# ---------------------------------------------------------------------------

def save_research_note(
    query: str,
    optimized_prompt: str,
    summary: str,
    key_concepts: str,
    applications: str,
    references_text: str,
    total_tokens: int,
    latency_ms: float,
    cost: float,
    model: str,
) -> int:
    """
    Insert a structured research result into the research_notes table.
    Also logs performance metrics into the metrics table.
    Returns the inserted research note id.
    """

    conn = get_connection()

    try:
        cur = conn.cursor()

        # --------------------------------------------------
        # Insert into research_notes table
        # --------------------------------------------------
        cur.execute(
            """
            INSERT INTO research_notes (
                query,
                optimized_prompt,
                summary,
                key_concepts,
                applications,
                references_text,
                total_tokens,
                latency_ms,
                cost,
                model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query,
                optimized_prompt,
                summary,
                key_concepts,
                applications,
                references_text,
                total_tokens,
                latency_ms,
                cost,
                model,
            ),
        )

        research_id = cur.lastrowid

        # --------------------------------------------------
        # Insert into metrics table
        # --------------------------------------------------
        cur.execute(
            """
            INSERT INTO metrics (
                feature_name,
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                latency_ms,
                cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "research",
                model,
                0,  # placeholder until prompt tokens tracked
                0,  # placeholder until completion tokens tracked
                total_tokens,
                latency_ms,
                cost,
            ),
        )

        conn.commit()
        return research_id

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read Operations
# ---------------------------------------------------------------------------

def search_research_notes(keyword: str) -> List[Dict[str, Any]]:
    """
    Search research notes whose query or summary contains the given keyword.
    Returns a list of dicts (ordered most-recent first).
    """

    conn = get_connection()

    try:
        cur = conn.cursor()
        q = f"%{keyword}%"

        cur.execute(
            """
            SELECT 
                id,
                query,
                optimized_prompt,
                summary,
                key_concepts,
                applications,
                references_text,
                total_tokens,
                latency_ms,
                cost,
                model,
                created_at
            FROM research_notes
            WHERE query LIKE ? OR summary LIKE ?
            ORDER BY created_at DESC
            """,
            (q, q),
        )

        rows = cur.fetchall()

        results = []
        for r in rows:
            results.append(
                {
                    "id": r[0],
                    "query": r[1],
                    "optimized_prompt": r[2],
                    "summary": r[3],
                    "key_concepts": r[4],
                    "applications": r[5],
                    "references_text": r[6],
                    "total_tokens": r[7],
                    "latency_ms": r[8],
                    "cost": r[9],
                    "model": r[10],
                    "created_at": r[11],
                }
            )

        return results

    finally:
        conn.close()