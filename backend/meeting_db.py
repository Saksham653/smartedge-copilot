# backend/meeting_db.py

"""
Database operations for Meeting Copilot.
Handles:
- Saving meeting notes
- Logging performance metrics
- Automatic task creation
"""

from backend.database import get_connection


def save_meeting_note(
    title: str,
    transcript: str,
    summary: str,
    key_topics: str,
    action_items: str,
    deadlines: str,
    decisions: str,
    total_tokens: int,
    latency_ms: float,
    cost: float,
    model: str,
    *,
    auto_create_tasks: bool = True,
) -> int:
    """
    Insert meeting note.
    Automatically logs metrics.
    Automatically creates tasks from action_items.
    Returns meeting_id.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        # --------------------------------------------------
        # Insert meeting note
        # --------------------------------------------------
        cursor.execute(
            """
            INSERT INTO meeting_notes (
                title,
                transcript,
                summary,
                key_topics,
                action_items,
                deadlines,
                decisions,
                total_tokens,
                latency_ms,
                cost,
                model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                transcript,
                summary,
                key_topics,
                action_items,
                deadlines,
                decisions,
                total_tokens,
                latency_ms,
                cost,
                model,
            ),
        )

        meeting_id = cursor.lastrowid

        # --------------------------------------------------
        # Insert metrics entry (correct schema)
        # --------------------------------------------------
        cursor.execute(
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
                "meeting",
                model,
                0,  # placeholder until prompt tokens tracked
                0,  # placeholder until completion tokens tracked
                total_tokens,
                latency_ms,
                cost,
            ),
        )

        # --------------------------------------------------
        # Auto-create tasks
        # --------------------------------------------------
        if auto_create_tasks and action_items and action_items.strip():
            try:
                from backend.tasks import create_tasks_from_meeting
                create_tasks_from_meeting(meeting_id, action_items, deadlines)
            except Exception:
                pass

        conn.commit()
        return meeting_id

    finally:
        conn.close()
