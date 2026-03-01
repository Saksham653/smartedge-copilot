import sqlite3
import textwrap


class NoteNotFoundError(Exception):
    """Raised when a requested note ID does not exist in the database."""
    pass


def _safe_str(value) -> str:
    """Safely converts a database value to a string, rendering None as an empty string."""
    return str(value) if value is not None else ""


def export_note_markdown(db_path: str, note_type: str, note_id: int) -> str:
    """
    Retrieves a note from the database and formats it as a Markdown string.

    Args:
        db_path (str): Path to SQLite database.
        note_type (str): "research" or "meeting".
        note_id (int): Primary key ID of the note.

    Returns:
        str: Formatted Markdown string.

    Raises:
        ValueError: If note_type is invalid.
        NoteNotFoundError: If note_id does not exist.
    """

    if note_type not in ("research", "meeting"):
        raise ValueError(
            f"Unsupported note_type '{note_type}'. Must be 'research' or 'meeting'."
        )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # ===============================
        # RESEARCH NOTE EXPORT
        # ===============================
        if note_type == "research":
            cursor.execute("""
                SELECT query, summary, key_concepts, applications,
                       references_text, created_at
                FROM research_notes
                WHERE id = ?
            """, (note_id,))

            row = cursor.fetchone()

            if not row:
                raise NoteNotFoundError(
                    f"Research note with ID {note_id} could not be found."
                )

            title = _safe_str(row["query"]) or f"Research Note {note_id}"

            markdown_content = f"""
# {title}

Created: {_safe_str(row["created_at"])}

## Summary
{_safe_str(row["summary"])}

## Key Concepts
{_safe_str(row["key_concepts"])}

## Applications
{_safe_str(row["applications"])}

## References
{_safe_str(row["references_text"])}
"""

        # ===============================
        # MEETING NOTE EXPORT
        # ===============================
        else:
            cursor.execute("""
                SELECT title, summary, key_topics, action_items,
                       deadlines, decisions, recommendations, risks, sentiment,
                       speaker_stats, followups, created_at
                FROM meeting_notes
                WHERE id = ?
            """, (note_id,))

            row = cursor.fetchone()

            if not row:
                raise NoteNotFoundError(
                    f"Meeting note with ID {note_id} could not be found."
                )

            title = _safe_str(row["title"]) or f"Meeting Note {note_id}"

            markdown_content = f"""
# {title}

Created: {_safe_str(row["created_at"])}

## Summary
{_safe_str(row["summary"])}

## Key Topics
{_safe_str(row["key_topics"])}

## Action Items
{_safe_str(row["action_items"])}

## Deadlines
{_safe_str(row["deadlines"])}

## Decisions
{_safe_str(row["decisions"])}

## Recommendations
{_safe_str(row["recommendations"])}

## Risks & Blockers
{_safe_str(row["risks"])}

## Sentiment & Tone
{_safe_str(row["sentiment"])}

## Speaker Contribution
{_safe_str(row["speaker_stats"])}

## Follow-up Questions
{_safe_str(row["followups"])}
"""

    return textwrap.dedent(markdown_content).strip()
