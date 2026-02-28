from backend.database import get_connection
from backend import research_db


def search_all(keyword: str) -> dict:
    """
    Search across research_notes, meeting_notes, and tasks tables.

    Args:
        keyword: The search term to query across all tables.

    Returns:
        A dictionary with keys "research", "meetings", and "tasks",
        each containing a list of matching row tuples.
    """
    empty = {"research": [], "meetings": [], "tasks": []}

    if not keyword or not keyword.strip():
        return empty

    research_results = research_db.search_research_notes(keyword)
    meeting_results = []

    task_results = []
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE LOWER(task_description) LIKE LOWER(?)", (f"%{keyword}%",))
        task_results = cursor.fetchall()
    finally:
        conn.close()

    return {
        "research": research_results,
        "meetings": meeting_results,
        "tasks": task_results,
    }
