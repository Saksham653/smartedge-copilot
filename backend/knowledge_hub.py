import sqlite3
from typing import List, Dict, Any

def search_knowledge_hub(db_path: str, query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Unified search across research_notes and meeting_notes.
    Case-insensitive LIKE search.
    """

    if not query or not query.strip():
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    search_term = f"%{query.strip()}%"

    sql = """
    SELECT 'research' AS type, id, query AS title, summary AS preview_text, created_at
    FROM research_notes
    WHERE (LOWER(query) LIKE LOWER(?) OR LOWER(summary) LIKE LOWER(?))

    UNION ALL

    SELECT 'meeting' AS type, id, title, summary AS preview_text, created_at
    FROM meeting_notes
    WHERE (LOWER(title) LIKE LOWER(?) OR LOWER(summary) LIKE LOWER(?))

    ORDER BY created_at DESC
    LIMIT ?
    """

    cursor.execute(sql, (search_term, search_term, search_term, search_term, limit))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for note_type, note_id, title, preview_text, _ in rows:
        full_text = preview_text or ""
        preview = full_text[:150].strip()
        if len(full_text) > 150:
            preview += "..."

        results.append({
            "type": note_type,
            "id": note_id,
            "title": title or "Untitled",
            "preview": preview
        })

    return results