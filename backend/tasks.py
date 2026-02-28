# backend/tasks.py

"""
tasks.py — SmartEdge Copilot
Task Automation Engine (improved parser).
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict
from backend.database import get_connection

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Patterns to strip common list prefixes like "- ", "* ", "1. ", "1) "
_PREFIX_RE = re.compile(r'^\s*(?:[-•*]|[0-9]+[.)])\s*')

# Candidate separators between assignee and task
_SEPARATORS = [r'—', r'–', r'-', r':']

def _strip_prefix(line: str) -> str:
    return _PREFIX_RE.sub('', line).strip()

def _split_assignee_task(text: str):
    """
    Try to split into (assignee, task_description) using known separators.
    If no clear assignee is found, return ('', full_text).
    """
    for sep in _SEPARATORS:
        parts = re.split(r'\s*' + sep + r'\s*', text, maxsplit=1)
        if len(parts) == 2:
            left, right = parts[0].strip(), parts[1].strip()
            # Heuristic: if left is short (<= 40 chars) treat as assignee
            if 0 < len(left) <= 40:
                return left, right
    # No separator or not clearly assignee/task -> treat whole as task
    return "", text.strip()

def extract_tasks_from_action_items(action_items_text: str) -> List[Dict[str, str]]:
    """
    Parse a free-form action_items block and return a list of task dicts:
      { "assignee": str, "task_description": str, "deadline": str }
    Accepts bullet lists, numbered lists, plain lines, with or without assignees.
    """
    tasks = []
    if not action_items_text:
        return tasks

    for raw_line in action_items_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # remove common bullet/number prefixes
        line = _strip_prefix(line)

        # If line is like "1) John - Do X", _strip_prefix removes "1)" -> left becomes "John - Do X"
        # Try to split assignee and task
        assignee, task_description = _split_assignee_task(line)

        # Try to extract " by <deadline>" from the task description
        deadline = ""
        by_marker = " by "
        lower_desc = task_description.lower()
        by_index = lower_desc.rfind(by_marker)
        if by_index != -1:
            deadline = task_description[by_index + len(by_marker):].strip()
            task_description = task_description[:by_index].strip()

        # If no assignee and task_description empty, skip
        if not task_description:
            continue

        tasks.append({
            "assignee": assignee,
            "task_description": task_description,
            "deadline": deadline,
        })

    return tasks


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def create_tasks_from_meeting(source_id: int, action_items_text: str, deadlines_text: str) -> None:
    """
    Extract tasks and attach deadlines from Deadlines section.
    """

    tasks = extract_tasks_from_action_items(action_items_text)

    # Build deadline mapping from Deadlines section
    deadline_map = {}

    for line in deadlines_text.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue

        content = line[2:].strip()

        if "—" in content:
            parts = content.split("—", 1)
        elif " - " in content:
            parts = content.split(" - ", 1)
        else:
            continue

        if len(parts) != 2:
            continue

        task_name = parts[0].strip()
        deadline = parts[1].strip()

        deadline_map[task_name.lower()] = deadline

    conn = get_connection()

    try:
        for task in tasks:
            task_name = task["task_description"].strip().lower()

            deadline = ""
            if task_name in deadline_map:
                deadline = deadline_map[task_name]

            conn.execute(
                """
                INSERT INTO tasks (
                    source_type,
                    source_id,
                    assignee,
                    task_description,
                    deadline,
                    status
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "meeting",
                    source_id,
                    task["assignee"],
                    task["task_description"],
                    deadline,
                    "pending",
                ),
            )

        conn.commit()

    finally:
        conn.close()
        
def update_task_status(task_id: int, new_status: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE tasks SET status = ? WHERE id = ?",
            (new_status, task_id),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def list_tasks(status: str = None):
    conn = get_connection()
    try:
        if status is None:
            cursor = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC"
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC",
                (status,),
            )
        return cursor.fetchall()
    finally:
        conn.close()

def get_due_soon(days: int) -> list:
    """
    Return pending tasks that are due within the next `days` days.

    Assumes deadline is stored in format: YYYY-MM-DD

    Args:
        days: Number of days from today to check.

    Returns:
        List of row tuples matching due-soon criteria.
    """

    if days <= 0:
        return []

    today = datetime.today().date()
    limit = today + timedelta(days=days)

    conn = get_connection()

    try:
        cursor = conn.execute(
            """
            SELECT * FROM tasks
            WHERE status = 'pending'
              AND deadline IS NOT NULL
              AND deadline != ''
            """
        )

        results = []
        rows = cursor.fetchall()

        for row in rows:
            # deadline column index (check your table order)
            # tasks table structure:
            # (id, source_type, source_id, assignee,
            #  task_description, deadline, status, created_at)

            deadline_str = row[5]

            try:
                deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d").date()

                if today <= deadline_date <= limit:
                    results.append(row)

            except ValueError:
                # Skip invalid date formats
                continue

        return results

    finally:
        conn.close()