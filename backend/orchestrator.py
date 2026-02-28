"""
orchestrator.py — SmartEdge Copilot

Internal workflow orchestration layer.

Coordinates existing backend modules to reduce friction
between meeting summarization, task creation, and
deadline intelligence.

Does NOT modify business logic.
Does NOT access database directly.
Uses existing backend modules only.
"""

from backend.meeting import generate_meeting_summary
from backend.tasks import get_due_soon, list_tasks


def process_meeting_workflow(title: str, transcript: str) -> dict:
    """
    Orchestrate full meeting workflow:

    1. Validate inputs
    2. Generate structured meeting summary
    3. Auto-create tasks (handled inside meeting module)
    4. Detect urgent tasks (due within 7 days)
    5. Count total pending tasks

    Args:
        title:      Meeting title (non-empty string)
        transcript: Raw meeting transcript (non-empty string)

    Returns:
        {
            "meeting_result": dict,
            "urgent_tasks": list,
            "total_urgent": int,
            "total_pending_tasks": int
        }

    Raises:
        ValueError: If title or transcript invalid.
    """

    # ---- Validation ----
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string.")

    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("transcript must be a non-empty string.")

    # ---- Step 1: Generate meeting summary (creates tasks internally) ----
    meeting_result = generate_meeting_summary(
        title.strip(),
        transcript.strip()
    )

    # ---- Step 2: Detect urgent tasks (next 7 days) ----
    urgent_tasks = get_due_soon(7)

    # ---- Step 3: Count only pending tasks ----
    pending_tasks = list_tasks(status="pending")

    return {
        "meeting_result": meeting_result,
        "urgent_tasks": urgent_tasks,
        "total_urgent": len(urgent_tasks),
        "total_pending_tasks": len(pending_tasks),
    }