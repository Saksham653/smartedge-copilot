"""
meeting.py — SmartEdge Copilot
Meeting / Class Summarizer feature.
Accepts a title and transcript, generates a structured summary via LLM,
parses the response into sections, persists to the database, and returns
structured sections + performance metrics.
"""
from backend.tasks import create_tasks_from_meeting
from backend.optimizer import generate_optimized_prompt as optimize_prompt
from backend.ai_wrapper import call_llm
from backend.meeting_db import save_meeting_note
from datetime import datetime


# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------

MEETING_PROMPT_TEMPLATE = """You are an expert meeting analyst and note-taker.
Today's date is: {today}
Analyse the transcript below and produce a structured summary.

Meeting Title: {title}

Transcript:
{transcript}

Your response MUST follow this exact structure with these exact section headers:

## Executive Summary
Write a concise paragraph summarising the overall meeting — what was discussed,
the context, and the key outcomes.

## Key Topics Discussed
List the main topics covered as bullet points (use "- " prefix for each item).

## Action Items
List every action item as a bullet point in the format:
- Person — Task

## Deadlines
List every deadline mentioned as a bullet point in the format:
- Task — YYYY-MM-DD

Rules:
- Convert natural language dates (e.g., Friday, Monday, next week) into exact ISO format dates (YYYY-MM-DD).
- If the date is relative (e.g., "Friday"), assume the next upcoming occurrence.
- If no specific date is mentioned, write "None identified."

## Important Decisions
Summarise every significant decision that was made or agreed upon.

Rules:
- Use the exact section headers shown above.
- Use bullet points for Action Items and Deadlines.
- Do not add any extra sections.
- If a section has no content, write "None identified."
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_meeting_output(text: str) -> dict:
    """
    Parse the LLM's structured response into individual section strings.

    Strips markdown heading markers (#) so headers are matched regardless
    of whether the model prefixed them with one or more '#' characters.
    Uses simple substring search and adjacent-header slicing — no regex.

    Args:
        text: Raw response string returned by the LLM.

    Returns:
        dict with keys: summary, key_topics, action_items, deadlines, decisions.
        Any section not found in the response is returned as an empty string.
    """

    # Strip '#' markers from heading lines so matching is format-agnostic.
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.lstrip("#").strip()
        cleaned_lines.append(stripped if stripped != line.strip() else line)
    cleaned_text = "\n".join(cleaned_lines)

    # Ordered (key, header_substring) pairs — order is used to delimit sections.
    SECTIONS = [
        ("summary",      "Executive Summary"),
        ("key_topics",   "Key Topics Discussed"),
        ("action_items", "Action Items"),
        ("deadlines",    "Deadlines"),
        ("decisions",    "Important Decisions"),
    ]

    # Locate the character position of each header.
    positions = {}
    for key, header in SECTIONS:
        idx = cleaned_text.find(header)
        if idx != -1:
            positions[key] = idx

    result = {}
    for i, (key, header) in enumerate(SECTIONS):
        if key not in positions:
            result[key] = ""
            continue

        # Content starts immediately after the header text.
        start = positions[key] + len(header)
        # Skip the newline that directly follows the header line.
        if start < len(cleaned_text) and cleaned_text[start] == "\n":
            start += 1

        # Content ends at the start of the next located section.
        end = len(cleaned_text)
        for next_key, _ in SECTIONS[i + 1:]:
            if next_key in positions:
                end = positions[next_key]
                break

        result[key] = cleaned_text[start:end].strip()

    # Ensure all keys are always present in the returned dict.
    for key, _ in SECTIONS:
        result.setdefault(key, "")

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_meeting_summary(title: str, transcript: str) -> dict:
    """
    Generate a structured meeting summary for the given title and transcript.

    Steps:
        1. Validate that title and transcript are non-empty.
        2. Insert title and transcript into MEETING_PROMPT_TEMPLATE.
        3. Optimise the prompt via optimizer.optimize_prompt().
        4. Send the optimised prompt to the LLM via ai_wrapper.call_llm().
        5. Parse the LLM response into structured sections.
        6. Persist the record via meeting_db.save_meeting_note().
        7. Return sections + selected performance metrics.

    Args:
        title:      Short descriptive title for the meeting or class session.
        transcript: Full text transcript of the meeting.

    Returns:
        dict:
            summary      (str)  — Executive summary paragraph.
            key_topics   (str)  — Bullet-point list of topics discussed.
            action_items (str)  — Bullet-point action items (Person — Task).
            deadlines    (str)  — Bullet-point deadlines (Task — Date).
            decisions    (str)  — Important decisions made.
            metrics      (dict) — total_tokens, latency_ms, cost, model.

    Raises:
        ValueError: If title or transcript is empty or whitespace.
        KeyError:   If the LLM response dict is missing expected keys.
    """

    # STEP 1 — Validate inputs.
    if not title or not title.strip():
        raise ValueError("title must be a non-empty string.")
    if not transcript or not transcript.strip():
        raise ValueError("transcript must be a non-empty string.")

    # STEP 2 — Build the structured prompt.
    today_str = datetime.today().strftime("%Y-%m-%d")

    structured_prompt = MEETING_PROMPT_TEMPLATE.format(
    title=title.strip(),
    transcript=transcript.strip(),
    today=today_str
    )
    
    # STEP 3 — Optimise the prompt to reduce tokens / improve latency.
    optimized_prompt = structured_prompt

    # STEP 4 — Call the LLM.
    response = call_llm(optimized_prompt)

    # STEP 5 — Parse the structured response into sections.
    sections = parse_meeting_output(response["response_text"])

    # STEP 6 — Persist the meeting note to the database.
    meeting_id = save_meeting_note(
    title=title.strip(),
    transcript=transcript.strip(),
    summary=sections["summary"],
    key_topics=sections["key_topics"],
    action_items=sections["action_items"],
    deadlines=sections["deadlines"],
    decisions=sections["decisions"],
    total_tokens=response["total_tokens"],
    latency_ms=response["latency_ms"],
    cost=response["cost"],
    model=response["model"],
    )

    # STEP 7 — Assemble and return the final result.
    return {
        "summary":      sections["summary"],
        "key_topics":   sections["key_topics"],
        "action_items": sections["action_items"],
        "deadlines":    sections["deadlines"],
        "decisions":    sections["decisions"],
        "metrics": {
            "total_tokens": response["total_tokens"],
            "latency_ms":   response["latency_ms"],
            "cost":         response["cost"],
            "model":        response["model"],
        },
    }
