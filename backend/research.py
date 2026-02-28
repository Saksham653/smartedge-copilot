# backend/research.py
"""
research.py — SmartEdge Copilot
Structured Research Copilot feature: accepts a user query, optimizes the prompt,
calls the LLM, parses the structured response, and returns sections + metrics.
"""

from backend.research_db import save_research_note
# optimizer exposes generate_optimized_prompt — alias to optimize_prompt for compatibility
from backend.optimizer import generate_optimized_prompt as optimize_prompt
from backend.ai_wrapper import call_llm

# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------

RESEARCH_PROMPT_TEMPLATE = """You are an expert research assistant. Provide a well-structured, 
accurate, and informative research response on the following topic.

Topic: {query}

Your response MUST follow this exact structure with these exact section headers:

## Concise Summary
Write 5–8 sentences giving a clear, concise overview of the topic.

## Key Concepts
Provide the most important concepts as bullet points (use "- " prefix for each bullet).

## Practical Applications
Describe real-world practical applications of this topic in clear paragraphs or bullet points.

## References
List credible sources with URLs where available. Use "- " prefix for each reference.

Do not add any extra sections. Follow the structure strictly.
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_research_output(text: str) -> dict:
    """
    Parse the LLM's structured output into individual section strings.

    Returns dict with keys: summary, key_concepts, applications, references.
    """
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.lstrip("#").strip()
        cleaned_lines.append(stripped if stripped != line.strip() else line)
    cleaned_text = "\n".join(cleaned_lines)

    variants = {
        "summary": ["Concise Summary", "Summary", "Executive Summary"],
        "key_concepts": ["Key Concepts", "Key Points", "Key Ideas", "Concepts"],
        "applications": ["Practical Applications", "Applications", "Use Cases", "Practical Use Cases", "Real-World Applications"],
        "references": ["References", "Sources", "Citations", "Further Reading"],
    }

    positions = {}
    matched = {}
    for key, headers in variants.items():
        best_idx = -1
        best_hdr = None
        for hdr in headers:
            idx = cleaned_text.find(hdr)
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx = idx
                best_hdr = hdr
        if best_idx != -1:
            positions[key] = best_idx
            matched[key] = best_hdr

    order = ["summary", "key_concepts", "applications", "references"]
    result = {}
    for i, key in enumerate(order):
        if key not in positions:
            result[key] = ""
            continue
        start = positions[key] + len(matched[key])
        if start < len(cleaned_text) and cleaned_text[start] == "\n":
            start += 1
        end = len(cleaned_text)
        for next_key in order[i + 1:]:
            if next_key in positions:
                end = positions[next_key]
                break
        result[key] = cleaned_text[start:end].strip()

    if not result.get("summary"):
        if positions:
            first_pos = min(positions.values())
            preface = cleaned_text[:first_pos].strip()
            result["summary"] = preface

    for key in order:
        result.setdefault(key, "")

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_research(query: str) -> dict:
    """
    Generate a structured research response for the given query.
    Returns sections + metrics.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")

    structured_prompt = RESEARCH_PROMPT_TEMPLATE.format(query=query.strip())

    # Use optimizer (aliased to optimize_prompt)
    optimized_prompt = optimize_prompt(structured_prompt)

    response = call_llm(
        optimized_prompt,
        feature_name="research",
        temperature=0.2,
        max_tokens=1400,
    )

    # Parse structured sections
    sections = parse_research_output(response.get("response_text", ""))

    # Persist to DB — best-effort: don't fail the function if DB write fails
    try:
        save_research_note(
            query=query.strip(),
            optimized_prompt=optimized_prompt,
            summary=sections.get("summary", ""),
            key_concepts=sections.get("key_concepts", ""),
            applications=sections.get("applications", ""),
            references_text=sections.get("references", ""),
            total_tokens=response.get("total_tokens") or response.get("total_tokens", 0),
            latency_ms=response.get("latency_ms", 0.0),
            cost=response.get("cost", 0.0),
            model=response.get("model", ""),
        )
    except Exception:
        # swallow DB errors (already logged by lower layers if any)
        pass

    return {
        "summary":      sections.get("summary", ""),
        "key_concepts": sections.get("key_concepts", ""),
        "applications": sections.get("applications", ""),
        "references":   sections.get("references", ""),
        "metrics": {
            "total_tokens": response.get("total_tokens", 0),
            "latency_ms":   response.get("latency_ms", 0.0),
            "cost":         response.get("cost", 0.0),
            "model":        response.get("model", ""),
        },
    }


def answer_followup(context: str, question: str) -> dict:
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string.")
    ctx = context or ""
    prompt = (
        "You are a precise research assistant. Use the provided context to answer the user's follow-up.\n\n"
        "Context:\n"
        f"{ctx}\n\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Answer clearly. Prefer short paragraphs and bullet points. Do not repeat the entire context."
    )
    resp = call_llm(
        prompt,
        feature_name="research_followup",
        temperature=0.3,
        max_tokens=800,
    )
    return {
        "answer": resp.get("response_text", ""),
        "metrics": {
            "total_tokens": resp.get("total_tokens", 0),
            "latency_ms": resp.get("latency_ms", 0.0),
            "cost": resp.get("cost", 0.0),
            "model": resp.get("model", ""),
        },
    }
