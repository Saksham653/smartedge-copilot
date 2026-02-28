# backend/utils.py
"""
SmartEdge Copilot – Utility Layer
Provides token counting, UUID generation, timestamps, env loading, and cost calculation.
"""

import os
import uuid
from datetime import datetime, timezone

import tiktoken
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

from pathlib import Path

def load_env() -> None:
    """
    Force-load .env from project root.
    """
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

def get_env(key: str, default: str | None = None) -> str | None:
    """Return an env variable, raising if required and missing."""
    value = os.getenv(key, default)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Add it to your .env file."
        )
    return value


# ─────────────────────────────────────────────
# Identity & Time
# ─────────────────────────────────────────────

def generate_request_id() -> str:
    """Return a globally unique request ID string (UUID4)."""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return utc_now().isoformat()


def utc_epoch_ms() -> int:
    """Return the current UTC time as integer milliseconds since epoch."""
    return int(utc_now().timestamp() * 1000)


# ─────────────────────────────────────────────
# Token Counting
# ─────────────────────────────────────────────

# Cache encoders so we don't reconstruct them on every call.
_ENCODER_CACHE: dict[str, tiktoken.Encoding] = {}

_MODEL_ENCODING_MAP: dict[str, str] = {
    # GPT-4 family
    "gpt-4":            "cl100k_base",
    "gpt-4-turbo":      "cl100k_base",
    "gpt-4o":           "o200k_base",
    "gpt-4o-mini":      "o200k_base",
    # GPT-3.5
    "gpt-3.5-turbo":    "cl100k_base",
    # Claude / generic fallback
    "default":          "cl100k_base",
}


def _get_encoder(model: str) -> tiktoken.Encoding:
    """Return (and cache) the tiktoken encoder for a given model name."""
    encoding_name = _MODEL_ENCODING_MAP.get(model) or _MODEL_ENCODING_MAP["default"]
    if encoding_name not in _ENCODER_CACHE:
        _ENCODER_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _ENCODER_CACHE[encoding_name]


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in *text* for the given *model*.

    Args:
        text:  The raw string to tokenize.
        model: OpenAI model name used to select the correct tokenizer.

    Returns:
        Integer token count.
    """
    if not text:
        return 0
    encoder = _get_encoder(model)
    return len(encoder.encode(text))


def count_messages_tokens(
    messages: list[dict],
    model: str = "gpt-4o",
) -> int:
    """
    Estimate token usage for a list of chat messages (OpenAI format).

    Follows the official token-counting heuristic:
    every message costs 4 tokens (role overhead) + content tokens.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        model:    Model name for tokenizer selection.

    Returns:
        Total estimated token count including per-message overhead.
    """
    tokens = 0
    for msg in messages:
        tokens += 4  # role + framing overhead
        for key, value in msg.items():
            if isinstance(value, str):
                tokens += count_tokens(value, model)
    tokens += 2  # reply primer
    return tokens


# ─────────────────────────────────────────────
# Cost Calculation
# ─────────────────────────────────────────────

# Prices in USD per 1 000 tokens (prompt / completion).
# Update this table as provider pricing changes.
_PRICE_TABLE: dict[str, dict[str, float]] = {
    "gpt-4o": {
        "prompt":     0.005,   # $5.00 / 1M → $0.005 / 1K
        "completion": 0.015,
    },
    "gpt-4o-mini": {
        "prompt":     0.00015,
        "completion": 0.0006,
    },
    "gpt-4-turbo": {
        "prompt":     0.01,
        "completion": 0.03,
    },
    "gpt-4": {
        "prompt":     0.03,
        "completion": 0.06,
    },
    "gpt-3.5-turbo": {
        "prompt":     0.0005,
        "completion": 0.0015,
    },
    # Generic / unknown model fallback
    "default": {
        "prompt":     0.002,
        "completion": 0.002,
    },
}


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gpt-4o",
) -> float:
    """
    Estimate USD cost for a single LLM call.

    Args:
        prompt_tokens:     Number of tokens in the prompt / input.
        completion_tokens: Number of tokens in the completion / output.
        model:             Model identifier (matched against _PRICE_TABLE).

    Returns:
        Estimated cost in USD as a float, rounded to 8 decimal places.

    Example:
        >>> calculate_cost(500, 200, "gpt-4o")
        0.00555
    """
    prices = _PRICE_TABLE.get(model, _PRICE_TABLE["default"])
    prompt_cost     = (prompt_tokens     / 1_000) * prices["prompt"]
    completion_cost = (completion_tokens / 1_000) * prices["completion"]
    return round(prompt_cost + completion_cost, 8)


def calculate_cost_from_response(response_usage: dict, model: str = "gpt-4o") -> float:
    """
    Convenience wrapper: compute cost directly from an OpenAI usage dict.

    Args:
        response_usage: The ``usage`` field from an OpenAI API response,
                        e.g. {"prompt_tokens": 100, "completion_tokens": 50, ...}
        model:          Model name for price lookup.

    Returns:
        Estimated cost in USD.
    """
    prompt_tokens     = response_usage.get("prompt_tokens", 0)
    completion_tokens = response_usage.get("completion_tokens", 0)
    return calculate_cost(prompt_tokens, completion_tokens, model)


# ─────────────────────────────────────────────
# Latency Helpers
# ─────────────────────────────────────────────

class LatencyTimer:
    """
    Context manager for measuring wall-clock latency in milliseconds.

    Usage::

        with LatencyTimer() as t:
            call_llm(...)
        print(t.elapsed_ms)  # e.g. 1234.5
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "LatencyTimer":
        import time
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        import time
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1_000, 3)


# ─────────────────────────────────────────────
# Misc Formatting
# ─────────────────────────────────────────────

def format_cost(cost_usd: float) -> str:
    """Return a human-readable cost string, e.g. '$0.00512'."""
    return f"${cost_usd:.5f}"


def format_latency(ms: float) -> str:
    """Return a human-readable latency string, e.g. '342.1 ms'."""
    return f"{ms:.1f} ms"


def truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate text for display/logging, appending '…' if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"