# backend/ai_wrapper.py
"""
SmartEdge Copilot – Universal AI Wrapper
Supports:
- OpenAI-compatible providers (OpenAI, Groq, Together, etc.)
- Gemini (basic)
- Local Ollama (offline fallback)

Provides:
  call_llm(prompt)           -> dict          (simple interface used by most modules)
  call_model_and_log(...)    -> LLMResponse   (full interface with metrics logging)
"""

# ─────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────
import os
import time
import traceback
from typing import Optional

# ─────────────────────────────────────────────
# Third-party
# ─────────────────────────────────────────────
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Internal backend modules
# ─────────────────────────────────────────────
from backend.utils import (
    load_env,
    get_env,
    count_tokens,
    calculate_cost,
    LatencyTimer,
    generate_request_id,
    utc_timestamp,
    truncate_text,
)
from backend.logger import log_metric
from backend.database import init_db as init_db_db

# ─────────────────────────────────────────────
# Bootstrap: load .env and initialise DB once
# ─────────────────────────────────────────────
load_env()     # loads from project-root .env via backend.utils
load_dotenv()  # fallback: also honour standard dotenv resolution

try:
    init_db_db()
except Exception:
    # Don't crash at import time if DB init fails; will surface when used
    pass

# ─────────────────────────────────────────────
# Configuration (read AFTER .env is loaded)
# ─────────────────────────────────────────────
PROVIDER        = os.getenv("PROVIDER", "openai").lower()
MODEL           = os.getenv("MODEL", "llama-3.1-8b-instant")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────────
# Singleton client
# ─────────────────────────────────────────────
_client = None


def get_client():
    """Return (and cache) the appropriate LLM client for the configured provider."""
    global _client
    if _client is not None:
        return _client

    if PROVIDER == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=get_env("GEMINI_API_KEY"))
        _client = genai
        return _client

    # OpenAI-compatible providers (OpenAI, Groq, Together, etc.)
    from importlib import import_module
    OpenAI = import_module("openai").OpenAI
    _client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    return _client


# ─────────────────────────────────────────────
# LLMResponse dataclass
# ─────────────────────────────────────────────
class LLMResponse:
    __slots__ = (
        "text",
        "tokens_used",
        "prompt_tokens",
        "completion_tokens",
        "latency_ms",
        "cost",
        "request_id",
        "timestamp",
        "model",
        "feature",
        "prompt_version",
        "success",
        "error",
    )

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        return {slot: getattr(self, slot, None) for slot in self.__slots__}

    def __repr__(self):
        tokens  = getattr(self, "tokens_used", None)
        latency = getattr(self, "latency_ms", None)
        cost    = getattr(self, "cost", None)
        return (
            f"LLMResponse(feature={getattr(self, 'feature', None)}, "
            f"model={getattr(self, 'model', None)}, "
            f"tokens={tokens}, latency={latency}ms, "
            f"cost={cost}, success={getattr(self, 'success', None)})"
        )


# ─────────────────────────────────────────────
# Full interface: call + metrics logging
# ─────────────────────────────────────────────
def call_model_and_log(
    feature_name: str,
    prompt: str,
    prompt_version: str = "v1",
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    keyword_tags: Optional[list] = None,
    log_response_text: bool = True,
    max_log_chars: int = 2000,
    db_path: str = "data/metrics.db",
) -> LLMResponse:
    """
    Call the configured LLM provider, measure performance, log metrics to SQLite,
    and return a structured LLMResponse.
    """
    request_id      = generate_request_id()
    timestamp       = utc_timestamp()
    effective_model = model or MODEL

    system_prompt = system_prompt or "You are SmartEdge Copilot. Be concise, structured, and accurate."

    estimated_prompt_tokens = (
        count_tokens(system_prompt, effective_model)
        + count_tokens(prompt, effective_model)
        + 8
    )

    with LatencyTimer() as timer:
        try:
            if PROVIDER == "gemini":
                client         = get_client()
                model_instance = client.GenerativeModel(effective_model)
                response       = model_instance.generate_content(prompt)
                text              = response.text or ""
                prompt_tokens     = estimated_prompt_tokens
                completion_tokens = count_tokens(text, effective_model)
                tokens_used       = prompt_tokens + completion_tokens

            else:
                # OpenAI-compatible (Groq, OpenAI, Together, etc.)
                client   = get_client()
                response = client.chat.completions.create(
                    model=effective_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                try:
                    text = response.choices[0].message.content or ""
                except Exception:
                    text = getattr(response, "text", "") or ""

                prompt_tokens     = getattr(response.usage, "prompt_tokens",     estimated_prompt_tokens)
                completion_tokens = getattr(response.usage, "completion_tokens", count_tokens(text, effective_model))
                tokens_used       = getattr(response.usage, "total_tokens",      prompt_tokens + completion_tokens)

            success   = True
            error_msg = None

        except Exception:
            text              = ""
            prompt_tokens     = estimated_prompt_tokens
            completion_tokens = 0
            tokens_used       = estimated_prompt_tokens
            success           = False
            error_msg         = traceback.format_exc(limit=2)

    latency_ms = timer.elapsed_ms
    cost       = calculate_cost(prompt_tokens, completion_tokens, effective_model)

    # Best-effort logging — never let a logging failure break the LLM call
    try:
        log_metric(
            feature=feature_name,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            cost=cost,
            response_text=truncate_text(text, max_log_chars) if log_response_text else None,
            prompt_version=prompt_version,
            model=effective_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            keyword_tags=keyword_tags,
            db_path=db_path,
            request_id=request_id,
        )
    except Exception:
        pass

    return LLMResponse(
        text=text,
        tokens_used=tokens_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        cost=cost,
        request_id=request_id,
        timestamp=timestamp,
        model=effective_model,
        feature=feature_name,
        prompt_version=prompt_version,
        success=success,
        error=error_msg,
    )


# ─────────────────────────────────────────────
# Simple interface: used by most backend modules
# ─────────────────────────────────────────────
def call_llm(
    prompt: str,
    feature_name: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> dict:
    """
    Unified LLM caller with automatic offline fallback to Ollama.

    Returns:
        {
            "response_text": str,
            "total_tokens":  int,
            "latency_ms":    float,
            "cost":          float,
            "model":         str,
        }
    """
    start_time = time.time()
    effective_model = model or MODEL

    # ── LOCAL MODE (Ollama) ──────────────────────
    if PROVIDER == "local":
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3", "prompt": prompt, "stream": False},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            latency_ms   = (time.time() - start_time) * 1000
            total_tokens = len(prompt.split()) + len(data.get("response", "").split())

            return {
                "response_text": data.get("response", ""),
                "total_tokens":  total_tokens,
                "latency_ms":    latency_ms,
                "cost":          0.0,
                "model":         "local-llama3",
            }

        except Exception:
            # Automatic fallback to cloud if Ollama is unreachable
            pass

    # ── CLOUD MODE (Groq / OpenAI) ───────────────
    client = get_client()

    completion = client.chat.completions.create(
        model=effective_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    latency_ms    = (time.time() - start_time) * 1000
    try:
        response_text = completion.choices[0].message.content or ""
    except Exception:
        response_text = getattr(completion, "text", "") or ""
    total_tokens  = getattr(getattr(completion, "usage", None), "total_tokens", 0)
    cost          = total_tokens * 0.000002   # rough estimate

    return {
        "response_text": response_text,
        "total_tokens":  total_tokens,
        "latency_ms":    latency_ms,
        "cost":          cost,
        "model":         effective_model,
    }
