# backend/optimizer.py
"""
SmartEdge Copilot – Prompt Optimizer
Runs a raw prompt and an AI-optimized version side-by-side, then produces
a structured comparison report with token savings %, latency improvement %,
and quality deltas.
"""

import re
import json
import logging
from typing import Optional
from dataclasses import dataclass, asdict, field

from backend.ai_wrapper import call_model_and_log, LLMResponse
from backend.utils import format_cost, format_latency

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ─────────────────────────────────────────────
# Data Contracts
# ─────────────────────────────────────────────

@dataclass
class PromptCallResult:
    """
    Outcome of a single prompt call (raw or optimized).

    Attributes:
        label:         Human tag, e.g. "raw" or "optimized".
        prompt:        The exact prompt text sent to the model.
        response_text: Decoded assistant reply.
        tokens_used:   Total tokens (prompt + completion).
        prompt_tokens: Prompt-side token count.
        completion_tokens: Completion-side token count.
        latency_ms:    Wall-clock round-trip in milliseconds.
        cost:          Estimated USD cost.
        success:       False if the API call failed.
        error:         Error message when success is False.
        request_id:    UUID of the metrics row in SQLite.
        model:         Model identifier used for this call.
        prompt_version: Version tag passed to the wrapper.
    """
    label:             str
    prompt:            str
    response_text:     str
    tokens_used:       int
    prompt_tokens:     int
    completion_tokens: int
    latency_ms:        float
    cost:              float
    success:           bool
    error:             Optional[str]
    request_id:        str
    model:             str
    prompt_version:    str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OptimizationReport:
    """
    Full side-by-side comparison between a raw and optimized prompt run.

    Delta fields use the convention: negative = improvement (smaller is better
    for tokens / latency / cost).  ``pct_`` fields are percentage changes
    relative to the raw baseline (negative = savings).
    """
    # ── Raw call ──────────────────────────────────────────────────────
    raw:               PromptCallResult

    # ── Optimized call ────────────────────────────────────────────────
    optimized:         PromptCallResult

    # ── Optimized prompt text ─────────────────────────────────────────
    optimized_prompt:  str

    # ── Absolute deltas (optimized − raw) ────────────────────────────
    delta_tokens:      int
    delta_latency_ms:  float
    delta_cost:        float

    # ── Percentage improvements (negative = optimized is better) ─────
    pct_token_savings:   float   # e.g. -18.5  →  18.5 % fewer tokens
    pct_latency_improvement: float
    pct_cost_savings:    float

    # ── Quality ───────────────────────────────────────────────────────
    response_length_delta: int   # chars: optimized.response − raw.response
    compression_ratio:     float # len(optimized_prompt) / len(raw_prompt)

    # ── Overall verdict ───────────────────────────────────────────────
    verdict:           str       # "better" | "neutral" | "worse"
    summary:           str       # one-line human-readable summary
    recommendations:   list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure nested dataclasses are also plain dicts
        d["raw"]       = self.raw.to_dict()
        d["optimized"] = self.optimized.to_dict()
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ─────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────

_OPTIMIZER_SYSTEM_PROMPT = """\
You are an expert prompt engineer specialising in token efficiency and \
output quality for production LLM systems.

Your task: rewrite the user's prompt so it achieves the same goal using \
fewer tokens, without sacrificing accuracy, completeness, or clarity.

Rules:
1. Remove filler words, redundant phrasing, and excessive politeness.
2. Use imperative voice (e.g. "List …" not "Could you please list …").
3. Prefer concrete, specific instructions over vague ones.
4. Preserve all factual constraints and output-format requirements.
5. Do NOT add new requirements the original prompt did not have.
6. Return ONLY the rewritten prompt – no preamble, no explanation, \
   no surrounding quotes.
"""

_QUALITY_JUDGE_SYSTEM_PROMPT = """\
You are an objective quality evaluator for LLM responses.

Compare two responses to the same underlying task and rate them.
Return a JSON object with this exact schema (no markdown fences):

{
  "raw_score":       <int 1-10>,
  "optimized_score": <int 1-10>,
  "raw_strengths":   [<string>, …],
  "optimized_strengths": [<string>, …],
  "winner":          "raw" | "optimized" | "tie",
  "reasoning":       "<one sentence>"
}
"""


# ─────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────

def _llm_to_result(resp: LLMResponse, label: str, prompt: str) -> PromptCallResult:
    """Convert an LLMResponse into a PromptCallResult."""
    return PromptCallResult(
        label             = label,
        prompt            = prompt,
        response_text     = resp.text,
        tokens_used       = resp.tokens_used       or 0,
        prompt_tokens     = resp.prompt_tokens     or 0,
        completion_tokens = resp.completion_tokens or 0,
        latency_ms        = resp.latency_ms        or 0.0,
        cost              = resp.cost              or 0.0,
        success           = resp.success,
        error             = resp.error,
        request_id        = resp.request_id,
        model             = resp.model             or "",
        prompt_version    = resp.prompt_version    or "v1",
    )


def _safe_pct(new_val: float, old_val: float) -> float:
    """
    Return percentage change (new − old) / old × 100.
    Returns 0.0 when the denominator is zero.
    """
    if old_val == 0:
        return 0.0
    return round((new_val - old_val) / old_val * 100, 2)


def _verdict(
    pct_tokens:  float,
    pct_latency: float,
    pct_cost:    float,
) -> str:
    """
    Classify the net outcome as "better", "neutral", or "worse".

    Logic:
    - "better"  → at least one metric improved ≥ 5 % and none degraded > 10 %
    - "worse"   → any metric degraded > 10 %
    - "neutral" → everything within ±5 %
    """
    metrics     = [pct_tokens, pct_latency, pct_cost]
    any_better  = any(v <= -5.0  for v in metrics)
    any_worse   = any(v >=  10.0 for v in metrics)

    if any_worse:
        return "worse"
    if any_better:
        return "better"
    return "neutral"


def _build_recommendations(
    pct_tokens:  float,
    pct_latency: float,
    pct_cost:    float,
    compression: float,
) -> list[str]:
    """Generate actionable bullet-point recommendations from the delta metrics."""
    recs: list[str] = []

    if pct_tokens <= -15:
        recs.append(
            f"Token count reduced by {abs(pct_tokens):.1f}% – consider adopting "
            "the optimized prompt in production to lower costs."
        )
    elif pct_tokens >= 10:
        recs.append(
            f"Optimized prompt uses {pct_tokens:.1f}% more tokens. "
            "Review the rewrite for unnecessary verbosity."
        )

    if pct_latency <= -10:
        recs.append(
            f"Latency improved by {abs(pct_latency):.1f}% – "
            "faster responses will improve perceived UX."
        )
    elif pct_latency >= 20:
        recs.append(
            f"Latency increased by {pct_latency:.1f}%. "
            "This may be caused by a longer completion; check max_tokens settings."
        )

    if pct_cost <= -10:
        recs.append(
            f"Cost reduced by {abs(pct_cost):.1f}% – "
            "meaningful savings at scale (1 M calls/day)."
        )

    if compression < 0.7:
        recs.append(
            f"Prompt was compressed to {compression * 100:.0f}% of its original length. "
            "Verify the rewrite retains all original intent."
        )
    elif compression > 1.1:
        recs.append(
            "Optimized prompt is longer than the original. "
            "Consider a more aggressive rewrite strategy."
        )

    if not recs:
        recs.append(
            "Performance is comparable across all metrics. "
            "Either version is suitable for production."
        )

    return recs


# ─────────────────────────────────────────────
# Step 1 – Generate Optimized Prompt
# ─────────────────────────────────────────────

def generate_optimized_prompt(
    raw_prompt:     str,
    model:          Optional[str] = None,
    prompt_version: str           = "optimizer-v1",
    db_path:        str           = "data/metrics.db",
) -> str:
    """
    Send the raw prompt to the LLM with a prompt-engineering system message
    and return the rewritten, token-efficient version.

    Args:
        raw_prompt:     The original user prompt to optimize.
        model:          Override the default model.
        prompt_version: Version tag for the optimizer call itself.
        db_path:        SQLite metrics path for logging.

    Returns:
        Rewritten prompt string. Falls back to ``raw_prompt`` on API failure.
    """
    resp = call_model_and_log(
        feature_name   = "optimizer",
        prompt         = raw_prompt,
        prompt_version = prompt_version,
        system_prompt  = _OPTIMIZER_SYSTEM_PROMPT,
        model          = model,
        temperature    = 0.3,      # low temp → deterministic rewrite
        max_tokens     = 512,
        keyword_tags   = ["prompt-optimization", "token-efficiency"],
        db_path        = db_path,
    )

    if not resp.success or not resp.text.strip():
        logger.warning(
            "Optimizer LLM call failed (%s). Returning raw prompt unchanged.",
            resp.error,
        )
        return raw_prompt

    optimized = resp.text.strip()

    # Strip any accidental surrounding quotes the model may have added
    if (optimized.startswith('"') and optimized.endswith('"')) or \
       (optimized.startswith("'") and optimized.endswith("'")):
        optimized = optimized[1:-1].strip()

    logger.info(
        "Prompt optimized: %d chars → %d chars  (%.0f%%)",
        len(raw_prompt), len(optimized),
        len(optimized) / max(len(raw_prompt), 1) * 100,
    )
    return optimized


# ─────────────────────────────────────────────
# Step 2 – Raw Call
# ─────────────────────────────────────────────

def call_raw(
    feature_name:   str,
    prompt:         str,
    prompt_version: str           = "raw-v1",
    model:          Optional[str] = None,
    system_prompt:  Optional[str] = None,
    max_tokens:     int           = 1024,
    temperature:    float         = 0.7,
    db_path:        str           = "data/metrics.db",
) -> PromptCallResult:
    """
    Execute the raw (unoptimized) prompt and return a PromptCallResult.

    Args:
        feature_name:   Feature tag for metrics logging.
        prompt:         Raw prompt text.
        prompt_version: Version tag for this baseline call.
        model:          Model override.
        system_prompt:  Optional system instruction.
        max_tokens:     Max completion tokens.
        temperature:    Sampling temperature.
        db_path:        SQLite path.

    Returns:
        PromptCallResult labelled "raw".
    """
    logger.info("Calling model with RAW prompt (%d chars) …", len(prompt))

    resp = call_model_and_log(
        feature_name   = feature_name,
        prompt         = prompt,
        prompt_version = prompt_version,
        system_prompt  = system_prompt,
        model          = model,
        temperature    = temperature,
        max_tokens     = max_tokens,
        keyword_tags   = ["comparison", "raw"],
        db_path        = db_path,
    )

    return _llm_to_result(resp, label="raw", prompt=prompt)


# ─────────────────────────────────────────────
# Step 3 – Optimized Call
# ─────────────────────────────────────────────

def call_optimized(
    feature_name:     str,
    optimized_prompt: str,
    prompt_version:   str           = "optimized-v1",
    model:            Optional[str] = None,
    system_prompt:    Optional[str] = None,
    max_tokens:       int           = 1024,
    temperature:      float         = 0.7,
    db_path:          str           = "data/metrics.db",
) -> PromptCallResult:
    """
    Execute the optimized prompt and return a PromptCallResult.

    Args:
        feature_name:     Feature tag for metrics logging.
        optimized_prompt: Rewritten prompt from generate_optimized_prompt().
        prompt_version:   Version tag for this optimized call.
        model:            Model override.
        system_prompt:    Optional system instruction.
        max_tokens:       Max completion tokens.
        temperature:      Sampling temperature.
        db_path:          SQLite path.

    Returns:
        PromptCallResult labelled "optimized".
    """
    logger.info(
        "Calling model with OPTIMIZED prompt (%d chars) …",
        len(optimized_prompt),
    )

    resp = call_model_and_log(
        feature_name   = feature_name,
        prompt         = optimized_prompt,
        prompt_version = prompt_version,
        system_prompt  = system_prompt,
        model          = model,
        temperature    = temperature,
        max_tokens     = max_tokens,
        keyword_tags   = ["comparison", "optimized"],
        db_path        = db_path,
    )

    return _llm_to_result(resp, label="optimized", prompt=optimized_prompt)


# ─────────────────────────────────────────────
# Step 4 – Quality Judge (optional)
# ─────────────────────────────────────────────

def judge_quality(
    original_task:  str,
    raw_response:   str,
    opt_response:   str,
    model:          Optional[str] = None,
    db_path:        str           = "data/metrics.db",
) -> dict:
    """
    Ask the LLM to score both responses on a 1-10 scale.

    Args:
        original_task:  The underlying task description (raw prompt).
        raw_response:   Response generated from the raw prompt.
        opt_response:   Response generated from the optimized prompt.
        model:          Model override for the judge call.
        db_path:        SQLite path.

    Returns:
        Parsed quality dict with keys:
            raw_score, optimized_score, raw_strengths,
            optimized_strengths, winner, reasoning
        Returns a safe default dict on parse failure.
    """
    judge_prompt = (
        f"TASK:\n{original_task}\n\n"
        f"RESPONSE A (raw):\n{raw_response[:1500]}\n\n"
        f"RESPONSE B (optimized):\n{opt_response[:1500]}"
    )

    resp = call_model_and_log(
        feature_name   = "optimizer",
        prompt         = judge_prompt,
        prompt_version = "quality-judge-v1",
        system_prompt  = _QUALITY_JUDGE_SYSTEM_PROMPT,
        model          = model,
        temperature    = 0.1,
        max_tokens     = 400,
        keyword_tags   = ["quality-judge"],
        log_response_text = False,   # judge output may be noisy – skip DB storage
        db_path        = db_path,
    )

    default = {
        "raw_score": 0, "optimized_score": 0,
        "raw_strengths": [], "optimized_strengths": [],
        "winner": "tie", "reasoning": "Quality evaluation unavailable.",
    }

    if not resp.success or not resp.text:
        return default

    # Strip accidental markdown fences
    clean = re.sub(r"```(?:json)?|```", "", resp.text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("Could not parse quality judge JSON: %s", resp.text[:200])
        return default


# ─────────────────────────────────────────────
# Core Comparison Orchestrator
# ─────────────────────────────────────────────

def compare_prompts(
    feature_name:     str,
    raw_prompt:       str,
    *,
    optimized_prompt: Optional[str] = None,
    model:            Optional[str] = None,
    system_prompt:    Optional[str] = None,
    max_tokens:       int           = 1024,
    temperature:      float         = 0.7,
    run_quality_judge: bool         = False,
    db_path:          str           = "data/metrics.db",
) -> OptimizationReport:
    """
    Full optimization comparison pipeline:

    1. (Optional) Generate optimized prompt via LLM rewrite.
    2. Call model with raw prompt → measure performance.
    3. Call model with optimized prompt → measure performance.
    4. Compute token savings %, latency improvement %, cost savings %.
    5. (Optional) Run quality judge to compare response quality.
    6. Return a fully populated OptimizationReport.

    Args:
        feature_name:      Feature tag used for both metric rows in SQLite.
        raw_prompt:        The original, unoptimized prompt text.
        optimized_prompt:  Pre-written optimized prompt. If None, it is
                           generated automatically by the LLM optimizer.
        model:             Model identifier for both calls.
        system_prompt:     Shared system instruction for both calls.
        max_tokens:        Completion token limit for both calls.
        temperature:       Sampling temperature for both calls.
        run_quality_judge: If True, a third LLM call scores both responses.
        db_path:           SQLite metrics database path.

    Returns:
        OptimizationReport with all delta metrics, verdict, and recommendations.

    Example::

        report = compare_prompts(
            feature_name="summarizer",
            raw_prompt="Could you please summarize the following article for me?",
        )
        print(report.pct_token_savings)   # e.g. -22.4
        print(report.verdict)             # "better"
        print(report.to_json())
    """
    # ── 1. Generate optimized prompt if not supplied ───────────────────
    if optimized_prompt is None:
        logger.info("No optimized prompt supplied – generating via LLM …")
        optimized_prompt = generate_optimized_prompt(
            raw_prompt = raw_prompt,
            model      = model,
            db_path    = db_path,
        )

    # ── 2. Raw call ───────────────────────────────────────────────────
    raw_result = call_raw(
        feature_name   = feature_name,
        prompt         = raw_prompt,
        prompt_version = "raw-v1",
        model          = model,
        system_prompt  = system_prompt,
        max_tokens     = max_tokens,
        temperature    = temperature,
        db_path        = db_path,
    )

    # ── 3. Optimized call ─────────────────────────────────────────────
    opt_result = call_optimized(
        feature_name     = feature_name,
        optimized_prompt = optimized_prompt,
        prompt_version   = "optimized-v1",
        model            = model,
        system_prompt    = system_prompt,
        max_tokens       = max_tokens,
        temperature      = temperature,
        db_path          = db_path,
    )

    # ── 4. Compute deltas ─────────────────────────────────────────────
    delta_tokens    = opt_result.tokens_used - raw_result.tokens_used
    delta_latency   = opt_result.latency_ms  - raw_result.latency_ms
    delta_cost      = opt_result.cost        - raw_result.cost

    pct_tokens      = _safe_pct(opt_result.tokens_used, raw_result.tokens_used)
    pct_latency     = _safe_pct(opt_result.latency_ms,  raw_result.latency_ms)
    pct_cost        = _safe_pct(opt_result.cost,        raw_result.cost)

    compression     = round(
        len(optimized_prompt) / max(len(raw_prompt), 1), 4
    )
    resp_len_delta  = len(opt_result.response_text) - len(raw_result.response_text)

    # ── 5. Quality judge (optional) ───────────────────────────────────
    quality: dict = {}
    if run_quality_judge:
        logger.info("Running quality judge …")
        quality = judge_quality(
            original_task = raw_prompt,
            raw_response  = raw_result.response_text,
            opt_response  = opt_result.response_text,
            model         = model,
            db_path       = db_path,
        )

    # ── 6. Verdict & recommendations ──────────────────────────────────
    verdict         = _verdict(pct_tokens, pct_latency, pct_cost)
    recommendations = _build_recommendations(
        pct_tokens, pct_latency, pct_cost, compression
    )

    # One-line human summary
    savings_parts = []
    if pct_tokens < 0:
        savings_parts.append(f"{abs(pct_tokens):.1f}% fewer tokens")
    if pct_latency < 0:
        savings_parts.append(f"{abs(pct_latency):.1f}% lower latency")
    if pct_cost < 0:
        savings_parts.append(f"{abs(pct_cost):.1f}% cost reduction")

    if savings_parts:
        summary = "Optimized prompt achieved: " + ", ".join(savings_parts) + "."
    elif verdict == "worse":
        summary = "Optimized prompt performed worse – review the rewrite."
    else:
        summary = "Performance parity between raw and optimized prompts."

    if quality.get("winner") == "optimized":
        summary += f"  Quality judge preferred the optimized response ({quality.get('reasoning', '')})."
    elif quality.get("winner") == "raw":
        summary += f"  Quality judge preferred the raw response ({quality.get('reasoning', '')})."

    logger.info(
        "Comparison complete | tokens: %+.1f%%  latency: %+.1f%%  cost: %+.1f%%  verdict: %s",
        pct_tokens, pct_latency, pct_cost, verdict,
    )

    return OptimizationReport(
        raw                      = raw_result,
        optimized                = opt_result,
        optimized_prompt         = optimized_prompt,
        delta_tokens             = delta_tokens,
        delta_latency_ms         = round(delta_latency, 3),
        delta_cost               = round(delta_cost,    8),
        pct_token_savings        = pct_tokens,
        pct_latency_improvement  = pct_latency,
        pct_cost_savings         = pct_cost,
        response_length_delta    = resp_len_delta,
        compression_ratio        = compression,
        verdict                  = verdict,
        summary                  = summary,
        recommendations          = recommendations,
    )


# ─────────────────────────────────────────────
# Batch Comparison
# ─────────────────────────────────────────────

def compare_prompts_batch(
    feature_name: str,
    raw_prompts:  list[str],
    *,
    model:        Optional[str] = None,
    db_path:      str           = "data/metrics.db",
    **kwargs,
) -> list[OptimizationReport]:
    """
    Run compare_prompts() for a list of prompts and return one report each.

    All additional keyword arguments are forwarded to compare_prompts().

    Args:
        feature_name: Shared feature tag for all comparison runs.
        raw_prompts:  List of raw prompt strings.
        model:        Shared model override.
        db_path:      SQLite path.

    Returns:
        List of OptimizationReport objects (same order as input).
    """
    reports = []
    for i, prompt in enumerate(raw_prompts, start=1):
        logger.info("Batch comparison %d / %d …", i, len(raw_prompts))
        reports.append(
            compare_prompts(
                feature_name = feature_name,
                raw_prompt   = prompt,
                model        = model,
                db_path      = db_path,
                **kwargs,
            )
        )
    return reports


# ─────────────────────────────────────────────
# Console Summary Printer
# ─────────────────────────────────────────────

def print_report(report: OptimizationReport) -> None:
    """Pretty-print an OptimizationReport to stdout."""
    SEP  = "─" * 62
    SEP2 = "═" * 62

    def arrow(val: float) -> str:
        if val < -0.5:  return f"▼ {abs(val):.2f}%  ✅"
        if val >  0.5:  return f"▲ {abs(val):.2f}%  ⚠️"
        return f"≈ {abs(val):.2f}%  –"

    print(f"\n{SEP2}")
    print(" SmartEdge Copilot  ·  Prompt Optimization Report")
    print(SEP2)

    print(f"\n{'RAW PROMPT':}")
    print(f"  {report.raw.prompt[:120]}{'…' if len(report.raw.prompt) > 120 else ''}")
    print(f"\n{'OPTIMIZED PROMPT':}")
    print(f"  {report.optimized_prompt[:120]}{'…' if len(report.optimized_prompt) > 120 else ''}")
    print(f"  Compression ratio: {report.compression_ratio:.2f}×")

    print(f"\n{SEP}")
    print(f"  {'Metric':<22} {'Raw':>12}  {'Optimized':>12}  {'Change':>18}")
    print(SEP)

    raw, opt = report.raw, report.optimized

    rows = [
        ("Tokens used",   f"{raw.tokens_used:,}",         f"{opt.tokens_used:,}",         arrow(report.pct_token_savings)),
        ("Prompt tokens", f"{raw.prompt_tokens:,}",        f"{opt.prompt_tokens:,}",        ""),
        ("Compl. tokens", f"{raw.completion_tokens:,}",    f"{opt.completion_tokens:,}",    ""),
        ("Latency",       format_latency(raw.latency_ms),  format_latency(opt.latency_ms),  arrow(report.pct_latency_improvement)),
        ("Cost (USD)",    format_cost(raw.cost),           format_cost(opt.cost),           arrow(report.pct_cost_savings)),
    ]

    for name, r_val, o_val, change in rows:
        print(f"  {name:<22} {r_val:>12}  {o_val:>12}  {change:>18}")

    print(SEP)
    print(f"\n  Verdict:  {report.verdict.upper()}")
    print(f"  Summary:  {report.summary}")

    if report.recommendations:
        print(f"\n  Recommendations:")
        for rec in report.recommendations:
            print(f"    • {rec}")

    print(f"\n{SEP2}\n")