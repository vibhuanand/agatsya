"""Single-call Claude client for production package generation."""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import anthropic

from app.config import settings
from app.services import call_tracker

# ─── Rate-limit retry constants ───────────────────────────────────────────────

# Wait times (seconds) between successive retry attempts after a 429.
# Attempt 1 → wait 20s → Attempt 2 → wait 45s → Attempt 3 → wait 90s → exhausted.
_RATE_LIMIT_RETRY_DELAYS: list[int] = [20, 45, 90]


class RateLimitExhaustedError(RuntimeError):
    """
    Raised when all retry attempts after a 429 RateLimitError are exhausted.

    Attributes:
        agent_name: the agent that failed.
        retry_count: number of retries attempted (always == len(_RATE_LIMIT_RETRY_DELAYS)).
        last_exception: the final anthropic.RateLimitError that caused exhaustion.
    """

    def __init__(
        self,
        agent_name: str,
        retry_count: int,
        last_exception: anthropic.RateLimitError,
    ) -> None:
        self.agent_name    = agent_name
        self.retry_count   = retry_count
        self.last_exception = last_exception
        super().__init__(
            f"Claude RateLimitError exhausted after {retry_count} retries "
            f"for agent '{agent_name}'. "
            "Re-run the pipeline when Anthropic capacity recovers, or reduce "
            "SAFE_CLAUDE_TOKENS_PER_MINUTE in .env."
        )

logger = logging.getLogger(__name__)

PROMPT_PATHS: dict[str, Path] = {
    "script_first": Path("app/prompts/script_first_package.txt"),
    "full_package": Path("app/prompts/production_package_single_call.txt"),
    "video_plan_only": Path("app/prompts/video_plan_from_approved_script.txt"),
}


def _load_prompt_template(package_level: str) -> str:
    path = PROMPT_PATHS.get(package_level, PROMPT_PATHS["script_first"])
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt template not found for package_level='{package_level}': {path}"
        )
    return path.read_text(encoding="utf-8")


def build_transcript_research_view(raw_transcript: str) -> str:
    """
    Build a cost-controlled research view of a transcript that preserves
    beginning, middle, and ending context — the three sections most important
    for true-crime episode production.

    If the transcript fits within the budget, return it unchanged.
    If it is longer, split the budget as:
      - 40% beginning  (setup, victim introduction, early facts)
      - 20% middle     (investigation, key events)
      - 40% ending     (verdict, appeal, legal outcome, final twist)

    The view is labelled so Claude knows the structure.
    Budget is set by TRANSCRIPT_RESEARCH_CHARS in .env (default 18000).
    """
    budget = settings.transcript_research_chars
    total = len(raw_transcript)

    if total <= budget:
        return raw_transcript

    begin_chars = int(budget * 0.40)
    middle_chars = int(budget * 0.20)
    end_chars    = budget - begin_chars - middle_chars  # remaining → 40%

    beginning = raw_transcript[:begin_chars].rstrip()

    mid_start = (total - middle_chars) // 2
    middle = raw_transcript[mid_start : mid_start + middle_chars].strip()

    ending = raw_transcript[-end_chars:].lstrip()

    return (
        "[BEGINNING EXCERPT]\n"
        f"{beginning}\n\n"
        "[MIDDLE EXCERPT]\n"
        f"{middle}\n\n"
        "[ENDING EXCERPT]\n"
        f"{ending}\n\n"
        "[NOTE: transcript was compressed for token efficiency — "
        "beginning/middle/ending sections included. "
        "If a fact is unclear, mark it in facts_to_verify.]"
    )


def _build_prompt(
    youtube_url: str,
    episode_number: str,
    case_hint: str,
    target_duration_min: int,
    transcript_research_view: str,
    style: str,
    cost_mode: str = "bootstrap",
    package_level: str = "script_first",
) -> str:
    template = _load_prompt_template(package_level)

    # Safe replacement: do NOT use str.format() — the prompt template contains
    # a JSON schema with many { } braces that Python would treat as placeholders
    # and crash with KeyError / IndexError.
    replacements = {
        "{youtube_url}": youtube_url,
        "{episode_number}": episode_number,
        "{case_hint}": case_hint,
        "{target_duration_min}": str(target_duration_min),
        "{raw_transcript_research_view}": transcript_research_view,
        "{style}": style,
        "{slug}": _slugify(case_hint),
        "{cost_mode}": cost_mode,
        "{package_level}": package_level,
    }

    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)

    return prompt


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:40]


def _extract_json(raw: str, agent_name: str = "agent") -> dict[str, Any]:
    """
    Robustly extract JSON from Claude response.
    Handles: raw JSON, ```json fenced blocks, stray text before/after JSON.

    Raises ValueError with a clear, actionable message on failure.
    The caller is responsible for saving raw_response to disk before calling this
    so the user can inspect it.
    """
    original_raw = raw

    # Strip markdown fences if present
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    if fenced:
        raw = fenced.group(1)

    starts_with_fence = original_raw.lstrip().startswith("```json") or \
                        original_raw.lstrip().startswith("```")

    # Find outermost JSON object
    start = raw.find("{")
    if start == -1:
        raise ValueError(
            f"[{agent_name}] Claude response contains no '{{' — cannot extract JSON. "
            f"Response starts with fence: {starts_with_fence}. "
            f"Raw size: {len(original_raw)} chars. "
            "Inspect the raw response file saved by the calling service."
        )

    # Walk to find matching closing brace
    depth = 0
    end = -1
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        # Never found a matching closing brace — truncated output
        raise ValueError(
            f"[{agent_name}] Claude JSON appears truncated or has unbalanced braces. "
            f"First '{{' found at position {start}. "
            f"Response starts with ```json fence: {starts_with_fence}. "
            f"Raw response size: {len(original_raw)} chars. "
            f"Brace depth at end of response: {depth} (should be 0 when complete). "
            "This usually means CLAUDE_MAX_TOKENS is too low for this output size, "
            "or the agent is being asked to produce too much in one call. "
            "Inspect the raw response file saved by the calling service."
        )

    json_str = raw[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        # Pass 1: light-touch repair (trailing commas, smart quotes, control chars)
        logger.warning("[%s] JSON parse failed (%s), attempting light repair", agent_name, exc)
        repaired = _repair_json(json_str)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Pass 2: json_repair library (handles unquoted keys, missing commas, etc.)
        try:
            from json_repair import repair_json  # type: ignore[import]
            repaired2 = repair_json(json_str)
            result = json.loads(repaired2) if isinstance(repaired2, str) else repaired2
            if isinstance(result, dict):
                logger.warning("[%s] JSON recovered via json_repair library", agent_name)
                return result
        except Exception as jr_exc:
            logger.debug("[%s] json_repair attempt failed: %s", agent_name, jr_exc)

        # All repair attempts exhausted
        raise ValueError(
            f"[{agent_name}] Could not parse Claude response as JSON after all repair attempts: {exc}. "
            f"Raw size: {len(original_raw)} chars. "
            "Inspect the raw response file saved by the calling service."
        ) from exc


def _repair_json(raw: str) -> str:
    """Light-touch JSON repair for common LLM output issues."""
    # Remove trailing commas before ] or }
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    # Replace smart / curly quotes
    raw = raw.replace("“", '"').replace("”", '"')
    raw = raw.replace("‘", "'").replace("’", "'")
    # Strip BOM if present
    raw = raw.lstrip("﻿")
    # Normalize invalid control characters (keep \t \n \r, strip the rest)
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", raw)
    return raw


def call_claude(
    youtube_url: str,
    episode_number: str,
    case_hint: str,
    target_duration_min: int,
    transcript_research_view: str,
    style: str,
    cost_mode: str = "bootstrap",
    package_level: str = "script_first",
) -> tuple[str, str]:
    """
    Make a single Claude API call.
    Accepts transcript_research_view — the pre-built beginning/middle/end
    research excerpt (built and saved by the caller before this call).
    Returns (raw_response_text, stop_reason).
    Does NOT parse JSON — caller must save raw_response to disk before parsing.
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    prompt = _build_prompt(
        youtube_url=youtube_url,
        episode_number=episode_number,
        case_hint=case_hint,
        target_duration_min=target_duration_min,
        transcript_research_view=transcript_research_view,
        style=style,
        cost_mode=cost_mode,
        package_level=package_level,
    )

    logger.info(
        "Calling Claude (%s) for episode %s — case: %s",
        settings.claude_model,
        episode_number,
        case_hint,
    )

    message = client.messages.create(
        model=settings.claude_model,
        max_tokens=settings.claude_max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_response = message.content[0].text
    stop_reason = message.stop_reason

    logger.info(
        "Claude response received — %d chars, stop_reason=%s",
        len(raw_response),
        stop_reason,
    )

    if stop_reason == "max_tokens":
        logger.warning(
            "Claude hit max_tokens limit (%d) — production package may be truncated. "
            "Increase CLAUDE_MAX_TOKENS in .env if JSON is cut off.",
            settings.claude_max_tokens,
        )

    return raw_response, stop_reason


def call_claude_agent(prompt: str, agent_name: str = "agent") -> tuple[str, str]:
    """
    Generic single-call agent interface with 429 retry/backoff.

    Accepts a fully-built prompt string and returns (raw_response_text, stop_reason).
    Use this for all multi-agent pipeline steps.
    Does NOT parse JSON — caller must save raw response before parsing.

    Increments the pipeline call counter (MAX_TOTAL_MODEL_CALLS budget guard).
    Raises BudgetExceededError before making the API call if the limit is exceeded.

    429 handling:
      - On RateLimitError, waits _RATE_LIMIT_RETRY_DELAYS[attempt] seconds and retries.
      - Up to len(_RATE_LIMIT_RETRY_DELAYS) = 3 retries (4 total attempts).
      - On exhaustion raises RateLimitExhaustedError — the pipeline converts this to
        status=rate_limited_retry_later with safe_to_voice=false.

    Model rate limiter (optional):
      - Registers the call with model_rate_limiter_service BEFORE the API call.
      - The limiter sleeps if the rolling 60-second token budget is exceeded.
      - Import is deferred to avoid a circular import at module load time.
    """
    call_tracker.inc_claude(agent_name)

    # Notify the rolling-window rate limiter before sending (non-fatal if unavailable)
    try:
        from app.services.model_rate_limiter_service import rate_limiter  # noqa: PLC0415
        rate_limiter.before_call(prompt, agent_name=agent_name)
    except Exception as rl_exc:  # noqa: BLE001
        logger.debug("model_rate_limiter not available: %s", rl_exc)

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    logger.info(
        "Calling Claude agent '%s' (%s)",
        agent_name,
        settings.claude_model,
    )

    last_rate_limit_exc: anthropic.RateLimitError | None = None

    for attempt in range(len(_RATE_LIMIT_RETRY_DELAYS) + 1):  # attempts: 0, 1, 2, 3
        if attempt > 0:
            delay = _RATE_LIMIT_RETRY_DELAYS[attempt - 1]
            logger.warning(
                "Claude RateLimitError (429) — waiting %ds before retry %d/%d (agent='%s')",
                delay,
                attempt,
                len(_RATE_LIMIT_RETRY_DELAYS),
                agent_name,
            )
            time.sleep(delay)

        try:
            message = client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            raw_response = message.content[0].text
            stop_reason  = message.stop_reason

            logger.info(
                "Agent '%s' response: %d chars, stop_reason=%s",
                agent_name,
                len(raw_response),
                stop_reason,
            )

            if stop_reason == "max_tokens":
                logger.warning(
                    "Agent '%s' hit max_tokens (%d) — output may be truncated.",
                    agent_name,
                    settings.claude_max_tokens,
                )

            # Notify rate limiter of successful call for telemetry
            try:
                from app.services.model_rate_limiter_service import rate_limiter  # noqa: PLC0415
                rate_limiter.after_call(prompt, agent_name=agent_name)
            except Exception:  # noqa: BLE001
                pass

            return raw_response, stop_reason

        except anthropic.RateLimitError as exc:
            last_rate_limit_exc = exc
            logger.warning(
                "RateLimitError on attempt %d/%d for agent '%s': %s",
                attempt + 1,
                len(_RATE_LIMIT_RETRY_DELAYS) + 1,
                agent_name,
                exc,
            )

    # All retries exhausted
    raise RateLimitExhaustedError(
        agent_name=agent_name,
        retry_count=len(_RATE_LIMIT_RETRY_DELAYS),
        last_exception=last_rate_limit_exc,  # type: ignore[arg-type]
    )


def call_claude_agent_cached(
    system_content: str,
    user_content: str,
    agent_name: str = "agent",
) -> tuple[str, str]:
    """
    Cached Claude agent call.

    Splits the prompt into a stable system message (marked for prompt caching)
    and an episode-specific user message. Uses the Anthropic prompt-caching beta
    to reduce cost and latency on repeated runs with the same system instructions.

    Only active when CLAUDE_PROMPT_CACHE_ENABLED=true (default).
    Falls back to call_claude_agent() with the combined prompt when disabled.

    Returns (raw_response_text, stop_reason).
    Increments the pipeline call counter (budget guard applies).
    """
    if not settings.claude_prompt_cache_enabled:
        # Fallback: combine into single user prompt
        combined = f"{system_content}\n\n{user_content}"
        return call_claude_agent(combined, agent_name=agent_name)

    call_tracker.inc_claude(agent_name)

    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    logger.info(
        "Calling Claude agent '%s' (%s) [cached system prompt]",
        agent_name,
        settings.claude_model,
    )

    last_rate_limit_exc: anthropic.RateLimitError | None = None

    for attempt in range(len(_RATE_LIMIT_RETRY_DELAYS) + 1):
        if attempt > 0:
            delay = _RATE_LIMIT_RETRY_DELAYS[attempt - 1]
            logger.warning(
                "Claude RateLimitError (429) [cached] — waiting %ds before retry %d/%d (agent='%s')",
                delay, attempt, len(_RATE_LIMIT_RETRY_DELAYS), agent_name,
            )
            time.sleep(delay)

        try:
            message = client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system_content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_content}],
            )

            raw_response = message.content[0].text
            stop_reason  = message.stop_reason

            logger.info(
                "Agent '%s' [cached] response: %d chars, stop_reason=%s",
                agent_name,
                len(raw_response),
                stop_reason,
            )

            if stop_reason == "max_tokens":
                logger.warning(
                    "Agent '%s' [cached] hit max_tokens (%d) — output may be truncated.",
                    agent_name,
                    settings.claude_max_tokens,
                )

            return raw_response, stop_reason

        except anthropic.RateLimitError as exc:
            last_rate_limit_exc = exc
            logger.warning(
                "RateLimitError [cached] on attempt %d/%d for agent '%s': %s",
                attempt + 1, len(_RATE_LIMIT_RETRY_DELAYS) + 1, agent_name, exc,
            )

    raise RateLimitExhaustedError(
        agent_name=agent_name,
        retry_count=len(_RATE_LIMIT_RETRY_DELAYS),
        last_exception=last_rate_limit_exc,  # type: ignore[arg-type]
    )


def parse_package_response(raw_response: str, agent_name: str = "agent") -> dict[str, Any]:
    """
    Parse a raw Claude response string into a dict.
    Call this only AFTER saving raw_response to disk so failures are inspectable.
    Pass agent_name for clearer error messages.
    """
    return _extract_json(raw_response, agent_name=agent_name)
