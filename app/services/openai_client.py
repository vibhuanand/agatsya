"""
OpenAI Client — shared base for all OpenAI gate services.

Wraps the OpenAI Chat Completions API with:
  - JSON output enforcement (response_format=json_object)
  - Robust JSON extraction from raw response
  - Raw response saved to caller-provided path
  - Graceful missing-key detection (raises RuntimeError, not KeyError)
  - Useful error messages on parse failure

Usage:
    from app.services.openai_client import call_openai_json

    result = call_openai_json(
        system_prompt="You are a...",
        user_content="...",
        raw_save_path=review_dir / "_my_raw_response.txt",
        agent_name="my_gate",
    )
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import openai

from app.config import settings
from app.services import call_tracker

logger = logging.getLogger(__name__)

_MAX_TOKENS = 8000
_TEMPERATURE = 0.2   # low — review/evaluate, don't hallucinate

# Models that use max_completion_tokens instead of max_tokens
# (o1, o3, o4, gpt-5, and gpt-4o families)
_MAX_COMPLETION_TOKENS_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-4o")


# ─── JSON extraction (robust) ─────────────────────────────────────────────────

def _repair_json(raw: str) -> str:
    """Light-touch JSON repair for common LLM output issues."""
    raw = re.sub(r",\s*([}\]])", r"\1", raw)     # trailing commas
    raw = raw.replace("“", '"').replace("”", '"')  # curly quotes
    raw = raw.replace("‘", "'").replace("’", "'")
    return raw


def _extract_json(raw: str, agent_name: str = "openai") -> dict[str, Any]:
    """
    Robustly extract a JSON object from a raw LLM response.
    Handles: raw JSON, ```json fenced blocks, stray text before/after.
    """
    # Strip markdown fences if present
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # fall through to brace-scanning

    start = raw.find("{")
    if start == -1:
        raise ValueError(
            f"[{agent_name}] OpenAI response contains no JSON object. "
            "Check the raw response file for details."
        )

    depth = 0
    end = start
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    json_str = raw[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("[%s] JSON parse failed (%s) — attempting repair", agent_name, exc)
        repaired = _repair_json(json_str)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc2:
            raise ValueError(
                f"[{agent_name}] Could not parse OpenAI response as JSON after repair: {exc2}"
            ) from exc2


# ─── Public API ───────────────────────────────────────────────────────────────

def call_openai_json(
    system_prompt: str,
    user_content: str,
    raw_save_path: Path,
    agent_name: str = "openai_gate",
    model: str | None = None,
    max_tokens: int = _MAX_TOKENS,
) -> dict[str, Any]:
    """
    Call OpenAI Chat Completions with JSON output mode. Save raw response.
    Return parsed JSON dict.

    Raises:
        RuntimeError  — if OPENAI_API_KEY is not set
        openai.OpenAIError — on API-level failures
        ValueError    — if response cannot be parsed as JSON
    """
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Set it in .env to enable OpenAI premium review gates."
        )

    # Budget guard: check before making the API call
    call_tracker.inc_openai(agent_name)

    effective_model = model or settings.openai_review_model
    client = openai.OpenAI(api_key=settings.openai_api_key)

    # Some model families (o1, o3, o4, gpt-5, gpt-4o) require max_completion_tokens
    # instead of max_tokens. Select the right parameter automatically.
    use_completion_tokens = any(
        effective_model.startswith(prefix)
        for prefix in _MAX_COMPLETION_TOKENS_PREFIXES
    )

    logger.info(
        "[%s] Calling OpenAI (%s) — system=%d chars user=%d chars token_param=%s",
        agent_name, effective_model, len(system_prompt), len(user_content),
        "max_completion_tokens" if use_completion_tokens else "max_tokens",
    )

    # Always ensure raw_save_path parent exists before any API call
    raw_save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if use_completion_tokens:
            response = client.chat.completions.create(
                model=effective_model,
                max_completion_tokens=max_tokens,
                temperature=_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
            )
        else:
            response = client.chat.completions.create(
                model=effective_model,
                max_tokens=max_tokens,
                temperature=_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
            )
    except openai.OpenAIError as api_exc:
        # Save the raw error to the review_dir for auditability
        error_text = (
            f"OpenAI API Error — agent={agent_name} model={effective_model}\n"
            f"Error type: {type(api_exc).__name__}\n"
            f"Message: {api_exc}\n"
        )
        err_path = raw_save_path.parent / f"_openai_error_{agent_name}.txt"
        try:
            err_path.write_text(error_text, encoding="utf-8")
        except Exception:
            pass  # best-effort
        logger.error("[%s] OpenAI API error: %s", agent_name, api_exc)
        # Return a safe failure dict rather than crashing the pipeline
        return {
            "approved": False,
            "safe_to_voice": False,
            "status": "needs_human_review",
            "_api_error": str(api_exc),
            "_warning": (
                f"OpenAI API call failed for agent '{agent_name}' — "
                f"error saved to {err_path}. "
                "Manual review required before audio generation."
            ),
        }

    raw = response.choices[0].message.content or ""
    finish_reason = response.choices[0].finish_reason

    # Always save raw response before any parsing
    raw_save_path.write_text(raw, encoding="utf-8")

    logger.info(
        "[%s] OpenAI response: %d chars, finish_reason=%s",
        agent_name, len(raw), finish_reason,
    )

    if finish_reason == "length":
        logger.warning(
            "[%s] OpenAI hit max_tokens=%d — response may be truncated",
            agent_name, max_tokens,
        )

    return _extract_json(raw, agent_name=agent_name)
