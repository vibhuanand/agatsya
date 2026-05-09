"""
Optional GPT review pass — second-pass production quality reviewer.

Receives the Claude production package dict, sends ONLY that JSON to OpenAI
(never the original transcript), and returns an improved package dict.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import openai

from app.config import settings

logger = logging.getLogger(__name__)

PROMPT_PATH = Path("app/prompts/gpt_package_reviewer.txt")

# Safety cap — GPT review should be a light pass, not a full rewrite
GPT_MAX_TOKENS = 10000


def _load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_review_prompt(package_dict: dict[str, Any]) -> str:
    """
    Inject the Claude package JSON into the reviewer prompt.
    Uses a single {production_package_json} placeholder — safe because
    the JSON itself is inserted verbatim, not via str.format().
    """
    template = _load_prompt_template()
    package_json_str = json.dumps(package_dict, ensure_ascii=False, indent=2)
    return template.replace("{production_package_json}", package_json_str)


def _extract_json(raw: str) -> dict[str, Any]:
    """
    Robustly extract JSON from GPT response.
    Handles: raw JSON, ```json fenced blocks, stray text before/after.
    """
    # Strip markdown fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    if fenced:
        raw = fenced.group(1)

    start = raw.find("{")
    if start == -1:
        raise ValueError(
            "Could not parse GPT review response as JSON — no '{' found. "
            "Check 02-package/_gpt_review_raw_response.txt for the raw output."
        )

    # Walk to matching closing brace
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
        logger.warning("GPT JSON parse failed (%s), attempting repair", exc)
        json_str = _repair_json(json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc2:
            raise ValueError(
                f"Could not parse GPT review response as JSON: {exc2}. "
                "Check 02-package/_gpt_review_raw_response.txt for the raw output."
            ) from exc2


def _repair_json(raw: str) -> str:
    """Light-touch JSON repair for common LLM output issues."""
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    raw = raw.replace("“", '"').replace("”", '"')
    raw = raw.replace("‘", "'").replace("’", "'")
    return raw


def review_package(
    package_dict: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """
    Send the Claude production package to GPT for a quality review pass.

    Args:
        package_dict: The parsed Claude production package dict.

    Returns:
        (reviewed_package_dict, raw_gpt_response_text)

    Raises:
        RuntimeError: if OPENAI_API_KEY is not configured.
        ValueError: if GPT response cannot be parsed as JSON.
        openai.OpenAIError: on API-level failures.
    """
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it in .env to use enable_gpt_review=true."
        )

    client = openai.OpenAI(api_key=settings.openai_api_key)
    prompt = _build_review_prompt(package_dict)

    logger.info(
        "Calling GPT review (%s) — package size: %d chars",
        settings.openai_review_model,
        len(prompt),
    )

    # gpt-5/gpt-5.5 and o-series models use max_completion_tokens, not max_tokens
    _new_api_prefixes = ("o1", "o3", "o4", "gpt-5", "gpt-4o")
    _use_completion_tokens = any(
        settings.openai_review_model.startswith(p) for p in _new_api_prefixes
    )
    _token_kwargs = (
        {"max_completion_tokens": GPT_MAX_TOKENS}
        if _use_completion_tokens
        else {"max_tokens": GPT_MAX_TOKENS}
    )

    response = client.chat.completions.create(
        model=settings.openai_review_model,
        **_token_kwargs,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a production quality reviewer for a Hindi true-crime YouTube channel. "
                    "Return only valid JSON. No markdown. No explanation. Pure JSON object only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.3,  # low temperature — improve, don't hallucinate
    )

    raw_response = response.choices[0].message.content or ""
    finish_reason = response.choices[0].finish_reason

    logger.info(
        "GPT review response received — %d chars, finish_reason=%s",
        len(raw_response),
        finish_reason,
    )

    if finish_reason == "length":
        logger.warning(
            "GPT review hit max_tokens (%d) — reviewed package may be truncated. "
            "Consider using a smaller package or increasing GPT_MAX_TOKENS.",
            GPT_MAX_TOKENS,
        )

    reviewed_dict = _extract_json(raw_response)
    return reviewed_dict, raw_response
