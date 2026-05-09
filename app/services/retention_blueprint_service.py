"""
Retention Blueprint Agent — Claude agent that designs the viewer experience arc
for audience retention, CTR, and subscriber conversion.

Runs after Story Blueprint and before Script Outline (premium mode only).
The output guides the outline agent in placing curiosity gaps, re-engagement
moments, pattern interrupts, and the subscriber conversion moment.

Produces:
  02-facts/retention_blueprint.json
  02-facts/_retention_blueprint_raw_response.txt
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.services.claude_client import call_claude_agent, parse_package_response

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/retention_blueprint_agent.txt")


def _build_prompt(
    fact_lock: dict,
    blueprint: dict,
    target_duration_min: int,
    case_hint: str,
    hinglish_level: int,
) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    replacements = {
        "{case_hint}":           case_hint,
        "{target_duration_min}": str(target_duration_min),
        "{hinglish_level}":      str(hinglish_level),
        "{fact_lock_json}":      json.dumps(fact_lock, ensure_ascii=False),
        "{story_blueprint_json}": json.dumps(blueprint, ensure_ascii=False),
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def run_retention_blueprint(
    fact_lock: dict,
    blueprint: dict,
    target_duration_min: int,
    case_hint: str,
    hinglish_level: int,
    facts_dir: Path,
) -> dict:
    """
    Generate a retention and revenue optimization blueprint via Claude.

    Non-fatal — if this agent fails, the pipeline continues with an empty blueprint
    and the script outline falls back to standard structure.

    Returns the retention_blueprint dict (empty dict on failure).
    Saves to 02-facts/retention_blueprint.json.
    """
    prompt = _build_prompt(
        fact_lock=fact_lock,
        blueprint=blueprint,
        target_duration_min=target_duration_min,
        case_hint=case_hint,
        hinglish_level=hinglish_level,
    )

    raw_response, stop_reason = call_claude_agent(prompt, agent_name="retention_blueprint")

    raw_path = facts_dir / "_retention_blueprint_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    logger.info("Retention blueprint raw response saved → %s", raw_path)

    if stop_reason == "max_tokens":
        logger.warning("retention_blueprint agent hit max_tokens — blueprint may be incomplete")

    try:
        retention_blueprint = parse_package_response(raw_response, agent_name="retention_blueprint")
    except ValueError as exc:
        logger.error("Retention blueprint JSON parse failed: %s", exc)
        raise

    out_path = facts_dir / "retention_blueprint.json"
    out_path.write_text(json.dumps(retention_blueprint, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Retention blueprint saved → %s | %d re-engagement moments, %d shorts candidates",
        out_path,
        len(retention_blueprint.get("re_engagement_moments", [])),
        len(retention_blueprint.get("shorts_candidates", [])),
    )

    return retention_blueprint
