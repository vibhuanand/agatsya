"""
Story Blueprint Agent — Claude agent that classifies the story type and creates
a narrative plan from the verified fact_lock.

Produces:
  02-facts/story_blueprint.json
  02-facts/_story_blueprint_raw_response.txt
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.services.claude_client import call_claude_agent, parse_package_response

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/story_blueprint_agent.txt")


def _build_prompt(case_hint: str, fact_lock: dict) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    # Compact JSON to save tokens — the blueprint agent only needs to read it
    fact_lock_json = json.dumps(fact_lock, ensure_ascii=False)
    replacements = {
        "{case_hint}": case_hint,
        "{fact_lock_json}": fact_lock_json,
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def run_story_blueprint(
    case_hint: str,
    fact_lock: dict,
    facts_dir: Path,
) -> dict:
    """
    Call the Story Blueprint Agent.

    Saves:
      _story_blueprint_raw_response.txt — raw Claude output
      story_blueprint.json              — parsed JSON

    Returns the story_blueprint dict.
    Raises ValueError on JSON parse failure (raw response already saved).
    """
    prompt = _build_prompt(case_hint=case_hint, fact_lock=fact_lock)

    raw_response, stop_reason = call_claude_agent(prompt, agent_name="story_blueprint")

    # Save raw response immediately before any parsing
    raw_path = facts_dir / "_story_blueprint_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    logger.info("Story blueprint raw response saved → %s", raw_path)

    if stop_reason == "max_tokens":
        logger.warning("story_blueprint agent hit max_tokens — output may be truncated")

    # Parse JSON
    try:
        blueprint = parse_package_response(raw_response)
    except ValueError as exc:
        raise ValueError(
            f"Story Blueprint Agent JSON parse failed: {exc}\n"
            f"Raw response saved at: {raw_path}"
        ) from exc

    # Save parsed JSON
    out_path = facts_dir / "story_blueprint.json"
    out_path.write_text(json.dumps(blueprint, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Story blueprint saved → %s", out_path)

    return blueprint
