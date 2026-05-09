"""
Script Outline Agent — Claude agent that creates a chunk-by-chunk outline
for the Hindi narration script.

No narration text is written here. This stage plans:
  - Which chunks to write (12–16)
  - Target word count per chunk
  - Which facts must appear in each chunk
  - Which recreated scenes to include

Produces:
  03-script/script_outline.json
  03-script/_script_outline_raw_response.txt
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from app.config import settings
from app.schemas import ScriptOutline
from app.services.claude_client import call_claude_agent, parse_package_response
from app.services.prompt_utils import get_channel_rules

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/script_outline_agent.txt")


def _build_prompt(
    case_hint: str,
    episode_id: str,
    target_duration_min: int,
    target_word_count_min: int,
    target_word_count_ideal: int,
    target_word_count_max: int,
    cost_mode: str,
    style: str,
    fact_lock: dict,
    blueprint: dict,
    hinglish_level: int = 2,
    retention_blueprint: dict | None = None,
) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    retention_json = json.dumps(retention_blueprint or {}, ensure_ascii=False)
    has_retention = bool(retention_blueprint)
    replacements = {
        "{channel_rules}": get_channel_rules(),
        "{case_hint}": case_hint,
        "{episode_id}": episode_id,
        "{target_duration_min}": str(target_duration_min),
        "{target_word_count_min}": str(target_word_count_min),
        "{target_word_count_ideal}": str(target_word_count_ideal),
        "{target_word_count_max}": str(target_word_count_max),
        "{cost_mode}": cost_mode,
        "{style}": style,
        "{hinglish_level}": str(hinglish_level),
        "{fact_lock_json}": json.dumps(fact_lock, ensure_ascii=False),
        "{story_blueprint_json}": json.dumps(blueprint, ensure_ascii=False),
        "{retention_blueprint_json}": retention_json,
        "{retention_blueprint_note}": (
            "A Retention Blueprint has been provided. Use it to assign retention_goal, "
            "curiosity_gap, viewer_payoff, and pattern_interrupt to each chunk."
            if has_retention else
            "No Retention Blueprint provided. Use standard narrative structure."
        ),
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def run_script_outline(
    case_hint: str,
    episode_id: str,
    target_duration_min: int,
    target_word_count_min: int,
    target_word_count_ideal: int,
    target_word_count_max: int,
    cost_mode: str,
    style: str,
    fact_lock: dict,
    blueprint: dict,
    script_dir: Path,
    hinglish_level: int = 2,
    retention_blueprint: dict | None = None,
) -> dict:
    """
    Call the Script Outline Agent to produce a chunk-by-chunk plan.
    Validates against ScriptOutline schema. Fatal on failure.

    Returns the outline dict.
    """
    prompt = _build_prompt(
        case_hint=case_hint,
        episode_id=episode_id,
        target_duration_min=target_duration_min,
        target_word_count_min=target_word_count_min,
        target_word_count_ideal=target_word_count_ideal,
        target_word_count_max=target_word_count_max,
        cost_mode=cost_mode,
        style=style,
        fact_lock=fact_lock,
        blueprint=blueprint,
        hinglish_level=hinglish_level,
        retention_blueprint=retention_blueprint,
    )

    raw_response, stop_reason = call_claude_agent(prompt, agent_name="script_outline")

    raw_path = script_dir / "_script_outline_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    logger.info("Script outline raw response saved → %s", raw_path)

    if stop_reason == "max_tokens":
        logger.warning("script_outline agent hit max_tokens — outline may be truncated")

    try:
        outline = parse_package_response(raw_response, agent_name="script_outline")
    except ValueError as exc:
        raise ValueError(
            f"Script Outline Agent JSON parse failed: {exc}\n"
            f"Raw response saved at: {raw_path}"
        ) from exc

    # Validate schema
    try:
        ScriptOutline.model_validate(outline)
    except ValidationError as exc:
        err_path = script_dir / "_script_outline_validation_error.txt"
        err_path.write_text(str(exc), encoding="utf-8")
        raise ValueError(
            f"Script Outline schema validation failed — pipeline stopped.\n"
            f"Error saved at: {err_path}\n{exc}"
        ) from exc

    out_path = script_dir / "script_outline.json"
    out_path.write_text(json.dumps(outline, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Script outline saved → %s (%d chunks planned, %d scenes)",
        out_path,
        len(outline.get("chunks", [])),
        len(outline.get("recreated_scene_plan", [])),
    )

    return outline
