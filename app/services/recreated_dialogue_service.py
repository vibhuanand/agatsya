"""
Recreated Dialogue Agent — Claude agent that writes short, clearly-labelled
recreated dialogue scenes from the outline's recreated_scene_plan.

Produces:
  03-script/recreated_dialogues_draft.json
  03-script/_recreated_dialogue_raw_response.txt
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.services.claude_client import call_claude_agent, parse_package_response
from app.services.prompt_utils import get_channel_rules

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/recreated_dialogue_agent.txt")

_EMPTY_DIALOGUES = {"items": []}


def _build_prompt(
    recreated_scene_plan: list,
    audio_moments: list,
    approved_scenes: list,
) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    replacements = {
        "{channel_rules}": get_channel_rules(),
        "{recreated_scene_plan_json}": json.dumps(recreated_scene_plan, ensure_ascii=False, indent=2),
        "{audio_moments_json}": json.dumps(audio_moments, ensure_ascii=False, indent=2),
        "{approved_scenes_json}": json.dumps(approved_scenes, ensure_ascii=False, indent=2),
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def run_recreated_dialogue(
    outline: dict,
    fact_lock: dict,
    blueprint: dict,
    script_dir: Path,
) -> dict:
    """
    Generate recreated dialogue scenes from the outline's recreated_scene_plan.

    If no scenes are planned, returns empty {"items": []} without calling Claude.
    Returns the dialogues dict.
    """
    recreated_scene_plan = outline.get("recreated_scene_plan", [])

    if not recreated_scene_plan:
        logger.info("No recreated scenes in outline — skipping Recreated Dialogue Agent")
        out_path = script_dir / "recreated_dialogues_draft.json"
        out_path.write_text(json.dumps(_EMPTY_DIALOGUES, ensure_ascii=False, indent=2), encoding="utf-8")
        return _EMPTY_DIALOGUES

    audio_moments = fact_lock.get("important_audio_or_call_moments", [])
    approved_scenes = blueprint.get("recreated_scenes_to_use", [])

    prompt = _build_prompt(
        recreated_scene_plan=recreated_scene_plan,
        audio_moments=audio_moments,
        approved_scenes=approved_scenes,
    )

    raw_response, stop_reason = call_claude_agent(prompt, agent_name="recreated_dialogue")

    raw_path = script_dir / "_recreated_dialogue_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    logger.info("Recreated dialogue raw response saved → %s", raw_path)

    if stop_reason == "max_tokens":
        logger.warning("recreated_dialogue agent hit max_tokens")

    try:
        dialogues = parse_package_response(raw_response, agent_name="recreated_dialogue")
    except ValueError as exc:
        logger.warning(
            "Recreated dialogue JSON parse failed: %s — using empty dialogues", exc
        )
        dialogues = _EMPTY_DIALOGUES

    out_path = script_dir / "recreated_dialogues_draft.json"
    out_path.write_text(json.dumps(dialogues, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Recreated dialogues saved → %s (%d scenes)", out_path, len(dialogues.get("items", [])))

    return dialogues
