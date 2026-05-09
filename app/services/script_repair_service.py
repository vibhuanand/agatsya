"""
Script Repair Agent — Claude agent that fixes only the issues listed in the
quality report's repair_instructions.

Runs at most MAX_REPAIR_PASSES times (default: 1).

Produces:
  03-script/script_final.json
  03-script/hindi_narration_full.txt
  03-script/hindi_narration_chunks.json
  03-script/recreated_dialogues.json
  03-script/elevenlabs_chunks.json
  03-script/youtube_metadata.json
  03-script/_script_repair_raw_response.txt
  04-review/script_repair_report.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.services.claude_client import call_claude_agent, parse_package_response
from app.services.prompt_utils import get_channel_rules
from app.services.script_writer_service import (
    _extract_full_narration,
    _extract_elevenlabs_chunks,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/script_repair_agent.txt")


def _build_prompt(
    fact_lock: dict,
    blueprint: dict,
    script_draft: dict,
    repair_instructions: list,
) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    replacements = {
        "{channel_rules}": get_channel_rules(),
        "{fact_lock_json}": json.dumps(fact_lock, ensure_ascii=False),
        "{story_blueprint_json}": json.dumps(blueprint, ensure_ascii=False),
        "{script_draft_json}": json.dumps(script_draft, ensure_ascii=False),
        "{repair_instructions_json}": json.dumps(repair_instructions, ensure_ascii=False, indent=2),
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def run_script_repair(
    fact_lock: dict,
    blueprint: dict,
    script_draft: dict,
    quality_report: dict,
    script_dir: Path,
    review_dir: Path,
) -> dict:
    """
    Call the Script Repair Agent for ONE pass.

    Saves:
      _script_repair_raw_response.txt
      script_final.json + companion text/chunk files
      script_repair_report.json (summary of what was repaired)

    Returns the repaired script dict (script_final).
    Raises ValueError on JSON parse failure (raw response already saved).
    """
    repair_instructions = quality_report.get("repair_instructions", [])
    logger.info(
        "Running script repair — %d instruction(s): %s",
        len(repair_instructions),
        repair_instructions,
    )

    prompt = _build_prompt(
        fact_lock=fact_lock,
        blueprint=blueprint,
        script_draft=script_draft,
        repair_instructions=repair_instructions,
    )

    raw_response, stop_reason = call_claude_agent(prompt, agent_name="script_repair")

    # Save raw response immediately before any parsing
    raw_path = script_dir / "_script_repair_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    logger.info("Script repair raw response saved → %s", raw_path)

    if stop_reason == "max_tokens":
        logger.warning(
            "script_repair agent hit max_tokens — repaired script may be truncated. "
            "Consider increasing CLAUDE_MAX_TOKENS in .env."
        )

    # Parse JSON
    try:
        script_final = parse_package_response(raw_response)
    except ValueError as exc:
        raise ValueError(
            f"Script Repair Agent JSON parse failed: {exc}\n"
            f"Raw response saved at: {raw_path}"
        ) from exc

    # Save final script files
    def _save_json(name: str, content) -> None:
        (script_dir / name).write_text(
            json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    _save_json("script_final.json", script_final)

    chunks = script_final.get("hindi_narration_chunks", [])
    (script_dir / "hindi_narration_full.txt").write_text(
        _extract_full_narration(chunks), encoding="utf-8"
    )
    _save_json("hindi_narration_chunks.json", chunks)
    _save_json("recreated_dialogues.json", script_final.get("recreated_dialogues", {}))
    _save_json("elevenlabs_chunks.json", _extract_elevenlabs_chunks(chunks))
    _save_json("youtube_metadata.json", script_final.get("youtube_metadata", {}))

    logger.info(
        "Repaired script saved → %s (%d chunks)",
        script_dir / "script_final.json",
        len(chunks),
    )

    # Write repair report for audit trail
    repair_report = {
        "repair_ran": True,
        "instructions_applied": repair_instructions,
        "stop_reason": stop_reason,
        "chunks_after_repair": len(chunks),
    }
    (review_dir / "script_repair_report.json").write_text(
        json.dumps(repair_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return script_final


def promote_draft_as_final(
    script_draft: dict,
    script_dir: Path,
) -> dict:
    """
    No repair needed — copy draft to final script location unchanged.
    """
    def _save_json(name: str, content) -> None:
        (script_dir / name).write_text(
            json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    _save_json("script_final.json", script_draft)

    chunks = script_draft.get("hindi_narration_chunks", [])
    (script_dir / "hindi_narration_full.txt").write_text(
        _extract_full_narration(chunks), encoding="utf-8"
    )
    _save_json("hindi_narration_chunks.json", chunks)
    _save_json("recreated_dialogues.json", script_draft.get("recreated_dialogues", {}))
    _save_json("elevenlabs_chunks.json", _extract_elevenlabs_chunks(chunks))
    _save_json("youtube_metadata.json", script_draft.get("youtube_metadata", {}))

    logger.info(
        "Draft promoted to final (no repair needed) → %s",
        script_dir / "script_final.json",
    )

    return script_draft
