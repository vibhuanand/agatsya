"""
Originality Transformation Service — Claude agent that produces a plan for writing
an original Hindi documentary episode from a source transcript.

Runs after fact_lock + story_blueprint, before script_outline / script_writer.

The plan tells the script outline and chunk writer:
  - What source structure to avoid (section order, opening style, sign-off language)
  - What original story structure to use instead
  - Which facts are safe to include (all from fact_lock — but stated in original Hindi framing)
  - Specific writer instructions for original documentary narration

Treating the transcript as reference, not blueprint, reduces YouTube reused-content risk
and produces more authentic Hindi documentary content.

Produces:
  02-facts/originality_transformation_plan.json
  02-facts/_originality_transformation_raw_response.txt
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.services.claude_client import call_claude_agent, parse_package_response

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/originality_transformation_agent.txt")

# Characters of the source transcript used for structure analysis only (not for copying).
# Enough to see the intro, sponsor, and first story section — not the full transcript.
_SOURCE_EXCERPT_CHARS = 3000


def _build_prompt(
    case_hint: str,
    target_duration_min: int,
    hinglish_level: int,
    fact_lock: dict,
    blueprint: dict,
    source_transcript: str,
) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    source_excerpt = source_transcript[:_SOURCE_EXCERPT_CHARS]
    replacements = {
        "{case_hint}":            case_hint,
        "{target_duration_min}":  str(target_duration_min),
        "{hinglish_level}":       str(hinglish_level),
        "{story_type}":           blueprint.get("primary_story_type", "true_crime"),
        "{fact_lock_json}":       json.dumps(fact_lock, ensure_ascii=False),
        "{story_blueprint_json}": json.dumps(blueprint, ensure_ascii=False),
        "{source_excerpt}":       source_excerpt,
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def run_originality_transformation(
    case_hint: str,
    target_duration_min: int,
    hinglish_level: int,
    fact_lock: dict,
    blueprint: dict,
    source_transcript: str,
    facts_dir: Path,
) -> dict:
    """
    Run the Originality Transformation Planner Claude agent.

    Produces a plan that the script outline and chunk writer use to create
    an original Hindi episode rather than translating or paraphrasing the
    source transcript.

    Args:
        case_hint:           Short case description for the prompt.
        target_duration_min: Target episode duration in minutes.
        hinglish_level:      Language level (1–5).
        fact_lock:           Verified facts dict (from Stage 2).
        blueprint:           Story blueprint dict (from Stage 3).
        source_transcript:   Cleaned source transcript (first excerpt used for analysis).
        facts_dir:           Path to 02-facts/ directory for saving outputs.

    Returns:
        Parsed transformation plan dict.

    Raises:
        ValueError: If the Claude response cannot be parsed as JSON.
    """
    prompt = _build_prompt(
        case_hint=case_hint,
        target_duration_min=target_duration_min,
        hinglish_level=hinglish_level,
        fact_lock=fact_lock,
        blueprint=blueprint,
        source_transcript=source_transcript,
    )

    raw_response, stop_reason = call_claude_agent(
        prompt, agent_name="originality_transformation"
    )

    raw_path = facts_dir / "_originality_transformation_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    logger.info(
        "Originality transformation raw response saved → %s", raw_path
    )

    if stop_reason == "max_tokens":
        logger.warning(
            "originality_transformation agent hit max_tokens — plan may be truncated"
        )

    try:
        plan = parse_package_response(
            raw_response, agent_name="originality_transformation"
        )
    except ValueError as exc:
        raise ValueError(
            f"Originality Transformation Agent JSON parse failed: {exc}\n"
            f"Raw response saved at: {raw_path}"
        ) from exc

    # Ensure required top-level keys are present with safe defaults.
    # Non-fatal: a partial plan is still useful for the writer.
    plan.setdefault("source_dependency_risk", "medium")
    plan.setdefault("source_structure_summary", [])
    plan.setdefault("source_sequence_to_avoid", [])
    plan.setdefault("safe_facts_to_use", [])
    plan.setdefault("original_story_structure", [])
    plan.setdefault("required_original_elements", [])
    plan.setdefault("phrases_or_patterns_to_avoid", [])
    plan.setdefault("commentary_angles", [])
    plan.setdefault("retention_angles", [])
    plan.setdefault("metadata_originality_rules", [])
    plan.setdefault("writer_instructions", [])

    out_path = facts_dir / "originality_transformation_plan.json"
    out_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Originality transformation plan saved → %s "
        "(source_risk=%s original_sections=%d writer_instructions=%d)",
        out_path,
        plan.get("source_dependency_risk", "?"),
        len(plan.get("original_story_structure", [])),
        len(plan.get("writer_instructions", [])),
    )

    return plan
