"""
Script Quality Critic Agent — Claude agent that reviews the script draft and
decides if it is ready for audio production.

Produces:
  04-review/script_quality_report.json
  04-review/_script_quality_raw_response.txt

  On the final (post-repair) review pass:
  04-review/final_script_quality_report.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.claude_client import call_claude_agent, parse_package_response
from app.services.prompt_utils import get_channel_rules
from app.services.report_normalization_service import safe_join_report_items

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/script_quality_critic_agent.txt")

# Minimum score thresholds per cost_mode — Python enforces these independent of Claude.
# Premium is the primary mode. Standard/bootstrap thresholds kept for backward-compat.
_SCORE_REQUIREMENTS: dict[str, dict[str, int]] = {
    "premium": {
        "factual_accuracy":        10,
        "safety":                  10,
        "monetization_safety":     10,
        "hindi_naturalness":        9,
        "story_structure":          9,
        "retention_hook":           9,
        "emotional_depth":          9,
        "recreated_scene_quality":  9,
    },
    "standard": {
        "factual_accuracy":     9,
        "safety":              10,
        "monetization_safety":  9,
        "hindi_naturalness":    8,
        "story_structure":      8,
        "retention_hook":       8,
    },
    "bootstrap": {
        "factual_accuracy":     9,
        "safety":              10,
        "monetization_safety":  9,
        "hindi_naturalness":    8,
        "story_structure":      8,
        "retention_hook":       8,
    },
}

_PREMIUM_EXTRA_CHECKS = [
    "legal_outcome_complete",   # all four legal outcome fields must be non-empty
    "no_date_errors",
    "no_generic_structure",
    "not_translated_transcript",
]


def _get_score_requirements(cost_mode: str) -> dict[str, int]:
    return _SCORE_REQUIREMENTS.get(cost_mode, _SCORE_REQUIREMENTS["standard"])


def _build_thresholds_text(cost_mode: str, min_dur: float, hinglish_level: int) -> str:
    reqs = _get_score_requirements(cost_mode)
    lines = [f"  {field:<30} >= {minimum}" for field, minimum in reqs.items()]
    lines.append(f"  {'estimated_duration_min':<30} >= {min_dur}")
    if cost_mode == "premium":
        lines.append("")
        lines.append("  PREMIUM ADDITIONAL RULES (any failure → repair_required=true):")
        lines.append("  — All four legal_outcome fields must be present in the script")
        lines.append("  — No invented or approximated dates")
        lines.append("  — Script must not sound like a close translation of the source")
        lines.append("  — Every recreated scene must carry both labels")
        lines.append(f"  — Hinglish level must match requested level {hinglish_level}")
        lines.append("  — Script structure must feel purposeful, not generic")
    return "\n".join(lines)


def _count_hindi_words(script_draft: dict) -> int:
    """Quick Python word count across all narration chunks."""
    total = 0
    for chunk in script_draft.get("hindi_narration_chunks", []):
        total += len(chunk.get("text", "").split())
    return total


def _python_validate_approval(
    quality_report: dict,
    target_duration_min: int,
    py_duration: float,
    cost_mode: str,
) -> tuple[bool, list[str]]:
    """
    Independent Python-side approval check based on numeric scores.
    Returns (approved, list_of_specific_failures).

    Runs after Claude's own judgement and can override approved=True
    if scores don't actually meet the bar.
    """
    reqs = _get_score_requirements(cost_mode)
    scores = quality_report.get("scores", {})
    failures: list[str] = []

    for field, minimum in reqs.items():
        score = scores.get(field, 0)
        if isinstance(score, str):
            try:
                score = int(score)
            except ValueError:
                score = 0
        if score < minimum:
            issue_key = (
                "fact_issues" if field == "factual_accuracy"
                else "language_issues" if field == "hindi_naturalness"
                else f"{field}_issues"
            )
            detail = safe_join_report_items(quality_report.get(issue_key, []), limit=2, sep=", ") or "see full report"
            failures.append(
                f"[{cost_mode.upper()}] Score {field}={score} below required {minimum}. "
                f"Issues: {detail}"
            )

    # Duration check using Python's word count (more reliable than Claude's estimate)
    min_dur = round(target_duration_min * settings.min_acceptable_duration_ratio, 1)
    if py_duration < min_dur:
        ideal_words = int(target_duration_min * settings.hindi_narration_wpm)
        current_words = quality_report.get("python_word_count", 0)
        needed = max(0, ideal_words - current_words)
        failures.append(
            f"Script duration {py_duration:.1f} min < required {min_dur:.1f} min. "
            f"Need ~{needed} more Hindi words. Expand chunks 004 and 005."
        )

    return len(failures) == 0, failures


def _build_prompt(
    target_duration_min: int,
    cost_mode: str,
    fact_lock: dict,
    blueprint: dict,
    script_draft: dict,
    case_glossary: dict | None = None,
    hinglish_level: int = 2,
) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")

    min_dur = round(target_duration_min * settings.min_acceptable_duration_ratio, 1)
    thresholds_text = _build_thresholds_text(cost_mode, min_dur, hinglish_level)

    replacements = {
        "{target_duration_min}": str(target_duration_min),
        "{min_duration_min}": str(min_dur),
        "{cost_mode}": cost_mode,
        "{hinglish_level}": str(hinglish_level),
        "{quality_thresholds_text}": thresholds_text,
        "{channel_rules}": get_channel_rules(),
        "{fact_lock_json}": json.dumps(fact_lock, ensure_ascii=False),
        "{story_blueprint_json}": json.dumps(blueprint, ensure_ascii=False),
        "{case_glossary_json}": json.dumps(case_glossary or {}, ensure_ascii=False),
        "{script_draft_json}": json.dumps(script_draft, ensure_ascii=False),
    }

    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def run_script_review(
    target_duration_min: int,
    cost_mode: str,
    fact_lock: dict,
    blueprint: dict,
    script_draft: dict,
    review_dir: Path,
    is_final_review: bool = False,
    hinglish_level: int = 2,
    case_glossary: dict | None = None,
) -> dict:
    """
    Call the Script Quality Critic Agent, then apply Python-side score validation.

    Python-side validation:
      - Picks score thresholds based on cost_mode (premium = stricter)
      - Checks each required score against its minimum threshold
      - Checks duration using Python word count (independent of Claude's estimate)
      - If Claude says approved=true but scores fail, overrides to approved=false

    Args:
        is_final_review: if True, saves as final_script_quality_report.json.

    Saves raw response and quality report JSON.
    Returns the quality_report dict (with python_word_count and python_duration_min added).
    """
    prompt = _build_prompt(
        target_duration_min=target_duration_min,
        cost_mode=cost_mode,
        fact_lock=fact_lock,
        blueprint=blueprint,
        script_draft=script_draft,
        case_glossary=case_glossary or {},
        hinglish_level=hinglish_level,
    )

    raw_response, stop_reason = call_claude_agent(prompt, agent_name="script_quality_critic")

    # Save raw response immediately before any parsing
    raw_path = review_dir / "_script_quality_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    logger.info("Quality critic raw response saved → %s", raw_path)

    if stop_reason == "max_tokens":
        logger.warning("script_quality_critic agent hit max_tokens")

    # Parse JSON
    try:
        quality_report = parse_package_response(raw_response)
    except ValueError as exc:
        raise ValueError(
            f"Script Quality Critic Agent JSON parse failed: {exc}\n"
            f"Raw response saved at: {raw_path}"
        ) from exc

    # Add Python word count (ground truth — more reliable than Claude's estimate)
    py_word_count = _count_hindi_words(script_draft)
    py_duration = round(py_word_count / settings.hindi_narration_wpm, 1)
    quality_report["python_word_count"] = py_word_count
    quality_report["python_duration_min"] = py_duration
    quality_report["cost_mode"] = cost_mode

    # Python-side score validation — overrides Claude's own approved decision if needed
    py_approved, py_failures = _python_validate_approval(
        quality_report=quality_report,
        target_duration_min=target_duration_min,
        py_duration=py_duration,
        cost_mode=cost_mode,
    )

    claude_approved = quality_report.get("approved", False)

    if not py_approved:
        quality_report["approved"] = False
        quality_report["repair_required"] = True
        existing_instructions = quality_report.get("repair_instructions", [])
        quality_report["repair_instructions"] = py_failures + [
            i for i in existing_instructions if i not in py_failures
        ]
        if claude_approved:
            logger.warning(
                "Python score validation OVERRODE Claude's approved=true (%s mode). "
                "Failures: %s", cost_mode, py_failures
            )
    elif not claude_approved:
        quality_report["repair_required"] = True

    # Log summary
    scores = quality_report.get("scores", {})
    logger.info(
        "Quality review [%s] — approved=%s (claude=%s, python=%s) | "
        "words: %d (%.1f min) | "
        "FA=%s SS=%s HN=%s ED=%s RH=%s S=%s MS=%s RSQ=%s",
        cost_mode,
        quality_report.get("approved", False),
        claude_approved,
        py_approved,
        py_word_count,
        py_duration,
        scores.get("factual_accuracy", "?"),
        scores.get("story_structure", "?"),
        scores.get("hindi_naturalness", "?"),
        scores.get("emotional_depth", "?"),
        scores.get("retention_hook", "?"),
        scores.get("safety", "?"),
        scores.get("monetization_safety", "?"),
        scores.get("recreated_scene_quality", "?"),
    )

    # Save quality report
    report_name = "final_script_quality_report.json" if is_final_review else "script_quality_report.json"
    out_path = review_dir / report_name
    out_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Quality report saved → %s", out_path)

    return quality_report
