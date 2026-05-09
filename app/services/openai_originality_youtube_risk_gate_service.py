"""
OpenAI Originality & YouTube Risk Gate Service.

Sends the final script, metadata, recreated dialogues, and a condensed
source transcript view to GPT for an independent originality/copyright/ad-safety
second opinion — after Claude's own originality and metadata gates have run.

Approval thresholds (Python-enforced):
  copying_risk_score           <= 2   (lower is better)
  reused_content_risk_score    <= 2   (lower is better)
  copyright_risk_score         <= 2   (lower is better)
  transformative_value_score   >= 9
  youtube_ad_safety_score      >= 9
  recreated_audio_safety_score >= 9
  metadata_safety_score        >= 9
  no HIGH severity issues

Produces:
  04-review/openai_originality_youtube_risk_report.json
  04-review/_openai_originality_youtube_risk_raw_response.txt
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import settings
from app.services.openai_client import call_openai_json

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/openai_originality_youtube_risk_gate.txt")

# (threshold, direction: "min" >= | "max" <=)
_THRESHOLDS: dict[str, tuple[int, str]] = {
    "copying_risk_score":           (2, "max"),
    "reused_content_risk_score":    (2, "max"),
    "copyright_risk_score":         (2, "max"),
    "transformative_value_score":   (9, "min"),
    "youtube_ad_safety_score":      (9, "min"),
    "recreated_audio_safety_score": (9, "min"),
    "metadata_safety_score":        (9, "min"),
}

_REPORT_NAME = "openai_originality_youtube_risk_report.json"
_RAW_NAME    = "_openai_originality_youtube_risk_raw_response.txt"


def _python_validate(report: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for field, (threshold, direction) in _THRESHOLDS.items():
        score = report.get(field, 0)
        if isinstance(score, str):
            try:
                score = int(score)
            except ValueError:
                score = 0
        if direction == "max" and score > threshold:
            failures.append(
                f"[OPENAI ORIGINALITY] {field}={score} exceeds maximum {threshold}"
            )
        elif direction == "min" and score < threshold:
            failures.append(
                f"[OPENAI ORIGINALITY] {field}={score} below required {threshold}"
            )

    high_issues = [i for i in report.get("issues", []) if i.get("severity") == "high"]
    if high_issues:
        failures.append(
            f"[OPENAI ORIGINALITY] {len(high_issues)} high-severity issue(s) — "
            "must resolve before audio generation."
        )
    return len(failures) == 0, failures


def run_openai_originality_youtube_risk_gate(
    script_draft: dict,
    source_transcript: str,
    fact_lock: dict,
    blueprint: dict,
    claude_originality_report: dict,
    claude_metadata_report: dict,
    claude_dialogue_report: dict,
    review_dir: Path,
) -> dict:
    """
    Run the OpenAI Originality & YouTube Risk Gate.

    Args:
        script_draft:              Final script dict (with metadata, dialogues, etc.)
        source_transcript:         Cleaned source transcript text (capped to first 6000 chars)
        fact_lock:                 Verified facts
        blueprint:                 Narrative blueprint
        claude_originality_report: Claude's own originality gate report (for context)
        claude_metadata_report:    Claude's own metadata gate report (for context)
        claude_dialogue_report:    Claude's own recreated dialogue gate report (for context)
        review_dir:                04-review/ directory

    Returns gate_report dict with approved and safe_to_voice bools.
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    # Build concise summaries to manage token budget
    fact_summary = {
        "case_title":   fact_lock.get("case_title", ""),
        "legal_outcome": fact_lock.get("legal_outcome", {}),
    }
    blueprint_summary = {
        "primary_story_type": blueprint.get("primary_story_type", ""),
        "main_hook":          blueprint.get("main_hook", ""),
        "sensitivity_rules":  blueprint.get("sensitivity_rules", []),
    }

    user_content = json.dumps({
        "source_transcript_excerpt": source_transcript[:6000],
        "hindi_narration_chunks":    script_draft.get("hindi_narration_chunks", []),
        "recreated_dialogues":       script_draft.get("recreated_dialogues", {}),
        "youtube_metadata":          script_draft.get("youtube_metadata", {}),
        "shorts_plan":               script_draft.get("shorts_plan", {}),
        "fact_lock_summary":         fact_summary,
        "blueprint_summary":         blueprint_summary,
        "claude_originality_report": {
            "gate_passed":    claude_originality_report.get("gate_passed", False),
            "scores":         claude_originality_report.get("scores", {}),
            "required_fixes": claude_originality_report.get("required_fixes", []),
        },
        "claude_metadata_report": {
            "gate_passed":    claude_metadata_report.get("gate_passed", False),
            "scores":         claude_metadata_report.get("scores", {}),
        },
        "claude_dialogue_report": {
            "gate_passed": claude_dialogue_report.get("gate_passed", False),
            "no_scenes":   claude_dialogue_report.get("no_recreated_scenes", False),
            "scores":      claude_dialogue_report.get("scores", {}),
        },
    }, ensure_ascii=False)

    raw_path = review_dir / _RAW_NAME

    try:
        report = call_openai_json(
            system_prompt=system_prompt,
            user_content=user_content,
            raw_save_path=raw_path,
            agent_name="openai_originality_youtube_risk",
        )
    except (RuntimeError, ValueError, Exception) as exc:
        raise exc  # caller handles

    py_passed, py_failures = _python_validate(report)
    gpt_approved = report.get("approved", False)
    gpt_safe     = report.get("safe_to_voice", False)

    if not py_passed:
        report["approved"]         = False
        report["safe_to_voice"]    = False
        report["_python_failures"] = py_failures
        if gpt_approved:
            logger.warning(
                "OpenAI originality gate: Python OVERRODE GPT's approved=true. Failures: %s",
                py_failures,
            )
    elif not gpt_approved or not gpt_safe:
        report["approved"]      = False
        report["safe_to_voice"] = False

    logger.info(
        "OpenAI originality gate: approved=%s safe_to_voice=%s | "
        "copying=%s reused=%s copyright=%s transformative=%s yt_safety=%s metadata=%s",
        report.get("approved", False),
        report.get("safe_to_voice", False),
        report.get("copying_risk_score", "?"),
        report.get("reused_content_risk_score", "?"),
        report.get("copyright_risk_score", "?"),
        report.get("transformative_value_score", "?"),
        report.get("youtube_ad_safety_score", "?"),
        report.get("metadata_safety_score", "?"),
    )

    out_path = review_dir / _REPORT_NAME
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("OpenAI originality risk report saved → %s", out_path)
    return report
