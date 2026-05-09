"""
OpenAI Premium Hindi Editor Gate Service.

Sends the final Hindi narration to GPT for an independent second-opinion
copyedit review — grammar, matra/nasalization, naturalness, Hinglish level
consistency, legal clarity, flow, and repetition.

This gate runs AFTER all Claude premium gates have passed. GPT acts as
an independent senior editor, not a replacement for Claude.

Approval thresholds (Python-enforced):
  overall_score                >= 9
  grammar_score                >= 9
  matra_nasalization_score     >= 9
  natural_hindi_score          >= 9
  hinglish_level_match_score   >= 9
  legal_language_clarity_score >= 8
  narration_flow_score         >= 9
  no HIGH severity issues

Produces:
  04-review/openai_premium_hindi_editor_report.json
  04-review/_openai_premium_hindi_editor_raw_response.txt
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import settings
from app.services.openai_client import call_openai_json

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/openai_premium_hindi_editor_gate.txt")

_THRESHOLDS: dict[str, tuple[int, str]] = {
    "overall_score":                (9, "min"),
    "grammar_score":                (9, "min"),
    "matra_nasalization_score":     (9, "min"),
    "natural_hindi_score":          (9, "min"),
    "hinglish_level_match_score":   (9, "min"),
    "legal_language_clarity_score": (8, "min"),
    "narration_flow_score":         (9, "min"),
}

_RAW_PATH   = "04-review/_openai_premium_hindi_editor_raw_response.txt"
_REPORT_NAME = "openai_premium_hindi_editor_report.json"


def _python_validate(report: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for field, (threshold, _) in _THRESHOLDS.items():
        score = report.get(field, 0)
        if isinstance(score, str):
            try:
                score = int(score)
            except ValueError:
                score = 0
        if score < threshold:
            failures.append(
                f"[OPENAI HINDI EDITOR] {field}={score} below required {threshold}"
            )

    high_issues = [i for i in report.get("issues", []) if i.get("severity") == "high"]
    if high_issues:
        failures.append(
            f"[OPENAI HINDI EDITOR] {len(high_issues)} high-severity issue(s) found "
            "— must resolve before audio generation."
        )
    return len(failures) == 0, failures


def _extract_narration_full_text(script_draft: dict) -> str:
    chunks = script_draft.get("hindi_narration_chunks", [])
    return "\n\n".join(
        f"[{c.get('chunk_id', '')}] {c.get('section_title', '')}\n{c.get('text', '')}"
        for c in chunks
    )


def run_openai_premium_hindi_editor_gate(
    script_draft: dict,
    fact_lock: dict,
    blueprint: dict,
    hinglish_level: int,
    lint_report: dict,
    copyedit_report: dict,
    quality_report: dict,
    review_dir: Path,
) -> dict:
    """
    Run the OpenAI Premium Hindi Editor Gate.

    Returns gate_report dict with approved and safe_to_voice bools.
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    narration_text = _extract_narration_full_text(script_draft)

    # Build lean summaries to stay within token budget
    fact_summary = {
        "case_title":    fact_lock.get("case_title", ""),
        "legal_outcome": fact_lock.get("legal_outcome", {}),
        "verified_dates": fact_lock.get("verified_dates", [])[:5],
    }
    blueprint_summary = {
        "primary_story_type": blueprint.get("primary_story_type", ""),
        "main_hook":          blueprint.get("main_hook", ""),
        "emotional_anchor":   blueprint.get("emotional_anchor", ""),
    }
    quality_summary = {
        "approved": quality_report.get("approved", False),
        "scores":   quality_report.get("scores", {}),
    }
    lint_summary = {
        "total_issues":  lint_report.get("total_issues", 0),
        "high_issues":   lint_report.get("high_issues", 0),
        "risk_level":    lint_report.get("risk_level", "none"),
        "issues":        lint_report.get("issues", []),
    }

    user_content = json.dumps({
        "hinglish_level":       hinglish_level,
        "hindi_narration_chunks": script_draft.get("hindi_narration_chunks", []),
        "narration_full_text":  narration_text,
        "fact_lock_summary":    fact_summary,
        "blueprint_summary":    blueprint_summary,
        "claude_quality_report": quality_summary,
        "python_lint_report":   lint_summary,
        "claude_copyedit_report": {
            "approved": copyedit_report.get("approved", False),
            "score":    copyedit_report.get("score", 0),
            "issues":   copyedit_report.get("issues", [])[:10],
        },
    }, ensure_ascii=False)

    raw_path = review_dir.parent / _RAW_PATH if not (review_dir / "_openai_premium_hindi_editor_raw_response.txt").parent == review_dir else review_dir / "_openai_premium_hindi_editor_raw_response.txt"
    raw_path = review_dir / "_openai_premium_hindi_editor_raw_response.txt"

    try:
        report = call_openai_json(
            system_prompt=system_prompt,
            user_content=user_content,
            raw_save_path=raw_path,
            agent_name="openai_premium_hindi_editor",
        )
    except (RuntimeError, ValueError, Exception) as exc:
        raise exc  # caller handles

    # Python threshold enforcement
    py_passed, py_failures = _python_validate(report)
    gpt_approved = report.get("approved", False)
    gpt_safe     = report.get("safe_to_voice", False)

    if not py_passed:
        report["approved"]       = False
        report["safe_to_voice"]  = False
        report["_python_failures"] = py_failures
        if gpt_approved:
            logger.warning(
                "OpenAI Hindi editor gate: Python OVERRODE GPT's approved=true. Failures: %s",
                py_failures,
            )
    elif not gpt_approved or not gpt_safe:
        # GPT itself flagged problems — respect
        report["approved"]      = False
        report["safe_to_voice"] = False

    scores_log = {k: report.get(k, "?") for k in _THRESHOLDS}
    logger.info(
        "OpenAI Hindi editor gate: approved=%s safe_to_voice=%s | scores=%s | issues=%d (%d high)",
        report.get("approved", False),
        report.get("safe_to_voice", False),
        scores_log,
        len(report.get("issues", [])),
        sum(1 for i in report.get("issues", []) if i.get("severity") == "high"),
    )

    out_path = review_dir / _REPORT_NAME
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("OpenAI Hindi editor report saved → %s", out_path)
    return report
