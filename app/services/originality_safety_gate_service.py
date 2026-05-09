"""
Originality & Safety Gate Service — premium mode gate.

Calls the Originality Safety Gate Agent (Claude) to evaluate whether the
script is sufficiently original, legally safe, and ad-safe.

Gate thresholds (Python-enforced, cost_mode=premium):
  copying_risk        <= 2
  reused_content_risk <= 2
  copyright_risk      <= 2
  transformative_value >= 9
  ad_safety            >= 9
  recreated_audio      >= 9
  metadata_safety      >= 9

Produces:
  04-review/originality_safety_gate_report.json
  04-review/_originality_safety_gate_raw_response.txt
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.services.claude_client import call_claude_agent, parse_package_response
from app.services.prompt_utils import get_channel_rules

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/originality_safety_gate_agent.txt")

# Thresholds: (minimum, direction) where direction is "min" (>=) or "max" (<=)
_THRESHOLDS: dict[str, tuple[int, str]] = {
    "copying_risk":         (2, "max"),
    "reused_content_risk":  (2, "max"),
    "copyright_risk":       (2, "max"),
    "transformative_value": (9, "min"),
    "ad_safety":            (9, "min"),
    "recreated_audio":      (9, "min"),
    "metadata_safety":      (9, "min"),
}


def _python_validate_gate(gate_report: dict) -> tuple[bool, list[str]]:
    """Independent Python threshold enforcement."""
    scores = gate_report.get("scores", {})
    failures: list[str] = []

    for field, (threshold, direction) in _THRESHOLDS.items():
        score = scores.get(field, 0)
        if isinstance(score, str):
            try:
                score = int(score)
            except ValueError:
                score = 0
        if direction == "max" and score > threshold:
            failures.append(
                f"[ORIGINALITY] {field}={score} exceeds maximum {threshold}. "
                f"Issues: {', '.join(gate_report.get('copying_issues', [])[:2]) or 'see report'}"
            )
        elif direction == "min" and score < threshold:
            failures.append(
                f"[ORIGINALITY] {field}={score} below required {threshold}. "
                f"Issues: {', '.join(gate_report.get('ad_safety_issues', [])[:2]) or 'see report'}"
            )

    return len(failures) == 0, failures


def run_originality_safety_gate(
    script_draft: dict,
    source_transcript: str,
    similarity_report: dict,
    review_dir: Path,
) -> dict:
    """
    Run the Originality & Safety Gate.

    Args:
        script_draft:       The script_final dict (post-repair if repair ran)
        source_transcript:  The cleaned source transcript text
        similarity_report:  Output from text_similarity_service
        review_dir:         04-review/ directory for output files

    Returns:
        gate_report dict with gate_passed bool.
    """
    template = _PROMPT_PATH.read_text(encoding="utf-8")

    prompt = template.replace("{channel_rules}", get_channel_rules())
    prompt = prompt.replace("{similarity_report_json}", json.dumps(similarity_report, ensure_ascii=False))
    prompt = prompt.replace("{script_draft_json}", json.dumps(script_draft, ensure_ascii=False))
    prompt = prompt.replace("{source_transcript}", source_transcript[:8000])   # cap to avoid token overflow

    raw_response, stop_reason = call_claude_agent(prompt, agent_name="originality_safety_gate")

    raw_path = review_dir / "_originality_safety_gate_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")

    if stop_reason == "max_tokens":
        logger.warning("originality_safety_gate hit max_tokens")

    try:
        gate_report = parse_package_response(raw_response)
    except ValueError as exc:
        raise ValueError(
            f"Originality Safety Gate JSON parse failed: {exc}\n"
            f"Raw response saved at: {raw_path}"
        ) from exc

    # Python threshold enforcement
    py_passed, py_failures = _python_validate_gate(gate_report)
    claude_passed = gate_report.get("gate_passed", False)

    if not py_passed:
        gate_report["gate_passed"] = False
        existing_fixes = gate_report.get("required_fixes", [])
        gate_report["required_fixes"] = py_failures + [
            f for f in existing_fixes if f not in py_failures
        ]
        if claude_passed:
            logger.warning(
                "Originality gate: Python OVERRODE Claude's gate_passed=true. Failures: %s",
                py_failures,
            )
    elif not claude_passed:
        pass  # Claude flagged failures, respect that

    scores = gate_report.get("scores", {})
    logger.info(
        "Originality gate: passed=%s | copying=%s reused=%s copyright=%s "
        "transformative=%s ad_safety=%s recreated=%s metadata=%s",
        gate_report.get("gate_passed", False),
        scores.get("copying_risk", "?"),
        scores.get("reused_content_risk", "?"),
        scores.get("copyright_risk", "?"),
        scores.get("transformative_value", "?"),
        scores.get("ad_safety", "?"),
        scores.get("recreated_audio", "?"),
        scores.get("metadata_safety", "?"),
    )

    out_path = review_dir / "originality_safety_gate_report.json"
    out_path.write_text(json.dumps(gate_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Originality safety gate report saved → %s", out_path)

    return gate_report
