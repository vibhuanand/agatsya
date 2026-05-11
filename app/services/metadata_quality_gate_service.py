"""
Metadata Quality Gate Service — premium mode gate.

Calls the Metadata Quality Gate Agent (Claude) to evaluate YouTube metadata,
shorts plan, and thumbnail options for clickability, respectfulness, ad-safety,
copyright risk, factual accuracy, CTR potential, and originality.

Gate thresholds (Python-enforced, premium):
  clickability           >= 8
  respectfulness         >= 9
  copyright_reuse_risk   <= 2
  monetization_safety    >= 10
  factual_accuracy       >= 9
  title_ctr_score        >= 7
  thumbnail_text_score   >= 7
  curiosity_score        >= 7
  originality_score      >= 7

Produces:
  04-review/metadata_quality_gate_report.json
  04-review/_metadata_quality_gate_raw_response.txt
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.services.claude_client import call_claude_agent, parse_package_response
from app.services.prompt_utils import get_channel_rules
from app.services.report_normalization_service import safe_join_report_items

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/metadata_quality_gate_agent.txt")

# (threshold, direction: "min" = >=, "max" = <=)
_THRESHOLDS: dict[str, tuple[int, str]] = {
    "clickability":          (8,  "min"),
    "respectfulness":        (9,  "min"),
    "copyright_reuse_risk":  (2,  "max"),
    "monetization_safety":   (10, "min"),
    "factual_accuracy":      (9,  "min"),
    "title_ctr_score":       (7,  "min"),
    "thumbnail_text_score":  (7,  "min"),
    "curiosity_score":       (7,  "min"),
    "originality_score":     (7,  "min"),
}


def _python_validate_gate(gate_report: dict) -> tuple[bool, list[str]]:
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
                f"[METADATA] {field}={score} exceeds maximum {threshold}. "
                f"Issues: {safe_join_report_items(gate_report.get('title_issues', []), limit=2, sep=', ') or 'see report'}"
            )
        elif direction == "min" and score < threshold:
            failures.append(
                f"[METADATA] {field}={score} below required {threshold}. "
                f"Issues: {safe_join_report_items(gate_report.get('monetization_risks', []), limit=2, sep=', ') or 'see report'}"
            )
    return len(failures) == 0, failures


def run_metadata_quality_gate(
    script_draft: dict,
    fact_lock: dict,
    review_dir: Path,
) -> dict:
    """
    Run the Metadata Quality Gate.

    Returns gate_report dict with gate_passed bool.
    """
    metadata  = script_draft.get("youtube_metadata", {})
    shorts    = script_draft.get("shorts_plan", {})

    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.replace("{channel_rules}", get_channel_rules())
    prompt = prompt.replace("{fact_lock_json}",   json.dumps(fact_lock, ensure_ascii=False))
    prompt = prompt.replace("{metadata_json}",    json.dumps(metadata,  ensure_ascii=False))
    prompt = prompt.replace("{shorts_plan_json}", json.dumps(shorts,    ensure_ascii=False))

    raw_response, stop_reason = call_claude_agent(prompt, agent_name="metadata_quality_gate")

    raw_path = review_dir / "_metadata_quality_gate_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")

    if stop_reason == "max_tokens":
        logger.warning("metadata_quality_gate hit max_tokens")

    try:
        gate_report = parse_package_response(raw_response)
    except ValueError as exc:
        raise ValueError(
            f"Metadata Quality Gate JSON parse failed: {exc}\n"
            f"Raw response saved at: {raw_path}"
        ) from exc

    py_passed, py_failures = _python_validate_gate(gate_report)
    claude_passed = gate_report.get("gate_passed", False)

    if not py_passed:
        gate_report["gate_passed"] = False
        # Stringify Claude's fixes before deduplication so dict items don't
        # cause TypeError when compared against the string py_failures entries.
        from app.services.report_normalization_service import stringify_report_item
        existing_fixes = [stringify_report_item(f) for f in gate_report.get("required_fixes", [])]
        gate_report["required_fixes"] = py_failures + [
            f for f in existing_fixes if f not in py_failures
        ]
        if claude_passed:
            logger.warning(
                "Metadata gate: Python OVERRODE Claude's gate_passed=true. Failures: %s",
                py_failures,
            )

    scores = gate_report.get("scores", {})
    logger.info(
        "Metadata gate: passed=%s | clickability=%s respectfulness=%s "
        "copyright_risk=%s monetization=%s factual=%s ctr=%s thumbnail=%s curiosity=%s originality=%s",
        gate_report.get("gate_passed", False),
        scores.get("clickability", "?"),
        scores.get("respectfulness", "?"),
        scores.get("copyright_reuse_risk", "?"),
        scores.get("monetization_safety", "?"),
        scores.get("factual_accuracy", "?"),
        scores.get("title_ctr_score", "?"),
        scores.get("thumbnail_text_score", "?"),
        scores.get("curiosity_score", "?"),
        scores.get("originality_score", "?"),
    )

    out_path = review_dir / "metadata_quality_gate_report.json"
    out_path.write_text(json.dumps(gate_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Metadata quality gate report saved → %s", out_path)

    return gate_report
