"""
Recreated Dialogue Quality Gate Service — premium mode gate.

Calls the Recreated Dialogue Quality Gate Agent (Claude) to evaluate every
recreated/simulated dialogue scene in the script for labelling compliance,
factual consistency, victim dignity, and naturalness.

Gate thresholds (Python-enforced, premium):
  overall_quality        >= 9
  labelling_compliance   >= 9
  factual_consistency    >= 9
  victim_dignity         >= 9

If the script has NO recreated scenes, the gate auto-passes (no Claude call).

Produces:
  04-review/recreated_dialogue_gate_report.json
  04-review/_recreated_dialogue_gate_raw_response.txt  (only when Claude is called)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.services.claude_client import call_claude_agent, parse_package_response
from app.services.prompt_utils import get_channel_rules
from app.services.report_normalization_service import safe_join_report_items

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/recreated_dialogue_quality_gate_agent.txt")

_THRESHOLDS: dict[str, int] = {
    "overall_quality":      9,
    "labelling_compliance": 9,
    "factual_consistency":  9,
    "victim_dignity":       9,
}


def _python_validate_gate(gate_report: dict) -> tuple[bool, list[str]]:
    scores = gate_report.get("scores", {})
    failures: list[str] = []
    for field, minimum in _THRESHOLDS.items():
        score = scores.get(field, 0)
        if isinstance(score, str):
            try:
                score = int(score)
            except ValueError:
                score = 0
        if score < minimum:
            failures.append(
                f"[DIALOGUE] {field}={score} below required {minimum}. "
                f"Issues: {safe_join_report_items(gate_report.get('scene_issues', []), limit=2, sep=', ') or 'see report'}"
            )
    return len(failures) == 0, failures


def run_recreated_dialogue_quality_gate(
    script_draft: dict,
    fact_lock: dict,
    review_dir: Path,
) -> dict:
    """
    Run the Recreated Dialogue Quality Gate.

    Returns gate_report dict with gate_passed bool.
    Auto-passes with no Claude call if script has no recreated scenes.
    """
    scenes = script_draft.get("recreated_dialogues", {}).get("items", [])

    # Auto-pass when there are no scenes — no charge, no latency
    if not scenes:
        gate_report = {
            "gate_passed": True,
            "no_recreated_scenes": True,
            "scores": {
                "labelling_compliance":  10,
                "factual_consistency":   10,
                "emotional_authenticity": 10,
                "victim_dignity":        10,
                "dialogue_naturalness":  10,
                "overall_quality":       10,
            },
            "scene_issues":        [],
            "labelling_failures":  [],
            "factual_violations":  [],
            "dignity_concerns":    [],
            "required_fixes":      [],
            "gate_notes":          "No recreated scenes present — gate auto-passed.",
        }
        out_path = review_dir / "recreated_dialogue_gate_report.json"
        out_path.write_text(json.dumps(gate_report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Recreated dialogue gate: AUTO-PASSED (no scenes)")
        return gate_report

    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.replace("{channel_rules}", get_channel_rules())
    prompt = prompt.replace("{fact_lock_json}", json.dumps(fact_lock, ensure_ascii=False))
    prompt = prompt.replace("{script_draft_json}", json.dumps(script_draft, ensure_ascii=False))

    raw_response, stop_reason = call_claude_agent(prompt, agent_name="recreated_dialogue_quality_gate")

    raw_path = review_dir / "_recreated_dialogue_gate_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")

    if stop_reason == "max_tokens":
        logger.warning("recreated_dialogue_quality_gate hit max_tokens")

    try:
        gate_report = parse_package_response(raw_response)
    except ValueError as exc:
        raise ValueError(
            f"Recreated Dialogue Gate JSON parse failed: {exc}\n"
            f"Raw response saved at: {raw_path}"
        ) from exc

    py_passed, py_failures = _python_validate_gate(gate_report)
    claude_passed = gate_report.get("gate_passed", False)

    if not py_passed:
        gate_report["gate_passed"] = False
        from app.services.report_normalization_service import stringify_report_item
        existing_fixes = [stringify_report_item(f) for f in gate_report.get("required_fixes", [])]
        gate_report["required_fixes"] = py_failures + [
            f for f in existing_fixes if f not in py_failures
        ]
        if claude_passed:
            logger.warning(
                "Dialogue gate: Python OVERRODE Claude's gate_passed=true. Failures: %s",
                py_failures,
            )

    scores = gate_report.get("scores", {})
    logger.info(
        "Dialogue gate: passed=%s | scenes=%d | overall=%s labelling=%s factual=%s dignity=%s",
        gate_report.get("gate_passed", False),
        len(scenes),
        scores.get("overall_quality", "?"),
        scores.get("labelling_compliance", "?"),
        scores.get("factual_consistency", "?"),
        scores.get("victim_dignity", "?"),
    )

    out_path = review_dir / "recreated_dialogue_gate_report.json"
    out_path.write_text(json.dumps(gate_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Recreated dialogue gate report saved → %s", out_path)

    return gate_report
