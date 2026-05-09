"""
Retention Quality Gate Service — premium mode gate.

Evaluates the final Hindi narration against the retention blueprint and
audience retention best practices. Checks opening hook, curiosity architecture,
pacing, emotional arc, midpoint re-engagement, ending payoff, and subscriber conversion.

Includes run_retention_repair() which repairs specific chunks using Claude,
following the same pattern as targeted_chunk_repair_service.py.

Gate thresholds (Python-enforced, premium):
  overall_retention_score    >= 9
  opening_hook_score         >= 9
  first_30_seconds_score     >= 9
  curiosity_gap_score        >= 8
  pacing_score               >= 8
  emotional_arc_score        >= 9
  midpoint_retention_score   >= 8
  ending_payoff_score        >= 8
  no HIGH severity issues

Produces:
  04-review/retention_quality_report.json
  04-review/_retention_quality_raw_response.txt

Repair produces (per chunk):
  03-script/chunks/repaired_retention_{chunk_id}.json
  03-script/chunks/_retention_repair_raw_{chunk_id}.txt
  04-review/retention_repair_report.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from app.schemas import NarrationChunk
from app.services.call_tracker import note_repair, BudgetExceededError
from app.services.claude_client import call_claude_agent, parse_package_response
from app.services.prompt_utils import get_channel_rules
from app.services.script_assembler_service import (
    _extract_full_narration,
    _extract_elevenlabs_chunks,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/retention_quality_gate_agent.txt")
_REPAIR_PROMPT_PATH = Path("app/prompts/targeted_chunk_repair_agent.txt")

_THRESHOLDS: dict[str, tuple[int, str]] = {
    "overall_retention_score":  (9, "min"),
    "opening_hook_score":       (9, "min"),
    "first_30_seconds_score":   (9, "min"),
    "curiosity_gap_score":      (8, "min"),
    "pacing_score":             (8, "min"),
    "emotional_arc_score":      (9, "min"),
    "midpoint_retention_score": (8, "min"),
    "ending_payoff_score":      (8, "min"),
}

_REPORT_NAME = "retention_quality_report.json"
_RAW_NAME    = "_retention_quality_raw_response.txt"


# ─── Gate ─────────────────────────────────────────────────────────────────────

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
                f"[RETENTION] {field}={score} below required {threshold}"
            )
    high_issues = [i for i in report.get("issues", []) if i.get("severity") == "high"]
    if high_issues:
        failures.append(
            f"[RETENTION] {len(high_issues)} high-severity retention issue(s) — must fix before audio."
        )
    return len(failures) == 0, failures


def _build_gate_prompt(
    script_draft: dict,
    retention_blueprint: dict,
    blueprint: dict,
    target_duration_min: int,
) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")

    chunks = script_draft.get("hindi_narration_chunks", [])
    narration_text = "\n\n".join(
        f"[{c.get('chunk_id', '')}] {c.get('section_title', '')}\n{c.get('text', '')}"
        for c in chunks
    )

    retention_summary = {
        "opening_hook":         retention_blueprint.get("opening_hook", ""),
        "central_question":     retention_blueprint.get("central_question", ""),
        "viewer_promise":       retention_blueprint.get("viewer_promise", ""),
        "retention_beats":      retention_blueprint.get("retention_beats", [])[:12],
        "re_engagement_moments": retention_blueprint.get("re_engagement_moments", []),
        "ending_strategy":      retention_blueprint.get("ending_strategy", ""),
    }

    user_content = json.dumps(
        {
            "target_duration_min":    target_duration_min,
            "hindi_narration_chunks": chunks,
            "narration_full_text":    narration_text,
            "youtube_metadata":       script_draft.get("youtube_metadata", {}),
            "retention_blueprint_summary": retention_summary,
            "story_type":             blueprint.get("primary_story_type", ""),
            "emotional_anchor":       blueprint.get("emotional_anchor", ""),
            "sensitivity_rules":      blueprint.get("sensitivity_rules", [])[:5],
        },
        ensure_ascii=False,
    )

    # System prompt is the template; user content is injected separately
    # We combine both into one prompt (Claude call pattern)
    combined = template + "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nINPUT DATA\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" + user_content
    return combined


def run_retention_quality_gate(
    script_draft: dict,
    retention_blueprint: dict,
    blueprint: dict,
    target_duration_min: int,
    review_dir: Path,
    is_recheck: bool = False,
) -> dict:
    """
    Run the Retention Quality Gate.

    Returns gate_report with approved bool and scores.
    """
    label = "retention_quality_recheck" if is_recheck else "retention_quality"
    prompt = _build_gate_prompt(
        script_draft=script_draft,
        retention_blueprint=retention_blueprint,
        blueprint=blueprint,
        target_duration_min=target_duration_min,
    )

    raw_response, stop_reason = call_claude_agent(prompt, agent_name=label)

    raw_path = review_dir / _RAW_NAME
    raw_path.write_text(raw_response, encoding="utf-8")

    if stop_reason == "max_tokens":
        logger.warning("[%s] hit max_tokens — report may be incomplete", label)

    try:
        report = parse_package_response(raw_response, agent_name=label)
    except ValueError as exc:
        raise ValueError(f"Retention quality gate JSON parse failed: {exc}") from exc

    py_passed, py_failures = _python_validate(report)

    if not py_passed:
        report["approved"] = False
        report["_python_failures"] = py_failures
        if report.get("approved", True):
            logger.warning("[%s] Python OVERRODE gate approved=true. Failures: %s", label, py_failures)
    elif not report.get("approved", False):
        report["approved"] = False

    scores_log = {k: report.get(k, "?") for k in _THRESHOLDS}
    logger.info(
        "[%s] approved=%s | scores=%s | issues=%d (%d high)",
        label,
        report.get("approved", False),
        scores_log,
        len(report.get("issues", [])),
        sum(1 for i in report.get("issues", []) if i.get("severity") == "high"),
    )

    out_path = review_dir / _REPORT_NAME
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Retention quality report saved → %s", out_path)
    return report


# ─── Retention Targeted Repair ────────────────────────────────────────────────

def _repair_one_retention_chunk(
    target: dict,
    current_chunk: dict,
    chunks_dir: Path,
    fact_lock: dict,
    blueprint: dict,
    retention_blueprint: dict,
    hinglish_level: int,
) -> tuple[dict | None, str | None]:
    """Repair one chunk for a retention issue using Claude. Retry once."""
    chunk_id = target.get("chunk_id", "unknown")
    template = _REPAIR_PROMPT_PATH.read_text(encoding="utf-8")

    sensitivity_rules = blueprint.get("sensitivity_rules", [])
    sensitivity_rules_text = "; ".join(sensitivity_rules[:5]) if sensitivity_rules else "none"

    # Enrich repair instruction with retention context
    base_instruction = target.get("repair_instruction", "")
    opening_hook = retention_blueprint.get("opening_hook", "")
    central_question = retention_blueprint.get("central_question", "")
    ending_strategy = retention_blueprint.get("ending_strategy", "")
    retention_context = ""
    if chunk_id.startswith("001") and (opening_hook or central_question):
        retention_context = (
            f" OPENING HOOK: '{opening_hook}'."
            + (f" CENTRAL QUESTION: '{central_question}'." if central_question else "")
        )
    if chunk_id.startswith("010") and ending_strategy:
        retention_context = f" ENDING STRATEGY: {ending_strategy}."

    enriched_instruction = base_instruction + retention_context

    replacements = {
        "{channel_rules}":         get_channel_rules(),
        "{hinglish_level}":        str(hinglish_level),
        "{chunk_id}":              chunk_id,
        "{section_title}":         current_chunk.get("section_title", ""),
        "{issue_type}":            "retention",
        "{problem}":               target.get("problem", ""),
        "{repair_instruction}":    enriched_instruction,
        "{current_chunk_json}":    json.dumps(current_chunk, ensure_ascii=False),
        "{fact_lock_json}":        json.dumps(fact_lock, ensure_ascii=False),
        "{main_hook}":             blueprint.get("main_hook", ""),
        "{emotional_anchor}":      blueprint.get("emotional_anchor", ""),
        "{sensitivity_rules_text}": sensitivity_rules_text,
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)

    raw_path = chunks_dir / f"_retention_repair_raw_{chunk_id}.txt"
    last_error = ""

    for attempt in range(1, 3):
        agent_label = f"retention_repair_{chunk_id}_a{attempt}"
        try:
            # Per-chunk budget check — counts each retention repair individually
            note_repair("claude", f"retention_repair:{chunk_id}")

            raw_response, stop_reason = call_claude_agent(prompt, agent_name=agent_label)
            raw_path.write_text(raw_response, encoding="utf-8")
            if stop_reason == "max_tokens":
                logger.warning("Retention repair '%s' attempt %d hit max_tokens", chunk_id, attempt)

            repaired = parse_package_response(raw_response, agent_name=agent_label)
            NarrationChunk.model_validate(repaired)

            out_path = chunks_dir / f"repaired_retention_{chunk_id}.json"
            out_path.write_text(json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(
                "Retention repair '%s' — %d words (attempt %d/2)",
                chunk_id, repaired.get("estimated_words", 0), attempt,
            )
            return repaired, None

        except (ValueError, ValidationError) as exc:
            last_error = str(exc)
            logger.warning("Retention repair '%s' attempt %d failed: %s", chunk_id, attempt, exc)

    return None, f"Retention repair of '{chunk_id}' failed after 2 attempts: {last_error}"


def run_retention_repair(
    script_draft: dict,
    fact_lock: dict,
    blueprint: dict,
    retention_blueprint: dict,
    repair_targets: list[dict],
    hinglish_level: int,
    script_dir: Path,
    review_dir: Path,
) -> tuple[dict, dict]:
    """
    Repair specific chunks for retention issues using Claude.
    Follows the same pattern as targeted_chunk_repair_service.run_targeted_chunk_repair().

    Returns (updated_script_draft, repair_report).
    """
    chunks_dir = script_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Build current chunk lookup
    chunks_by_id: dict[str, dict] = {
        c.get("chunk_id", ""): c
        for c in script_draft.get("hindi_narration_chunks", [])
        if c.get("chunk_id")
    }

    logger.info(
        "Retention Repair START — %d target(s)", len(repair_targets)
    )

    (review_dir / "retention_repair_targets.json").write_text(
        json.dumps(repair_targets, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    repair_results: list[dict] = []
    repaired_chunks: dict[str, dict] = {}

    for target in repair_targets:
        chunk_id = target.get("chunk_id", "")
        if not chunk_id:
            continue

        current_chunk = chunks_by_id.get(chunk_id)
        if current_chunk is None:
            repair_results.append({
                "chunk_id": chunk_id, "status": "failed_kept_original",
                "error": f"chunk '{chunk_id}' not found in script",
            })
            continue

        words_before = current_chunk.get("estimated_words") or len(current_chunk.get("text", "").split())

        repaired, error = _repair_one_retention_chunk(
            target=target,
            current_chunk=current_chunk,
            chunks_dir=chunks_dir,
            fact_lock=fact_lock,
            blueprint=blueprint,
            retention_blueprint=retention_blueprint,
            hinglish_level=hinglish_level,
        )

        if repaired is not None:
            repaired_chunks[chunk_id] = repaired
            repair_results.append({
                "chunk_id": chunk_id, "status": "repaired",
                "words_before": words_before,
                "words_after": repaired.get("estimated_words", 0),
            })
        else:
            logger.warning("Retention repair failed for '%s' — keeping original: %s", chunk_id, error)
            repair_results.append({
                "chunk_id": chunk_id, "status": "failed_kept_original", "error": error,
            })

    # Merge repaired chunks back
    original_chunks = script_draft.get("hindi_narration_chunks", [])
    merged_chunks = [
        repaired_chunks.get(c.get("chunk_id", ""), c)
        for c in original_chunks
    ]

    script_final = dict(script_draft)
    script_final["hindi_narration_chunks"] = merged_chunks

    failed = [r for r in repair_results if r["status"] == "failed_kept_original"]
    has_failures = len(failed) > 0
    repaired_count = len(repair_results) - len(failed)
    total_words = sum(c.get("estimated_words", 0) for c in merged_chunks)

    def _save(path: Path, data: object) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    _save(script_dir / "script_final.json", script_final)
    _save(script_dir / "hindi_narration_chunks.json", merged_chunks)
    _save(script_dir / "elevenlabs_chunks.json", _extract_elevenlabs_chunks(merged_chunks))
    (script_dir / "hindi_narration_full.txt").write_text(
        _extract_full_narration(merged_chunks), encoding="utf-8"
    )

    repair_report = {
        "status":                   "needs_human_review" if has_failures else "retention_repair_complete",
        "chunks_repaired":          repaired_count,
        "chunks_failed":            len(failed),
        "has_failures":             has_failures,
        "total_words_after_repair": total_words,
        "repair_results":           repair_results,
        "warnings": (
            [f"{len(failed)} retention repair(s) failed — original kept. Manual review required."]
            if has_failures else []
        ),
    }
    _save(review_dir / "retention_repair_report.json", repair_report)

    logger.info(
        "Retention Repair COMPLETE — %d repaired, %d failed, %d total words",
        repaired_count, len(failed), total_words,
    )
    return script_final, repair_report
