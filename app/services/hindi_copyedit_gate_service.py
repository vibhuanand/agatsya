"""
Hindi Copyedit Gate Service — premium mode gate.

Calls the Hindi Copyedit Gate Agent (Claude) to evaluate the final Hindi
narration for grammar, matra/nasalization, sentence flow, legal clarity,
Hinglish consistency, and repetition.

Also provides run_copyedit_repair() — targeted chunk repair for copyedit
issues, reusing the same repair agent prompt as the quality repair system
but saving to distinct output files.

Gate thresholds (Python-enforced, premium):
  score                           >= 9
  grammar_score                   >= 9
  matra_nasalization_score        >= 9
  sentence_flow_score             >= 9
  legal_language_clarity_score    >= 8
  hinglish_level_consistency_score >= 9
  no high-severity issues

Produces:
  04-review/hindi_copyedit_report.json
  04-review/_hindi_copyedit_raw_response.txt

  04-review/hindi_copyedit_repair_report.json          (if repair runs)
  03-script/chunks/repaired_copyedit_<chunk_id>.json   (if repair runs)
  03-script/chunks/_copyedit_repair_raw_<chunk_id>.txt (if repair runs)
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

_GATE_PROMPT_PATH   = Path("app/prompts/hindi_copyedit_gate_agent.txt")
_REPAIR_PROMPT_PATH = Path("app/prompts/targeted_chunk_repair_agent.txt")

# (threshold, direction: "min" >= or "max" <=)
_THRESHOLDS: dict[str, tuple[int, str]] = {
    "score":                           (9, "min"),
    "grammar_score":                   (9, "min"),
    "matra_nasalization_score":        (9, "min"),
    "sentence_flow_score":             (9, "min"),
    "legal_language_clarity_score":    (8, "min"),
    "hinglish_level_consistency_score":(9, "min"),
}


# ─── Python gate threshold enforcement ───────────────────────────────────────

def _python_validate_gate(gate_report: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for field, (threshold, direction) in _THRESHOLDS.items():
        score = gate_report.get(field, 0)
        if isinstance(score, str):
            try:
                score = int(score)
            except ValueError:
                score = 0
        if direction == "min" and score < threshold:
            failures.append(
                f"[COPYEDIT] {field}={score} below required {threshold}"
            )

    # Any high-severity issue also fails the gate
    high_issues = [
        i for i in gate_report.get("issues", [])
        if i.get("severity") == "high"
    ]
    if high_issues:
        failures.append(
            f"[COPYEDIT] {len(high_issues)} high-severity issue(s) found — "
            "must be resolved before audio generation."
        )

    return len(failures) == 0, failures


# ─── Gate ─────────────────────────────────────────────────────────────────────

def run_hindi_copyedit_gate(
    script_draft: dict,
    fact_lock: dict,
    blueprint: dict,
    hinglish_level: int,
    lint_report: dict,
    review_dir: Path,
    is_recheck: bool = False,
) -> dict:
    """
    Call the Hindi Copyedit Gate Agent and apply Python threshold enforcement.

    Args:
        is_recheck: True when this is the post-repair re-run (saves as *_recheck.json).
    """
    template = _GATE_PROMPT_PATH.read_text(encoding="utf-8")

    prompt = template.replace("{channel_rules}",      get_channel_rules())
    prompt = prompt.replace("{hinglish_level}",       str(hinglish_level))
    prompt = prompt.replace("{lint_report_json}",     json.dumps(lint_report,   ensure_ascii=False))
    prompt = prompt.replace("{fact_lock_json}",       json.dumps(fact_lock,     ensure_ascii=False))
    prompt = prompt.replace("{script_draft_json}",    json.dumps(script_draft,  ensure_ascii=False))

    agent_label = "hindi_copyedit_gate_recheck" if is_recheck else "hindi_copyedit_gate"
    raw_response, stop_reason = call_claude_agent(prompt, agent_name=agent_label)

    raw_filename = "_hindi_copyedit_recheck_raw_response.txt" if is_recheck else "_hindi_copyedit_raw_response.txt"
    raw_path = review_dir / raw_filename
    raw_path.write_text(raw_response, encoding="utf-8")

    if stop_reason == "max_tokens":
        logger.warning("%s hit max_tokens", agent_label)

    try:
        gate_report = parse_package_response(raw_response)
    except ValueError as exc:
        raise ValueError(
            f"Hindi Copyedit Gate JSON parse failed: {exc}\n"
            f"Raw response saved at: {raw_path}"
        ) from exc

    py_passed, py_failures = _python_validate_gate(gate_report)
    claude_approved = gate_report.get("approved", False)

    if not py_passed:
        gate_report["approved"] = False
        existing_targets = gate_report.get("chunk_repair_targets", [])
        gate_report["_python_failures"] = py_failures
        if claude_approved:
            logger.warning(
                "Hindi copyedit gate: Python OVERRODE Claude's approved=true. Failures: %s",
                py_failures,
            )
    elif not claude_approved:
        pass  # Claude flagged failures — respect

    scores_log = {k: gate_report.get(k, "?") for k in _THRESHOLDS}
    logger.info(
        "Hindi copyedit gate [%s]: approved=%s | scores=%s | issues=%d (%d high)",
        "recheck" if is_recheck else "initial",
        gate_report.get("approved", False),
        scores_log,
        len(gate_report.get("issues", [])),
        sum(1 for i in gate_report.get("issues", []) if i.get("severity") == "high"),
    )

    report_filename = "hindi_copyedit_recheck_report.json" if is_recheck else "hindi_copyedit_report.json"
    out_path = review_dir / report_filename
    out_path.write_text(json.dumps(gate_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Always keep canonical report as hindi_copyedit_report.json (overwrite with latest)
    (review_dir / "hindi_copyedit_report.json").write_text(
        json.dumps(gate_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Hindi copyedit report saved → %s", review_dir / "hindi_copyedit_report.json")
    return gate_report


# ─── Copyedit targeted repair ─────────────────────────────────────────────────

def _load_chunk(chunks_dir: Path, chunk_id: str) -> dict | None:
    path = chunks_dir / f"{chunk_id}.json"
    if not path.exists():
        logger.warning("Copyedit repair: chunk file not found: %s", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load chunk '%s': %s", chunk_id, exc)
        return None


def _build_copyedit_repair_prompt(
    target: dict,
    current_chunk: dict,
    fact_lock: dict,
    blueprint: dict,
    hinglish_level: int,
) -> str:
    """Build a repair prompt using the shared targeted_chunk_repair_agent template."""
    template = _REPAIR_PROMPT_PATH.read_text(encoding="utf-8")

    chunk_id       = target.get("chunk_id", "")
    section_title  = current_chunk.get("section_title", "")
    issue_type     = target.get("issue_type", "hindi_copyedit")
    problem        = target.get("problem", "")
    repair_instr   = target.get("repair_instruction", "")

    sensitivity_rules     = blueprint.get("sensitivity_rules", [])
    sensitivity_rules_text = "; ".join(sensitivity_rules[:5]) or "none"

    replacements = {
        "{channel_rules}":          get_channel_rules(),
        "{hinglish_level}":         str(hinglish_level),
        "{chunk_id}":               chunk_id,
        "{section_title}":          section_title,
        "{issue_type}":             issue_type,
        "{problem}":                problem,
        "{repair_instruction}":     repair_instr,
        "{current_chunk_json}":     json.dumps(current_chunk, ensure_ascii=False),
        "{fact_lock_json}":         json.dumps(fact_lock, ensure_ascii=False),
        "{main_hook}":              blueprint.get("main_hook", ""),
        "{emotional_anchor}":       blueprint.get("emotional_anchor", ""),
        "{sensitivity_rules_text}": sensitivity_rules_text,
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def _repair_one_copyedit_chunk(
    target: dict,
    chunks_dir: Path,
    fact_lock: dict,
    blueprint: dict,
    hinglish_level: int,
) -> tuple[dict | None, str | None]:
    """Repair a single chunk for copyedit issues. Retries once on failure."""
    chunk_id = target.get("chunk_id", "unknown")
    current_chunk = _load_chunk(chunks_dir, chunk_id)
    if current_chunk is None:
        return None, f"Chunk file not found for '{chunk_id}'"

    last_error = ""
    for attempt in range(1, 3):
        agent_label = f"copyedit_repair_{chunk_id}_a{attempt}"
        try:
            # Per-chunk budget check — counts each copyedit repair individually
            note_repair("claude", f"copyedit_repair:{chunk_id}")

            prompt = _build_copyedit_repair_prompt(
                target=target,
                current_chunk=current_chunk,
                fact_lock=fact_lock,
                blueprint=blueprint,
                hinglish_level=hinglish_level,
            )
            raw_response, stop_reason = call_claude_agent(prompt, agent_name=agent_label)

            raw_path = chunks_dir / f"_copyedit_repair_raw_{chunk_id}.txt"
            raw_path.write_text(raw_response, encoding="utf-8")

            if stop_reason == "max_tokens":
                logger.warning("Copyedit repair '%s' attempt %d hit max_tokens", chunk_id, attempt)

            repaired = parse_package_response(raw_response, agent_name=agent_label)
            NarrationChunk.model_validate(repaired)

            repaired_path = chunks_dir / f"repaired_copyedit_{chunk_id}.json"
            repaired_path.write_text(json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8")

            logger.info(
                "Copyedit repair '%s' done — %d words (attempt %d/2)",
                chunk_id, repaired.get("estimated_words", 0), attempt,
            )
            return repaired, None

        except (ValueError, ValidationError) as exc:
            last_error = str(exc)
            logger.warning("Copyedit repair '%s' attempt %d failed: %s", chunk_id, attempt, exc)

    return None, f"Copyedit repair '{chunk_id}' failed after 2 attempts: {last_error}"


def run_copyedit_repair(
    script_draft: dict,
    fact_lock: dict,
    blueprint: dict,
    copyedit_targets: list[dict],
    hinglish_level: int,
    script_dir: Path,
    review_dir: Path,
) -> tuple[dict, dict]:
    """
    Run targeted copyedit repair on the chunks listed in copyedit_targets.

    Returns (updated_script_draft, copyedit_repair_report).
    Saves updated assembly files (hindi_narration_chunks.json, etc.) so the
    post-repair gate re-run sees the corrected text.
    """
    chunks_dir = script_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate targets by chunk_id — consolidate multiple issues per chunk
    # into a single repair call with a combined repair_instruction.
    consolidated: dict[str, dict] = {}
    for t in copyedit_targets:
        cid = t.get("chunk_id", "")
        if not cid:
            continue
        if cid not in consolidated:
            consolidated[cid] = dict(t)
        else:
            # Append problem and instruction to the existing entry
            existing = consolidated[cid]
            existing["problem"] += f" | {t.get('problem', '')}"
            existing["repair_instruction"] += f" Also: {t.get('repair_instruction', '')}"

    logger.info(
        "Copyedit repair START — %d unique chunks to repair (from %d targets)",
        len(consolidated), len(copyedit_targets),
    )

    repair_results: list[dict] = []
    repaired_chunks: dict[str, dict] = {}

    for chunk_id, target in consolidated.items():
        original_chunk = _load_chunk(chunks_dir, chunk_id)
        words_before = 0
        if original_chunk:
            words_before = (
                original_chunk.get("estimated_words")
                or len(original_chunk.get("text", "").split())
            )

        repaired, error = _repair_one_copyedit_chunk(
            target=target,
            chunks_dir=chunks_dir,
            fact_lock=fact_lock,
            blueprint=blueprint,
            hinglish_level=hinglish_level,
        )

        if repaired is not None:
            repaired_chunks[chunk_id] = repaired
            repair_results.append({
                "chunk_id":    chunk_id,
                "status":      "repaired",
                "issue_type":  "hindi_copyedit",
                "words_before": words_before,
                "words_after": repaired.get("estimated_words", 0),
            })
        else:
            logger.warning("Copyedit repair failed for '%s' — keeping original: %s", chunk_id, error)
            repair_results.append({
                "chunk_id":    chunk_id,
                "status":      "failed_kept_original",
                "issue_type":  "hindi_copyedit",
                "error":       error,
            })

    # Merge repaired chunks back into script
    original_chunks = script_draft.get("hindi_narration_chunks", [])
    merged_chunks = [
        repaired_chunks.get(c.get("chunk_id", ""), c)
        for c in original_chunks
    ]

    script_updated = dict(script_draft)
    script_updated["hindi_narration_chunks"] = merged_chunks

    # Re-save assembly files so downstream gates see corrected text
    def _save_json(path: Path, content: object) -> None:
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")

    _save_json(script_dir / "script_final.json", script_updated)
    _save_json(script_dir / "hindi_narration_chunks.json", merged_chunks)
    _save_json(script_dir / "elevenlabs_chunks.json", _extract_elevenlabs_chunks(merged_chunks))
    (script_dir / "hindi_narration_full.txt").write_text(
        _extract_full_narration(merged_chunks), encoding="utf-8"
    )

    failed = [r for r in repair_results if r["status"] == "failed_kept_original"]
    has_failures = len(failed) > 0

    copyedit_repair_report = {
        "status":          "needs_human_review" if has_failures else "copyedit_repair_complete",
        "chunks_repaired": sum(1 for r in repair_results if r["status"] == "repaired"),
        "chunks_failed":   len(failed),
        "has_failures":    has_failures,
        "repair_results":  repair_results,
        "warnings": (
            [f"{len(failed)} copyedit chunk(s) could not be repaired — original content kept."]
            if has_failures else []
        ),
    }

    _save_json(review_dir / "hindi_copyedit_repair_report.json", copyedit_repair_report)
    logger.info(
        "Copyedit repair COMPLETE — %d repaired, %d failed",
        copyedit_repair_report["chunks_repaired"],
        copyedit_repair_report["chunks_failed"],
    )
    return script_updated, copyedit_repair_report
