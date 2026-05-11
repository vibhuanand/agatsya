"""
Targeted Chunk Repair Service — repairs only the specific chunks identified
by the Script Quality Critic, one chunk at a time.

This avoids the full-script truncation problem that plagued the old repair agent.
Each Claude call returns a single small NarrationChunk JSON (< 1 KB), not a 40 KB script.

Repair flow:
  1. Read chunk_repair_targets from quality_report
  2. For each target: load the chunk from 03-script/chunks/<chunk_id>.json
  3. Send only that chunk + fact_lock + blueprint summary + hinglish_level + repair_instruction
  4. Get back one repaired NarrationChunk JSON
  5. Retry once on failure
  6. If repair fails → keep original chunk, add warning, status=needs_human_review
  7. Merge all repaired chunks back into the full script
  8. Re-save hindi_narration_full.txt, hindi_narration_chunks.json, elevenlabs_chunks.json
  9. Save audit files: chunk_repair_targets.json, script_repair_report.json,
                       hinglish_level_assessment.json

Produces (in 04-review/):
  chunk_repair_targets.json
  script_repair_report.json
  hinglish_level_assessment.json

Produces (in 03-script/chunks/):
  repaired_{chunk_id}.json
  _repair_raw_{chunk_id}.txt

Produces (updated in 03-script/):
  script_draft.json
  hindi_narration_full.txt
  hindi_narration_chunks.json
  elevenlabs_chunks.json
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

_PROMPT_PATH = Path("app/prompts/targeted_chunk_repair_agent.txt")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_chunk(chunks_dir: Path, chunk_id: str) -> dict | None:
    """Load a chunk JSON from disk. Returns None if not found."""
    path = chunks_dir / f"{chunk_id}.json"
    if not path.exists():
        logger.warning("Chunk file not found: %s", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load chunk '%s': %s", chunk_id, exc)
        return None


def _build_repair_prompt(
    target: dict,
    current_chunk: dict,
    fact_lock: dict,
    blueprint: dict,
    hinglish_level: int,
) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")

    chunk_id = target.get("chunk_id", "")
    section_title = current_chunk.get("section_title", "")
    issue_type = target.get("issue_type", "")
    problem = target.get("problem", "")
    repair_instruction = target.get("repair_instruction", "")

    sensitivity_rules = blueprint.get("sensitivity_rules", [])
    sensitivity_rules_text = "; ".join(sensitivity_rules[:5]) if sensitivity_rules else "none"

    replacements = {
        "{channel_rules}": get_channel_rules(),
        "{hinglish_level}": str(hinglish_level),
        "{chunk_id}": chunk_id,
        "{section_title}": section_title,
        "{issue_type}": issue_type,
        "{problem}": problem,
        "{repair_instruction}": repair_instruction,
        "{current_chunk_json}": json.dumps(current_chunk, ensure_ascii=False),
        "{fact_lock_json}": json.dumps(fact_lock, ensure_ascii=False),
        "{main_hook}": blueprint.get("main_hook", ""),
        "{emotional_anchor}": blueprint.get("emotional_anchor", ""),
        "{sensitivity_rules_text}": sensitivity_rules_text,
    }

    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def _repair_one_chunk(
    target: dict,
    chunks_dir: Path,
    fact_lock: dict,
    blueprint: dict,
    hinglish_level: int,
) -> tuple[dict | None, str | None]:
    """
    Attempt to repair a single chunk. Returns (repaired_chunk, None) on success,
    or (None, error_message) on failure after retries.

    Retries once on parse/validation failure.
    """
    chunk_id = target.get("chunk_id", "unknown")
    current_chunk = _load_chunk(chunks_dir, chunk_id)

    if current_chunk is None:
        return None, f"Chunk file not found for '{chunk_id}'"

    last_error: str = ""

    for attempt in range(1, 3):  # max 2 attempts
        agent_label = f"chunk_repair_{chunk_id}_a{attempt}"
        try:
            # Per-chunk budget check — counts each chunk repair individually
            note_repair("claude", f"targeted_chunk_repair:{chunk_id}")

            prompt = _build_repair_prompt(
                target=target,
                current_chunk=current_chunk,
                fact_lock=fact_lock,
                blueprint=blueprint,
                hinglish_level=hinglish_level,
            )

            raw_response, stop_reason = call_claude_agent(prompt, agent_name=agent_label)

            # Save raw response
            raw_path = chunks_dir / f"_repair_raw_{chunk_id}.txt"
            raw_path.write_text(raw_response, encoding="utf-8")

            if stop_reason == "max_tokens":
                logger.warning("Chunk repair '%s' attempt %d hit max_tokens", chunk_id, attempt)

            repaired = parse_package_response(raw_response, agent_name=agent_label)
            NarrationChunk.model_validate(repaired)

            # Save repaired chunk
            repaired_path = chunks_dir / f"repaired_{chunk_id}.json"
            repaired_path.write_text(json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8")

            logger.info(
                "Chunk '%s' repaired — %d words (attempt %d/2)",
                chunk_id, repaired.get("estimated_words", 0), attempt,
            )
            return repaired, None

        except (ValueError, ValidationError) as exc:
            last_error = str(exc)
            logger.warning(
                "Chunk repair '%s' attempt %d failed: %s",
                chunk_id, attempt, exc,
            )

    return None, f"Chunk '{chunk_id}' repair failed after 2 attempts: {last_error}"


# ─── Main targeted repair function ────────────────────────────────────────────

def run_targeted_chunk_repair(
    fact_lock: dict,
    blueprint: dict,
    script_draft: dict,
    quality_report: dict,
    hinglish_level: int,
    script_dir: Path,
    review_dir: Path,
) -> tuple[dict, dict]:
    """
    Repair only the chunks identified in quality_report.chunk_repair_targets.

    - For each target: load chunk, call Claude, validate, save repaired file.
    - Failed repairs keep the original chunk — script is not abandoned.
    - After all repairs: merge chunks, re-save assembly files, save audit files.
    - Returns (updated_script_draft, repair_report) so the pipeline can gate on failures.

    repair_report schema:
      {
        "status": "targeted_repair_complete" | "needs_human_review",
        "chunks_repaired": int,
        "chunks_failed": int,
        "has_failures": bool,
        "repair_results": [...],
        "warnings": [...]
      }
    """
    chunks_dir = script_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_repair_targets = quality_report.get("chunk_repair_targets", [])
    hinglish_assessment = quality_report.get("hinglish_level_assessment", {})

    logger.info(
        "Targeted Chunk Repair START — %d targets, hinglish_level=%d",
        len(chunk_repair_targets), hinglish_level,
    )

    # ── Save chunk_repair_targets.json for audit ──────────────────────────────
    (review_dir / "chunk_repair_targets.json").write_text(
        json.dumps(chunk_repair_targets, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # If no repair targets, promote draft as-is
    if not chunk_repair_targets:
        logger.info("No chunk_repair_targets — promoting draft as final without changes")
        return _promote_and_save(script_draft, script_dir, review_dir, [], hinglish_assessment)
        # returns (script_draft, repair_report) with has_failures=False

    # ── Repair each targeted chunk ────────────────────────────────────────────
    repair_results: list[dict] = []
    repaired_chunks: dict[str, dict] = {}  # chunk_id → repaired chunk
    valid_chunk_ids = {
        chunk.get("chunk_id")
        for chunk in script_draft.get("hindi_narration_chunks", [])
        if chunk.get("chunk_id")
    }

    for target in chunk_repair_targets:
        chunk_id = target.get("chunk_id", "")
        if not chunk_id:
            logger.warning("Skipping repair target with missing chunk_id: %s", target)
            continue
        if chunk_id not in valid_chunk_ids:
            logger.warning(
                "Skipping non-narration repair target '%s' — targeted chunk repair only handles narration chunks",
                chunk_id,
            )
            repair_results.append({
                "chunk_id": chunk_id,
                "status": "skipped_non_chunk_target",
                "issue_type": target.get("issue_type", ""),
                "repair_instruction": target.get("repair_instruction", ""),
                "note": (
                    "Not counted as a chunk repair failure. Route metadata/dialogue "
                    "issues to their dedicated repair gates."
                ),
            })
            continue

        # Load original chunk now so we can record words_before accurately
        original_chunk = _load_chunk(chunks_dir, chunk_id)
        words_before = 0
        if original_chunk:
            words_before = (
                original_chunk.get("estimated_words")
                or len(original_chunk.get("text", "").split())
            )

        repaired, error = _repair_one_chunk(
            target=target,
            chunks_dir=chunks_dir,
            fact_lock=fact_lock,
            blueprint=blueprint,
            hinglish_level=hinglish_level,
        )

        if repaired is not None:
            repaired_chunks[chunk_id] = repaired
            repair_results.append({
                "chunk_id": chunk_id,
                "status": "repaired",
                "issue_type": target.get("issue_type", ""),
                "words_before": words_before,
                "words_after": repaired.get("estimated_words", 0),
            })
        else:
            logger.warning("Repair failed for chunk '%s' — keeping original: %s", chunk_id, error)
            repair_results.append({
                "chunk_id": chunk_id,
                "status": "failed_kept_original",
                "issue_type": target.get("issue_type", ""),
                "error": error,
            })

    # ── Merge repaired chunks back into the script ────────────────────────────
    original_chunks = script_draft.get("hindi_narration_chunks", [])
    merged_chunks = []
    for chunk in original_chunks:
        chunk_id = chunk.get("chunk_id", "")
        if chunk_id in repaired_chunks:
            merged_chunks.append(repaired_chunks[chunk_id])
            logger.info("Merged repaired chunk: %s", chunk_id)
        else:
            merged_chunks.append(chunk)

    # Update script_draft in-place with merged chunks
    script_final = dict(script_draft)
    script_final["hindi_narration_chunks"] = merged_chunks

    # Any failed repairs → flag for human review
    failed = [r for r in repair_results if r["status"] == "failed_kept_original"]

    return _promote_and_save(
        script_final, script_dir, review_dir, repair_results, hinglish_assessment,
        has_failures=len(failed) > 0,
    )  # returns (script_final, repair_report)


def _promote_and_save(
    script_draft: dict,
    script_dir: Path,
    review_dir: Path,
    repair_results: list[dict],
    hinglish_assessment: dict,
    has_failures: bool = False,
) -> tuple[dict, dict]:
    """
    Save the (repaired) script_draft as script_final.json and update assembly files.
    Also save audit reports. Returns (script_draft, repair_report).
    """
    chunks = script_draft.get("hindi_narration_chunks", [])

    def _save_json(path: Path, content: object) -> None:
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Primary output: script_final.json ────────────────────────────────────
    _save_json(script_dir / "script_final.json", script_draft)

    # ── Re-save assembly files so downstream stages see updated chunks ────────
    _save_json(script_dir / "hindi_narration_chunks.json", chunks)
    _save_json(script_dir / "elevenlabs_chunks.json", _extract_elevenlabs_chunks(chunks))
    (script_dir / "hindi_narration_full.txt").write_text(
        _extract_full_narration(chunks), encoding="utf-8"
    )

    # ── Audit: script_repair_report.json ──────────────────────────────────────
    total_words = sum(c.get("estimated_words", 0) for c in chunks)
    repaired_count = sum(1 for r in repair_results if r.get("status") == "repaired")
    failed_count = sum(1 for r in repair_results if r.get("status") == "failed_kept_original")
    skipped_non_chunk_count = sum(
        1 for r in repair_results if r.get("status") == "skipped_non_chunk_target"
    )

    repair_report = {
        "status": "needs_human_review" if has_failures else "targeted_repair_complete",
        "chunks_repaired": repaired_count,
        "chunks_failed": failed_count,
        "non_chunk_targets_skipped": skipped_non_chunk_count,
        "has_failures": has_failures,
        "total_words_after_repair": total_words,
        "repair_results": repair_results,
        "warnings": (
            [f"{failed_count} chunk(s) could not be repaired — original content kept. Automated retry exhausted — safe_to_voice=false."]
            if has_failures else []
        ),
    }
    _save_json(review_dir / "script_repair_report.json", repair_report)

    # ── Audit: hinglish_level_assessment.json ─────────────────────────────────
    if hinglish_assessment:
        _save_json(review_dir / "hinglish_level_assessment.json", hinglish_assessment)

    logger.info(
        "Targeted Chunk Repair COMPLETE — %d repaired, %d failed, %d total words",
        repaired_count, failed_count, total_words,
    )

    return script_draft, repair_report


def promote_draft_as_final(script_draft: dict, script_dir: Path) -> dict:
    """
    Copy script_draft to script_final.json without any repair.
    Used when repair is not needed or as fallback.
    """
    (script_dir / "script_final.json").write_text(
        json.dumps(script_draft, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Draft promoted as final (no repair) → %s", script_dir / "script_final.json")
    return script_draft
