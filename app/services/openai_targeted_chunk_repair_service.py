"""
OpenAI Targeted Chunk Repair Service.

Uses GPT to repair specific Hindi narration chunks identified by:
  - OpenAI Premium Hindi Editor Gate (grammar, matra, naturalness issues)
  - OpenAI Originality / YouTube Risk Gate (source-copying, originality issues)

Never rewrites the full script. Repairs only targeted chunks.

Constraints:
  - Max chunks = settings.openai_repair_max_chunks (default 6)
  - If target count exceeds max → caller must set needs_human_review, no repair runs
  - Each chunk: one OpenAI call + one retry on failure
  - Failed repairs keep original chunk (no data loss)

Produces (per repaired chunk):
  03-script/chunks/openai_repaired_{chunk_id}.json
  03-script/chunks/_openai_repair_raw_{chunk_id}.txt

Produces (after all repairs):
  04-review/openai_repair_targets.json
  04-review/openai_repair_report.json
  03-script/script_final.json           (updated)
  03-script/hindi_narration_chunks.json (updated)
  03-script/elevenlabs_chunks.json      (updated)
  03-script/hindi_narration_full.txt    (updated)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from app.schemas import NarrationChunk
from app.services.call_tracker import note_repair, BudgetExceededError
from app.services.openai_client import call_openai_json
from app.services.script_assembler_service import (
    _extract_full_narration,
    _extract_elevenlabs_chunks,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/openai_targeted_chunk_repair_agent.txt")


# ─── Repair helpers ───────────────────────────────────────────────────────────

def _build_user_content(
    target: dict,
    current_chunk: dict,
    fact_lock: dict,
    blueprint: dict,
    hinglish_level: int,
) -> str:
    """Build the user content JSON for a single OpenAI chunk repair call."""
    return json.dumps(
        {
            "chunk_to_repair": current_chunk,
            "repair_target": {
                "chunk_id":          target.get("chunk_id", ""),
                "issue_type":        target.get("issue_type", ""),
                "problem":           target.get("problem", ""),
                "repair_instruction": target.get("repair_instruction", ""),
            },
            "hinglish_level": hinglish_level,
            "fact_lock_summary": {
                "case_title":      fact_lock.get("case_title", ""),
                "legal_outcome":   fact_lock.get("legal_outcome", {}),
                "verified_people": fact_lock.get("verified_people", [])[:10],
            },
            "story_context": {
                "main_hook":         blueprint.get("main_hook", ""),
                "emotional_anchor":  blueprint.get("emotional_anchor", ""),
                "sensitivity_rules": blueprint.get("sensitivity_rules", [])[:5],
            },
        },
        ensure_ascii=False,
    )


def _repair_one_chunk(
    target: dict,
    current_chunk: dict,
    chunks_dir: Path,
    fact_lock: dict,
    blueprint: dict,
    hinglish_level: int,
) -> tuple[dict | None, str | None]:
    """
    Repair a single chunk via OpenAI. Retry once on failure.
    Returns (repaired_chunk, None) on success, (None, error_msg) on failure.
    """
    chunk_id = target.get("chunk_id", "unknown")
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    user_content = _build_user_content(
        target, current_chunk, fact_lock, blueprint, hinglish_level
    )

    raw_path = chunks_dir / f"_openai_repair_raw_{chunk_id}.txt"
    last_error = ""

    for attempt in range(1, 3):  # max 2 attempts
        try:
            # Per-chunk budget check — counts each OpenAI repair individually
            note_repair("openai", f"openai_chunk_repair:{chunk_id}")

            repaired = call_openai_json(
                system_prompt=system_prompt,
                user_content=user_content,
                raw_save_path=raw_path,
                agent_name=f"openai_chunk_repair_{chunk_id}_a{attempt}",
            )
            # Validate against NarrationChunk schema
            NarrationChunk.model_validate(repaired)

            out_path = chunks_dir / f"openai_repaired_{chunk_id}.json"
            out_path.write_text(
                json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(
                "OpenAI chunk repair '%s' — %d words (attempt %d/2)",
                chunk_id, repaired.get("estimated_words", 0), attempt,
            )
            return repaired, None

        except (ValueError, ValidationError, Exception) as exc:
            last_error = str(exc)
            logger.warning(
                "OpenAI chunk repair '%s' attempt %d failed: %s",
                chunk_id, attempt, exc,
            )

    return None, f"OpenAI repair of chunk '{chunk_id}' failed after 2 attempts: {last_error}"


# ─── Main repair function ─────────────────────────────────────────────────────

def run_openai_targeted_chunk_repair(
    script_draft: dict,
    repair_targets: list[dict],
    fact_lock: dict,
    blueprint: dict,
    hinglish_level: int,
    script_dir: Path,
    review_dir: Path,
) -> tuple[dict, dict]:
    """
    Repair specific chunks using OpenAI. Called when OpenAI gates fail with
    chunk_repair_targets. Max targets enforced by caller.

    Deduplicates targets by chunk_id, combining repair instructions for same chunk.
    Loads chunks directly from script_draft (in-memory, most current state).

    Returns (updated_script_draft, repair_report).

    repair_report schema:
    {
      "status": "openai_repair_complete" | "needs_human_review",
      "chunks_attempted": int,
      "chunks_repaired": int,
      "chunks_failed": int,
      "has_failures": bool,
      "total_words_after_repair": int,
      "repair_results": [...],
      "warnings": [...]
    }
    """
    chunks_dir = script_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Build current chunk lookup from the passed script_draft (latest state)
    chunks_by_id: dict[str, dict] = {
        c.get("chunk_id", ""): c
        for c in script_draft.get("hindi_narration_chunks", [])
        if c.get("chunk_id")
    }

    # Deduplicate by chunk_id — combine repair_instruction for same chunk
    combined: dict[str, dict] = {}
    for t in repair_targets:
        cid = t.get("chunk_id", "")
        if not cid:
            continue
        if cid in combined:
            # Merge repair instructions
            combined[cid]["repair_instruction"] = (
                combined[cid].get("repair_instruction", "")
                + " | "
                + t.get("repair_instruction", "")
            )
        else:
            combined[cid] = dict(t)

    deduped_targets = list(combined.values())

    logger.info(
        "OpenAI Targeted Repair START — %d unique chunk(s) to repair (from %d target(s))",
        len(deduped_targets), len(repair_targets),
    )

    # Save targets for audit
    (review_dir / "openai_repair_targets.json").write_text(
        json.dumps(deduped_targets, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    repair_results: list[dict] = []
    repaired_chunks: dict[str, dict] = {}

    for target in deduped_targets:
        chunk_id = target.get("chunk_id", "")
        if not chunk_id:
            continue

        current_chunk = chunks_by_id.get(chunk_id)
        if current_chunk is None:
            logger.warning(
                "OpenAI repair target '%s' not found in script — skipping", chunk_id
            )
            repair_results.append({
                "chunk_id":   chunk_id,
                "status":     "failed_kept_original",
                "issue_type": target.get("issue_type", ""),
                "error":      f"chunk '{chunk_id}' not found in script_draft",
            })
            continue

        words_before = current_chunk.get("estimated_words") or len(
            current_chunk.get("text", "").split()
        )

        repaired, error = _repair_one_chunk(
            target=target,
            current_chunk=current_chunk,
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
                "issue_type":  target.get("issue_type", ""),
                "words_before": words_before,
                "words_after":  repaired.get("estimated_words", 0),
            })
        else:
            logger.warning(
                "OpenAI repair failed for '%s' — keeping original. %s", chunk_id, error
            )
            repair_results.append({
                "chunk_id":   chunk_id,
                "status":     "failed_kept_original",
                "issue_type": target.get("issue_type", ""),
                "error":      error,
            })

    # Merge repaired chunks back into the script
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

    # Re-save all assembly files so downstream stages see updated chunks
    def _save(path: Path, data: object) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    _save(script_dir / "script_final.json", script_final)
    _save(script_dir / "hindi_narration_chunks.json", merged_chunks)
    _save(script_dir / "elevenlabs_chunks.json", _extract_elevenlabs_chunks(merged_chunks))
    (script_dir / "hindi_narration_full.txt").write_text(
        _extract_full_narration(merged_chunks), encoding="utf-8"
    )

    repair_report = {
        "status":                  "needs_human_review" if has_failures else "openai_repair_complete",
        "chunks_attempted":        len(deduped_targets),
        "chunks_repaired":         repaired_count,
        "chunks_failed":           len(failed),
        "has_failures":            has_failures,
        "total_words_after_repair": total_words,
        "repair_results":          repair_results,
        "warnings": (
            [
                f"{len(failed)} chunk(s) could not be repaired by OpenAI — "
                "original content kept. Automated retry exhausted — safe_to_voice=false."
            ]
            if has_failures else []
        ),
    }
    _save(review_dir / "openai_repair_report.json", repair_report)

    logger.info(
        "OpenAI Targeted Repair COMPLETE — %d repaired, %d failed, %d total words",
        repaired_count, len(failed), total_words,
    )
    return script_final, repair_report
