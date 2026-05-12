"""
Premium Section Rebuild Service.

When the OpenAI Final Premium Gate finds too many repair targets for targeted repair,
this service rebuilds only the problematic grouped sections via a single Claude call
(one call per root-cause group, up to MAX_CLAUDE_REPAIR_TARGETS_PER_ROUND groups).

After rebuild:
  1. Runs deterministic_auto_fix_service to catch any residual mechanical issues.
  2. Reassembles hindi_narration_full.txt and updates script_final.json.
  3. The pipeline caller is responsible for re-running Claude gates and OAI recheck.

Produces:
  04-review/premium_section_rebuild_report.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.schemas import NarrationChunk
from app.services.claude_client import call_claude_agent, parse_package_response
from app.services.prompt_utils import get_channel_rules
from app.services.script_assembler_service import (
    _extract_full_narration,
    _extract_elevenlabs_chunks,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/premium_section_rebuild_agent.txt")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_target_chunk_ids(routing_plan: dict) -> list[str]:
    """Collect all unique chunk IDs from Claude repair targets."""
    ids: list[str] = []
    for target in routing_plan.get("claude_repair_targets", []):
        for cid in target.get("chunk_ids", []):
            if cid and cid not in ids:
                ids.append(cid)
    return ids


def _filter_chunks(chunks: list[dict], target_ids: list[str]) -> list[dict]:
    """Return only the chunks that need rebuilding."""
    if not target_ids:
        return chunks
    return [c for c in chunks if c.get("chunk_id", "") in target_ids]


def _merge_rebuilt_chunks(
    original_chunks: list[dict],
    rebuilt: list[dict],
) -> list[dict]:
    """Replace original chunks with their rebuilt versions; keep untouched chunks."""
    rebuilt_map = {c["chunk_id"]: c for c in rebuilt if c.get("chunk_id")}
    return [rebuilt_map.get(c.get("chunk_id", ""), c) for c in original_chunks]


def _assemble_full_narration(chunks: list[dict]) -> str:
    """Join all chunk texts into a single narration string.

    Delegates to the shared assembler helper so the format stays consistent
    across all pipeline stages (includes section titles).
    """
    return _extract_full_narration(chunks)


# ─── Public entry point ───────────────────────────────────────────────────────

def run_premium_section_rebuild(
    script_draft: dict,
    routing_plan: dict,
    fact_lock: dict,
    blueprint: dict,
    retention_blueprint: dict,
    originality_transformation_plan: dict,
    case_glossary: dict,
    hinglish_level: int,
    review_dir: Path,
    script_dir: Path,
) -> tuple[dict, dict]:
    """
    Rebuild only the sections/chunks identified in routing_plan as needing Claude repair.

    Parameters
    ----------
    script_draft                 : current script dict
    routing_plan                 : output of repair_routing_service
    fact_lock                    : verified facts (source of truth)
    blueprint                    : story blueprint
    retention_blueprint          : retention blueprint
    originality_transformation_plan : originality plan
    case_glossary                : case glossary
    hinglish_level               : 1–5
    review_dir                   : 04-review/ directory
    script_dir                   : 03-script/ directory

    Returns
    -------
    (updated_script_draft, rebuild_report)
    """
    from app.config import settings

    target_ids = _get_target_chunk_ids(routing_plan)
    root_causes = routing_plan.get("claude_repair_targets", [])

    if not root_causes:
        logger.info("Premium section rebuild: no claude repair targets in routing plan — skipping")
        empty_report = {
            "rebuilt_count": 0,
            "skipped": True,
            "reason": "No claude_repair_targets in routing plan",
        }
        return script_draft, empty_report

    # Safety cap: never rebuild more chunks than the limit
    max_targets = settings.max_auto_rebuild_targets
    if len(target_ids) > max_targets:
        logger.warning(
            "Premium rebuild: %d targets exceed MAX_AUTO_REBUILD_TARGETS=%d — capping",
            len(target_ids), max_targets,
        )
        target_ids = target_ids[:max_targets]

    # Respect MAX_CLAUDE_REPAIR_TARGETS_PER_ROUND
    max_round_targets = settings.max_claude_repair_targets_per_round
    if len(root_causes) > max_round_targets:
        logger.warning(
            "Premium rebuild: %d root causes exceed MAX_CLAUDE_REPAIR_TARGETS_PER_ROUND=%d",
            len(root_causes), max_round_targets,
        )
        root_causes = root_causes[:max_round_targets]

    # Build and send one grouped Claude call
    all_chunks = script_draft.get("hindi_narration_chunks", [])
    target_chunks = _filter_chunks(all_chunks, target_ids)

    if not target_chunks:
        logger.warning("Premium rebuild: target chunk IDs not found in script — skipping")
        empty_report = {
            "rebuilt_count": 0,
            "skipped": True,
            "reason": f"Target IDs {target_ids} not found in script chunks",
        }
        return script_draft, empty_report

    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.replace("{channel_rules}", get_channel_rules())
    prompt = prompt.replace("{root_causes_json}", json.dumps(root_causes, ensure_ascii=False, indent=2))
    prompt = prompt.replace("{target_chunk_ids}", ", ".join(target_ids))
    prompt = prompt.replace("{fact_lock_json}", json.dumps(fact_lock, ensure_ascii=False))
    prompt = prompt.replace("{story_blueprint_json}", json.dumps(blueprint, ensure_ascii=False))
    prompt = prompt.replace(
        "{retention_blueprint_json}",
        json.dumps(retention_blueprint, ensure_ascii=False),
    )
    prompt = prompt.replace(
        "{originality_transformation_json}",
        json.dumps(originality_transformation_plan, ensure_ascii=False),
    )
    prompt = prompt.replace("{case_glossary_json}", json.dumps(case_glossary, ensure_ascii=False))
    prompt = prompt.replace("{current_chunks_json}", json.dumps(target_chunks, ensure_ascii=False, indent=2))
    prompt = prompt.replace("{hinglish_level}", str(hinglish_level))

    logger.info(
        "Premium section rebuild: %d chunks, %d root causes, prompt=%d chars",
        len(target_chunks), len(root_causes), len(prompt),
    )

    raw_response, stop_reason = call_claude_agent(prompt, agent_name="premium_section_rebuild")

    raw_path = review_dir / "_premium_section_rebuild_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")

    if stop_reason == "max_tokens":
        logger.warning("premium_section_rebuild hit max_tokens — output may be truncated")

    try:
        rebuild_result = parse_package_response(raw_response, agent_name="premium_section_rebuild")
    except ValueError as exc:
        raise ValueError(
            f"Premium Section Rebuild JSON parse failed: {exc}\n"
            f"Raw response saved at: {raw_path}"
        ) from exc

    rebuilt_chunks_raw: list[dict] = rebuild_result.get("rebuilt_chunks", [])

    if not rebuilt_chunks_raw:
        logger.warning("Premium section rebuild: Claude returned no rebuilt chunks")
        report = {
            "rebuilt_count": 0,
            "target_chunk_ids": target_ids,
            "root_cause_count": len(root_causes),
            "child_victim_safety_applied": rebuild_result.get("child_victim_safety_applied", False),
            "rebuild_notes": rebuild_result.get("rebuild_notes", ""),
            "warning": "Claude returned no rebuilt chunks",
        }
        return script_draft, report

    # ── Validate each rebuilt chunk before merging ────────────────────────────
    # A chunk that fails NarrationChunk schema validation is skipped rather than
    # merged — rebuilt_count counts only valid chunks so callers can trust the
    # number.  Failures are recorded in the report for auditability.
    rebuilt_chunks: list[dict] = []
    invalid_rebuilt: list[dict] = []
    for _rc in rebuilt_chunks_raw:
        try:
            NarrationChunk.model_validate(_rc)
            rebuilt_chunks.append(_rc)
        except ValidationError as _ve:
            _cid = _rc.get("chunk_id", "?")
            logger.warning(
                "Premium rebuild: chunk %s failed NarrationChunk validation — skipping merge: %s",
                _cid, str(_ve)[:200],
            )
            invalid_rebuilt.append({
                "chunk_id":            _cid,
                "validation_error":    str(_ve)[:300],
                "raw_keys":            list(_rc.keys()),
            })

    if invalid_rebuilt:
        logger.warning(
            "Premium rebuild: %d/%d rebuilt chunk(s) failed validation and were skipped",
            len(invalid_rebuilt), len(rebuilt_chunks_raw),
        )

    if not rebuilt_chunks:
        logger.warning(
            "Premium rebuild: all %d rebuilt chunk(s) failed validation — no merge performed",
            len(rebuilt_chunks_raw),
        )
        report = {
            "rebuilt_count":   0,
            "target_chunk_ids": target_ids,
            "root_cause_count": len(root_causes),
            "invalid_rebuilt_chunks": invalid_rebuilt,
            "child_victim_safety_applied": rebuild_result.get("child_victim_safety_applied", False),
            "rebuild_notes": rebuild_result.get("rebuild_notes", ""),
            "warning": "All rebuilt chunks failed NarrationChunk schema validation",
        }
        return script_draft, report

    # Merge rebuilt chunks back into the full chunk list
    import copy
    updated_draft = copy.deepcopy(script_draft)
    updated_chunks = _merge_rebuilt_chunks(all_chunks, rebuilt_chunks)
    updated_draft["hindi_narration_chunks"] = updated_chunks

    # Reassemble full narration
    full_narration = _assemble_full_narration(updated_chunks)
    updated_draft["hindi_narration_full"] = full_narration

    # ── Persist updated outputs to 03-script/ ────────────────────────────────
    # script_final.json, narration chunks, full text, and ElevenLabs chunks
    # must all be in sync after rebuild so downstream stages read consistent data.
    elevenlabs_chunks = _extract_elevenlabs_chunks(updated_chunks)
    try:
        (script_dir / "script_final.json").write_text(
            json.dumps(updated_draft, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (script_dir / "hindi_narration_chunks.json").write_text(
            json.dumps(updated_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (script_dir / "hindi_narration_full.txt").write_text(full_narration, encoding="utf-8")
        (script_dir / "elevenlabs_chunks.json").write_text(
            json.dumps(elevenlabs_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as save_exc:
        logger.warning("Could not persist rebuilt 03-script/ files: %s", save_exc)

    # ── Mirror to 02-package/ so downstream reads are consistent ─────────────
    # 02-package/ is the backward-compat location read by the renderer and
    # other tooling.  Rebuild must write here too — same data, same format.
    pkg_dir = script_dir.parent / "02-package"
    if pkg_dir.exists():
        try:
            (pkg_dir / "hindi_narration_chunks.json").write_text(
                json.dumps(updated_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (pkg_dir / "hindi_narration_full.txt").write_text(full_narration, encoding="utf-8")
            (pkg_dir / "elevenlabs_chunks.json").write_text(
                json.dumps(elevenlabs_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (pkg_dir / "production_package.json").write_text(
                json.dumps(updated_draft, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info("Premium rebuild — mirrored outputs to 02-package/")
        except Exception as pkg_exc:
            logger.warning("Could not mirror rebuilt files to 02-package/: %s", pkg_exc)

    report: dict[str, Any] = {
        "rebuilt_count":              len(rebuilt_chunks),   # only valid chunks
        "target_chunk_ids":           target_ids,
        "rebuilt_chunk_ids":          [c.get("chunk_id", "") for c in rebuilt_chunks],
        "root_cause_count":           len(root_causes),
        "root_causes_addressed":      rebuild_result.get("root_causes_addressed", []),
        "child_victim_safety_applied": rebuild_result.get("child_victim_safety_applied", False),
        "rebuild_notes":              rebuild_result.get("rebuild_notes", ""),
        "stop_reason":                stop_reason,
        "invalid_rebuilt_chunks":     invalid_rebuilt,
        "invalid_rebuilt_count":      len(invalid_rebuilt),
    }

    out_path = review_dir / "premium_section_rebuild_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Premium section rebuild complete — rebuilt %d/%d chunks, root_causes=%d",
        len(rebuilt_chunks), len(target_chunks), len(root_causes),
    )

    return updated_draft, report
