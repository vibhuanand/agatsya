"""
Script Writer Service — orchestrates the 5-stage script generation pipeline.

Replaces the single monolithic Claude call with focused sub-agents:
  1. Script Outline Agent    — plans chunk structure, word targets, fact assignments
  2. Narration Chunk Writer  — writes one chunk at a time (12–16 calls)
  3. Recreated Dialogue Agent — writes short labelled dialogue scenes
  4. Metadata Agent          — writes minimal YouTube metadata
  5. Script Assembler        — pure Python combines all pieces into script_draft.json

Each sub-agent produces a small, focused JSON output — no more 40KB truncations.
Individual chunk failures trigger retry before the pipeline stops.

Produces (draft files in 03-script/):
  script_outline.json
  chunks/{chunk_id}.json  (one per chunk)
  hindi_narration_full_draft.txt
  hindi_narration_chunks_draft.json
  recreated_dialogues_draft.json
  elevenlabs_chunks_draft.json
  youtube_metadata_draft.json
  script_draft.json               ← final assembled output
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from app.config import settings
from app.services.script_outline_service import run_script_outline
from app.services.narration_chunk_writer_service import (
    run_narration_chunk,
    _get_last_sentence,
)
from app.services.recreated_dialogue_service import run_recreated_dialogue
from app.services.metadata_service import run_metadata
from app.services.script_assembler_service import (
    assemble_script_package,
    _extract_full_narration,
    _extract_elevenlabs_chunks,
)

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:40]


def _compute_word_targets(target_duration_min: int) -> tuple[int, int, int]:
    """Return (min, ideal, max) word count targets based on hindi_narration_wpm."""
    wpm = settings.hindi_narration_wpm
    t_min   = int(target_duration_min * wpm)
    t_ideal = int(target_duration_min * round(wpm * 1.15))
    t_max   = int(target_duration_min * round(wpm * 1.30))
    return t_min, t_ideal, t_max


def run_script_writer(
    case_hint: str,
    episode_number: str,
    episode_id: str,
    target_duration_min: int,
    cost_mode: str,
    style: str,
    fact_lock: dict,
    blueprint: dict,
    script_dir: Path,
    hinglish_level: int = 2,
    retention_blueprint: dict | None = None,
    case_glossary: dict | None = None,
) -> dict:
    """
    Orchestrate the 5-stage script generation pipeline.

    Public interface is identical to the old monolithic version —
    agent_pipeline_service.py calls this and receives the same script_draft dict.

    Returns the assembled script_draft dict.
    """
    slug = _slugify(case_hint)
    t_min, t_ideal, t_max = _compute_word_targets(target_duration_min)

    logger.info(
        "Script Writer Pipeline START — episode: %s  wpm: %d  targets: min=%d ideal=%d max=%d  hinglish_level=%d",
        episode_id, settings.hindi_narration_wpm, t_min, t_ideal, t_max, hinglish_level,
    )

    chunks_dir = script_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage A: Script Outline ───────────────────────────────────────────────
    logger.info("Script Stage A — Script Outline Agent")
    outline = run_script_outline(
        case_hint=case_hint,
        episode_id=episode_id,
        target_duration_min=target_duration_min,
        target_word_count_min=t_min,
        target_word_count_ideal=t_ideal,
        target_word_count_max=t_max,
        cost_mode=cost_mode,
        style=style,
        fact_lock=fact_lock,
        blueprint=blueprint,
        script_dir=script_dir,
        hinglish_level=hinglish_level,
        retention_blueprint=retention_blueprint,
    )

    chunk_specs = outline.get("chunks", [])
    logger.info("Outline produced %d chunk specs", len(chunk_specs))

    # ── Stage B: Narration Chunk Writer (one call per chunk) ──────────────────
    logger.info("Script Stage B — Narration Chunk Writer (%d chunks)", len(chunk_specs))
    written_chunks: list[dict] = []
    prev_last_sentence = ""

    for i, chunk_spec in enumerate(chunk_specs):
        chunk_id = chunk_spec.get("chunk_id", f"chunk_{i:03d}")

        # Idempotency: reuse existing chunk file if present
        chunk_file = chunks_dir / f"{chunk_id}.json"
        if chunk_file.exists():
            try:
                existing = json.loads(chunk_file.read_text(encoding="utf-8"))
                logger.info("Chunk '%s' — reusing existing file", chunk_id)
                written_chunks.append(existing)
                prev_last_sentence = _get_last_sentence(existing.get("text", ""))
                continue
            except Exception:
                logger.warning("Could not load existing chunk '%s' — regenerating", chunk_id)

        # Next chunk purpose (for transition guidance)
        next_purpose = ""
        if i + 1 < len(chunk_specs):
            next_purpose = chunk_specs[i + 1].get("purpose", "")

        chunk_data = run_narration_chunk(
            chunk_spec=chunk_spec,
            fact_lock=fact_lock,
            blueprint=blueprint,
            script_dir=script_dir,
            case_glossary=case_glossary or {},
            prev_last_sentence=prev_last_sentence,
            next_chunk_purpose=next_purpose,
            hinglish_level=hinglish_level,
        )

        written_chunks.append(chunk_data)
        prev_last_sentence = _get_last_sentence(chunk_data.get("text", ""))
        logger.info(
            "  Chunk %d/%d '%s' — %d words",
            i + 1, len(chunk_specs), chunk_id, chunk_data.get("estimated_words", 0),
        )

    total_words = sum(c.get("estimated_words", 0) for c in written_chunks)
    logger.info("Stage B complete — %d chunks, ~%d words total", len(written_chunks), total_words)

    # ── Stage C: Recreated Dialogue Agent ────────────────────────────────────
    logger.info("Script Stage C — Recreated Dialogue Agent")
    dialogues = run_recreated_dialogue(
        outline=outline,
        fact_lock=fact_lock,
        blueprint=blueprint,
        script_dir=script_dir,
    )

    # ── Stage D: Metadata Agent ───────────────────────────────────────────────
    logger.info("Script Stage D — Metadata Agent")
    metadata = run_metadata(
        case_hint=case_hint,
        episode_id=episode_id,
        cost_mode=cost_mode,
        fact_lock=fact_lock,
        blueprint=blueprint,
        chunks=written_chunks,
        script_dir=script_dir,
        case_glossary=case_glossary or {},
        hinglish_level=hinglish_level,
    )

    # ── Stage E: Python Assembler ─────────────────────────────────────────────
    logger.info("Script Stage E — Script Assembler (Python)")
    script_draft = assemble_script_package(
        episode_id=episode_id,
        slug=slug,
        fact_lock=fact_lock,
        chunks=written_chunks,
        dialogues=dialogues,
        metadata=metadata,
        script_dir=script_dir,
    )

    logger.info(
        "Script Writer Pipeline COMPLETE — %d chunks, ~%d words, status: assembled",
        len(written_chunks), total_words,
    )

    return script_draft
