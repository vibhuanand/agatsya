"""
Script Assembler — Pure Python stage that combines individual chunk JSON files
and metadata into the final script_draft.json.

No Claude call in this stage.

Reads:
  03-script/script_outline.json     — chunk order
  03-script/chunks/{chunk_id}.json  — one file per chunk
  03-script/recreated_dialogues_draft.json
  03-script/youtube_metadata_draft.json

Produces:
  03-script/script_draft.json
  03-script/hindi_narration_full_draft.txt
  03-script/hindi_narration_chunks_draft.json
  03-script/elevenlabs_chunks_draft.json
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─── Shared helpers (re-exported for backward compat with repair/pipeline) ─────

def _extract_full_narration(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"## {c.get('section_title', c.get('chunk_id', ''))}\n{c.get('text', '')}"
        for c in chunks
    )


def _extract_elevenlabs_chunks(chunks: list[dict]) -> list[dict]:
    result = []
    for c in chunks:
        result.append({
            "chunk_id": c.get("chunk_id"),
            "voice_id": "{ELEVENLABS_NARRATOR_VOICE_ID}",
            "model_id": "{ELEVENLABS_MODEL_ID}",
            "text": c.get("text", ""),
            "voice_settings": {
                "stability": 0.55,
                "similarity_boost": 0.75,
                "style": 0.4,
                "use_speaker_boost": True,
            },
        })
    return result


# ─── Case summary builder ──────────────────────────────────────────────────────

def _extract_year(fact_lock: dict) -> str:
    year_pattern = re.compile(r"\b(19|20)\d{2}\b")
    for src in [fact_lock.get("verified_timeline", []), fact_lock.get("verified_dates", [])]:
        for item in src:
            m = year_pattern.search(item.get("date_or_period", ""))
            if m:
                return m.group()
    return ""


def _build_case_summary(fact_lock: dict) -> dict:
    legal = fact_lock.get("legal_outcome", {})
    legal_summary = " | ".join(filter(None, [
        legal.get("trial_result", ""),
        legal.get("appeal_result", ""),
        legal.get("sentence_or_parole", ""),
    ]))

    return {
        "case_title": fact_lock.get("case_name", ""),
        "location": ", ".join(
            loc.get("location", "")
            for loc in fact_lock.get("verified_locations", [])[:2]
            if loc.get("location")
        ),
        "year": _extract_year(fact_lock),
        "people": [
            {"name": p.get("name", ""), "role": p.get("role", "")}
            for p in fact_lock.get("verified_people", [])
        ],
        "timeline": [
            {"date": e.get("date_or_period", ""), "event": e.get("event", "")}
            for e in fact_lock.get("verified_timeline", [])
        ],
        "core_story": fact_lock.get("source_summary", ""),
        "legal_outcome": legal_summary,
        "sensitive_topics": [],
        "facts_to_verify": [
            item.get("fact", str(item)) if isinstance(item, dict) else str(item)
            for item in fact_lock.get("facts_to_verify_externally", [])
        ],
        "avoid_in_final_video": fact_lock.get("must_not_say", []),
    }


# ─── Main assembler ────────────────────────────────────────────────────────────

def assemble_script_package(
    episode_id: str,
    slug: str,
    fact_lock: dict,
    chunks: list[dict],
    dialogues: dict,
    metadata: dict,
    script_dir: Path,
) -> dict:
    """
    Assemble the final script_draft from individual pieces.
    Pure Python — no Claude call.

    Saves all draft files and returns the complete script_draft dict.
    """
    case_summary = _build_case_summary(fact_lock)

    script_draft = {
        "episode_id": episode_id,
        "folder_name": slug,
        "case_summary": case_summary,
        "hindi_narration_chunks": chunks,
        "recreated_dialogues": dialogues,
        "youtube_metadata": metadata,
        "quality_checklist": [],
    }

    def _save_json(name: str, content: Any) -> None:
        (script_dir / name).write_text(
            json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    _save_json("script_draft.json", script_draft)
    _save_json("hindi_narration_chunks_draft.json", chunks)
    _save_json("recreated_dialogues_draft.json", dialogues)
    _save_json("youtube_metadata_draft.json", metadata)
    _save_json("elevenlabs_chunks_draft.json", _extract_elevenlabs_chunks(chunks))

    (script_dir / "hindi_narration_full_draft.txt").write_text(
        _extract_full_narration(chunks), encoding="utf-8"
    )

    total_words = sum(c.get("estimated_words", 0) for c in chunks)
    logger.info(
        "Script assembled — %d chunks, ~%d words → %s",
        len(chunks), total_words, script_dir / "script_draft.json",
    )

    return script_draft
