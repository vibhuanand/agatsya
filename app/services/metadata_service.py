"""
Metadata Agent — Claude agent that generates minimal YouTube metadata
(titles and description only) for the episode.

Produces:
  03-script/youtube_metadata_draft.json
  03-script/_metadata_raw_response.txt
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from app.schemas import YoutubeMetadataMinimal
from app.services.claude_client import call_claude_agent, parse_package_response
from app.services.prompt_utils import get_channel_rules

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/metadata_agent.txt")


def _build_chunk_summaries(chunks: list[dict]) -> str:
    """Build a brief summary of each narration chunk for metadata context."""
    lines = []
    for c in chunks:
        chunk_id = c.get("chunk_id", "")
        title = c.get("section_title", "")
        words = c.get("estimated_words", 0)
        text_preview = c.get("text", "")[:120].replace("\n", " ")
        lines.append(f"  [{chunk_id}] {title} (~{words} words) — {text_preview}…")
    return "\n".join(lines) if lines else "  (no chunks)"


def _build_prompt(
    case_hint: str,
    episode_id: str,
    cost_mode: str,
    fact_lock: dict,
    blueprint: dict,
    chunks: list[dict],
    case_glossary: dict,
    hinglish_level: int = 2,
) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")

    replacements = {
        "{channel_rules}": get_channel_rules(),
        "{episode_id}": episode_id,
        "{case_hint}": case_hint,
        "{cost_mode}": cost_mode,
        "{hinglish_level}": str(hinglish_level),
        "{fact_lock_json}": json.dumps(fact_lock, ensure_ascii=False),
        "{case_glossary_json}": json.dumps(case_glossary, ensure_ascii=False),
        "{title_angle}": blueprint.get("title_angle", ""),
        "{main_hook}": blueprint.get("main_hook", ""),
        "{closing_style}": blueprint.get("closing_style", ""),
        "{chunk_summaries_text}": _build_chunk_summaries(chunks),
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def run_metadata(
    case_hint: str,
    episode_id: str,
    cost_mode: str,
    fact_lock: dict,
    blueprint: dict,
    chunks: list[dict],
    script_dir: Path,
    case_glossary: dict | None = None,
    hinglish_level: int = 2,
) -> dict:
    """
    Generate minimal YouTube metadata (titles + description only).

    Returns the metadata dict.
    """
    prompt = _build_prompt(
        case_hint=case_hint,
        episode_id=episode_id,
        cost_mode=cost_mode,
        fact_lock=fact_lock,
        blueprint=blueprint,
        chunks=chunks,
        case_glossary=case_glossary or {},
        hinglish_level=hinglish_level,
    )

    raw_response, stop_reason = call_claude_agent(prompt, agent_name="metadata")

    raw_path = script_dir / "_metadata_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    logger.info("Metadata raw response saved → %s", raw_path)

    if stop_reason == "max_tokens":
        logger.warning("metadata agent hit max_tokens")

    try:
        metadata = parse_package_response(raw_response, agent_name="metadata")
        YoutubeMetadataMinimal.model_validate(metadata)
    except (ValueError, ValidationError) as exc:
        logger.warning("Metadata parse/validation failed: %s — using fallback", exc)
        metadata = {
            "title_options": [case_hint],
            "recommended_title": case_hint,
            "description": f"सच्ची घटना पर आधारित: {case_hint}",
        }

    out_path = script_dir / "youtube_metadata_draft.json"
    out_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Metadata saved → %s", out_path)

    return metadata
