"""
Narration Chunk Writer Agent — Claude agent that writes ONE Hindi narration chunk.

Called once per chunk in the outline. Small, focused, retryable.
If a chunk fails JSON parsing or schema validation, it retries once
(controlled by SCRIPT_CHUNK_RETRY_LIMIT).

Produces per chunk:
  03-script/chunks/{chunk_id}.json
  03-script/chunks/_raw_{chunk_id}.txt

On validation error (after all retries):
  03-script/chunks/_validation_error_{chunk_id}.txt
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from app.config import settings
from app.schemas import NarrationChunk
from app.services.claude_client import call_claude_agent, parse_package_response
from app.services.prompt_utils import get_channel_rules

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/narration_chunk_writer_agent.txt")


def _build_chunk_prompt(
    chunk_spec: dict,
    fact_lock: dict,
    blueprint: dict,
    prev_last_sentence: str,
    next_chunk_purpose: str,
    hinglish_level: int = 2,
) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")

    chunk_id = chunk_spec.get("chunk_id", "")
    section_title = chunk_spec.get("section_title", "")
    purpose = chunk_spec.get("purpose", "")
    tone = chunk_spec.get("tone", "cinematic_sober")
    target_words = chunk_spec.get("target_words", 150)
    min_words = int(target_words * 0.80)
    max_words = int(target_words * 1.20)
    must_include = chunk_spec.get("must_include_points", [])
    safety_notes = chunk_spec.get("safety_notes", [])

    # Format must_include_points as numbered list
    if must_include:
        must_include_text = "\n".join(f"  {i+1}. {pt}" for i, pt in enumerate(must_include))
    else:
        must_include_text = "  (Use the relevant verified facts from fact_lock for this section)"

    if safety_notes:
        safety_notes_text = "\n".join(f"  ⚠ {note}" for note in safety_notes)
    else:
        safety_notes_text = "  (No special safety notes for this chunk)"

    # Build continuity context
    if prev_last_sentence:
        continuity_text = (
            f"Previous chunk ended with: \"{prev_last_sentence}\"\n"
            "Begin this chunk with a natural transition from that point."
        )
    else:
        continuity_text = "This is the first chunk. No previous context."

    if next_chunk_purpose:
        continuity_text += f"\nNext chunk will cover: {next_chunk_purpose}"

    # Blueprint context (key fields only — not the full blueprint)
    main_hook = blueprint.get("main_hook", "")
    emotional_anchor = blueprint.get("emotional_anchor", "")
    closing_style = blueprint.get("closing_style", "")
    sensitivity_rules = blueprint.get("sensitivity_rules", [])
    sensitivity_rules_text = "; ".join(sensitivity_rules[:5]) if sensitivity_rules else "none"

    # Retention fields from outline chunk spec (present in premium mode only)
    retention_goal = chunk_spec.get("retention_goal", "")
    curiosity_gap = chunk_spec.get("curiosity_gap", "")
    viewer_payoff = chunk_spec.get("viewer_payoff", "")
    pattern_interrupt = chunk_spec.get("pattern_interrupt", "")

    # Compose retention guidance block (empty string if no retention data)
    if any([retention_goal, curiosity_gap, viewer_payoff, pattern_interrupt]):
        retention_lines = []
        if retention_goal:
            retention_lines.append(f"Retention goal:    {retention_goal}")
        if curiosity_gap:
            retention_lines.append(f"Curiosity gap:     {curiosity_gap}")
        if viewer_payoff:
            retention_lines.append(f"Viewer payoff:     {viewer_payoff}")
        if pattern_interrupt:
            retention_lines.append(f"Pattern interrupt: {pattern_interrupt}")
        retention_guidance = "\n".join(retention_lines)
    else:
        retention_guidance = ""

    replacements = {
        "{channel_rules}": get_channel_rules(),
        "{chunk_id}": chunk_id,
        "{section_title}": section_title,
        "{purpose}": purpose,
        "{tone}": tone,
        "{target_words}": str(target_words),
        "{min_words}": str(min_words),
        "{max_words}": str(max_words),
        "{must_include_text}": must_include_text,
        "{safety_notes_text}": safety_notes_text,
        "{continuity_text}": continuity_text,
        "{hinglish_level}": str(hinglish_level),
        "{fact_lock_json}": json.dumps(fact_lock, ensure_ascii=False),
        "{main_hook}": main_hook,
        "{emotional_anchor}": emotional_anchor,
        "{closing_style}": closing_style,
        "{sensitivity_rules_text}": sensitivity_rules_text,
        "{retention_guidance}": retention_guidance,
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def _get_last_sentence(text: str, max_chars: int = 200) -> str:
    """Extract the last sentence from a narration chunk for continuity context."""
    if not text:
        return ""
    tail = text[-max_chars:] if len(text) > max_chars else text
    for sep in ["।", "॥", ".", "!", "?"]:
        pos = tail.rfind(sep)
        if pos != -1:
            candidate = tail[pos + 1:].strip()
            if candidate:
                return candidate
    return tail.strip()


def run_narration_chunk(
    chunk_spec: dict,
    fact_lock: dict,
    blueprint: dict,
    script_dir: Path,
    prev_last_sentence: str = "",
    next_chunk_purpose: str = "",
    hinglish_level: int = 2,
) -> dict:
    """
    Write one narration chunk with retry logic.

    Retries up to settings.script_chunk_retry_limit times on parse or validation failure.
    If all attempts fail, saves validation error and raises ValueError.

    Returns the parsed chunk dict.
    """
    chunk_id = chunk_spec.get("chunk_id", "unknown")
    chunks_dir = script_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    max_attempts = settings.script_chunk_retry_limit + 1  # e.g. retry_limit=1 → 2 total attempts
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        agent_label = f"chunk_{chunk_id}_a{attempt}"
        try:
            prompt = _build_chunk_prompt(
                chunk_spec=chunk_spec,
                fact_lock=fact_lock,
                blueprint=blueprint,
                prev_last_sentence=prev_last_sentence,
                next_chunk_purpose=next_chunk_purpose,
                hinglish_level=hinglish_level,
            )

            raw_response, stop_reason = call_claude_agent(prompt, agent_name=agent_label)

            raw_path = chunks_dir / f"_raw_{chunk_id}.txt"
            raw_path.write_text(raw_response, encoding="utf-8")

            if stop_reason == "max_tokens":
                logger.warning("Chunk '%s' attempt %d hit max_tokens", chunk_id, attempt)

            data = parse_package_response(raw_response, agent_name=agent_label)
            NarrationChunk.model_validate(data)

            # Success — save chunk
            chunk_path = chunks_dir / f"{chunk_id}.json"
            chunk_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(
                "Chunk '%s' written — %d words (attempt %d/%d)",
                chunk_id, data.get("estimated_words", 0), attempt, max_attempts,
            )
            return data

        except (ValueError, ValidationError) as exc:
            last_exc = exc
            logger.warning(
                "Chunk '%s' attempt %d/%d failed: %s",
                chunk_id, attempt, max_attempts, exc,
            )

    # All attempts exhausted
    err_path = chunks_dir / f"_validation_error_{chunk_id}.txt"
    err_path.write_text(str(last_exc), encoding="utf-8")
    raise ValueError(
        f"Narration chunk '{chunk_id}' failed after {max_attempts} attempt(s). "
        f"Validation error saved at: {err_path}. "
        "Check the raw response file in 03-script/chunks/ for the Claude output."
    ) from last_exc
