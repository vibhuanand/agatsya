"""
Fact Lock Agent — Claude agent that extracts verified facts from a transcript.

Supports two modes (controlled by FACT_LOCK_MODE in .env):

  research_view (default, cheap):
    Sends the pre-built beginning/middle/end research view to one Claude call.
    Good for most transcripts. Budget-conscious.

  segmented (thorough):
    Splits the clean transcript into ~7000-char segments.
    Runs a compact fact extraction per segment (one Claude call each).
    Merges segment results into a unified fact_lock.
    Better for very long or complex transcripts where details may fall in the middle.
    Saves individual segment files for debugging.

Produces:
  02-facts/fact_lock.json
  02-facts/_fact_lock_raw_response.txt      (research_view mode)
  02-facts/segments/fact_segment_NNN.json   (segmented mode only)
  02-facts/segments/_segment_NNN_raw.txt    (segmented mode only)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import settings
from app.services.claude_client import call_claude_agent, parse_package_response

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/fact_lock_agent.txt")

# ─── Compact segment extraction prompt (embedded — no separate file needed) ───

_SEGMENT_PROMPT_TEMPLATE = """You are a fact extraction assistant. Extract facts ONLY from this transcript segment.
This is segment {segment_num} of {total_segments} from the case: {case_hint}

Return ONLY a valid JSON object. No markdown. No explanation. No code fences.

SEGMENT TEXT:
{segment_text}

Extract whatever facts appear in this segment. Use confidence=low if unsure.
Return the same JSON schema as the full fact_lock (legal_outcome may be empty if not in this segment):

{schema}"""

_FACT_LOCK_SCHEMA = """{
  "case_name": "",
  "source_summary": "",
  "verified_people": [{"name": "", "role": "", "confidence": "high|medium|low", "source_phrase": ""}],
  "verified_dates": [{"date_or_period": "", "event": "", "confidence": "high|medium|low", "source_phrase": ""}],
  "verified_locations": [{"location": "", "context": "", "confidence": "high|medium|low"}],
  "verified_timeline": [{"order": 1, "date_or_period": "", "event": "", "confidence": "high|medium|low", "source_phrase": ""}],
  "legal_outcome": {"trial_result": "", "appeal_result": "", "supreme_court_or_final_result": "", "sentence_or_parole": "", "confidence": "high|medium|low", "source_phrase": ""},
  "key_evidence_or_turning_points": [{"evidence": "", "source_phrase": "", "confidence": "high|medium|low", "why_it_matters": ""}],
  "important_audio_or_call_moments": [{"call_type": "", "description": "", "source_phrase": "", "confidence": "high|medium|low", "safety_note": ""}],
  "emotional_details": [{"detail": "", "source_phrase": "", "confidence": "high|medium|low", "story_use": ""}],
  "recreated_scene_candidates": [{"scene_type": "", "why_useful": "", "safety_note": ""}],
  "facts_to_verify_externally": [{"fact": "", "reason": "", "confidence": "high|medium|low"}],
  "must_not_say": []
}"""


# ─── Research-view mode ───────────────────────────────────────────────────────

def _build_research_view_prompt(
    case_hint: str,
    episode_number: str,
    source_url: str,
    transcript_research_view: str,
) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    replacements = {
        "{case_hint}": case_hint,
        "{episode_number}": episode_number,
        "{source_url}": source_url,
        "{transcript_research_view}": transcript_research_view,
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def _run_research_view_mode(
    case_hint: str,
    episode_number: str,
    source_url: str,
    transcript_research_view: str,
    facts_dir: Path,
) -> dict:
    prompt = _build_research_view_prompt(
        case_hint=case_hint,
        episode_number=episode_number,
        source_url=source_url,
        transcript_research_view=transcript_research_view,
    )
    raw_response, stop_reason = call_claude_agent(prompt, agent_name="fact_lock")

    raw_path = facts_dir / "_fact_lock_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    logger.info("Fact lock raw response saved → %s", raw_path)

    if stop_reason == "max_tokens":
        logger.warning("fact_lock agent hit max_tokens — output may be truncated")

    try:
        return parse_package_response(raw_response)
    except ValueError as exc:
        raise ValueError(
            f"Fact Lock Agent JSON parse failed: {exc}\n"
            f"Raw response saved at: {raw_path}"
        ) from exc


# ─── Segmented mode ───────────────────────────────────────────────────────────

def _split_into_segments(text: str, segment_chars: int) -> list[str]:
    """Split text into segments of approximately segment_chars characters.
    Tries to break on sentence boundaries to avoid cutting mid-sentence."""
    if len(text) <= segment_chars:
        return [text]

    segments = []
    start = 0
    while start < len(text):
        end = start + segment_chars
        if end >= len(text):
            segments.append(text[start:].strip())
            break
        # Try to find a good sentence break near the end of the segment
        # Look back up to 300 chars for ". " or "\n"
        break_pos = end
        for ch in ["। ", ". ", "\n\n", "\n"]:
            pos = text.rfind(ch, start + segment_chars - 300, end)
            if pos != -1:
                break_pos = pos + len(ch)
                break
        segments.append(text[start:break_pos].strip())
        start = break_pos

    return [s for s in segments if s.strip()]


def _merge_segment_facts(segments: list[dict], case_hint: str) -> dict:
    """Merge multiple segment fact_lock dicts into one unified fact_lock."""
    merged: dict = {
        "case_name": case_hint,
        "source_summary": f"Merged from {len(segments)} transcript segments.",
        "verified_people": [],
        "verified_dates": [],
        "verified_locations": [],
        "verified_timeline": [],
        "legal_outcome": {
            "trial_result": "",
            "appeal_result": "",
            "supreme_court_or_final_result": "",
            "sentence_or_parole": "",
            "confidence": "low",
            "source_phrase": "",
        },
        "key_evidence_or_turning_points": [],
        "important_audio_or_call_moments": [],
        "emotional_details": [],
        "recreated_scene_candidates": [],
        "facts_to_verify_externally": [],
        "must_not_say": [],
    }

    # Deduplication helpers
    seen_people: set[str] = set()
    seen_dates: set[str] = set()
    seen_locations: set[str] = set()

    _CONF_ORDER = {"high": 3, "medium": 2, "low": 1}

    timeline_order = 0

    for seg in segments:
        # People — deduplicate by name (keep highest confidence)
        for p in seg.get("verified_people", []):
            key = p.get("name", "").lower().strip()
            if key and key not in seen_people:
                seen_people.add(key)
                merged["verified_people"].append(p)

        # Dates — deduplicate by date + event
        for d in seg.get("verified_dates", []):
            key = f"{d.get('date_or_period','')}|{d.get('event','')}".lower().strip()
            if key and key not in seen_dates:
                seen_dates.add(key)
                merged["verified_dates"].append(d)

        # Locations — deduplicate by location name
        for loc in seg.get("verified_locations", []):
            key = loc.get("location", "").lower().strip()
            if key and key not in seen_locations:
                seen_locations.add(key)
                merged["verified_locations"].append(loc)

        # Timeline — append and renumber at the end
        for ev in seg.get("verified_timeline", []):
            timeline_order += 1
            ev_copy = dict(ev)
            ev_copy["order"] = timeline_order
            merged["verified_timeline"].append(ev_copy)

        # Legal outcome — take the highest-confidence one
        seg_legal = seg.get("legal_outcome", {})
        if seg_legal.get("trial_result") or seg_legal.get("sentence_or_parole"):
            seg_conf = _CONF_ORDER.get(seg_legal.get("confidence", "low"), 1)
            cur_conf = _CONF_ORDER.get(merged["legal_outcome"].get("confidence", "low"), 1)
            if seg_conf > cur_conf:
                merged["legal_outcome"] = seg_legal
            elif seg_conf == cur_conf:
                # Merge fields — take non-empty from segment
                for field in ["trial_result", "appeal_result", "supreme_court_or_final_result", "sentence_or_parole"]:
                    if seg_legal.get(field) and not merged["legal_outcome"].get(field):
                        merged["legal_outcome"][field] = seg_legal[field]

        # Lists — append all, deduplication by content
        for field in ["key_evidence_or_turning_points", "important_audio_or_call_moments",
                      "emotional_details", "facts_to_verify_externally", "must_not_say"]:
            existing_set = set(str(x) for x in merged[field])
            for item in seg.get(field, []):
                if str(item) not in existing_set:
                    merged[field].append(item)
                    existing_set.add(str(item))

        # Recreated scene candidates
        for cand in seg.get("recreated_scene_candidates", []):
            key = cand.get("scene_type", "").lower()
            existing_types = {c.get("scene_type", "").lower() for c in merged["recreated_scene_candidates"]}
            if key and key not in existing_types:
                merged["recreated_scene_candidates"].append(cand)

    logger.info(
        "Merged %d segments: %d people, %d dates, %d timeline events",
        len(segments),
        len(merged["verified_people"]),
        len(merged["verified_dates"]),
        len(merged["verified_timeline"]),
    )
    return merged


def _run_segmented_mode(
    case_hint: str,
    clean_transcript: str,
    facts_dir: Path,
) -> dict:
    """Split transcript into segments and run one fact extraction per segment."""
    seg_dir = facts_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    segment_chars = settings.fact_lock_segment_chars
    segments = _split_into_segments(clean_transcript, segment_chars)
    total = len(segments)
    logger.info("Segmented mode: %d segments from %d chars", total, len(clean_transcript))

    segment_facts: list[dict] = []

    for i, seg_text in enumerate(segments, 1):
        seg_num_str = str(i).zfill(3)
        logger.info("Processing segment %s/%d (%d chars)", seg_num_str, total, len(seg_text))

        prompt = _SEGMENT_PROMPT_TEMPLATE.format(
            segment_num=i,
            total_segments=total,
            case_hint=case_hint,
            segment_text=seg_text,
            schema=_FACT_LOCK_SCHEMA,
        )

        raw_response, stop_reason = call_claude_agent(
            prompt, agent_name=f"fact_lock_segment_{seg_num_str}"
        )

        # Save raw segment response
        (seg_dir / f"_segment_{seg_num_str}_raw.txt").write_text(raw_response, encoding="utf-8")

        if stop_reason == "max_tokens":
            logger.warning("Segment %s hit max_tokens", seg_num_str)

        try:
            seg_facts = parse_package_response(raw_response)
        except ValueError as exc:
            logger.warning("Segment %s JSON parse failed: %s — skipping", seg_num_str, exc)
            continue

        # Save parsed segment
        (seg_dir / f"fact_segment_{seg_num_str}.json").write_text(
            json.dumps(seg_facts, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        segment_facts.append(seg_facts)

    if not segment_facts:
        raise ValueError("All fact_lock segments failed to parse. Check 02-facts/segments/ for raw responses.")

    return _merge_segment_facts(segment_facts, case_hint)


# ─── Public entry point ───────────────────────────────────────────────────────

def run_fact_lock(
    case_hint: str,
    episode_number: str,
    source_url: str,
    transcript_research_view: str,
    facts_dir: Path,
    clean_transcript: str = "",
    override_mode: str | None = None,
) -> dict:
    """
    Call the Fact Lock Agent using the configured mode.

    Mode is controlled by FACT_LOCK_MODE in .env:
      research_view (default) — one call using transcript_research_view
      segmented               — multiple calls over clean_transcript segments

    override_mode: if provided (e.g. "segmented"), takes precedence over FACT_LOCK_MODE.
    Used by the pipeline to auto-switch to segmented for long premium transcripts.

    Saves:
      fact_lock.json                    — parsed, merged fact lock
      _fact_lock_raw_response.txt       — (research_view mode)
      segments/fact_segment_NNN.json    — (segmented mode)
      segments/_segment_NNN_raw.txt     — (segmented mode)

    Returns the fact_lock dict.
    Raises ValueError on failure (raw response already saved).
    """
    mode = (override_mode or settings.fact_lock_mode).lower().strip()

    if mode == "segmented" and clean_transcript:
        logger.info("Fact Lock: using SEGMENTED mode")
        fact_lock = _run_segmented_mode(
            case_hint=case_hint,
            clean_transcript=clean_transcript,
            facts_dir=facts_dir,
        )
    else:
        if mode == "segmented" and not clean_transcript:
            logger.warning(
                "FACT_LOCK_MODE=segmented but clean_transcript was not passed — "
                "falling back to research_view mode"
            )
        logger.info("Fact Lock: using RESEARCH_VIEW mode")
        fact_lock = _run_research_view_mode(
            case_hint=case_hint,
            episode_number=episode_number,
            source_url=source_url,
            transcript_research_view=transcript_research_view,
            facts_dir=facts_dir,
        )

    # Save unified fact_lock.json
    out_path = facts_dir / "fact_lock.json"
    out_path.write_text(json.dumps(fact_lock, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Fact lock saved → %s", out_path)

    return fact_lock
