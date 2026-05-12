"""
Fact Lock Agent — Claude agent that extracts verified facts from a transcript.

Supports three modes (controlled by FACT_LOCK_MODE in .env):

  research_view (default, cheap):
    Sends the pre-built beginning/middle/end research view to one Claude call.
    Good for most transcripts. Budget-conscious.

  segmented (thorough):
    Splits the clean transcript into ~7000-char segments.
    Runs a compact fact extraction per segment (one Claude call each).
    Merges segment results into a unified fact_lock.
    Better for very long or complex transcripts where details may fall in the middle.

  auto (recommended for production):
    Picks the best mode based on transcript size using prompt_budget_service.
    Uses segmented when clean_chars >= LONG_TRANSCRIPT_CLEAN_CHARS_THRESHOLD (default 30 000).
    Falls back to research_view for smaller transcripts.
    Users never need to change .env per episode when FACT_LOCK_MODE=auto.

Produces:
  02-facts/fact_lock.json                               (all modes)
  02-facts/_fact_lock_raw_response.txt                  (research_view mode)
  02-facts/fact_lock_segments/fact_segment_NNN.json     (segmented / auto-segmented)
  02-facts/fact_lock_segments/_segment_NNN_raw.txt      (segmented / auto-segmented)
  02-facts/fact_lock_segment_index.json                 (segmented / auto-segmented)
  02-facts/fact_lock_merged.json                        (segmented / auto-segmented)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import settings
from app.services.claude_client import call_claude_agent, parse_package_response
from app.services.prompt_budget_service import estimate_tokens, should_use_segmented_mode

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
  "verified_people": [{"name": "", "role": "", "confidence": "high|medium|low", "source_note": ""}],
  "verified_dates": [{"date_or_period": "", "event": "", "confidence": "high|medium|low", "source_note": ""}],
  "verified_locations": [{"location": "", "context": "", "confidence": "high|medium|low"}],
  "verified_timeline": [{"order": 1, "date_or_period": "", "event": "", "confidence": "high|medium|low", "source_note": ""}],
  "legal_outcome": {"trial_result": "", "appeal_result": "", "supreme_court_or_final_result": "", "sentence_or_parole": "", "confidence": "high|medium|low", "source_note": ""},
  "key_evidence_or_turning_points": [{"evidence": "", "source_note": "", "confidence": "high|medium|low", "why_it_matters": ""}],
  "important_audio_or_call_moments": [{"call_type": "", "description": "", "source_note": "", "confidence": "high|medium|low", "safety_note": ""}],
  "emotional_details": [{"detail": "", "source_note": "", "confidence": "high|medium|low", "story_use": ""}],
  "recreated_scene_candidates": [{"scene_type": "", "why_useful": "", "safety_note": ""}],
  "facts_to_verify_externally": [{"fact": "", "reason": "", "confidence": "high|medium|low"}],
  "must_not_say": []
}"""

# Threshold above which we warn that the fact lock output is unexpectedly large
_FACT_LOCK_LARGE_OUTPUT_CHARS = 25_000

# Threshold above which we inject a compact-mode instruction into the prompt
_COMPACT_NOTE_THRESHOLD_CHARS = 16_000

_COMPACT_EXTRACTION_NOTE = (
    "IMPORTANT — COMPACT MODE ACTIVE: This transcript view is large. "
    "Extract only the most essential facts. "
    "Do NOT produce an exhaustive timeline entry for every sentence. "
    "Do NOT include every emotional detail. "
    "Prioritise: key people, confirmed dates, legal outcome, top evidence, "
    "and recreated scene candidates. Stay strictly within the hard output limits."
)


# ─── Research-view mode ───────────────────────────────────────────────────────

def _build_research_view_prompt(
    case_hint: str,
    episode_number: str,
    source_url: str,
    transcript_research_view: str,
) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    # Inject compact-mode note when the research view is large
    compact_note = (
        _COMPACT_EXTRACTION_NOTE
        if len(transcript_research_view) > _COMPACT_NOTE_THRESHOLD_CHARS
        else ""
    )
    replacements = {
        "{case_hint}": case_hint,
        "{episode_number}": episode_number,
        "{source_url}": source_url,
        "{transcript_research_view}": transcript_research_view,
        "{compact_extraction_note}": compact_note,
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
    logger.info(
        "Fact Lock preflight: research_view=%d chars, prompt=%d chars, "
        "compact_mode=%s, max_tokens=%d",
        len(transcript_research_view),
        len(prompt),
        len(transcript_research_view) > _COMPACT_NOTE_THRESHOLD_CHARS,
        settings.claude_max_tokens,
    )

    raw_response, stop_reason = call_claude_agent(prompt, agent_name="fact_lock")

    raw_path = facts_dir / "_fact_lock_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    logger.info("Fact lock raw response saved → %s  (%d chars)", raw_path, len(raw_response))

    if stop_reason == "max_tokens":
        logger.warning("fact_lock agent hit max_tokens — output may be truncated")

    if len(raw_response) > _FACT_LOCK_LARGE_OUTPUT_CHARS:
        logger.warning(
            "fact_lock output is large (%d chars > %d threshold). "
            "Consider using FACT_LOCK_MODE=segmented or compacting the prompt. "
            "Continuing parse attempt.",
            len(raw_response),
            _FACT_LOCK_LARGE_OUTPUT_CHARS,
        )

    try:
        return parse_package_response(raw_response, agent_name="fact_lock")
    except ValueError as exc:
        error_path = facts_dir / "_fact_lock_parse_error.txt"
        try:
            error_path.write_text(
                f"Parse failed: {exc}\nRaw response: {raw_path}\n",
                encoding="utf-8",
            )
            logger.error("Fact lock parse error saved → %s", error_path)
        except Exception as save_exc:
            logger.warning("Could not save fact_lock parse error file: %s", save_exc)
        raise ValueError(
            f"Fact Lock Agent JSON parse failed: {exc}\n"
            f"Raw response saved at: {raw_path}\n"
            f"Parse error details saved at: {error_path}"
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
    seg_dir = facts_dir / "fact_lock_segments"
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

        # Save raw segment response (legacy path alias kept for compatibility)
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
        raise ValueError(
            "All fact_lock segments failed to parse. "
            "Check 02-facts/fact_lock_segments/ for raw responses."
        )

    # Write segment index (list of segment files with char counts)
    segment_index = [
        {
            "segment_num": i + 1,
            "chars": len(segments[i]),
            "parsed": (i + 1) <= len(segment_facts),
            "file": f"fact_lock_segments/fact_segment_{str(i + 1).zfill(3)}.json",
        }
        for i in range(total)
    ]
    (facts_dir / "fact_lock_segment_index.json").write_text(
        json.dumps({"total_segments": total, "segments": segment_index}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    merged = _merge_segment_facts(segment_facts, case_hint)

    # Write the merged result before returning so callers can inspect it separately
    (facts_dir / "fact_lock_merged.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Segmented fact lock: merged file saved → %s/fact_lock_merged.json", facts_dir)

    return merged


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
      auto                    — uses prompt_budget_service to pick the best mode:
                                segmented when clean_chars >= long_transcript_clean_chars_threshold,
                                research_view otherwise.

    override_mode: if provided (e.g. "segmented"), takes precedence over FACT_LOCK_MODE.
    Used by the pipeline to auto-switch to segmented for long premium transcripts.

    Saves:
      fact_lock.json                              — parsed, merged fact lock (all modes)
      _fact_lock_raw_response.txt                 — (research_view mode)
      fact_lock_segments/fact_segment_NNN.json    — (segmented / auto-segmented mode)
      fact_lock_segments/_segment_NNN_raw.txt     — (segmented / auto-segmented mode)
      fact_lock_segment_index.json                — (segmented / auto-segmented mode)
      fact_lock_merged.json                       — (segmented / auto-segmented mode)

    Returns the fact_lock dict.
    Raises ValueError on failure (raw response already saved).
    """
    requested_mode = (override_mode or settings.fact_lock_mode).lower().strip()
    mode = requested_mode  # may be overridden below for "auto"

    clean_chars = len(clean_transcript) if clean_transcript else 0
    clean_tokens = estimate_tokens(clean_chars)

    logger.info(
        "Fact Lock: requested_mode=%s, clean_transcript=%d chars (~%d tokens), "
        "research_view=%d chars, FACT_LOCK_MODE=%s, CLAUDE_MAX_TOKENS=%d",
        requested_mode,
        clean_chars,
        clean_tokens,
        len(transcript_research_view),
        settings.fact_lock_mode,
        settings.claude_max_tokens,
    )

    # Auto mode: decide based on transcript size
    if requested_mode == "auto":
        if clean_transcript and should_use_segmented_mode(clean_chars):
            logger.info(
                "Fact Lock auto mode: clean_transcript %d chars >= threshold %d — "
                "switching to segmented for thorough coverage",
                clean_chars,
                settings.long_transcript_clean_chars_threshold,
            )
            mode = "segmented"
        else:
            logger.info(
                "Fact Lock auto mode: clean_transcript %d chars < threshold %d — "
                "using research_view (fits in budget)",
                clean_chars,
                settings.long_transcript_clean_chars_threshold,
            )
            mode = "research_view"

    if mode == "segmented" and clean_transcript:
        logger.info("Fact Lock: using SEGMENTED mode")
        fact_lock = _run_segmented_mode(
            case_hint=case_hint,
            clean_transcript=clean_transcript,
            facts_dir=facts_dir,
        )
    else:
        if mode in ("segmented", "auto") and not clean_transcript:
            logger.warning(
                "FACT_LOCK_MODE=%s but clean_transcript was not passed — "
                "falling back to research_view mode",
                mode,
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
