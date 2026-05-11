"""
Transcript Cleaner — deterministic Python cleanup, no Claude call.

Removes YouTube-transcript timestamp noise, bracketed markers, sponsor/ad
reads, outro/supporter sections, and platform UI junk while preserving all
spoken case content.

Input:  raw_transcript (str) — as provided in the API payload
Output: cleaned transcript (str)

Optionally writes a JSON cleanup report to report_path.

Conservative design rules:
  - Only removes sponsor blocks when a clear trigger phrase is present.
  - Only removes outro sections from the last 25% of the transcript.
  - Never removes case names, locations, dates, court details, or evidence.
  - Paragraphs mentioning YouTube/internet/phone as part of the case are kept.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Timestamp patterns ────────────────────────────────────────────────────────
# Matches YouTube auto-transcript / caption export timestamps:
#   "0:011 second"          "0:2020 seconds"
#   "1:001 minute"          "4:004 minutes"
#   "1:091 minute, 9 seconds"   "44:5944 minutes, 59 seconds"
# Digits run together with no separator between code and label.

_TS_PATTERN = re.compile(
    r"\d{1,2}:\d{2}"                               # timestamp code, e.g. "0:09"
    r"\d*"                                          # run-together extra digits
    r"\s*\d*\s*"
    r"(?:minutes?|seconds?)"                        # unit word
    r"(?:\s*,\s*\d+\s*(?:minutes?|seconds?))?"     # optional ", 9 seconds"
    r"\s*",
    re.IGNORECASE,
)

# ── Bracketed non-story markers ───────────────────────────────────────────────
_BRACKET_PATTERN = re.compile(
    r"\["
    r"(?:Music|Applause|Laughter|Sponsor|Ad|Advertisement|"
    r"Intro|Outro|Background Music|Upbeat Music|Sad Music|"
    r"Dramatic Music|Sound Effect|Silence|Inaudible)"
    r"[^\]]*\]",
    re.IGNORECASE,
)

# ── YouTube UI / playlist junk (line-level patterns) ─────────────────────────
_UI_PATTERNS: list[re.Pattern] = [
    re.compile(r"Sync to video time\s*", re.IGNORECASE),
    re.compile(r"Crime Beat TV\s*[-–—]\s*Season\s+\d+\s*", re.IGNORECASE),
    re.compile(r"Crime Beat TV\s*", re.IGNORECASE),
    re.compile(r"\d+\s*/\s*\d+\s*"),                                # "1 / 13"
    re.compile(r"^\s*Season\s+\d+\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bwatch next\b[^\n]*", re.IGNORECASE),
    re.compile(r"\bup next\b[^\n]*", re.IGNORECASE),
    # "subscribe" only when it's an isolated UI element (its own line / surrounded by newlines)
    re.compile(r"(?:^|\n)[ \t]*subscribe[ \t]*(?=\n|$)", re.IGNORECASE | re.MULTILINE),
]

# ── Sponsor block detection ───────────────────────────────────────────────────
# Trigger phrases: any of these → begin scanning for a sponsor block.
_SPONSOR_TRIGGERS: list[str] = [
    "today's sponsor",
    "today’s sponsor",   # curly apostrophe variant
    "sponsored by",
    "this episode is sponsored",
    "this video is sponsored",
    "thanks to our sponsor",
    "thank you to our sponsor",
    "brought to you by",
    "in partnership with",
]

# Keywords that strongly indicate a sentence belongs to a sponsor block.
# Deliberately narrow to avoid false-positives in case content.
_SPONSOR_SENTENCE_KEYWORDS: list[str] = [
    "free trial",
    "promo code",
    "use code",
    "discount code",
    "coupon code",
    "link in the description",
    "description box",
    "click my link",
    "aura.com",
    "nordvpn",
    "expressvpn",
    "betterhelp",
    "hellofresh",
    "squarespace",
    "wix.com",
    "manscaped",
    "data broker",
    "opt out request",
    "antivirus",
    "identity theft insurance",
    "password manager",
    "parental control",
    "/month",
    "% off",
    "go to aura",
]

# Maximum characters to consume per sponsor block (safety cap).
_MAX_SPONSOR_BLOCK_CHARS = 2500

# Minimum sponsor content (chars) before we look for a story-resume sentence.
_MIN_SPONSOR_CHARS = 150

# ── Outro / supporter section triggers ───────────────────────────────────────
# Only applied within the last _OUTRO_ZONE_PCT% of the transcript.
_OUTRO_TRIGGERS: list[str] = [
    "thank you so much for watching",
    "thank you for watching",
    "thank you to my supporters",
    "thank you to my patrons",
    "huge thank you to my supporters",
    "huge thank you to my patrons",
    "a huge thank you",
    "like and subscribe",
    "please subscribe",
    "check out my merch",
    "follow me on",
    "stay safe stay spooky",
    "stay safe, stay spooky",
    "devils in the detail",
    "devil's in the detail",
    "devil’s in the detail",
]

# Outro zone: only trigger outro removal within the last N% of transcript.
_OUTRO_ZONE_PCT = 25

# ── Post-cleanup validation terms ────────────────────────────────────────────
# Warn if any of these survive after cleanup.
_LEFTOVER_JUNK_TERMS: list[str] = [
    "[music]",
    "sponsored by",
    "aura.com",
    "patreon",
    "supporters",
    "stay safe stay spooky",
    "like and subscribe",
    "merch",
    "follow me on",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _strip_timestamps(text: str) -> str:
    """Remove YouTube timestamp artifacts."""
    return _TS_PATTERN.sub(" ", text)


def _strip_brackets(text: str) -> str:
    """Remove bracketed non-story markers like [Music], [Applause]."""
    return _BRACKET_PATTERN.sub(" ", text)


def _strip_ui_junk(text: str) -> str:
    """Remove YouTube UI noise and playlist junk."""
    for pattern in _UI_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def _strip_sponsor_blocks(text: str) -> tuple[str, int, list[str]]:
    """
    Remove sponsor/ad-read sections. Returns (cleaned_text, count, markers).

    Algorithm:
    1. Find the earliest sponsor trigger phrase.
    2. Walk back to the nearest sentence boundary (avoids orphaning half-sentences).
    3. Scan forward sentence-by-sentence up to _MAX_SPONSOR_BLOCK_CHARS.
    4. Stop when a "story-resume" sentence is found: ≥8 words, no sponsor keywords,
       after at least _MIN_SPONSOR_CHARS of sponsor content.
    5. Remove the block. Repeat until no more triggers remain.

    Conservative: only activates when a clear trigger phrase is present.
    Never removes more than _MAX_SPONSOR_BLOCK_CHARS per block.
    """
    removed_count = 0
    removed_markers: list[str] = []
    result = text

    for _attempt in range(10):  # max 10 sponsor blocks per transcript
        lower = result.lower()
        trigger_pos = -1
        for phrase in _SPONSOR_TRIGGERS:
            pos = lower.find(phrase)
            if pos != -1 and (trigger_pos == -1 or pos < trigger_pos):
                trigger_pos = pos
        if trigger_pos == -1:
            break   # no more triggers

        # Walk back to sentence/line boundary (within 400 chars)
        block_start = trigger_pos
        for i in range(trigger_pos - 1, max(0, trigger_pos - 400), -1):
            if result[i] in ".!?\n" and i < trigger_pos - 3:
                block_start = i + 1
                break
        # Skip leading whitespace
        while block_start < trigger_pos and result[block_start] in " \t\n":
            block_start += 1

        # Scan forward to find where sponsor content ends
        max_end = min(len(result), trigger_pos + _MAX_SPONSOR_BLOCK_CHARS)
        scan_region = result[trigger_pos:max_end]

        # Rough sentence split on ". " or double newline
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", scan_region)

        chars_consumed = 0
        block_end_rel = len(scan_region)  # default: cut to max

        for sent in raw_sentences:
            chars_consumed += len(sent) + 1
            if chars_consumed < _MIN_SPONSOR_CHARS:
                continue  # always absorb at least _MIN_SPONSOR_CHARS

            sent_lower = sent.lower()
            is_sponsor = any(kw in sent_lower for kw in _SPONSOR_SENTENCE_KEYWORDS)
            word_count = len(sent.split())

            if not is_sponsor and word_count >= 8:
                # Story-resume sentence found — stop here
                block_end_rel = chars_consumed - len(sent) - 1
                break

        block_end = min(trigger_pos + max(block_end_rel, 0), len(result))

        if block_end <= block_start:
            break

        snippet = result[block_start:block_end].strip()
        removed_markers.append(
            f"sponsor_block: {snippet[:80]}{'...' if len(snippet) > 80 else ''}"
        )
        logger.info(
            "Sponsor block removed: %d chars at pos %d (trigger in %r)",
            block_end - block_start,
            block_start,
            snippet[:40],
        )
        result = result[:block_start].rstrip() + "\n\n" + result[block_end:].lstrip()
        removed_count += 1

    return result, removed_count, removed_markers


def _strip_outro(text: str) -> tuple[str, int, list[str]]:
    """
    Remove outro/supporter sections from the last _OUTRO_ZONE_PCT% of the transcript.

    Only triggers within the outro zone — prevents accidentally removing
    case content that mentions YouTube, Patreon, or social media as part of the story.
    """
    removed_markers: list[str] = []
    total_len = len(text)
    outro_zone_start = int(total_len * (1 - _OUTRO_ZONE_PCT / 100))

    outro_section = text[outro_zone_start:]
    outro_lower = outro_section.lower()

    earliest_trigger_rel = -1
    matched_phrase = ""
    for phrase in _OUTRO_TRIGGERS:
        pos = outro_lower.find(phrase)
        if pos != -1 and (earliest_trigger_rel == -1 or pos < earliest_trigger_rel):
            earliest_trigger_rel = pos
            matched_phrase = phrase

    if earliest_trigger_rel == -1:
        return text, 0, []

    # Walk back to nearest sentence/line boundary in the outro section
    cut_pos_rel = earliest_trigger_rel
    for i in range(earliest_trigger_rel - 1, max(0, earliest_trigger_rel - 300), -1):
        if outro_section[i] in ".!?\n" and i < earliest_trigger_rel - 3:
            cut_pos_rel = i + 1
            break

    cut_pos = outro_zone_start + cut_pos_rel
    removed_text = text[cut_pos:].strip()

    removed_markers.append(
        f"outro_block (trigger={matched_phrase!r}): "
        f"{removed_text[:80]}{'...' if len(removed_text) > 80 else ''}"
    )
    logger.info(
        "Outro block removed: %d chars from pos %d (trigger=%r)",
        total_len - cut_pos,
        cut_pos,
        matched_phrase,
    )
    return text[:cut_pos].rstrip(), 1, removed_markers


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated spaces/tabs and excessive blank lines."""
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _validate_leftover_junk(clean_text: str) -> list[str]:
    """Scan cleaned text for leftover junk terms. Returns list of warning strings."""
    warnings = []
    lower = clean_text.lower()
    for term in _LEFTOVER_JUNK_TERMS:
        if term in lower:
            warnings.append(f"Leftover junk term found after cleanup: {term!r}")
    return warnings


# ── Public API ────────────────────────────────────────────────────────────────

def clean_transcript(
    raw_transcript: str,
    report_path: Optional[Path] = None,
) -> str:
    """
    Remove timestamp clutter, sponsor blocks, outro sections, and platform noise
    from a raw YouTube/podcast transcript.

    Preserves:
      - All spoken case content
      - Names, dates, locations, legal terms
      - Quoted phone calls, interviews, reactions
      - Emotional details and memorial descriptions
      - Mentions of YouTube/internet/phone/social media that are part of the case

    Args:
        raw_transcript: Raw transcript string from the API payload.
        report_path:    Optional path to save a JSON cleanup report.
                        Parent directory is created if needed.

    Returns:
        Cleaned transcript as a single string.
    """
    text = raw_transcript
    raw_chars = len(raw_transcript)
    removed_markers: list[str] = []

    # 1. Strip timestamp artifacts
    text = _strip_timestamps(text)

    # 2. Strip bracketed markers — [Music], [Applause], etc.
    text = _strip_brackets(text)

    # 3. Strip YouTube UI / playlist junk
    text = _strip_ui_junk(text)

    # 4. Intermediate whitespace normalization (before sponsor/outro removal)
    text = _normalize_whitespace(text)

    # 5. Remove sponsor/ad blocks (conservative — only when clear trigger present)
    text, sponsor_blocks_removed, sponsor_markers = _strip_sponsor_blocks(text)
    removed_markers.extend(sponsor_markers)

    # 6. Remove outro/supporter sections (last 25% only)
    text, outro_blocks_removed, outro_markers = _strip_outro(text)
    removed_markers.extend(outro_markers)

    # 7. Final whitespace normalization
    text = _normalize_whitespace(text)

    # 8. Build stats
    clean_chars = len(text)
    removed_chars = raw_chars - clean_chars
    removed_pct = round(100.0 * removed_chars / max(raw_chars, 1), 1)

    # 9. Post-cleanup validation
    warnings: list[str] = []
    warnings.extend(_validate_leftover_junk(text))

    if removed_pct > 35.0:
        msg = (
            f"Transcript cleanup removed {removed_pct:.1f}% of content "
            f"({removed_chars:,} chars). Review transcript_cleanup_report.json."
        )
        warnings.append(msg)
        logger.warning(msg)

    logger.info(
        "Transcript cleaned: %d → %d chars (%.1f%% removed) | "
        "sponsor_blocks=%d outro_blocks=%d markers=%d warnings=%d",
        raw_chars,
        clean_chars,
        removed_pct,
        sponsor_blocks_removed,
        outro_blocks_removed,
        len(removed_markers),
        len(warnings),
    )

    # 10. Save cleanup report if path provided
    if report_path is not None:
        report = {
            "raw_chars": raw_chars,
            "clean_chars": clean_chars,
            "removed_chars": removed_chars,
            "removed_pct": removed_pct,
            "removed_markers": removed_markers,
            "sponsor_blocks_removed": sponsor_blocks_removed,
            "outro_blocks_removed": outro_blocks_removed,
            "warnings": warnings,
        }
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Cleanup report saved → %s", report_path)
        except Exception as exc:
            logger.warning("Could not save cleanup report: %s", exc)

    return text
