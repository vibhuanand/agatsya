"""
Transcript Cleaner — deterministic Python cleanup, no Claude call.

Removes YouTube-transcript timestamp noise, platform footers, and playlist
junk while preserving every word of spoken content.

Input:  raw_transcript (str) — as provided in the API payload
Output: cleaned transcript (str) — spoken content only
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# ─── Timestamp patterns ───────────────────────────────────────────────────────
# Matches patterns produced by YouTube auto-transcript / caption exports:
#   "0:099 seconds"          "0:1919 seconds"
#   "1:001 minute"           "4:004 minutes"
#   "1:091 minute, 9 seconds"   "44:5944 minutes, 59 seconds"
# The format is:  <m>:<ss><total_seconds_or_minutes_label>
# — the digits run together with no separator between the timestamp code and the
#   human-readable label, so we match them as one unit.

_TS_PATTERN = re.compile(
    r"\d+:\d+"          # timestamp code  e.g. "0:09" or "44:59"
    r"\d*"              # repeated/extra digits that run into the label
    r"\s*"
    r"\d*"              # optional leading digit of the label number
    r"\s*"
    r"(?:minutes?|seconds?)"   # unit word
    r"(?:\s*,\s*\d+\s*(?:minutes?|seconds?))?"  # optional ", 9 seconds" suffix
    r"\s*",
    re.IGNORECASE,
)

# ─── Footer / platform noise ──────────────────────────────────────────────────
_FOOTER_PATTERNS: list[re.Pattern] = [
    re.compile(r"Sync to video time\s*", re.IGNORECASE),
    re.compile(r"Crime Beat TV\s*[-–—]\s*Season\s+\d+\s*", re.IGNORECASE),
    re.compile(r"Crime Beat TV\s*", re.IGNORECASE),
    re.compile(r"\d+\s*/\s*\d+\s*"),            # "1 / 13"  (playlist position)
    re.compile(r"^\s*Season\s+\d+\s*$", re.IGNORECASE | re.MULTILINE),
]


def clean_transcript(raw_transcript: str) -> str:
    """
    Remove timestamp clutter and platform noise from a raw YouTube transcript.

    Preserves:
      - All spoken content
      - Names, dates, locations, legal terms
      - Quoted phone calls, interviews, reactions
      - Emotional details and memorial descriptions

    Returns the cleaned transcript as a single string.
    """
    text = raw_transcript

    # 1. Strip timestamp codes
    text = _TS_PATTERN.sub(" ", text)

    # 2. Strip footer / platform patterns
    for pattern in _FOOTER_PATTERNS:
        text = pattern.sub(" ", text)

    # 3. Collapse multiple blank lines → single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Collapse excessive internal whitespace on each line
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)

    # 5. Final strip
    text = text.strip()

    original_chars = len(raw_transcript)
    cleaned_chars = len(text)
    removed_pct = round(100 * (1 - cleaned_chars / max(original_chars, 1)), 1)
    logger.info(
        "Transcript cleaned: %d → %d chars (%.1f%% removed)",
        original_chars,
        cleaned_chars,
        removed_pct,
    )
    return text
