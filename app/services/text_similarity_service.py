"""
Text Similarity Service — Python-only English phrase detection.

Detects English phrases from the source transcript that have been
copied verbatim (or near-verbatim) into the Hindi script or metadata.

No Claude calls. Pure Python string/n-gram matching.

Produces a similarity_report dict that the Originality Safety Gate
Service passes into its Claude prompt and saves to disk.
"""
from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

# Minimum consecutive English words to flag as a "phrase match"
_MIN_NGRAM = 5

# Strip these from extracted tokens before matching
_STRIP_CHARS = re.compile(r"[^\w\s]")

# Regex that matches a run of ASCII-alphabet words (English phrases embedded
# in a largely Devanagari / Hindi document)
_ENGLISH_RUN = re.compile(r"(?<!\w)([A-Za-z]+(?:\s+[A-Za-z]+){%d,})(?!\w)" % (_MIN_NGRAM - 1))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation."""
    text = _STRIP_CHARS.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _extract_english_ngrams(text: str, n: int) -> set[str]:
    """
    Return a set of normalised n-grams from English-looking runs in *text*.
    Only considers words that are ASCII letters (i.e. English).
    """
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _extract_script_text(script_draft: dict) -> str:
    """Concatenate all narration chunk text from the script draft."""
    parts: list[str] = []
    for chunk in script_draft.get("hindi_narration_chunks", []):
        parts.append(chunk.get("text", ""))
    for scene in script_draft.get("recreated_dialogues", {}).get("items", []):
        for line in scene.get("dialogue", []):
            parts.append(line.get("text", ""))
    return " ".join(parts)


def _extract_metadata_text(script_draft: dict) -> str:
    """Concatenate YouTube metadata strings."""
    meta = script_draft.get("youtube_metadata", {})
    parts: list[str] = []
    parts.extend(meta.get("title_options", []))
    parts.append(meta.get("description", ""))
    parts.extend(meta.get("tags", []))
    return " ".join(parts)


def _find_matches(
    source_ngrams: set[str],
    target_text: str,
    n: int,
) -> list[str]:
    """Return source n-grams that appear in target_text."""
    target_ngrams = _extract_english_ngrams(target_text, n)
    hits = sorted(source_ngrams & target_ngrams)
    return hits


# ─── Public API ───────────────────────────────────────────────────────────────

def run_text_similarity_check(
    source_transcript: str,
    script_draft: dict,
) -> dict:
    """
    Compare English phrases in the source transcript against the script
    and metadata. Returns a similarity_report dict.

    similarity_report keys:
      script_matches       — list of English phrases copied into the script
      metadata_matches     — list of English phrases copied into metadata
      script_match_count   — int
      metadata_match_count — int
      total_match_count    — int
      risk_level           — "none" | "low" | "medium" | "high"
      summary              — human-readable one-liner
    """
    n = _MIN_NGRAM
    source_ngrams = _extract_english_ngrams(source_transcript, n)

    script_text   = _extract_script_text(script_draft)
    metadata_text = _extract_metadata_text(script_draft)

    script_matches   = _find_matches(source_ngrams, script_text, n)
    metadata_matches = _find_matches(source_ngrams, metadata_text, n)

    total = len(script_matches) + len(metadata_matches)

    if total == 0:
        risk_level = "none"
        summary    = "No verbatim English phrases detected from source transcript."
    elif total <= 3:
        risk_level = "low"
        summary    = f"{total} short phrase match(es) — likely proper nouns or unavoidable."
    elif total <= 8:
        risk_level = "medium"
        summary    = f"{total} phrase matches — review whether these are transformative."
    else:
        risk_level = "high"
        summary    = f"{total} phrase matches — significant reuse of source English content."

    report = {
        "script_matches":        script_matches,
        "metadata_matches":      metadata_matches,
        "script_match_count":    len(script_matches),
        "metadata_match_count":  len(metadata_matches),
        "total_match_count":     total,
        "risk_level":            risk_level,
        "summary":               summary,
        "ngram_size":            n,
    }

    logger.info(
        "Text similarity: %d script matches, %d metadata matches — risk=%s",
        len(script_matches), len(metadata_matches), risk_level,
    )
    return report
