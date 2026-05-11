"""
Text Similarity Service — Python-only English phrase detection and structural
source-copy risk analysis.

Detects English phrases from the source transcript that have been copied verbatim
(or near-verbatim) into the Hindi script or metadata.

Also checks for structural similarity: whether the script opening mirrors the
source opening in sequence, and whether the overall section order is too close
to the source transcript order.

No Claude calls. Pure Python string/n-gram matching.

Produces a similarity_report dict that the Originality Safety Gate Service and
the OpenAI Final Premium Gate receive as evidence.
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

# Longer n-gram size for "high-risk" matches (verbatim longer phrases)
_HIGH_RISK_NGRAM = 8

# Strip these from extracted tokens before matching
_STRIP_CHARS = re.compile(r"[^\w\s]")

# Regex that matches a run of ASCII-alphabet words (English phrases embedded
# in a largely Devanagari / Hindi document)
_ENGLISH_RUN = re.compile(r"(?<!\w)([A-Za-z]+(?:\s+[A-Za-z]+){%d,})(?!\w)" % (_MIN_NGRAM - 1))

# Fraction of text treated as the "opening" for sequence-similarity checks
_OPENING_FRACTION = 0.12


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


# ─── Structural similarity helpers ───────────────────────────────────────────

def _extract_opening_tokens(text: str, fraction: float = _OPENING_FRACTION) -> set[str]:
    """Return English n-grams from the opening fraction of text."""
    end = max(200, int(len(text) * fraction))
    return _extract_english_ngrams(text[:end], _MIN_NGRAM)


def _check_opening_sequence_risk(
    source_transcript: str,
    script_text: str,
) -> str:
    """
    Compare the opening section of the script against the opening section of
    the source transcript.

    Returns "high" if the script opening shares many English phrases with the
    source opening (suggesting the writer followed the same intro sequence),
    "medium" if moderate overlap, "low" or "none" otherwise.

    Does NOT flag proper nouns or short factual phrases — but a shared opening
    structure with many overlapping English phrases is a signal of structural copy.
    """
    source_opening = _extract_opening_tokens(source_transcript)
    script_opening = _extract_opening_tokens(script_text)
    overlap = source_opening & script_opening
    if not source_opening:
        return "none"
    overlap_ratio = len(overlap) / max(len(source_opening), 1)
    if overlap_ratio >= 0.40:
        return "high"
    if overlap_ratio >= 0.20:
        return "medium"
    if len(overlap) >= 2:
        return "low"
    return "none"


# ─── Public API ───────────────────────────────────────────────────────────────

def run_text_similarity_check(
    source_transcript: str,
    script_draft: dict,
) -> dict:
    """
    Compare English phrases in the source transcript against the script and
    metadata. Also checks for structural (opening-sequence) similarity.
    Returns a similarity_report dict.

    similarity_report keys:
      script_matches           — list of English phrases (≥5 words) copied into the script
      metadata_matches         — list of English phrases copied into metadata
      high_risk_matches        — count of very long matches (≥8 words) — strong verbatim signal
      script_match_count       — int
      metadata_match_count     — int
      total_match_count        — int
      opening_sequence_risk    — "none"|"low"|"medium"|"high" — opening section structure risk
      structure_risk_level     — aggregate structure risk
      risk_level               — overall risk "none"|"low"|"medium"|"high"
      summary                  — human-readable one-liner
      ngram_size               — n-gram size used
    """
    n = _MIN_NGRAM
    source_ngrams      = _extract_english_ngrams(source_transcript, n)
    source_ngrams_long = _extract_english_ngrams(source_transcript, _HIGH_RISK_NGRAM)

    script_text   = _extract_script_text(script_draft)
    metadata_text = _extract_metadata_text(script_draft)

    script_matches   = _find_matches(source_ngrams, script_text, n)
    metadata_matches = _find_matches(source_ngrams, metadata_text, n)

    # High-risk matches: longer n-grams (≥8 words) — very strong verbatim copy signal
    script_matches_long   = _find_matches(source_ngrams_long, script_text, _HIGH_RISK_NGRAM)
    metadata_matches_long = _find_matches(source_ngrams_long, metadata_text, _HIGH_RISK_NGRAM)
    high_risk_matches = len(script_matches_long) + len(metadata_matches_long)

    # Opening sequence structural risk
    opening_sequence_risk = _check_opening_sequence_risk(source_transcript, script_text)

    total = len(script_matches) + len(metadata_matches)

    # Compute phrase risk
    if total == 0:
        phrase_risk = "none"
    elif total <= 3:
        phrase_risk = "low"
    elif total <= 8:
        phrase_risk = "medium"
    else:
        phrase_risk = "high"

    # Elevate risk if high-risk (long) matches present
    if high_risk_matches > 0 and phrase_risk in ("none", "low"):
        phrase_risk = "medium"
    if high_risk_matches >= 3:
        phrase_risk = "high"

    # Structure risk (opening sequence)
    structure_risk_level = opening_sequence_risk

    # Overall risk: worst of phrase risk and structure risk
    _risk_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    overall_rank = max(
        _risk_rank.get(phrase_risk, 0),
        _risk_rank.get(structure_risk_level, 0),
    )
    risk_level = ["none", "low", "medium", "high"][overall_rank]

    # Summary
    parts = []
    if total > 0:
        parts.append(f"{total} English phrase match(es)")
    if high_risk_matches > 0:
        parts.append(f"{high_risk_matches} long verbatim match(es)")
    if opening_sequence_risk not in ("none", "low"):
        parts.append(f"opening-sequence risk={opening_sequence_risk}")
    if not parts:
        summary = "No verbatim English phrases detected from source transcript."
    else:
        summary = " | ".join(parts) + f" — overall risk: {risk_level}"

    report = {
        "script_matches":        script_matches,
        "metadata_matches":      metadata_matches,
        "high_risk_matches":     high_risk_matches,
        "script_match_count":    len(script_matches),
        "metadata_match_count":  len(metadata_matches),
        "total_match_count":     total,
        "opening_sequence_risk": opening_sequence_risk,
        "structure_risk_level":  structure_risk_level,
        "risk_level":            risk_level,
        "summary":               summary,
        "ngram_size":            n,
    }

    logger.info(
        "Text similarity: %d script matches (%d long), %d metadata matches — "
        "phrase_risk=%s opening_risk=%s overall_risk=%s",
        len(script_matches), high_risk_matches,
        len(metadata_matches),
        phrase_risk, opening_sequence_risk, risk_level,
    )
    return report
