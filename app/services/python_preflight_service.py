"""Python preflight gate for zero-cost script/metadata checks.

Runs before expensive final model gates. Catches deterministic issues:
unsupported legal superlatives, forbidden terms from the case glossary,
forbidden name variants, metadata shape problems, Hinglish-level leaks.

All checks are driven by the case_glossary — no case-specific hardcoding.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import settings


_FIRST_CLAIM_PATTERNS = [
    "पहला मामला",
    "पहली बार",
    "पहले कभी नहीं",
    "कानून बदल दिया",
    "क़ानून बदल दिया",
    "first ever",
    "never before",
    "changed the law",
]

# Generic sensational/clickbait Hindi phrases — always blocked regardless of glossary.
# These violate YouTube's true-crime content guidelines for all cases.
_SENSATIONAL_HINDI_PHRASES = [
    "सबसे खौफनाक",
    "रूह कांप जाएगी",
    "हिला देने वाला सच",
    "पूरी सच्चाई पहली बार",
    # Hindi YouTube-risk equivalents
    "रूह कंपा देने वाला",
    "सच्चाई पहली बार",
    "देखकर कांप जाएंगे",
]

# Unverified media authenticity claims — high severity.
# Blocked unless case_glossary marks allow_verified_media_claims=True.
# ("first ever" / "never before" are already caught by _FIRST_CLAIM_PATTERNS.)
_UNVERIFIED_CLAIM_PHRASES = [
    "real voice",
    "real audio",
    "real scream",
    "actual scream",
    "last words",
    "caught on camera",
    "leaked",
    "leaked footage",
    "never seen before",
    "law changed forever",
    # Hindi equivalents
    "असली आवाज़",
    "असली चीख",
    "आख़िरी शब्द",
    "लीक वीडियो",
    "लीक फुटेज",
    "कैमरे में कैद",
    "CCTV में कैद",
]

# Generic graphic/gory content — always high severity regardless of glossary.
# Specific case-level terms belong in case_glossary.do_not_use instead.
_GRAPHIC_CONTENT_PHRASES = [
    "dismembered",
    "body parts scattered",
    "mutilated body",
    "torture footage",
    "graphic violence",
    # Hindi equivalents
    "खून से लथपथ",
    "शरीर के टुकड़े",
    "दरिंदगी की हद",
]

# Sexualized victim framing — always high severity.
_SEXUALIZED_VICTIM_PHRASES = [
    "sexual assault details",
    "rape scene recreation",
    "sexual abuse recreation",
    "sexual violence description",
]

# Child suffering recreation — always high severity.
_CHILD_HARM_PHRASES = [
    "child abuse scene",
    "child suffering recreation",
    "child pain recreation",
]

# Generic reconstruction-like markers. These are intentionally broad and
# case-agnostic; they only become blocking when paired with copied/source-shaped
# wording or unsafe detail, not ordinary factual chronology alone.
_RECONSTRUCTION_MARKERS = [
    "reconstruction",
    "crime",
    "incident",
    "discovery",
    "final_hours",
    "final moments",
    "confession",
    "assault",
    "attack",
    "abduction",
    "captivity",
    "events",
    "interrogation",
    "murder",
    "killing",
    "timeline",
    "what_happened",
    "evidence sequence",
    "court reconstruction",
    "reconstruct",
    "जुर्म",
    "घटना",
    "हमला",
    "हत्या",
    "अपहरण",
    "पूछताछ",
    "स्वीकारोक्ति",
    "समयरेखा",
]

_RECONSTRUCTION_RELATED_MARKERS = [
    "investigation",
    "discovery",
    "reconstruction",
    "confession",
    "interrogation",
    "court",
    "evidence",
    "timeline",
    "aftermath",
    "जाँच",
    "तलाश",
    "अदालत",
    "साक्ष्य",
    "सबूत",
    "समयरेखा",
]

_PLAY_BY_PLAY_MECHANICS_PHRASES = [
    "then he",
    "then she",
    "after that he",
    "after that she",
    "grabbed her",
    "held her down",
    "tied her",
    "dragged her",
    "strangled",
    "stabbed",
    "hit her",
    "hit him",
    "उसके बाद",
    "फिर उसने",
    "खींचा",
    "बाँधा",
    "दबोचा",
    "गला",
    "वार किया",
]

_EXPLICIT_SENSITIVE_VIOLENCE_PHRASES = [
    "rape",
    "raped",
    "sexual assault",
    "sexual violence",
    "molested",
    "undressed",
    "naked body",
    "bound and gagged",
    "hands tied",
    "legs tied",
    "body position",
    "positioned body",
    "बलात्कार",
    "यौन हमला",
    "यौन हिंसा",
    "कपड़े उतारे",
    "नग्न",
    "हाथ बाँध",
    "पैर बाँध",
]

# Shock/clickbait words in thumbnail text — medium severity.
_THUMBNAIL_SHOCK_WORDS: frozenset[str] = frozenset({
    "shocking", "brutal", "graphic", "exposed", "leaked",
    "gruesome", "disturbing", "horror", "explicit", "nsfw",
})

# Tags that signal off-topic keyword stuffing.
# Blocked unless youtube_metadata_rules.allow_unrelated_tags contains the tag.
_UNRELATED_TAGS: frozenset[str] = frozenset({"bollywood", "cricket", "trending", "viral"})

_DESCRIPTION_MIN_WORDS = 100
_THUMBNAIL_TEXT_MIN_WORDS = 2
_THUMBNAIL_TEXT_MAX_WORDS = 5

_METADATA_SAFE_REPLACEMENTS: list[tuple[re.Pattern, str, str]] = [
    (
        re.compile(r"\bmost\s+(?:brutal|chilling|dangerous|shocking|infamous|horrific)\b", re.IGNORECASE),
        "a serious case",
        "Replace unsupported English superlative with factual neutral framing",
    ),
    (
        re.compile(r"सबसे\s+(?:क्रूर|खतरनाक|भयानक|डरावना|कुख्यात|खौफनाक)", re.UNICODE | re.IGNORECASE),
        "एक गंभीर मामला",
        "Replace unsupported Hindi superlative with factual neutral framing",
    ),
    (
        re.compile(r"\b\d+\s+(?:stab\s+wounds?|knife\s+wounds?|injur(?:y|ies)|fractures?|bruises?|cuts?|wounds?)\b", re.IGNORECASE),
        "case evidence",
        "Remove graphic injury-count metadata wording",
    ),
    (
        re.compile(r"\d+\s*(?:चोटें|चोट|घाव|वार|फ्रैक्चर|नील)", re.UNICODE | re.IGNORECASE),
        "मामले के साक्ष्य",
        "Remove graphic injury-count metadata wording",
    ),
    (
        re.compile(r"\b(?:ruptured\s+liver|autopsy\s+details?|body\s+details?|body\s+position|restraint\s+details?)\b", re.IGNORECASE),
        "case evidence",
        "Remove graphic body-detail metadata wording",
    ),
    (
        re.compile(r"(?:फटा\s+हुआ\s+जिगर|पोस्टमॉर्टम\s+का\s+विवरण|शरीर\s+की\s+स्थिति)", re.UNICODE | re.IGNORECASE),
        "मामले के साक्ष्य",
        "Remove graphic body-detail metadata wording",
    ),
    (
        re.compile(r"\b(?:rape|raped|sexual\s+assault|sexual\s+violence|molestation)\b", re.IGNORECASE),
        "sensitive violence",
        "Replace explicit sexual-violence metadata wording with restrained phrasing",
    ),
    (
        re.compile(r"(?:बलात्कार|यौन\s+हमला|यौन\s+हिंसा|यौन\s+उत्पीड़न)", re.UNICODE | re.IGNORECASE),
        "संवेदनशील हिंसा",
        "Replace explicit sexual-violence metadata wording with restrained phrasing",
    ),
    (
        re.compile(r"\b(?:media|journalism|police|court)\s+(?:killed|murdered)\b", re.IGNORECASE),
        "raised serious questions",
        "Replace unsupported legal-blame metadata framing",
    ),
    (
        re.compile(r"(?:मीडिया|पत्रकारिता|पुलिस|अदालत)\s+ने\s+(?:मारा|हत्या\s+की)", re.UNICODE | re.IGNORECASE),
        "ने गंभीर सवाल खड़े किए",
        "Replace unsupported legal-blame metadata framing",
    ),
]

_METADATA_UNSAFE_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (
        re.compile(r"\bmost\s+(?:brutal|chilling|dangerous|shocking|infamous|horrific)\b", re.IGNORECASE),
        "metadata_unsupported_superlative",
        "Unsupported English superlative in metadata.",
    ),
    (
        re.compile(r"सबसे\s+(?:क्रूर|खतरनाक|भयानक|डरावना|कुख्यात|खौफनाक)", re.UNICODE | re.IGNORECASE),
        "metadata_unsupported_superlative",
        "Unsupported Hindi superlative in metadata.",
    ),
    (
        re.compile(r"\b\d+\s+(?:stab\s+wounds?|knife\s+wounds?|injur(?:y|ies)|fractures?|bruises?|cuts?|wounds?)\b", re.IGNORECASE),
        "metadata_graphic_injury_detail",
        "Graphic injury-count wording in metadata.",
    ),
    (
        re.compile(r"\d+\s*(?:चोटें|चोट|घाव|वार|फ्रैक्चर|नील)", re.UNICODE | re.IGNORECASE),
        "metadata_graphic_injury_detail",
        "Graphic injury-count wording in metadata.",
    ),
    (
        re.compile(r"\b(?:rape|raped|sexual\s+assault|sexual\s+violence|molestation)\b|(?:बलात्कार|यौन\s+हमला|यौन\s+हिंसा|यौन\s+उत्पीड़न)", re.IGNORECASE),
        "metadata_explicit_sexual_violence",
        "Explicit sexual-violence wording in metadata.",
    ),
    (
        re.compile(r"\b(?:media|journalism|police|court)\s+(?:killed|murdered)\b|(?:मीडिया|पत्रकारिता|पुलिस|अदालत)\s+ने\s+(?:मारा|हत्या\s+की)", re.IGNORECASE),
        "metadata_unsupported_legal_blame",
        "Unsupported legal-blame wording in metadata.",
    ),
]


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _title_too_long(title: str, max_chars: int) -> bool:
    return len(title) > max_chars


def _chunk_targets_from_issue(chunk_id: str, issue_type: str, problem: str, instruction: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "issue_type": issue_type,
        "problem": problem,
        "repair_instruction": instruction,
    }


def _meta_target(field: str, issue_type: str, problem: str, instruction: str) -> dict:
    return {
        "field": field,
        "issue_type": issue_type,
        "problem": problem,
        "repair_instruction": instruction,
    }


def _metadata_field_items(metadata: Any, path: str = "youtube_metadata") -> list[tuple[str, str]]:
    """Return string metadata fields with stable dotted paths."""
    items: list[tuple[str, str]] = []
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            items.extend(_metadata_field_items(value, f"{path}.{key}"))
    elif isinstance(metadata, list):
        for idx, value in enumerate(metadata):
            items.extend(_metadata_field_items(value, f"{path}[{idx}]"))
    elif isinstance(metadata, str):
        items.append((path, metadata))
    return items


def _apply_metadata_safety_autofix(metadata: Any, path: str = "youtube_metadata") -> tuple[Any, list[dict[str, str]]]:
    """Apply safe deterministic metadata fixes only where replacement is obvious."""
    fixes: list[dict[str, str]] = []

    def _fix_text(value: str, current_path: str) -> str:
        text = value
        for pattern, replacement, description in _METADATA_SAFE_REPLACEMENTS:
            updated, count = pattern.subn(replacement, text)
            if count:
                fixes.append({
                    "field": current_path,
                    "description": description,
                    "occurrences": str(count),
                    "before": text[:180],
                    "after": updated[:180],
                })
                text = updated
        return text

    def _walk(value: Any, current_path: str) -> Any:
        if isinstance(value, dict):
            return {k: _walk(v, f"{current_path}.{k}") for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v, f"{current_path}[{i}]") for i, v in enumerate(value)]
        if isinstance(value, str):
            return _fix_text(value, current_path)
        return value

    return _walk(metadata, path), fixes


def _normalise_for_source_match(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _contains_any(text: str, phrases: list[str]) -> bool:
    text_lower = text.lower()
    return any(p.lower() in text_lower for p in phrases)


def _is_reconstruction_like(chunk: dict[str, Any]) -> bool:
    probe = " ".join(
        str(chunk.get(k, ""))
        for k in ("chunk_id", "section_title", "purpose", "summary", "tone")
    )
    return _contains_any(probe, _RECONSTRUCTION_MARKERS)


def _find_source_english_phrases(text: str, source_transcript: str, min_words: int = 7) -> list[str]:
    """Find English phrases in text that also appear in the source transcript.

    This catches copied documentary quotes/dialogue without case-specific strings.
    Names/dates/locations usually fall below the word threshold and are ignored.
    """
    if not text or not source_transcript:
        return []
    source_norm = _normalise_for_source_match(source_transcript)
    hits: list[str] = []
    for match in re.finditer(r"\b[A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){6,}\b", text):
        phrase = match.group(0).strip()
        words = re.findall(r"[A-Za-z][A-Za-z'’-]*", phrase)
        if len(words) < min_words:
            continue
        phrase_norm = _normalise_for_source_match(phrase)
        if phrase_norm and phrase_norm in source_norm and phrase not in hits:
            hits.append(phrase)
    return hits


def run_python_preflight(
    script_draft: dict[str, Any],
    fact_lock: dict[str, Any],
    case_glossary: dict[str, Any],
    review_dir: Path,
    target_duration_min: int,
    hinglish_level: int,
    label: str = "",
    source_transcript: str = "",
) -> dict[str, Any]:
    """Run deterministic checks and save 04-review/python_preflight_report{label}.json.

    label="" → python_preflight_report.json (initial run)
    label="_after_repair" → python_preflight_report_after_repair.json (post-repair recheck)

    Output shape:
      passed                  bool — True only when no issues at all
      blocking                bool — True when high or medium issues exist
      issues                  list — per-chunk narration issues
      chunk_repair_targets    list — structured targets for Claude repair
      metadata_issues         list — metadata-level issues
      metadata_repair_targets list — structured targets for metadata repair
      severity_counts         dict — {high, medium, low} counts across all issues
      estimated_duration_min  float
      target_duration_min     int
      hinglish_level          int
    """
    chunks = script_draft.get("hindi_narration_chunks", [])
    metadata = script_draft.get("youtube_metadata", {})
    do_not_use: set[str] = set(case_glossary.get("do_not_use", []))
    forbidden_name_variants: list[str] = case_glossary.get("forbidden_name_variants", [])
    allow_first_claim = case_glossary.get("legal_claim_rules", {}).get(
        "allow_first_case_claim", False
    )
    metadata_rules = case_glossary.get("youtube_metadata_rules", {})
    title_max = int(metadata_rules.get("recommended_title_max_chars", 100))
    tags_min = int(metadata_rules.get("tags_min", 15))
    tags_max = int(metadata_rules.get("tags_max", 25))

    allow_verified_media_claims: bool = case_glossary.get("allow_verified_media_claims", False)
    allow_unrelated_tags: set[str] = {
        t.lower() for t in metadata_rules.get("allow_unrelated_tags", [])
    }

    issues: list[dict[str, Any]] = []
    chunk_targets: list[dict[str, str]] = []
    metadata_issues: list[dict[str, str]] = []
    metadata_targets: list[dict[str, str]] = []
    metadata_python_fixes_applied: list[dict[str, str]] = []
    source_shaped_detected = False
    reconstruction_cluster_candidates: list[dict[str, Any]] = []

    # ── Narration chunk checks ────────────────────────────────────────────────
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        text = chunk.get("text", "")
        reconstruction_like = _is_reconstruction_like(chunk)

        # Forbidden terms from case glossary (generic — derived from case data)
        for forbidden in do_not_use:
            if forbidden and forbidden in text:
                problem = f"Forbidden term '{forbidden}' in narration chunk."
                instruction = f"Remove or rephrase '{forbidden}' using appropriate language."
                issues.append({
                    "severity": "medium",
                    "type": "case_glossary",
                    "chunk_id": chunk_id,
                    "problem": problem,
                })
                chunk_targets.append(
                    _chunk_targets_from_issue(chunk_id, "case_glossary", problem, instruction)
                )

        # Forbidden name variants (e.g. wrong alias for a verified person)
        for variant in forbidden_name_variants:
            if variant and variant in text:
                problem = f"Forbidden name variant '{variant}' in narration chunk."
                instruction = (
                    f"Replace '{variant}' with the correct verified name from the glossary."
                )
                issues.append({
                    "severity": "high",
                    "type": "forbidden_name_variant",
                    "chunk_id": chunk_id,
                    "problem": problem,
                })
                chunk_targets.append(
                    _chunk_targets_from_issue(chunk_id, "name_variant", problem, instruction)
                )

        # Unsupported first/never-before legal framing
        if not allow_first_claim and any(p in text for p in _FIRST_CLAIM_PATTERNS):
            problem = "Unsupported first/never-before legal framing."
            instruction = (
                "Remove definitive first/never-before framing. Use qualified language such as "
                "'यह मामला एक महत्वपूर्ण कानूनी मिसाल बना'."
            )
            issues.append({
                "severity": "high",
                "type": "unsupported_legal_claim",
                "chunk_id": chunk_id,
                "problem": problem,
            })
            chunk_targets.append(
                _chunk_targets_from_issue(chunk_id, "safety", problem, instruction)
            )

        # Hinglish leakage — only when hinglish_level is low (≤2)
        if hinglish_level <= 2 and re.search(r"\bmiss\s+कर", text, flags=re.IGNORECASE):
            problem = "Unnecessary English phrase 'miss कर' at Hinglish level 2."
            instruction = (
                "Replace 'miss कर' phrasing with 'याद कर' / 'याद आना' "
                "while keeping natural Hindi."
            )
            issues.append({
                "severity": "medium",
                "type": "hinglish_level",
                "chunk_id": chunk_id,
                "problem": problem,
            })
            chunk_targets.append(
                _chunk_targets_from_issue(chunk_id, "hinglish_level_mismatch", problem, instruction)
            )

        # Generic sensational/clickbait Hindi phrases (always blocked)
        for phrase in _SENSATIONAL_HINDI_PHRASES:
            if phrase in text:
                problem = f"Sensational phrase '{phrase}' in narration — YouTube policy risk."
                instruction = (
                    f"Remove or replace '{phrase}' with factual, dignity-first language."
                )
                issues.append({
                    "severity": "medium",
                    "type": "youtube_safety_phrase",
                    "chunk_id": chunk_id,
                    "problem": problem,
                })
                chunk_targets.append(
                    _chunk_targets_from_issue(chunk_id, "youtube_safety_phrase", problem, instruction)
                )

        # Unverified media authenticity claims
        if not allow_verified_media_claims:
            text_lower = text.lower()
            for phrase in _UNVERIFIED_CLAIM_PHRASES:
                if phrase.lower() in text_lower:
                    problem = f"Unverified media claim '{phrase}' in narration."
                    instruction = (
                        f"Remove '{phrase}' unless this is fact-checked and sourced. "
                        "Set allow_verified_media_claims=True in case_glossary if verified."
                    )
                    issues.append({
                        "severity": "high",
                        "type": "unverified_media_claim",
                        "chunk_id": chunk_id,
                        "problem": problem,
                    })
                    chunk_targets.append(
                        _chunk_targets_from_issue(
                            chunk_id, "unverified_media_claim", problem, instruction
                        )
                    )

        # Graphic/gory content wording (always high severity — YouTube policy)
        text_lower_for_graphic = text.lower()
        for phrase in _GRAPHIC_CONTENT_PHRASES:
            if phrase.lower() in text_lower_for_graphic:
                problem = f"Graphic content phrase '{phrase}' in narration — YouTube policy violation."
                instruction = (
                    f"Remove or rewrite '{phrase}' using respectful, factual language "
                    "that does not dwell on graphic physical detail."
                )
                issues.append({
                    "severity": "high",
                    "type": "graphic_content",
                    "chunk_id": chunk_id,
                    "problem": problem,
                })
                chunk_targets.append(
                    _chunk_targets_from_issue(chunk_id, "graphic_content", problem, instruction)
                )

        # Sexualized victim framing (always high severity)
        for phrase in _SEXUALIZED_VICTIM_PHRASES:
            if phrase.lower() in text_lower_for_graphic:
                problem = f"Sexualized victim framing '{phrase}' in narration."
                instruction = (
                    f"Remove '{phrase}'. Report facts without recreating or describing "
                    "sexual violence in exploitative detail."
                )
                issues.append({
                    "severity": "high",
                    "type": "sexualized_victim_framing",
                    "chunk_id": chunk_id,
                    "problem": problem,
                })
                chunk_targets.append(
                    _chunk_targets_from_issue(
                        chunk_id, "sexualized_victim_framing", problem, instruction
                    )
                )

        # Child harm recreation (always high severity)
        for phrase in _CHILD_HARM_PHRASES:
            if phrase.lower() in text_lower_for_graphic:
                problem = f"Child harm recreation phrase '{phrase}' in narration."
                instruction = (
                    f"Remove '{phrase}'. Child suffering must never be recreated or "
                    "described in graphic detail."
                )
                issues.append({
                    "severity": "high",
                    "type": "child_harm_content",
                    "chunk_id": chunk_id,
                    "problem": problem,
                })
                chunk_targets.append(
                    _chunk_targets_from_issue(
                        chunk_id, "child_harm_content", problem, instruction
                    )
                )

        # Source-shaped reconstruction / exact English source quote risk.
        # This is generic and source-aware: no one case or quote is hardcoded.
        copied_english_phrases = _find_source_english_phrases(text, source_transcript)
        if copied_english_phrases:
            for phrase in copied_english_phrases[:3]:
                is_dramatic_quote = any(q in text for q in ('"', "“", "”", "'"))
                severity = "high" if reconstruction_like or is_dramatic_quote else "medium"
                issue_type = (
                    "source_shaped_reconstruction"
                    if reconstruction_like
                    else "exact_source_quote_copy"
                )
                problem = (
                    "English source phrase copied into narration"
                    + (" inside a reconstruction-like chunk" if reconstruction_like else "")
                    + f": '{phrase[:160]}'"
                )
                instruction = (
                    "Translate/paraphrase this into original Hindi. If legally important, "
                    "use legal/court/evidence framing. Do not preserve exact English source "
                    "wording unless it is a short unavoidable legal phrase."
                )
                issues.append({
                    "severity": severity,
                    "type": issue_type,
                    "chunk_id": chunk_id,
                    "problem": problem,
                    "source_phrase": phrase,
                    "reconstruction_like": reconstruction_like,
                })
                chunk_targets.append(
                    _chunk_targets_from_issue(chunk_id, issue_type, problem, instruction)
                )
                if reconstruction_like:
                    source_shaped_detected = True
                    reconstruction_cluster_candidates.append({
                        "chunk_id": chunk_id,
                        "issue_type": issue_type,
                        "source_phrase": phrase,
                    })

        # Unsafe reconstruction mechanics: do not block factual chronology alone,
        # only reconstruction-like chunks with play-by-play/physical detail.
        lower_text = text.lower()
        has_play_by_play = _contains_any(lower_text, _PLAY_BY_PLAY_MECHANICS_PHRASES)
        has_explicit_sensitive = _contains_any(lower_text, _EXPLICIT_SENSITIVE_VIOLENCE_PHRASES)
        if reconstruction_like and (has_play_by_play or has_explicit_sensitive):
            source_shaped_detected = True
            severity = "critical" if has_explicit_sensitive else "high"
            issue_type = (
                "explicit_sensitive_violence"
                if has_explicit_sensitive
                else "source_shaped_reconstruction"
            )
            problem = (
                "Reconstruction-like chunk uses play-by-play crime mechanics or explicit "
                "physical/sensitive detail instead of restrained documentary framing."
            )
            instruction = (
                "Rebuild this section as original Hindi documentary narration using "
                "legal/evidence framing. Do not follow source sequence mechanically, "
                "do not recreate crime mechanics, and soften explicit body/violence wording."
            )
            issues.append({
                "severity": severity,
                "type": issue_type,
                "chunk_id": chunk_id,
                "problem": problem,
                "reconstruction_like": True,
            })
            chunk_targets.append(
                _chunk_targets_from_issue(chunk_id, issue_type, problem, instruction)
            )
            reconstruction_cluster_candidates.append({
                "chunk_id": chunk_id,
                "issue_type": issue_type,
            })

    # ── Metadata checks ───────────────────────────────────────────────────────
    original_metadata_text = json.dumps(metadata, ensure_ascii=False)

    # First normalize obvious unsafe/clickbait metadata without spending model
    # calls. Complex cases still become repair targets below.
    if isinstance(metadata, dict):
        fixed_metadata, metadata_python_fixes_applied = _apply_metadata_safety_autofix(metadata)
        if metadata_python_fixes_applied:
            script_draft["youtube_metadata"] = fixed_metadata
            metadata = fixed_metadata
            metadata_issues.append({
                "severity": "low",
                "type": "metadata_python_autofix_applied",
                "problem": (
                    f"Applied {len(metadata_python_fixes_applied)} deterministic metadata safety fix(es)."
                ),
            })

    recommended_title = metadata.get("recommended_title", "")
    if _title_too_long(recommended_title, title_max):
        problem = f"recommended_title is {len(recommended_title)} chars; max is {title_max}."
        instruction = f"Shorten recommended_title to at most {title_max} characters."
        metadata_issues.append({
            "severity": "medium",
            "type": "title_length",
            "problem": problem,
        })
        metadata_targets.append(
            _meta_target("recommended_title", "title_length", problem, instruction)
        )

    metadata_text = json.dumps(metadata, ensure_ascii=False)

    # Generic unsafe metadata patterns that remain after deterministic cleanup.
    # These are blocking because they either require context-aware repair or
    # indicate the simple replacement did not fully sanitize the field.
    for field_path, field_text in _metadata_field_items(metadata):
        for pattern, issue_type, issue_problem in _METADATA_UNSAFE_PATTERNS:
            if pattern.search(field_text):
                problem = f"{issue_problem} Field: {field_path}."
                instruction = (
                    "Remove graphic/clickbait/legal-blame wording and use factual, "
                    "respectful documentary framing. Do not include injury counts, "
                    "body-detail phrases, unsupported superlatives, or explicit "
                    "sexual-violence terms in title, tags, thumbnail, or description."
                )
                metadata_issues.append({
                    "severity": "high",
                    "type": issue_type,
                    "field": field_path,
                    "problem": problem,
                })
                metadata_targets.append(
                    _meta_target(field_path, issue_type, problem, instruction)
                )

    # Forbidden terms in metadata
    for forbidden in do_not_use:
        if forbidden and (forbidden in metadata_text or forbidden in original_metadata_text):
            problem = f"Metadata contains forbidden term: {forbidden}"
            instruction = f"Remove '{forbidden}' from all metadata fields."
            fixed_by_python = forbidden not in metadata_text and forbidden in original_metadata_text
            metadata_issues.append({
                "severity": "low" if fixed_by_python else "medium",
                "type": "metadata_forbidden_term",
                "problem": (
                    f"{problem} Deterministically fixed before AI repair."
                    if fixed_by_python else problem
                ),
            })
            if not fixed_by_python:
                metadata_targets.append(
                    _meta_target("youtube_metadata", "metadata_forbidden_term", problem, instruction)
                )

    # Unsupported first-claim in metadata
    if not allow_first_claim and any(p in metadata_text for p in _FIRST_CLAIM_PATTERNS):
        problem = "Metadata uses unsupported first/never-before legal framing."
        instruction = "Remove or qualify first/never-before framing from metadata fields."
        metadata_issues.append({
            "severity": "high",
            "type": "metadata_unsupported_legal_claim",
            "problem": problem,
        })
        metadata_targets.append(
            _meta_target("youtube_metadata", "metadata_unsupported_legal_claim", problem, instruction)
        )

    # Forbidden name variants in metadata
    for variant in forbidden_name_variants:
        if variant and variant in metadata_text:
            problem = f"Metadata contains forbidden name variant: {variant}"
            instruction = f"Replace '{variant}' with the verified name spelling in all metadata fields."
            metadata_issues.append({
                "severity": "high",
                "type": "metadata_forbidden_name_variant",
                "problem": problem,
            })
            metadata_targets.append(
                _meta_target("youtube_metadata", "metadata_forbidden_name_variant", problem, instruction)
            )

    tags = metadata.get("tags", [])
    if tags and not (tags_min <= len(tags) <= tags_max):
        problem = f"Metadata has {len(tags)} tags; expected {tags_min}–{tags_max}."
        instruction = f"Adjust tags list to between {tags_min} and {tags_max} entries."
        metadata_issues.append({
            "severity": "medium",
            "type": "tag_count",
            "problem": problem,
        })
        metadata_targets.append(
            _meta_target("tags", "tag_count", problem, instruction)
        )

    # Duplicate tags
    if tags:
        seen: set[str] = set()
        duplicates: list[str] = []
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in seen:
                duplicates.append(tag)
            seen.add(tag_lower)
        if duplicates:
            problem = f"Duplicate tags detected: {', '.join(duplicates[:5])}."
            instruction = "Remove duplicate tags so each tag appears at most once."
            metadata_issues.append({
                "severity": "medium",
                "type": "duplicate_tags",
                "problem": problem,
            })
            metadata_targets.append(_meta_target("tags", "duplicate_tags", problem, instruction))

    # Unrelated tags (keyword stuffing)
    if tags:
        bad_tags = [
            t for t in tags
            if t.lower() in _UNRELATED_TAGS and t.lower() not in allow_unrelated_tags
        ]
        if bad_tags:
            problem = f"Off-topic tags detected (keyword stuffing): {', '.join(bad_tags)}."
            instruction = (
                "Remove tags unrelated to the case. "
                "Add them to youtube_metadata_rules.allow_unrelated_tags only if genuinely relevant."
            )
            metadata_issues.append({
                "severity": "medium",
                "type": "unrelated_tags",
                "problem": problem,
            })
            metadata_targets.append(_meta_target("tags", "unrelated_tags", problem, instruction))

    # Thumbnail text word count (2–5 words) and shock/clickbait word detection
    for thumb in metadata.get("thumbnail_options", []):
        thumb_text = thumb.get("thumbnail_text", "")
        if thumb_text:
            wc = _word_count(thumb_text)
            if not (_THUMBNAIL_TEXT_MIN_WORDS <= wc <= _THUMBNAIL_TEXT_MAX_WORDS):
                problem = (
                    f"thumbnail_text '{thumb_text}' has {wc} word(s); "
                    f"expected {_THUMBNAIL_TEXT_MIN_WORDS}–{_THUMBNAIL_TEXT_MAX_WORDS}."
                )
                instruction = (
                    f"Rewrite thumbnail text to {_THUMBNAIL_TEXT_MIN_WORDS}–"
                    f"{_THUMBNAIL_TEXT_MAX_WORDS} punchy words."
                )
                metadata_issues.append({
                    "severity": "medium",
                    "type": "thumbnail_text_length",
                    "problem": problem,
                })
                metadata_targets.append(
                    _meta_target("thumbnail_options", "thumbnail_text_length", problem, instruction)
                )
            # Shock/clickbait word detection in thumbnail
            thumb_words_lower = {w.lower().strip(".,!?") for w in thumb_text.split()}
            shock_found = thumb_words_lower & _THUMBNAIL_SHOCK_WORDS
            if shock_found:
                problem = (
                    f"Thumbnail text contains shock/clickbait words: {', '.join(sorted(shock_found))}."
                )
                instruction = (
                    "Remove shock words from thumbnail text. Use factual, dignity-first language."
                )
                metadata_issues.append({
                    "severity": "medium",
                    "type": "thumbnail_shock_word",
                    "problem": problem,
                })
                metadata_targets.append(
                    _meta_target("thumbnail_options", "thumbnail_shock_word", problem, instruction)
                )

    # Unverified media authenticity claims in metadata (mirrors narration check)
    if not allow_verified_media_claims:
        meta_text_lower_claims = metadata_text.lower()
        for phrase in _UNVERIFIED_CLAIM_PHRASES:
            if phrase.lower() in meta_text_lower_claims:
                problem = f"Metadata contains unverified media claim: '{phrase}'."
                instruction = (
                    f"Remove '{phrase}' from all metadata fields unless fact-checked and sourced. "
                    "Set allow_verified_media_claims=True in case_glossary if verified."
                )
                metadata_issues.append({
                    "severity": "high",
                    "type": "unverified_media_claim_metadata",
                    "problem": problem,
                })
                metadata_targets.append(
                    _meta_target(
                        "youtube_metadata", "unverified_media_claim_metadata", problem, instruction
                    )
                )

    # Graphic content / sexualized framing / child harm in metadata text
    meta_text_lower = metadata_text.lower()
    for phrase in _GRAPHIC_CONTENT_PHRASES + _SEXUALIZED_VICTIM_PHRASES + _CHILD_HARM_PHRASES:
        if phrase.lower() in meta_text_lower:
            problem = f"Metadata contains graphic/sensitive phrase: '{phrase}'."
            instruction = (
                f"Remove or rewrite '{phrase}' in metadata using factual, respectful language."
            )
            metadata_issues.append({
                "severity": "high",
                "type": "graphic_content_metadata",
                "problem": problem,
            })
            metadata_targets.append(
                _meta_target("youtube_metadata", "graphic_content_metadata", problem, instruction)
            )

    # Description word count
    description = metadata.get("description", "")
    if description:
        desc_wc = _word_count(description)
        if desc_wc < _DESCRIPTION_MIN_WORDS:
            problem = (
                f"description has {desc_wc} word(s); "
                f"minimum is {_DESCRIPTION_MIN_WORDS} for YouTube SEO."
            )
            instruction = (
                f"Expand description to at least {_DESCRIPTION_MIN_WORDS} words with "
                "relevant case details, timestamps, and a call-to-action."
            )
            metadata_issues.append({
                "severity": "medium",
                "type": "description_too_short",
                "problem": problem,
            })
            metadata_targets.append(
                _meta_target("description", "description_too_short", problem, instruction)
            )

    # Pinned comment
    pinned_comment = metadata.get("pinned_comment", "")
    if not pinned_comment or not pinned_comment.strip():
        metadata_issues.append({
            "severity": "low",
            "type": "pinned_comment_missing",
            "problem": "pinned_comment is absent or empty.",
        })
        # low severity — no repair target needed

    # Sensational/clickbait phrases in metadata text
    for phrase in _SENSATIONAL_HINDI_PHRASES:
        if phrase in metadata_text:
            problem = f"Sensational phrase '{phrase}' in metadata — YouTube policy risk."
            instruction = f"Remove or rephrase '{phrase}' from all metadata fields."
            metadata_issues.append({
                "severity": "medium",
                "type": "youtube_safety_phrase",
                "problem": problem,
            })
            metadata_targets.append(
                _meta_target("youtube_metadata", "youtube_safety_phrase", problem, instruction)
            )

    estimated_duration = round(
        sum(_word_count(c.get("text", "")) for c in chunks) / settings.hindi_narration_wpm,
        1,
    )
    chapters = metadata.get("chapters", [])
    if chapters:
        metadata_issues.append({
            "severity": "low",
            "type": "chapters_before_audio",
            "problem": (
                f"Chapters exist before final audio timing. Estimated narration duration is "
                f"{estimated_duration} min; verify timestamps after ElevenLabs."
            ),
        })
        # chapters_before_audio is low severity — no metadata_repair_target needed

    # ── Severity aggregation ──────────────────────────────────────────────────
    all_issues = issues + metadata_issues
    severity_counts = {
        "critical": sum(1 for i in all_issues if i.get("severity") == "critical"),
        "high":   sum(1 for i in all_issues if i.get("severity") == "high"),
        "medium": sum(1 for i in all_issues if i.get("severity") == "medium"),
        "low":    sum(1 for i in all_issues if i.get("severity") == "low"),
    }
    blocking = (
        severity_counts["critical"] > 0
        or severity_counts["high"] > 0
        or severity_counts["medium"] > 0
    )
    passed = len(all_issues) == 0

    report = {
        "passed": passed,
        "blocking": blocking,
        "issues": issues,
        "chunk_repair_targets": _dedupe_targets(chunk_targets),
        "metadata_issues": metadata_issues,
        "metadata_repair_targets": metadata_targets,
        "metadata_python_fixes_applied": metadata_python_fixes_applied,
        "severity_counts": severity_counts,
        "estimated_duration_min": estimated_duration,
        "target_duration_min": target_duration_min,
        "hinglish_level": hinglish_level,
        "source_shaped_reconstruction_detected": source_shaped_detected,
        "reconstruction_cluster_candidates": reconstruction_cluster_candidates,
    }

    review_dir.mkdir(parents=True, exist_ok=True)
    filename = f"python_preflight_report{label}.json"
    (review_dir / filename).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _dedupe_targets(targets: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for target in targets:
        key = (target.get("chunk_id", ""), target.get("issue_type", ""))
        if key in by_key:
            by_key[key]["problem"] += " | " + target.get("problem", "")
            by_key[key]["repair_instruction"] += " | " + target.get("repair_instruction", "")
        else:
            by_key[key] = dict(target)
    return list(by_key.values())
