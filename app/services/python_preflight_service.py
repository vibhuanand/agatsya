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


def run_python_preflight(
    script_draft: dict[str, Any],
    fact_lock: dict[str, Any],
    case_glossary: dict[str, Any],
    review_dir: Path,
    target_duration_min: int,
    hinglish_level: int,
    label: str = "",
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

    # ── Narration chunk checks ────────────────────────────────────────────────
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        text = chunk.get("text", "")

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

    # ── Metadata checks ───────────────────────────────────────────────────────
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

    # Forbidden terms in metadata
    for forbidden in do_not_use:
        if forbidden and forbidden in metadata_text:
            problem = f"Metadata contains forbidden term: {forbidden}"
            instruction = f"Remove '{forbidden}' from all metadata fields."
            metadata_issues.append({
                "severity": "medium",
                "type": "metadata_forbidden_term",
                "problem": problem,
            })
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
        "high":   sum(1 for i in all_issues if i.get("severity") == "high"),
        "medium": sum(1 for i in all_issues if i.get("severity") == "medium"),
        "low":    sum(1 for i in all_issues if i.get("severity") == "low"),
    }
    blocking = severity_counts["high"] > 0 or severity_counts["medium"] > 0
    passed = len(all_issues) == 0

    report = {
        "passed": passed,
        "blocking": blocking,
        "issues": issues,
        "chunk_repair_targets": _dedupe_targets(chunk_targets),
        "metadata_issues": metadata_issues,
        "metadata_repair_targets": metadata_targets,
        "severity_counts": severity_counts,
        "estimated_duration_min": estimated_duration,
        "target_duration_min": target_duration_min,
        "hinglish_level": hinglish_level,
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
