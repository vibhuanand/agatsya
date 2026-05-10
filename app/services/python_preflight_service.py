"""Python preflight gate for zero-cost script/metadata checks.

Runs before expensive final model gates. It catches deterministic issues that
Claude/OpenAI should not be paid to rediscover: wrong motif terms, unsupported
legal superlatives, metadata shape problems, and known Hinglish-level leaks.
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
]


_GRAPHIC_CHILD_TERMS = [
    "फफोला",
    "काला या सफेद",
    "काला या सफ़ेद",
    "पूरे हाथ",
    "30 सेकंड",
    "तीसरे दर्जे की जलन",
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


def run_python_preflight(
    script_draft: dict[str, Any],
    fact_lock: dict[str, Any],
    case_glossary: dict[str, Any],
    review_dir: Path,
    target_duration_min: int,
    hinglish_level: int,
) -> dict[str, Any]:
    """Run deterministic checks and save 04-review/python_preflight_report.json."""
    chunks = script_draft.get("hindi_narration_chunks", [])
    metadata = script_draft.get("youtube_metadata", {})
    preferred = case_glossary.get("preferred_terms", {})
    do_not_use = set(case_glossary.get("do_not_use", []))
    allow_first_claim = case_glossary.get("legal_claim_rules", {}).get(
        "allow_first_case_claim", False
    )
    metadata_rules = case_glossary.get("youtube_metadata_rules", {})
    title_max = int(metadata_rules.get("recommended_title_max_chars", 100))
    tags_min = int(metadata_rules.get("tags_min", 15))
    tags_max = int(metadata_rules.get("tags_max", 25))

    issues: list[dict[str, Any]] = []
    chunk_targets: list[dict[str, str]] = []
    metadata_issues: list[dict[str, str]] = []

    # Narration checks
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        text = chunk.get("text", "")

        if "झींगुर" in text or "तितल" in text:
            problem = "Ladybug motif mistranslated as झींगुर/तितली."
            instruction = "Replace incorrect insect words with लेडीबग consistently."
            issues.append({"severity": "high", "type": "case_glossary", "chunk_id": chunk_id, "problem": problem})
            chunk_targets.append(_chunk_targets_from_issue(chunk_id, "hindi_naturalness", problem, instruction))

        if hinglish_level <= 2 and re.search(r"\bmiss\s+कर", text, flags=re.IGNORECASE):
            problem = "Unnecessary English phrase 'miss कर' at Hinglish level 2."
            instruction = "Replace 'miss कर' phrasing with 'याद कर' / 'याद आना' while keeping natural Hindi."
            issues.append({"severity": "medium", "type": "hinglish_level", "chunk_id": chunk_id, "problem": problem})
            chunk_targets.append(_chunk_targets_from_issue(chunk_id, "hinglish_level_mismatch", problem, instruction))

        if not allow_first_claim and any(p in text for p in _FIRST_CLAIM_PATTERNS):
            problem = "Unsupported first/never-before legal framing."
            instruction = (
                "Remove definitive first/never-before framing. Use qualified language such as "
                "'यह मामला एक महत्वपूर्ण कानूनी मिसाल बना'."
            )
            issues.append({"severity": "high", "type": "unsupported_legal_claim", "chunk_id": chunk_id, "problem": problem})
            chunk_targets.append(_chunk_targets_from_issue(chunk_id, "safety", problem, instruction))

        if any(term in text for term in _GRAPHIC_CHILD_TERMS):
            problem = "Potentially graphic child-harm phrasing."
            instruction = "Rewrite in restrained legal/forensic language without visual injury detail."
            issues.append({"severity": "medium", "type": "youtube_safety", "chunk_id": chunk_id, "problem": problem})
            chunk_targets.append(_chunk_targets_from_issue(chunk_id, "safety", problem, instruction))

    # Metadata checks
    recommended_title = metadata.get("recommended_title", "")
    if _title_too_long(recommended_title, title_max):
        metadata_issues.append({
            "severity": "medium",
            "type": "title_length",
            "problem": f"recommended_title is {len(recommended_title)} chars; max is {title_max}.",
        })

    metadata_text = json.dumps(metadata, ensure_ascii=False)
    if "झींगुर" in metadata_text or "तितल" in metadata_text:
        metadata_issues.append({
            "severity": "high",
            "type": "metadata_motif",
            "problem": "Metadata uses wrong motif term for ladybug.",
        })

    if not allow_first_claim and any(p in metadata_text for p in _FIRST_CLAIM_PATTERNS):
        metadata_issues.append({
            "severity": "high",
            "type": "metadata_unsupported_legal_claim",
            "problem": "Metadata uses unsupported first/never-before legal framing.",
        })

    for forbidden in do_not_use:
        if forbidden and forbidden in metadata_text:
            metadata_issues.append({
                "severity": "medium",
                "type": "metadata_forbidden_term",
                "problem": f"Metadata contains forbidden term: {forbidden}",
            })

    tags = metadata.get("tags", [])
    if tags and not (tags_min <= len(tags) <= tags_max):
        metadata_issues.append({
            "severity": "medium",
            "type": "tag_count",
            "problem": f"Metadata has {len(tags)} tags; expected {tags_min}-{tags_max}.",
        })

    estimated_duration = round(
        sum(_word_count(c.get("text", "")) for c in chunks) / settings.hindi_narration_wpm,
        1,
    )
    chapters = metadata.get("chapters", [])
    if chapters:
        # If chapters contain timestamps beyond estimated duration, flag as estimated-only risk.
        metadata_issues.append({
            "severity": "low",
            "type": "chapters_before_audio",
            "problem": (
                f"Chapters exist before final audio timing. Estimated narration duration is "
                f"{estimated_duration} min; verify timestamps after ElevenLabs."
            ),
        })

    report = {
        "passed": not issues and not metadata_issues,
        "issues": issues,
        "chunk_repair_targets": _dedupe_targets(chunk_targets),
        "metadata_issues": metadata_issues,
        "estimated_duration_min": estimated_duration,
        "target_duration_min": target_duration_min,
        "hinglish_level": hinglish_level,
        "notes": [
            "Deterministic preflight runs before final premium gates.",
            "If this report fails, repair should happen before OpenAI final review.",
        ],
    }

    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "python_preflight_report.json").write_text(
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

