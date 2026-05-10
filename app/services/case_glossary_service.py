"""Case glossary builder.

Deterministic, zero-model-cost stage that turns Fact Lock + Story Blueprint
into a compact writing constraint sheet for downstream agents.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_BASE_PREFERRED_TERMS = {
    "investigation": "जाँच",
    "evidence": "साक्ष्य / सबूत",
    "trial": "मुक़दमा / सुनवाई",
    "appeal": "अपील",
    "parole": "पैरोल / रिहाई की संभावना",
    "custody": "अभिरक्षा / बच्चों की देखभाल का विवाद",
    "victim": "पीड़िता / मृतका",
    "suspect": "संदिग्ध",
    "accused": "आरोपी",
    "court": "अदालत / न्यायालय",
    "sentence": "सज़ा",
    "conviction": "दोषसिद्धि",
    "confession": "स्वीकारोक्ति / इक़बालिया बयान",
    "unlawful confinement": "ग़ैरक़ानूनी क़ैद",
    "first-degree murder": "प्रथम-श्रेणी की हत्या",
    "second-degree murder": "द्वितीय-श्रेणी की हत्या",
}


_BASE_DO_NOT_USE = [
    "भारत की पहली",
    "सबसे भयानक",
    "सबसे दर्दनाक",
    "आप यकीन नहीं करेंगे",
    "पहली बार",
    "पहले कभी नहीं",
]


def _contains(text: str, *needles: str) -> bool:
    lowered = text.lower()
    return any(n.lower() in lowered for n in needles)


def _verified_names(fact_lock: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for person in fact_lock.get("verified_people", []):
        name = person.get("name", "")
        if name and name not in names:
            names.append(name)
    return names


def _has_high_confidence_first_claim(fact_lock: dict[str, Any]) -> bool:
    """Return true only when source facts explicitly support first-time framing."""
    sources: list[str] = []
    for item in fact_lock.get("verified_timeline", []):
        if item.get("confidence") == "high":
            sources.append(item.get("event", ""))
            sources.append(item.get("source_phrase", ""))
    legal = fact_lock.get("legal_outcome", {})
    if legal.get("confidence") == "high":
        sources.extend(str(v) for v in legal.values() if isinstance(v, str))
    joined = " ".join(sources)
    return _contains(joined, "first", "पहली बार", "पहला मामला", "पहले कभी")


def build_case_glossary(
    fact_lock: dict[str, Any],
    blueprint: dict[str, Any],
    facts_dir: Path,
) -> dict[str, Any]:
    """Build and save 02-facts/case_glossary.json."""
    text_blob = json.dumps(
        {"fact_lock": fact_lock, "blueprint": blueprint},
        ensure_ascii=False,
    )
    preferred_terms = dict(_BASE_PREFERRED_TERMS)
    do_not_use = list(_BASE_DO_NOT_USE)

    if _contains(text_blob, "ladybug", "ladybugs", "लेडीबग"):
        preferred_terms["ladybug"] = "लेडीबग"
        preferred_terms["ladybugs"] = "लेडीबग"
        do_not_use.extend(["झींगुर", "तितलियाँ"])

    verified_names = _verified_names(fact_lock)
    lower_names = {n.lower() for n in verified_names}
    if "kyla woodhouse" in lower_names:
        do_not_use.append("Kyla Jordan")

    legal_claim_rules = {
        "allow_first_case_claim": _has_high_confidence_first_claim(fact_lock),
        "safe_legal_framing": [
            "यह मामला एक महत्वपूर्ण कानूनी मिसाल बना",
            "इस मामले ने कनाडा के क़ानूनी विमर्श में अहम जगह बनाई",
            "अदालत ने इस मामले में ग़ैरक़ानूनी क़ैद और हत्या के संबंध को गंभीरता से देखा",
        ],
        "avoid_unless_high_confidence": [
            "पहला मामला",
            "पहली बार",
            "पहले कभी नहीं",
            "कानून बदल दिया",
        ],
    }

    glossary = {
        "preferred_terms": preferred_terms,
        "do_not_use": sorted(set(do_not_use)),
        "verified_name_spellings": verified_names,
        "legal_claim_rules": legal_claim_rules,
        "youtube_metadata_rules": {
            "recommended_title_max_chars": 100,
            "tags_min": 15,
            "tags_max": 25,
            "thumbnail_text_words_min": 2,
            "thumbnail_text_words_max": 5,
            "chapters_before_audio": "estimated_only",
        },
        "safety_rules": [
            "Child harm must be restrained, legal/forensic, and non-graphic.",
            "Do not describe child suffering sounds.",
            "Do not use sensational titles or unsupported superlatives.",
            "Use victim dignity first; remember the person, not only the crime.",
        ],
    }

    facts_dir.mkdir(parents=True, exist_ok=True)
    (facts_dir / "case_glossary.json").write_text(
        json.dumps(glossary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return glossary

