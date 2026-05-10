"""Case glossary builder.

Deterministic, zero-model-cost stage that turns Fact Lock + Story Blueprint
into a compact writing constraint sheet for downstream agents.

All restrictions are derived from the case data (fact_lock + blueprint).
No case-specific names, motifs, or objects are hardcoded here.
"""
from __future__ import annotations

import json
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


def _forbidden_name_variants(fact_lock: dict[str, Any]) -> list[str]:
    """Collect forbidden name variants from fact_lock.verified_people.

    Each person may optionally carry a 'forbidden_name_variants' list —
    alternate spellings or aliases that must not appear in the script.
    """
    variants: list[str] = []
    for person in fact_lock.get("verified_people", []):
        for v in person.get("forbidden_name_variants", []):
            if v and v not in variants:
                variants.append(v)
    return variants


def _has_high_confidence_first_claim(fact_lock: dict[str, Any]) -> bool:
    """Return True only when source facts explicitly support first-time framing.

    Searches for explicit "first case / पहली बार / पहला मामला" phrasing in
    high-confidence timeline events and source phrases. Legal charge labels
    like 'first-degree murder' do NOT count as evidence of a first-time claim.
    """
    _FIRST_CLAIM_PHRASES = ["पहली बार", "पहला मामला", "पहले कभी", "first ever", "never before"]
    sources: list[str] = []
    for item in fact_lock.get("verified_timeline", []):
        if item.get("confidence") == "high":
            sources.append(item.get("source_phrase", ""))
            # Only use event text if it explicitly contains a first-claim phrase
            evt = item.get("event", "")
            if any(p.lower() in evt.lower() for p in _FIRST_CLAIM_PHRASES):
                sources.append(evt)
    legal = fact_lock.get("legal_outcome", {})
    if legal.get("confidence") == "high":
        # Exclude charge field — "first-degree murder" is a charge type, not a
        # first-ever-case claim; checking it causes false positives.
        for k, v in legal.items():
            if k != "charge" and isinstance(v, str):
                sources.append(v)
    joined = " ".join(sources)
    return _contains(joined, *_FIRST_CLAIM_PHRASES)


def _motif_constraints_from_blueprint(
    blueprint: dict[str, Any],
    preferred_terms: dict[str, str],
    do_not_use: list[str],
) -> dict[str, Any]:
    """Derive motif constraints from blueprint.motif_terms (if present).

    blueprint.motif_terms is an optional list of dicts:
      {"english": "ladybug", "preferred_hindi": "लेडीबग", "forbidden_hindi": ["झींगुर", "तितलियाँ"]}

    This keeps motif restrictions case-driven, not hardcoded.
    """
    constraints: dict[str, Any] = {}
    for motif in blueprint.get("motif_terms", []):
        english = motif.get("english", "")
        preferred_hindi = motif.get("preferred_hindi", "")
        forbidden_hindi: list[str] = motif.get("forbidden_hindi", [])

        if english and preferred_hindi:
            preferred_terms[english] = preferred_hindi

        for term in forbidden_hindi:
            if term and term not in do_not_use:
                do_not_use.append(term)

        if english:
            constraints[english] = {
                "preferred_hindi": preferred_hindi,
                "forbidden_hindi": forbidden_hindi,
            }
    return constraints


def build_case_glossary(
    fact_lock: dict[str, Any],
    blueprint: dict[str, Any],
    facts_dir: Path,
) -> dict[str, Any]:
    """Build and save 02-facts/case_glossary.json.

    All restrictions are derived from fact_lock and blueprint data.
    No case-specific names, motifs, or objects are hardcoded.
    """
    preferred_terms = dict(_BASE_PREFERRED_TERMS)
    do_not_use = list(_BASE_DO_NOT_USE)

    # Motif constraints — driven by blueprint.motif_terms (case-specific data)
    motif_constraints = _motif_constraints_from_blueprint(blueprint, preferred_terms, do_not_use)

    # Verified names from fact_lock
    verified_names = _verified_names(fact_lock)

    # Forbidden name variants — from fact_lock.verified_people[].forbidden_name_variants
    forbidden_variants = _forbidden_name_variants(fact_lock)
    for v in forbidden_variants:
        if v not in do_not_use:
            do_not_use.append(v)

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
        "forbidden_name_variants": forbidden_variants,
        "motif_constraints": motif_constraints,
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
