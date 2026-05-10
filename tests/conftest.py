"""Shared pytest fixtures for Agatsya Automation test suite.

All fixtures load static JSON from tests/fixtures/ — no API calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def fact_lock_generic() -> dict:
    return json.loads((FIXTURES_DIR / "fact_lock_generic.json").read_text(encoding="utf-8"))


@pytest.fixture
def blueprint_generic() -> dict:
    return json.loads((FIXTURES_DIR / "blueprint_generic.json").read_text(encoding="utf-8"))


def _make_chunk(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "section_title": "Test Section",
        "voice": "narrator",
        "tone": "grave",
        "estimated_words": len(text.split()),
        "text": text,
    }


def _make_metadata(
    title: str = "देविका राठी हत्याकांड — न्याय की कहानी",
    tags: list | None = None,
    extra: dict | None = None,
) -> dict:
    base = {
        "recommended_title": title,
        "title_options": ["Option A", "Option B", "Option C"],
        # 100+ words so description_too_short check passes in clean-script tests.
        "description": (
            "यह कहानी देविका राठी की है जो नागपुर की एक साहसी युवती थी। "
            "उनकी हत्या ने पूरे शहर को हिलाकर रख दिया। "
            "न्याय की लंबी और कठिन लड़ाई शुरू हुई। "
            "डीएनए साक्ष्य और सीसीटीवी फुटेज ने इस मामले में महत्वपूर्ण भूमिका निभाई। "
            "अदालत ने आरोपी प्रकाश सोनी को दोषी ठहराया और उचित सजा सुनाई। "
            "यह मामला भारतीय न्यायपालिका के लिए एक महत्वपूर्ण मिसाल बन गया। "
            "परिवार के लिए यह संघर्ष बहुत लंबा और कठिन था। "
            "लेकिन अंत में सच्चाई और न्याय की जीत हुई। "
            "इस सच्ची घटना पर आधारित कहानी में देखिए कैसे साक्ष्य और कानूनी प्रक्रिया "
            "ने एक परिवार को न्याय दिलाया। "
            "सच जानने के लिए पूरा वीडियो अवश्य देखें।"
        ),
        "tags": tags if tags is not None else [
            "देविका राठी", "नागपुर", "हत्याकांड", "true crime Hindi",
            "Hindi crime", "सच्ची घटना", "प्रकाश सोनी", "Devika Rathi",
            "Nagpur murder", "Hindi true crime", "DNA evidence", "CCTV",
            "justice", "murder", "conviction",
        ],
        "thumbnail_options": [
            {"thumbnail_text": "देविका का सच", "angle": "Justice theme"}
        ],
        "chapters": [],
        "pinned_comment": "नमन।",
    }
    if extra:
        base.update(extra)
    return base


def _make_script(chunks: list[dict], metadata: dict | None = None) -> dict:
    return {
        "hindi_narration_chunks": chunks,
        "youtube_metadata": metadata or _make_metadata(),
    }


def _make_glossary(
    do_not_use: list[str] | None = None,
    allow_first_claim: bool = False,
    extra_preferred: dict | None = None,
) -> dict:
    return {
        "preferred_terms": extra_preferred or {},
        "do_not_use": do_not_use if do_not_use is not None else [
            "सबसे भयानक", "सबसे दर्दनाक", "आप यकीन नहीं करेंगे",
            "भारत की पहली",
        ],
        "verified_name_spellings": ["Devika Rathi", "Prakash Soni", "Justice Arvind Nair"],
        "forbidden_name_variants": [],
        "legal_claim_rules": {
            "allow_first_case_claim": allow_first_claim,
            "safe_legal_framing": [
                "यह मामला एक महत्वपूर्ण कानूनी मिसाल बना",
            ],
            "avoid_unless_high_confidence": [
                "पहला मामला", "पहली बार", "पहले कभी नहीं",
            ],
        },
        "youtube_metadata_rules": {
            "recommended_title_max_chars": 100,
            "tags_min": 15,
            "tags_max": 25,
        },
        "safety_rules": [
            "Do not use sensational titles.",
            "Use victim dignity first.",
        ],
    }


# Export helpers so individual test files can import from conftest
pytest.make_chunk = _make_chunk
pytest.make_metadata = _make_metadata
pytest.make_script = _make_script
pytest.make_glossary = _make_glossary
