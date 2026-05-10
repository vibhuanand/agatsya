"""Tests for case_glossary_service.build_case_glossary().

Verifies generic behavior — no case-specific hardcoding should leak
into unrelated cases.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.case_glossary_service import build_case_glossary


def test_base_do_not_use_terms_present(tmp_path, fact_lock_generic, blueprint_generic):
    facts_dir = tmp_path / "02-facts"
    glossary = build_case_glossary(fact_lock_generic, blueprint_generic, facts_dir)
    for term in ["सबसे भयानक", "सबसे दर्दनाक", "आप यकीन नहीं करेंगे"]:
        assert term in glossary["do_not_use"], f"Base term '{term}' missing from do_not_use"


def test_jhingur_not_added_for_non_ladybug_case(tmp_path, fact_lock_generic, blueprint_generic):
    facts_dir = tmp_path / "02-facts"
    glossary = build_case_glossary(fact_lock_generic, blueprint_generic, facts_dir)
    assert "झींगुर" not in glossary["do_not_use"], (
        "झींगुर should only be forbidden for ladybug cases, not a generic murder case"
    )
    assert "तितलियाँ" not in glossary["do_not_use"], (
        "तितलियाँ should only be forbidden for ladybug cases"
    )


def test_jhingur_added_for_ladybug_case(tmp_path, blueprint_generic):
    facts_dir = tmp_path / "02-facts"
    fact_lock_with_ladybug = {
        "case_name": "Ladybug Killer Case",
        "verified_people": [],
        "verified_dates": [],
        "verified_timeline": [],
        "legal_outcome": {"verdict": "convicted", "confidence": "high"},
        "facts_to_verify_externally": [],
        "key_evidence_or_turning_points": ["ladybug found at scene"],
        "emotional_details": [],
    }
    glossary = build_case_glossary(fact_lock_with_ladybug, blueprint_generic, facts_dir)
    assert "झींगुर" in glossary["do_not_use"]
    assert "तितलियाँ" in glossary["do_not_use"]


def test_kyla_jordan_not_added_for_non_kyla_woodhouse_case(tmp_path, fact_lock_generic, blueprint_generic):
    facts_dir = tmp_path / "02-facts"
    glossary = build_case_glossary(fact_lock_generic, blueprint_generic, facts_dir)
    assert "Kyla Jordan" not in glossary["do_not_use"]


def test_kyla_jordan_added_for_kyla_woodhouse_case(tmp_path, blueprint_generic):
    facts_dir = tmp_path / "02-facts"
    fact_lock_kyla = {
        "case_name": "Kyla Woodhouse Case",
        "verified_people": [
            {"name": "Kyla Woodhouse", "role": "victim"},
        ],
        "verified_dates": [],
        "verified_timeline": [],
        "legal_outcome": {"verdict": "convicted", "confidence": "high"},
        "facts_to_verify_externally": [],
        "key_evidence_or_turning_points": [],
        "emotional_details": [],
    }
    glossary = build_case_glossary(fact_lock_kyla, blueprint_generic, facts_dir)
    assert "Kyla Jordan" in glossary["do_not_use"]


def test_verified_names_from_fact_lock(tmp_path, fact_lock_generic, blueprint_generic):
    facts_dir = tmp_path / "02-facts"
    glossary = build_case_glossary(fact_lock_generic, blueprint_generic, facts_dir)
    spellings = glossary["verified_name_spellings"]
    assert "Devika Rathi" in spellings
    assert "Prakash Soni" in spellings
    assert "Justice Arvind Nair" in spellings


def test_glossary_saved_to_facts_dir(tmp_path, fact_lock_generic, blueprint_generic):
    facts_dir = tmp_path / "02-facts"
    build_case_glossary(fact_lock_generic, blueprint_generic, facts_dir)
    saved = json.loads((facts_dir / "case_glossary.json").read_text(encoding="utf-8"))
    assert "do_not_use" in saved
    assert "preferred_terms" in saved
    assert "verified_name_spellings" in saved


def test_preferred_terms_include_base_legal_terms(tmp_path, fact_lock_generic, blueprint_generic):
    facts_dir = tmp_path / "02-facts"
    glossary = build_case_glossary(fact_lock_generic, blueprint_generic, facts_dir)
    preferred = glossary["preferred_terms"]
    assert "conviction" in preferred
    assert "evidence" in preferred
    assert "trial" in preferred


def test_no_first_claim_for_generic_case(tmp_path, fact_lock_generic, blueprint_generic):
    facts_dir = tmp_path / "02-facts"
    glossary = build_case_glossary(fact_lock_generic, blueprint_generic, facts_dir)
    assert glossary["legal_claim_rules"]["allow_first_case_claim"] is False
