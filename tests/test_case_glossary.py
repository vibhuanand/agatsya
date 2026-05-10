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


def test_jhingur_added_for_ladybug_case_via_blueprint_motif_terms(tmp_path, fact_lock_generic):
    """झींगुर/तितलियाँ are added only when the blueprint explicitly lists them
    as forbidden_hindi for a motif — no hardcoding."""
    facts_dir = tmp_path / "02-facts"
    blueprint_with_motif = {
        "primary_story_type": "murder_conviction",
        "sensitivity_level": "medium",
        "narrative_sections": [],
        "main_hook": "A ladybug case.",
        "emotional_anchor": "Family seeking justice.",
        "sensitivity_rules": {"child_victim": False, "graphic_details": "minimize"},
        "motif_terms": [
            {
                "english": "ladybug",
                "preferred_hindi": "लेडीबग",
                "forbidden_hindi": ["झींगुर", "तितलियाँ"],
            }
        ],
    }
    glossary = build_case_glossary(fact_lock_generic, blueprint_with_motif, facts_dir)
    assert "झींगुर" in glossary["do_not_use"]
    assert "तितलियाँ" in glossary["do_not_use"]
    assert glossary["motif_constraints"]["ladybug"]["preferred_hindi"] == "लेडीबग"


def test_kyla_jordan_not_added_for_non_kyla_woodhouse_case(tmp_path, fact_lock_generic, blueprint_generic):
    facts_dir = tmp_path / "02-facts"
    glossary = build_case_glossary(fact_lock_generic, blueprint_generic, facts_dir)
    assert "Kyla Jordan" not in glossary["do_not_use"]


def test_forbidden_name_variant_added_via_fact_lock(tmp_path, blueprint_generic):
    """Forbidden name variants come from fact_lock.verified_people[].forbidden_name_variants —
    no hardcoding of specific names."""
    facts_dir = tmp_path / "02-facts"
    fact_lock_with_variants = {
        "case_name": "Kyla Woodhouse Case",
        "verified_people": [
            {
                "name": "Kyla Woodhouse",
                "role": "victim",
                "forbidden_name_variants": ["Kyla Jordan"],
            }
        ],
        "verified_dates": [],
        "verified_timeline": [],
        "legal_outcome": {"verdict": "convicted", "confidence": "high"},
        "facts_to_verify_externally": [],
        "key_evidence_or_turning_points": [],
        "emotional_details": [],
    }
    glossary = build_case_glossary(fact_lock_with_variants, blueprint_generic, facts_dir)
    assert "Kyla Jordan" in glossary["do_not_use"]
    assert "Kyla Jordan" in glossary["forbidden_name_variants"]


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


def test_first_degree_murder_charge_does_not_trigger_first_claim(tmp_path, blueprint_generic):
    """'first-degree murder' in the charge field must NOT be mistaken for a
    first-ever-case claim. The charge is a legal label, not a historical claim."""
    facts_dir = tmp_path / "02-facts"
    fact_lock_with_charge = {
        "case_name": "Test Murder Case",
        "verified_people": [{"name": "Victim A", "role": "victim"}],
        "verified_dates": [],
        "verified_timeline": [
            {"event": "Arrest", "date": "2021-01-01", "confidence": "high", "source_phrase": "arrested"}
        ],
        "legal_outcome": {
            "verdict": "convicted",
            "charge": "first-degree murder",
            "sentence": "life imprisonment",
            "confidence": "high",
        },
        "facts_to_verify_externally": [],
        "key_evidence_or_turning_points": [],
        "emotional_details": [],
    }
    glossary = build_case_glossary(fact_lock_with_charge, blueprint_generic, facts_dir)
    assert glossary["legal_claim_rules"]["allow_first_case_claim"] is False


def test_glossary_includes_new_generic_fields(tmp_path, fact_lock_generic, blueprint_generic):
    """Glossary must include forbidden_name_variants and motif_constraints fields."""
    facts_dir = tmp_path / "02-facts"
    glossary = build_case_glossary(fact_lock_generic, blueprint_generic, facts_dir)
    assert "forbidden_name_variants" in glossary
    assert "motif_constraints" in glossary
    # Generic case without motif_terms in blueprint → empty motif_constraints
    assert isinstance(glossary["motif_constraints"], dict)
    assert isinstance(glossary["forbidden_name_variants"], list)
