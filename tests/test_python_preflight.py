"""Tests for python_preflight_service.run_python_preflight().

The xfail test documents the known झींगुर hardcoding bug (Phase 3 fix).
All other tests must pass on the current codebase.
"""
from __future__ import annotations

import json

import pytest

from app.services.python_preflight_service import run_python_preflight

make_chunk = pytest.make_chunk
make_metadata = pytest.make_metadata
make_script = pytest.make_script
make_glossary = pytest.make_glossary


# ── Helper ────────────────────────────────────────────────────────────────────

def _run(script, glossary, tmp_path, *, target=20, hinglish=2):
    review_dir = tmp_path / "04-review"
    return run_python_preflight(
        script_draft=script,
        fact_lock={},
        case_glossary=glossary,
        review_dir=review_dir,
        target_duration_min=target,
        hinglish_level=hinglish,
    )


# ── Passing cases ─────────────────────────────────────────────────────────────

def test_clean_script_passes(tmp_path):
    script = make_script([make_chunk("001", "नागपुर अदालत ने फैसला सुनाया।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert result["passed"] is True
    assert result["issues"] == []
    assert result["metadata_issues"] == []


def test_report_saved_to_disk(tmp_path):
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    _run(script, glossary, tmp_path)
    saved = json.loads((tmp_path / "04-review" / "python_preflight_report.json").read_text())
    assert "passed" in saved
    assert "issues" in saved


def test_miss_kar_flagged_at_hinglish_level_2(tmp_path):
    script = make_script([make_chunk("001", "वह उसे miss कर रही थी।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path, hinglish=2)
    assert result["passed"] is False
    types = [i["type"] for i in result["issues"]]
    assert "hinglish_level" in types


def test_miss_kar_not_flagged_at_hinglish_level_4(tmp_path):
    script = make_script([make_chunk("001", "वह उसे miss कर रही थी।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path, hinglish=4)
    types = [i["type"] for i in result["issues"]]
    assert "hinglish_level" not in types


def test_unsupported_first_claim_flagged(tmp_path):
    script = make_script([make_chunk("001", "यह भारत में पहला मामला था।")])
    glossary = make_glossary(allow_first_claim=False)
    result = _run(script, glossary, tmp_path)
    assert result["passed"] is False
    types = [i["type"] for i in result["issues"]]
    assert "unsupported_legal_claim" in types


def test_first_claim_allowed_when_flag_set(tmp_path):
    script = make_script([make_chunk("001", "यह भारत में पहला मामला था।")])
    glossary = make_glossary(allow_first_claim=True)
    result = _run(script, glossary, tmp_path)
    types = [i["type"] for i in result["issues"]]
    assert "unsupported_legal_claim" not in types


def test_generic_do_not_use_term_in_narration_flagged(tmp_path):
    """Forbidden terms from case_glossary.do_not_use are caught in narration chunks.

    The old hardcoded _GRAPHIC_CHILD_TERMS list was case-specific. The generic
    mechanism is: add the term to do_not_use in the glossary → preflight catches it.
    """
    script = make_script([make_chunk("001", "यह सबसे भयानक मामला था।")])
    # "सबसे भयानक" is in the default do_not_use list
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert result["passed"] is False
    assert result["blocking"] is True
    types = [i["type"] for i in result["issues"]]
    assert "case_glossary" in types


def test_title_too_long_in_metadata(tmp_path):
    long_title = "अ" * 101
    script = make_script(
        [make_chunk("001", "साफ़ पाठ।")],
        metadata=make_metadata(title=long_title),
    )
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert result["passed"] is False
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "title_length" in meta_types


def test_tag_count_too_few_flagged(tmp_path):
    script = make_script(
        [make_chunk("001", "साफ़ पाठ।")],
        metadata=make_metadata(tags=["tag1", "tag2"]),
    )
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "tag_count" in meta_types


def test_forbidden_term_in_metadata_flagged(tmp_path):
    metadata = make_metadata()
    metadata["description"] = "सबसे भयानक मामला।"
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=metadata)
    glossary = make_glossary(do_not_use=["सबसे भयानक"])
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "metadata_forbidden_term" in meta_types


def test_estimated_duration_in_report(tmp_path):
    # 110 wpm default; 220 words ≈ 2.0 min
    words = " ".join(["word"] * 220)
    script = make_script([make_chunk("001", words)])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert "estimated_duration_min" in result
    assert result["estimated_duration_min"] > 0


# ── New output shape (Phase 4) ────────────────────────────────────────────────

def test_report_has_blocking_field(tmp_path):
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert "blocking" in result
    assert isinstance(result["blocking"], bool)


def test_report_has_severity_counts(tmp_path):
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    sc = result.get("severity_counts", {})
    assert "high" in sc and "medium" in sc and "low" in sc


def test_clean_script_has_blocking_false(tmp_path):
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert result["blocking"] is False


def test_high_issue_sets_blocking_true(tmp_path):
    """A high-severity issue (unsupported legal claim) must set blocking=True."""
    script = make_script([make_chunk("001", "यह भारत का पहला मामला था।")])
    glossary = make_glossary(allow_first_claim=False)
    result = _run(script, glossary, tmp_path)
    assert result["blocking"] is True
    assert result["severity_counts"]["high"] > 0


def test_medium_issue_sets_blocking_true(tmp_path):
    """A medium-severity issue must also set blocking=True."""
    script = make_script([make_chunk("001", "वह उसे miss कर रही थी।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path, hinglish=2)
    assert result["blocking"] is True
    assert result["severity_counts"]["medium"] > 0


def test_low_only_issue_does_not_block(tmp_path):
    """A low-severity issue alone must NOT set blocking=True."""
    metadata = make_metadata()
    metadata["chapters"] = [{"title": "भाग 1", "timestamp": "0:00"}]
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=metadata)
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    # chapters_before_audio is low — should not block
    assert result["blocking"] is False
    assert result["severity_counts"]["low"] >= 1


def test_metadata_repair_targets_populated_on_title_too_long(tmp_path):
    """Metadata issues must create structured metadata_repair_targets entries."""
    long_title = "अ" * 101
    script = make_script(
        [make_chunk("001", "साफ़ पाठ।")],
        metadata=make_metadata(title=long_title),
    )
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert "metadata_repair_targets" in result
    assert len(result["metadata_repair_targets"]) > 0
    first = result["metadata_repair_targets"][0]
    assert "field" in first
    assert "issue_type" in first
    assert "repair_instruction" in first


def test_metadata_repair_targets_empty_when_clean(tmp_path):
    """No metadata issues → metadata_repair_targets must be an empty list."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert result["metadata_repair_targets"] == []


def test_forbidden_name_variant_in_narration_high_severity(tmp_path):
    """Forbidden name variants in narration must be high severity and blocking."""
    script = make_script([make_chunk("001", "Kyla Jordan ने खेला।")])
    glossary = make_glossary()
    # Inject forbidden_name_variants into the glossary directly
    glossary["forbidden_name_variants"] = ["Kyla Jordan"]
    result = _run(script, glossary, tmp_path)
    assert result["blocking"] is True
    types = [i["type"] for i in result["issues"]]
    assert "forbidden_name_variant" in types


# ── Known bug: झींगुर hardcoding fires for ANY case ──────────────────────────

def test_jhingur_in_non_ladybug_script_is_not_flagged(tmp_path):
    """A non-ladybug case mentioning झींगुर (cricket sounds as atmosphere) must NOT
    be flagged — झींगुर is only forbidden when the case glossary says so.

    Phase 3/4 fix: the hardcoded '`if "झींगुर" in text`' check was removed.
    Forbidden terms now come exclusively from case_glossary.do_not_use.
    """
    script = make_script([
        make_chunk("001", "रात को झींगुर की आवाज़ें सुनाई देती थीं।")
    ])
    # do_not_use does NOT contain झींगुर — this is a generic murder case
    glossary = make_glossary(do_not_use=["सबसे भयानक"])
    result = _run(script, glossary, tmp_path)
    assert result["passed"] is True
    assert result["blocking"] is False
