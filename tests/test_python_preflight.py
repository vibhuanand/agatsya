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


def test_graphic_child_term_flagged(tmp_path):
    script = make_script([make_chunk("001", "तीसरे दर्जे की जलन के निशान मिले।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert result["passed"] is False
    types = [i["type"] for i in result["issues"]]
    assert "youtube_safety" in types


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


# ── Known bug: झींगुर hardcoding fires for ANY case ──────────────────────────

@pytest.mark.xfail(
    reason="Phase 3 needed: python_preflight hardcodes झींगुर check for any case, "
           "not just ladybug cases. Non-ladybug scripts mentioning insects fail incorrectly.",
    strict=True,
)
def test_jhingur_in_non_ladybug_script_should_not_be_flagged(tmp_path):
    """A non-ladybug case that happens to mention झींगुर (cricket / insect sound
    as local atmosphere) should NOT be flagged as a motif error.

    Currently FAILS: the hardcoded `if "झींगुर" in text` check in
    python_preflight_service.py fires regardless of case type.
    Fix: derive forbidden insect terms from the case glossary (Phase 3).
    """
    script = make_script([
        make_chunk("001", "रात को झींगुर की आवाज़ें सुनाई देती थीं।")
    ])
    # do_not_use does NOT contain झींगुर — this is a murder case, not a ladybug case
    glossary = make_glossary(do_not_use=["सबसे भयानक"])
    result = _run(script, glossary, tmp_path)
    assert result["passed"] is True
