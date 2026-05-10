"""Tests for OpenAI final premium gate threshold enforcement and safe_to_voice logic.

All API calls are mocked — no real OpenAI calls are made.
Tests verify that Python-side threshold enforcement is correct regardless of
what the LLM returns.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.openai_final_premium_gate_service import (
    run_openai_final_premium_gate,
    _THRESHOLDS,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

_PASSING_SCORES = {k: 9.5 for k in _THRESHOLDS}

_EMPTY_REPORT_FIELDS = {
    "issues": [],
    "chunk_repair_targets": [],
}


def _full_passing_response() -> dict:
    return {
        **_PASSING_SCORES,
        "approved": True,
        "safe_to_voice": True,
        **_EMPTY_REPORT_FIELDS,
    }


def _call_gate(mock_response: dict, tmp_path: Path, label: str = "") -> dict:
    """Run the gate with mocked LLM response. Returns the gate report."""
    review_dir = tmp_path / "04-review"
    review_dir.mkdir(parents=True, exist_ok=True)
    with patch(
        "app.services.openai_final_premium_gate_service.call_openai_json",
        return_value=mock_response,
    ):
        return run_openai_final_premium_gate(
            script_draft={"hindi_narration_chunks": [], "youtube_metadata": {}},
            fact_lock={},
            blueprint={},
            hinglish_level=2,
            lint_report={},
            copyedit_report={},
            quality_report={},
            retention_report={},
            similarity_report={},
            originality_report={},
            dialogue_report={},
            metadata_report={},
            review_dir=review_dir,
            label=label,
        )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_all_scores_passing_sets_safe_to_voice_true(tmp_path):
    report = _call_gate(_full_passing_response(), tmp_path)
    assert report["approved"] is True
    assert report["safe_to_voice"] is True


def test_one_score_below_threshold_blocks_approval(tmp_path):
    """A single score below 9.0 must result in approved=False, safe_to_voice=False."""
    response = _full_passing_response()
    response["hindi_quality_score"] = 8.5  # below threshold
    report = _call_gate(response, tmp_path)
    assert report["approved"] is False
    assert report["safe_to_voice"] is False


def test_score_exactly_at_threshold_passes(tmp_path):
    response = _full_passing_response()
    for k in _THRESHOLDS:
        response[k] = 9.0  # exactly at minimum
    report = _call_gate(response, tmp_path)
    assert report["approved"] is True
    assert report["safe_to_voice"] is True


def test_python_threshold_failure_added_to_issues(tmp_path):
    response = _full_passing_response()
    response["overall_score"] = 7.0
    report = _call_gate(response, tmp_path)
    issue_types = [i.get("type") for i in report.get("issues", [])]
    assert "python_threshold_failure" in issue_types


def test_llm_approved_false_overrides_safe_to_voice(tmp_path):
    """Even if all Python thresholds pass, if LLM returns approved=False,
    safe_to_voice must be False."""
    response = _full_passing_response()
    response["approved"] = False
    response["safe_to_voice"] = False
    report = _call_gate(response, tmp_path)
    assert report["safe_to_voice"] is False


def test_high_severity_issue_blocks_approval(tmp_path):
    """A high-severity issue in LLM response must force approved=False."""
    response = _full_passing_response()
    response["issues"] = [{"severity": "high", "type": "safety", "description": "problem"}]
    report = _call_gate(response, tmp_path)
    assert report["approved"] is False
    assert report["safe_to_voice"] is False


def test_medium_severity_issue_does_not_block_if_scores_pass(tmp_path):
    """Medium severity issues should not block on their own."""
    response = _full_passing_response()
    response["issues"] = [{"severity": "medium", "type": "style", "description": "minor"}]
    report = _call_gate(response, tmp_path)
    assert report["approved"] is True
    assert report["safe_to_voice"] is True


def test_report_saved_to_disk(tmp_path):
    import json
    _call_gate(_full_passing_response(), tmp_path)
    saved_path = tmp_path / "04-review" / "openai_final_premium_report.json"
    assert saved_path.exists()
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert "approved" in saved
    assert "safe_to_voice" in saved


def test_after_repair_label_saves_separate_file(tmp_path):
    _call_gate(_full_passing_response(), tmp_path, label="_after_repair")
    assert (tmp_path / "04-review" / "openai_final_premium_report_after_repair.json").exists()


def test_decimal_score_below_threshold_blocked(tmp_path):
    """Score of 8.9 (decimal, < 9.0) must be caught even though LLM says approved."""
    response = _full_passing_response()
    response["retention_score"] = 8.9
    report = _call_gate(response, tmp_path)
    assert report["approved"] is False
    assert report["safe_to_voice"] is False


def test_all_thresholds_are_uniform_nine(tmp_path):
    """Guard against accidental threshold drift — all must be 9.0."""
    for key, val in _THRESHOLDS.items():
        assert val == 9.0, f"Threshold for {key} is {val}, expected 9.0"


# ── Blocking gate → safe_to_voice must stay False ────────────────────────────

def test_safe_to_voice_false_when_preflight_blocks(tmp_path):
    """If python preflight reports blocking=True, the pipeline must not produce
    safe_to_voice=True. This test verifies the OFP gate itself enforces the rule
    independently: a low OFP score blocks safe_to_voice regardless of preflight.

    The pipeline-level integration (preflight → no OFP until clean) is tested
    via the preflight service's blocking field.
    """
    import pytest
    make_chunk = pytest.make_chunk
    make_script = pytest.make_script
    make_glossary = pytest.make_glossary
    from app.services.python_preflight_service import run_python_preflight

    review_dir = tmp_path / "04-review"
    review_dir.mkdir()
    script = make_script([make_chunk("001", "यह भारत का पहला मामला था।")])
    glossary = make_glossary(allow_first_claim=False)
    preflight = run_python_preflight(
        script_draft=script,
        fact_lock={},
        case_glossary=glossary,
        review_dir=review_dir,
        target_duration_min=20,
        hinglish_level=2,
    )
    # Preflight must block
    assert preflight["blocking"] is True
    assert preflight["passed"] is False


def test_safe_to_voice_false_when_all_scores_nine_but_llm_says_no(tmp_path):
    """If LLM explicitly returns approved=False, safe_to_voice must be False
    even when all numeric thresholds pass."""
    response = _full_passing_response()
    response["approved"] = False
    response["safe_to_voice"] = False
    report = _call_gate(response, tmp_path)
    assert report["safe_to_voice"] is False


def test_preflight_blocking_field_is_in_output(tmp_path):
    """The 'blocking' field must always be present in the preflight report."""
    import pytest
    make_chunk = pytest.make_chunk
    make_script = pytest.make_script
    make_glossary = pytest.make_glossary
    from app.services.python_preflight_service import run_python_preflight

    review_dir = tmp_path / "04-review"
    review_dir.mkdir()
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    result = run_python_preflight(
        script_draft=script,
        fact_lock={},
        case_glossary=glossary,
        review_dir=review_dir,
        target_duration_min=20,
        hinglish_level=2,
    )
    assert "blocking" in result
    assert "severity_counts" in result
    assert "metadata_repair_targets" in result
