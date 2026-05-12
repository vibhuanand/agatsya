"""
Tests for TASK 2 — refresh exceptions must create blocking stale reports.

Verifies:
- When a gate refresh raises an exception, the result dict contains a
  refresh_failed=True stale report (not the old passing report)
- The stale report has passed=False, stale=True, refresh_failed=True, reason=<str>
- _finalize_reports_before_openai treats a refresh_failed similarity report as blocking
- safe_to_voice cannot become True when required gate refresh_failed
- Each gate (lint, similarity, quality, copyedit, retention, originality,
  dialogue, metadata) produces a structured stale failure on exception
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

from app.services.agent_pipeline_service import (
    ArtifactState,
    REQUIRED_SAFE_TO_VOICE_GATES,
    _gate_passed_for_safe_to_voice,
    _refresh_reports_after_script_mutation,
    _finalize_reports_before_openai,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _script() -> dict:
    return {
        "hindi_narration_chunks": [{"chunk_id": "001", "text": "पाठ।"}],
        "youtube_metadata": {"title": "T"},
        "recreated_dialogues": {"items": []},
    }


def _fact_lock() -> dict:
    return {"case_name": "T", "facts": [], "people": []}


def _blueprint() -> dict:
    return {"title": "T", "sections": []}


def _assert_stale_failure(report: dict, gate_name: str) -> None:
    """Assert the structured stale-failure shape."""
    assert report.get("passed") is False,      f"{gate_name}: expected passed=False"
    assert report.get("stale") is True,        f"{gate_name}: expected stale=True"
    assert report.get("refresh_failed") is True, f"{gate_name}: expected refresh_failed=True"
    assert "reason" in report,                 f"{gate_name}: expected reason key"
    assert len(report["reason"]) > 0,          f"{gate_name}: reason must not be empty"


# ─── Tests: lint exception creates stale report ───────────────────────────────

class TestLintExceptionCreatesStaleReport:
    def test_lint_exception_sets_refresh_failed(self, tmp_path):
        original_lint = {"total_issues": 0, "passed": True}
        with patch(
            "app.services.agent_pipeline_service.run_hindi_text_lint",
            side_effect=RuntimeError("lint service crashed"),
        ):
            result = _refresh_reports_after_script_mutation(
                script_final=_script(),
                fact_lock=_fact_lock(),
                blueprint=_blueprint(),
                review_dir=tmp_path,
                gate_summary={},
                warnings=[],
                lint_report=original_lint,
                similarity_report={},
                quality_report={},
                copyedit_report={},
                rerun_lint=True,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
            )
        fresh_lint = result["lint_report"]
        _assert_stale_failure(fresh_lint, "lint")

    def test_lint_exception_does_not_keep_old_passing_report(self, tmp_path):
        original_lint = {"total_issues": 0, "passed": True}
        with patch(
            "app.services.agent_pipeline_service.run_hindi_text_lint",
            side_effect=RuntimeError("crash"),
        ):
            result = _refresh_reports_after_script_mutation(
                script_final=_script(),
                fact_lock=_fact_lock(),
                blueprint=_blueprint(),
                review_dir=tmp_path,
                gate_summary={},
                warnings=[],
                lint_report=original_lint,
                similarity_report={},
                quality_report={},
                copyedit_report={},
                rerun_lint=True,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
            )
        assert result["lint_report"] is not original_lint, (
            "Exception must replace the old passing lint report — "
            "keeping it would allow OFP to see stale evidence"
        )
        assert result["lint_report"].get("passed") is False


# ─── Tests: similarity exception creates stale report ────────────────────────

class TestSimilarityExceptionCreatesStaleReport:
    def test_similarity_exception_sets_refresh_failed(self, tmp_path):
        original_sim = {"risk_level": "none", "passed": True}
        with patch(
            "app.services.agent_pipeline_service.run_text_similarity_check",
            side_effect=RuntimeError("similarity service down"),
        ):
            result = _refresh_reports_after_script_mutation(
                script_final=_script(),
                fact_lock=_fact_lock(),
                blueprint=_blueprint(),
                review_dir=tmp_path,
                gate_summary={},
                warnings=[],
                lint_report={},
                similarity_report=original_sim,
                quality_report={},
                copyedit_report={},
                source_transcript="some transcript",
                rerun_lint=False,
                rerun_similarity=True,
                rerun_quality=False,
                rerun_copyedit=False,
            )
        _assert_stale_failure(result["similarity_report"], "similarity")

    def test_similarity_exception_blocks_ofp_via_finalization(self, tmp_path):
        """A similarity exception cascades to block OFP in _finalize_reports_before_openai."""
        state = ArtifactState()
        state.mark_script_mutated("stage6_repair")

        with patch(
            "app.services.agent_pipeline_service.run_text_similarity_check",
            side_effect=RuntimeError("similarity crashed"),
        ):
            result = _finalize_reports_before_openai(
                artifact_state=state,
                script_final=_script(),
                fact_lock=_fact_lock(),
                blueprint=_blueprint(),
                review_dir=tmp_path,
                gate_summary={},
                warnings=[],
                lint_report={},
                similarity_report={},
                copyedit_report={},
                quality_report={},
                source_transcript="some transcript",
            )
        assert result["blocking"] is True, (
            "similarity exception must block OFP — stale evidence must not reach OpenAI gate"
        )
        assert result["refresh_ok"] is False

    def test_similarity_exception_safe_to_voice_false(self, tmp_path):
        """After similarity exception, gate_summary must not allow safe_to_voice=True."""
        gs = {
            "originality_transformation": {"passed": True},
            "script_quality":             {"passed": True},
            "python_preflight":           {"passed": False, "blocking": False},
            "hindi_copyedit":             {"passed": True},
            "originality_safety":         {"passed": True},
            "recreated_dialogue":         {"passed": True},
            "metadata_quality":           {"passed": True},
            "retention_quality":          {"passed": True},
            "openai_final_premium":       {"passed": False, "refresh_failed": True,
                                           "reason": "similarity refresh exception"},
        }
        # openai_final_premium has refresh_failed=True — must not pass gate
        assert _gate_passed_for_safe_to_voice("openai_final_premium",
                                              gs["openai_final_premium"]) is False
        all_pass = all(
            _gate_passed_for_safe_to_voice(name, gs.get(name, {"passed": False}))
            for name in REQUIRED_SAFE_TO_VOICE_GATES
        )
        assert all_pass is False, (
            "safe_to_voice must be False when OFP gate has refresh_failed=True"
        )


# ─── Tests: retention exception creates stale report ─────────────────────────

class TestRetentionExceptionCreatesStaleReport:
    def test_retention_exception_sets_refresh_failed(self, tmp_path):
        original_ret = {"approved": True, "passed": True}
        with patch(
            "app.services.agent_pipeline_service.run_retention_quality_gate",
            side_effect=RuntimeError("retention gate error"),
        ):
            result = _refresh_reports_after_script_mutation(
                script_final=_script(),
                fact_lock=_fact_lock(),
                blueprint=_blueprint(),
                review_dir=tmp_path,
                gate_summary={},
                warnings=[],
                lint_report={},
                similarity_report={},
                quality_report={},
                copyedit_report={},
                retention_report=original_ret,
                retention_blueprint={"sections": []},
                rerun_lint=False,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
                rerun_retention=True,
            )
        _assert_stale_failure(result["retention_report"], "retention")

    def test_retention_exception_replaces_passing_report(self, tmp_path):
        passing_ret = {"approved": True, "passed": True}
        with patch(
            "app.services.agent_pipeline_service.run_retention_quality_gate",
            side_effect=RuntimeError("crash"),
        ):
            result = _refresh_reports_after_script_mutation(
                script_final=_script(),
                fact_lock=_fact_lock(),
                blueprint=_blueprint(),
                review_dir=tmp_path,
                gate_summary={},
                warnings=[],
                lint_report={},
                similarity_report={},
                quality_report={},
                copyedit_report={},
                retention_report=passing_ret,
                retention_blueprint={"sections": []},
                rerun_lint=False,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
                rerun_retention=True,
            )
        assert result["retention_report"] is not passing_ret
        assert result["retention_report"].get("passed") is False


# ─── Tests: quality exception creates stale report ────────────────────────────

class TestQualityExceptionCreatesStaleReport:
    def test_quality_exception_sets_refresh_failed(self, tmp_path):
        with patch(
            "app.services.agent_pipeline_service.run_script_review",
            side_effect=RuntimeError("quality gate error"),
        ):
            result = _refresh_reports_after_script_mutation(
                script_final=_script(),
                fact_lock=_fact_lock(),
                blueprint=_blueprint(),
                review_dir=tmp_path,
                gate_summary={},
                warnings=[],
                lint_report={},
                similarity_report={},
                quality_report={"approved": True},
                copyedit_report={},
                rerun_lint=False,
                rerun_similarity=False,
                rerun_quality=True,
                rerun_copyedit=False,
            )
        _assert_stale_failure(result["quality_report"], "quality")


# ─── Tests: originality exception creates stale report ────────────────────────

class TestOriginalityExceptionCreatesStaleReport:
    def test_originality_exception_sets_refresh_failed(self, tmp_path):
        with patch(
            "app.services.agent_pipeline_service.run_originality_safety_gate",
            side_effect=RuntimeError("originality gate error"),
        ):
            result = _refresh_reports_after_script_mutation(
                script_final=_script(),
                fact_lock=_fact_lock(),
                blueprint=_blueprint(),
                review_dir=tmp_path,
                gate_summary={},
                warnings=[],
                lint_report={},
                similarity_report={"risk_level": "none"},
                quality_report={},
                copyedit_report={},
                originality_report={"passed": True},
                source_transcript="some transcript",
                rerun_lint=False,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
                rerun_originality=True,
            )
        _assert_stale_failure(result["originality_report"], "originality")


# ─── Tests: dialogue and metadata exceptions ──────────────────────────────────

class TestDialogueExceptionCreatesStaleReport:
    def test_dialogue_exception_sets_refresh_failed(self, tmp_path):
        with patch(
            "app.services.agent_pipeline_service.run_recreated_dialogue_quality_gate",
            side_effect=RuntimeError("dialogue gate error"),
        ):
            result = _refresh_reports_after_script_mutation(
                script_final=_script(),
                fact_lock=_fact_lock(),
                blueprint=_blueprint(),
                review_dir=tmp_path,
                gate_summary={},
                warnings=[],
                lint_report={},
                similarity_report={},
                quality_report={},
                copyedit_report={},
                dialogue_report={"passed": True},
                rerun_lint=False,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
                rerun_dialogue=True,
            )
        _assert_stale_failure(result["dialogue_report"], "dialogue")


class TestMetadataExceptionCreatesStaleReport:
    def test_metadata_exception_sets_refresh_failed(self, tmp_path):
        with patch(
            "app.services.agent_pipeline_service.run_metadata_quality_gate",
            side_effect=RuntimeError("metadata gate error"),
        ):
            result = _refresh_reports_after_script_mutation(
                script_final=_script(),
                fact_lock=_fact_lock(),
                blueprint=_blueprint(),
                review_dir=tmp_path,
                gate_summary={},
                warnings=[],
                lint_report={},
                similarity_report={},
                quality_report={},
                copyedit_report={},
                metadata_report={"passed": True},
                rerun_lint=False,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
                rerun_metadata=True,
            )
        _assert_stale_failure(result["metadata_report"], "metadata")


# ─── Tests: stale report from exception blocks gate ───────────────────────────

class TestStaleReportFromExceptionBlocksGate:
    """_gate_passed_for_safe_to_voice rejects refresh_failed reports regardless of context."""

    def test_refresh_failed_from_exception_blocks_originality_gate(self):
        gate = {"passed": False, "stale": True, "refresh_failed": True,
                "reason": "originality_safety refresh exception: timeout"}
        assert _gate_passed_for_safe_to_voice("originality_safety", gate) is False

    def test_refresh_failed_from_exception_blocks_similarity_derived_gate(self):
        """Even if passed was True before, refresh_failed overrides it."""
        gate = {"passed": True, "refresh_failed": True,
                "reason": "text_similarity refresh exception: crash"}
        assert _gate_passed_for_safe_to_voice("openai_final_premium", gate) is False

    def test_warning_appended_on_exception(self, tmp_path):
        warnings: list = []
        with patch(
            "app.services.agent_pipeline_service.run_hindi_text_lint",
            side_effect=RuntimeError("lint crash"),
        ):
            _refresh_reports_after_script_mutation(
                script_final=_script(),
                fact_lock=_fact_lock(),
                blueprint=_blueprint(),
                review_dir=tmp_path,
                gate_summary={},
                warnings=warnings,
                lint_report={},
                similarity_report={},
                quality_report={},
                copyedit_report={},
                rerun_lint=True,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
            )
        assert len(warnings) > 0
        assert any("lint" in w.lower() for w in warnings)
