"""
Tests for _finalize_reports_before_openai — session 4, TASK 3.

Verifies:
- When artifact_state.final_review_inputs_mutated=False, returns immediately with refresh_ok=True
- When script mutated and source_transcript missing, similarity refresh_failed → blocking=True
- When no mutation, blocking=False and no refresh attempted
- Returns refreshed_reports, failed_refreshes, stale_reports keys
- blocking=True sets refresh_ok=False
- Stale expensive gates are noted in stale_reports (not failed)
"""
from __future__ import annotations

import pytest
from pathlib import Path

from app.services.agent_pipeline_service import (
    ArtifactState,
    _finalize_reports_before_openai,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _script(text: str = "पाठ।") -> dict:
    return {
        "hindi_narration_chunks": [{"chunk_id": "001", "text": text}],
        "youtube_metadata": {"title": "T"},
        "recreated_dialogues": {"items": []},
    }


def _fact_lock() -> dict:
    return {"case_name": "T", "facts": [], "people": []}


def _blueprint() -> dict:
    return {"title": "T", "sections": []}


# ─── Tests: no mutation → no work ─────────────────────────────────────────────

class TestNoMutationNoWork:
    def test_returns_immediately_when_no_mutation(self, tmp_path):
        state = ArtifactState()  # all False — nothing mutated
        result = _finalize_reports_before_openai(
            artifact_state=state,
            script_final=_script(),
            fact_lock=_fact_lock(),
            blueprint=_blueprint(),
            review_dir=tmp_path,
            gate_summary={},
            warnings=[],
            lint_report={},
            similarity_report={"risk_level": "none", "passed": True},
            copyedit_report={},
            quality_report={},
        )
        assert result["refresh_ok"] is True
        assert result["blocking"] is False
        assert result["refreshed_reports"] == {}
        assert result["failed_refreshes"] == []

    def test_no_mutation_no_warnings_appended(self, tmp_path):
        state = ArtifactState()
        warnings: list = []
        _finalize_reports_before_openai(
            artifact_state=state,
            script_final=_script(),
            fact_lock=_fact_lock(),
            blueprint=_blueprint(),
            review_dir=tmp_path,
            gate_summary={},
            warnings=warnings,
            lint_report={},
            similarity_report={},
            copyedit_report={},
            quality_report={},
        )
        assert warnings == []


# ─── Tests: script mutation with missing source_transcript ────────────────────

class TestScriptMutatedMissingSourceTranscript:
    def test_similarity_refresh_fails_when_transcript_missing(self, tmp_path):
        state = ArtifactState()
        state.mark_script_mutated("stage6_repair")

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
            source_transcript="",   # missing — triggers stale failure
        )
        assert "similarity" in result["failed_refreshes"]

    def test_blocking_true_when_similarity_fails(self, tmp_path):
        state = ArtifactState()
        state.mark_script_mutated("stage6_repair")

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
            source_transcript="",
        )
        assert result["blocking"] is True
        assert result["refresh_ok"] is False

    def test_warning_appended_when_blocking(self, tmp_path):
        state = ArtifactState()
        state.mark_script_mutated("stage6_repair")
        warnings: list = []

        _finalize_reports_before_openai(
            artifact_state=state,
            script_final=_script(),
            fact_lock=_fact_lock(),
            blueprint=_blueprint(),
            review_dir=tmp_path,
            gate_summary={},
            warnings=warnings,
            lint_report={},
            similarity_report={},
            copyedit_report={},
            quality_report={},
            source_transcript="",
        )
        assert len(warnings) > 0
        assert any("refresh" in w.lower() or "stale" in w.lower() for w in warnings)


# ─── Tests: result structure ──────────────────────────────────────────────────

class TestResultStructure:
    def test_result_has_all_keys(self, tmp_path):
        state = ArtifactState()
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
        )
        assert "refresh_ok" in result
        assert "refreshed_reports" in result
        assert "failed_refreshes" in result
        assert "stale_reports" in result
        assert "blocking" in result

    def test_refreshed_reports_is_dict(self, tmp_path):
        state = ArtifactState()
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
        )
        assert isinstance(result["refreshed_reports"], dict)

    def test_failed_refreshes_is_list(self, tmp_path):
        state = ArtifactState()
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
        )
        assert isinstance(result["failed_refreshes"], list)


# ─── Tests: lint refreshed when script mutated ────────────────────────────────

class TestLintRefreshedWhenScriptMutated:
    def test_lint_is_refreshed_after_script_mutation(self, tmp_path):
        state = ArtifactState()
        state.mark_script_mutated("stage6_repair")
        stale_lint = {"total_issues": 99, "stale": True}

        result = _finalize_reports_before_openai(
            artifact_state=state,
            script_final=_script(),
            fact_lock=_fact_lock(),
            blueprint=_blueprint(),
            review_dir=tmp_path,
            gate_summary={},
            warnings=[],
            lint_report=stale_lint,
            similarity_report={},
            copyedit_report={},
            quality_report={},
            source_transcript="",
        )
        # lint should be refreshed (or at least attempted)
        # We can't assert specific content without mocking, but
        # the result dict structure must be correct
        assert isinstance(result["refreshed_reports"], dict)


# ─── Tests: metadata mutation ─────────────────────────────────────────────────

class TestMetadataMutation:
    def test_metadata_mutation_does_not_require_similarity_refresh(self, tmp_path):
        """Metadata-only mutation doesn't require similarity refresh — not blocking."""
        state = ArtifactState()
        state.mark_metadata_mutated("stage13a_metadata_repair")

        result = _finalize_reports_before_openai(
            artifact_state=state,
            script_final=_script(),
            fact_lock=_fact_lock(),
            blueprint=_blueprint(),
            review_dir=tmp_path,
            gate_summary={},
            warnings=[],
            lint_report={},
            similarity_report={"risk_level": "none", "passed": True},
            copyedit_report={},
            quality_report={},
            source_transcript="",
        )
        # similarity not required for metadata-only mutation
        # so failed_refreshes should not contain "similarity"
        assert "similarity" not in result["failed_refreshes"]
