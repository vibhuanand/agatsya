"""
Tests for _refresh_reports_after_script_mutation — verifies that the helper
calls each gate service with the REAL current signatures.

Tests are intentionally NOT mock-only: they pass real args that match the live
function signatures so that any future signature mismatch is caught at import/
call time.  Live Claude/OpenAI calls are avoided by:
  - Using rerun_*=False flags for expensive gates in integration-style tests.
  - Using patch() only for tests that verify a specific gate path was invoked.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, call as mcall

import pytest

from app.services.agent_pipeline_service import _refresh_reports_after_script_mutation


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _script(chunks: list[dict] | None = None) -> dict:
    return {
        "hindi_narration_chunks": chunks or [
            {"chunk_id": "001_hook", "text": "हुक पाठ।", "section_title": "Hook",
             "voice": "narrator", "tone": "neutral", "estimated_words": 2},
        ],
        "youtube_metadata": {"title": "Test Episode"},
        "recreated_dialogues": {"items": []},
    }


def _fact_lock() -> dict:
    return {"case_name": "Test Case", "facts": [], "people": []}


def _blueprint() -> dict:
    return {"title": "Test", "sections": []}


def _retention_blueprint() -> dict:
    return {"re_engagement_moments": [], "shorts_candidates": []}


SOURCE_TRANSCRIPT = "This is a test English source transcript for similarity checking."


# ─── Tests: return structure (no flags — all disabled) ────────────────────────

class TestReturnStructure:
    def test_all_keys_present_when_no_gates_rerun(self, tmp_path):
        result = _refresh_reports_after_script_mutation(
            script_final=_script(),
            fact_lock=_fact_lock(),
            blueprint=_blueprint(),
            review_dir=tmp_path,
            gate_summary={},
            warnings=[],
            lint_report={"gate": "lint"},
            similarity_report={"gate": "sim"},
            quality_report={"gate": "q"},
            copyedit_report={"gate": "copy"},
            rerun_lint=False,
            rerun_similarity=False,
            rerun_quality=False,
            rerun_copyedit=False,
        )
        for key in ("lint_report", "similarity_report", "quality_report",
                    "copyedit_report", "retention_report", "originality_report",
                    "dialogue_report", "metadata_report"):
            assert key in result

    def test_originals_returned_when_all_disabled(self, tmp_path):
        orig_lint = {"gate": "lint", "v": 1}
        orig_sim  = {"gate": "sim",  "v": 2}
        result = _refresh_reports_after_script_mutation(
            script_final=_script(),
            fact_lock=_fact_lock(),
            blueprint=_blueprint(),
            review_dir=tmp_path,
            gate_summary={},
            warnings=[],
            lint_report=orig_lint,
            similarity_report=orig_sim,
            quality_report={},
            copyedit_report={},
            rerun_lint=False,
            rerun_similarity=False,
        )
        assert result["lint_report"] is orig_lint
        assert result["similarity_report"] is orig_sim

    def test_none_retention_returned_when_not_provided(self, tmp_path):
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
            rerun_lint=False,
            rerun_similarity=False,
        )
        assert result["retention_report"] is None


# ─── Tests: lint (Python-only, real call) ─────────────────────────────────────

class TestLintRealSignature:
    """Lint is Python-only — we can call it without mocking."""

    def test_lint_refreshed_with_real_call(self, tmp_path):
        result = _refresh_reports_after_script_mutation(
            script_final=_script(),
            fact_lock=_fact_lock(),
            blueprint=_blueprint(),
            review_dir=tmp_path,
            gate_summary={},
            warnings=[],
            lint_report={"stale": True},
            similarity_report={},
            quality_report={},
            copyedit_report={},
            hinglish_level=2,
            rerun_lint=True,
            rerun_similarity=False,
            rerun_quality=False,
            rerun_copyedit=False,
        )
        assert result["lint_report"].get("refreshed_after_script_mutation") is True
        assert "stale" not in result["lint_report"]

    def test_lint_written_to_disk(self, tmp_path):
        _refresh_reports_after_script_mutation(
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
            hinglish_level=2,
            rerun_lint=True,
            rerun_similarity=False,
            rerun_quality=False,
            rerun_copyedit=False,
        )
        assert (tmp_path / "hindi_text_lint_report.json").exists()
        data = json.loads((tmp_path / "hindi_text_lint_report.json").read_text())
        assert data.get("refreshed_after_script_mutation") is True

    def test_lint_receives_hinglish_level_as_int(self, tmp_path):
        """hinglish_level must be passed as int, not str."""
        with patch("app.services.agent_pipeline_service.run_hindi_text_lint") as mock_lint:
            mock_lint.return_value = {"total_issues": 0}
            _refresh_reports_after_script_mutation(
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
                hinglish_level=3,
                rerun_lint=True,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
            )
        _, kwargs = mock_lint.call_args
        assert isinstance(kwargs.get("hinglish_level", 2), int)

    def test_lint_exception_creates_stale_failure_and_warns(self, tmp_path):
        """TASK 2: exception must produce a stale-failure report, not keep the old one."""
        orig = {"gate": "lint", "v": "original", "passed": True}
        warnings: list[str] = []
        with patch(
            "app.services.agent_pipeline_service.run_hindi_text_lint",
            side_effect=RuntimeError("lint crashed"),
        ):
            result = _refresh_reports_after_script_mutation(
                script_final=_script(),
                fact_lock=_fact_lock(),
                blueprint=_blueprint(),
                review_dir=tmp_path,
                gate_summary={},
                warnings=warnings,
                lint_report=orig,
                similarity_report={},
                quality_report={},
                copyedit_report={},
                rerun_lint=True,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
            )
        # Old behavior (keep original) is gone: exception → stale failure
        assert result["lint_report"] is not orig
        assert result["lint_report"].get("refresh_failed") is True
        assert result["lint_report"].get("passed") is False
        assert result["lint_report"].get("stale") is True
        assert any("lint" in w.lower() for w in warnings)


# ─── Tests: similarity — requires source_transcript ───────────────────────────

class TestSimilarityRealSignature:
    def test_similarity_stale_failure_when_source_transcript_missing(self, tmp_path):
        warnings: list[str] = []
        result = _refresh_reports_after_script_mutation(
            script_final=_script(),
            fact_lock=_fact_lock(),
            blueprint=_blueprint(),
            review_dir=tmp_path,
            gate_summary={},
            warnings=warnings,
            lint_report={},
            similarity_report={"old": True},
            quality_report={},
            copyedit_report={},
            source_transcript="",    # missing!
            rerun_lint=False,
            rerun_similarity=True,
            rerun_quality=False,
            rerun_copyedit=False,
        )
        sim = result["similarity_report"]
        assert sim.get("refresh_failed") is True
        assert sim.get("stale") is True
        assert sim.get("passed") is False
        assert any("source_transcript" in w for w in warnings)

    def test_similarity_called_with_correct_signature(self, tmp_path):
        """Verify the real run_text_similarity_check signature is used."""
        with patch(
            "app.services.agent_pipeline_service.run_text_similarity_check",
        ) as mock_sim:
            mock_sim.return_value = {"risk_level": "low", "high_risk_matches": 0}
            _refresh_reports_after_script_mutation(
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
                source_transcript=SOURCE_TRANSCRIPT,
                rerun_lint=False,
                rerun_similarity=True,
                rerun_quality=False,
                rerun_copyedit=False,
            )
        _, kwargs = mock_sim.call_args
        # Must pass source_transcript AND script_draft — the two required args
        assert "source_transcript" in kwargs
        assert "script_draft" in kwargs
        assert isinstance(kwargs["source_transcript"], str)
        assert isinstance(kwargs["script_draft"], dict)

    def test_similarity_marker_on_fresh_report(self, tmp_path):
        with patch(
            "app.services.agent_pipeline_service.run_text_similarity_check",
            return_value={"risk_level": "none", "high_risk_matches": 0},
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
                source_transcript=SOURCE_TRANSCRIPT,
                rerun_lint=False,
                rerun_similarity=True,
                rerun_quality=False,
                rerun_copyedit=False,
            )
        assert result["similarity_report"].get("refreshed_after_script_mutation") is True


# ─── Tests: retention — requires retention_blueprint ──────────────────────────

class TestRetentionRealSignature:
    def test_retention_stale_failure_when_blueprint_missing(self, tmp_path):
        warnings: list[str] = []
        result = _refresh_reports_after_script_mutation(
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
            retention_blueprint=None,    # missing!
            rerun_lint=False,
            rerun_similarity=False,
            rerun_quality=False,
            rerun_copyedit=False,
            rerun_retention=True,
        )
        ret = result["retention_report"]
        assert ret is not None
        assert ret.get("refresh_failed") is True
        assert ret.get("passed") is False
        assert any("retention_blueprint" in w for w in warnings)

    def test_retention_called_with_correct_signature(self, tmp_path):
        """run_retention_quality_gate needs retention_blueprint, not fact_lock."""
        with patch(
            "app.services.agent_pipeline_service.run_retention_quality_gate",
        ) as mock_rq:
            mock_rq.return_value = {"approved": True, "overall_retention_score": 8}
            _refresh_reports_after_script_mutation(
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
                retention_blueprint=_retention_blueprint(),
                retention_report={},
                target_duration_min=12,
                rerun_lint=False,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
                rerun_retention=True,
            )
        _, kwargs = mock_rq.call_args
        # Must NOT pass fact_lock — retention gate doesn't take it
        assert "fact_lock" not in kwargs
        # Must pass retention_blueprint and target_duration_min
        assert "retention_blueprint" in kwargs
        assert "target_duration_min" in kwargs
        assert isinstance(kwargs["target_duration_min"], int)


# ─── Tests: originality — requires source_transcript + similarity ──────────────

class TestOriginalityRealSignature:
    def test_originality_stale_failure_when_source_transcript_missing(self, tmp_path):
        warnings: list[str] = []
        result = _refresh_reports_after_script_mutation(
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
            source_transcript="",    # missing
            originality_report={},
            rerun_lint=False,
            rerun_similarity=False,
            rerun_quality=False,
            rerun_copyedit=False,
            rerun_originality=True,
        )
        orig = result["originality_report"]
        assert orig.get("refresh_failed") is True

    def test_originality_called_with_correct_signature(self, tmp_path):
        """run_originality_safety_gate needs source_transcript+similarity_report, not fact_lock/blueprint."""
        with patch(
            "app.services.agent_pipeline_service.run_originality_safety_gate",
        ) as mock_orig:
            mock_orig.return_value = {"gate_passed": True, "approved": True}
            _refresh_reports_after_script_mutation(
                script_final=_script(),
                fact_lock=_fact_lock(),
                blueprint=_blueprint(),
                review_dir=tmp_path,
                gate_summary={},
                warnings=[],
                lint_report={},
                similarity_report={"risk_level": "low", "high_risk_matches": 0},
                quality_report={},
                copyedit_report={},
                source_transcript=SOURCE_TRANSCRIPT,
                originality_report={},
                rerun_lint=False,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
                rerun_originality=True,
            )
        _, kwargs = mock_orig.call_args
        # Must NOT pass fact_lock or blueprint
        assert "fact_lock" not in kwargs
        assert "blueprint" not in kwargs
        # Must pass source_transcript and similarity_report
        assert "source_transcript" in kwargs
        assert "similarity_report" in kwargs


# ─── Tests: recreated_dialogue — NO blueprint arg ─────────────────────────────

class TestDialogueRealSignature:
    def test_dialogue_called_without_blueprint(self, tmp_path):
        """run_recreated_dialogue_quality_gate takes (script_draft, fact_lock, review_dir) only."""
        with patch(
            "app.services.agent_pipeline_service.run_recreated_dialogue_quality_gate",
        ) as mock_dial:
            mock_dial.return_value = {"gate_passed": True, "approved": True}
            _refresh_reports_after_script_mutation(
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
                dialogue_report={},
                rerun_lint=False,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
                rerun_dialogue=True,
            )
        _, kwargs = mock_dial.call_args
        # blueprint must NOT be passed
        assert "blueprint" not in kwargs
        assert "script_draft" in kwargs
        assert "fact_lock" in kwargs
        assert "review_dir" in kwargs

    def test_dialogue_marker_set(self, tmp_path):
        with patch(
            "app.services.agent_pipeline_service.run_recreated_dialogue_quality_gate",
            return_value={"gate_passed": True},
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
                dialogue_report={},
                rerun_lint=False,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
                rerun_dialogue=True,
            )
        assert result["dialogue_report"].get("refreshed_after_script_mutation") is True


# ─── Tests: metadata ──────────────────────────────────────────────────────────

class TestMetadataRealSignature:
    def test_metadata_called_with_correct_signature(self, tmp_path):
        with patch(
            "app.services.agent_pipeline_service.run_metadata_quality_gate",
        ) as mock_meta:
            mock_meta.return_value = {"gate_passed": True, "approved": True}
            _refresh_reports_after_script_mutation(
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
                metadata_report={},
                rerun_lint=False,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
                rerun_metadata=True,
            )
        _, kwargs = mock_meta.call_args
        assert "script_draft" in kwargs
        assert "fact_lock" in kwargs
        assert "review_dir" in kwargs

    def test_metadata_marker_set(self, tmp_path):
        with patch(
            "app.services.agent_pipeline_service.run_metadata_quality_gate",
            return_value={"gate_passed": True},
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
                metadata_report={},
                rerun_lint=False,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
                rerun_metadata=True,
            )
        assert result["metadata_report"].get("refreshed_after_script_mutation") is True


# ─── Tests: quality refresh updates gate_summary ──────────────────────────────

class TestQualityGateSummaryUpdate:
    def test_gate_summary_updated_on_quality_refresh(self, tmp_path):
        gs: dict = {}
        with patch(
            "app.services.agent_pipeline_service.run_script_review",
            return_value={"approved": True, "scores": {"overall": 90}},
        ):
            _refresh_reports_after_script_mutation(
                script_final=_script(),
                fact_lock=_fact_lock(),
                blueprint=_blueprint(),
                review_dir=tmp_path,
                gate_summary=gs,
                warnings=[],
                lint_report={},
                similarity_report={},
                quality_report={},
                copyedit_report={},
                rerun_lint=False,
                rerun_similarity=False,
                rerun_quality=True,
                rerun_copyedit=False,
            )
        assert gs.get("script_quality", {}).get("passed") is True
        assert gs["script_quality"].get("refreshed_after_script_mutation") is True

    def test_ofp_never_called_inside_helper(self, tmp_path):
        with patch(
            "app.services.agent_pipeline_service.run_openai_final_premium_gate",
        ) as mock_ofp:
            _refresh_reports_after_script_mutation(
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
                rerun_lint=False,
                rerun_similarity=False,
                rerun_quality=False,
                rerun_copyedit=False,
            )
        mock_ofp.assert_not_called()
