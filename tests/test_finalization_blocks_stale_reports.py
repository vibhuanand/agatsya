from __future__ import annotations

from app.services.agent_pipeline_service import ArtifactState, _finalize_reports_before_openai


def test_finalization_blocks_when_similarity_refresh_has_no_source(tmp_path):
    state = ArtifactState()
    state.mark_script_mutated("test")

    result = _finalize_reports_before_openai(
        artifact_state=state,
        script_final={"hindi_narration_chunks": []},
        fact_lock={},
        blueprint={},
        review_dir=tmp_path,
        gate_summary={"originality_safety": {"passed": True}},
        warnings=[],
        lint_report={},
        similarity_report={"risk_level": "none"},
        copyedit_report={},
        quality_report={},
        source_transcript="",
    )

    assert result["blocking"] is True
    assert "similarity" in result["failed_refreshes"]
