from __future__ import annotations

from app.services import call_tracker
from app.services.agent_pipeline_service import _record_soft_budget_stop, _soft_repair_budget_exceeded
from app.services.repair_routing_service import run_repair_routing


def _script() -> dict:
    return {
        "hindi_narration_chunks": [
            {"chunk_id": "001_investigation", "section_title": "Investigation", "text": ""},
            {"chunk_id": "002_reconstruction", "section_title": "Court reconstruction", "text": ""},
            {"chunk_id": "003_evidence", "section_title": "Evidence sequence", "text": ""},
        ]
    }


def _preflight() -> dict:
    return {
        "issues": [{"severity": "high", "type": "source_shaped_reconstruction", "chunk_id": "002_reconstruction", "problem": "source shaped"}],
        "chunk_repair_targets": [{
            "chunk_id": "002_reconstruction",
            "issue_type": "source_shaped_reconstruction",
            "problem": "source shaped",
            "repair_instruction": "repair cluster",
        }],
    }


def test_routing_telemetry_records_repeated_root_cause_and_openai_route():
    first = run_repair_routing({"python_preflight": _preflight()}, 4, script_draft=_script(), max_cluster_size=4)
    key = first["source_copy_reconstruction_clusters"][0]["root_cause_key"]
    second = run_repair_routing(
        {"python_preflight": _preflight()},
        4,
        script_draft=_script(),
        max_cluster_size=4,
        previous_root_cause_attempts={key: 1},
    )

    assert second["claude_repair_skipped_due_previous_failure"] is True
    assert second["repeat_root_cause_detected"] == [key]
    assert second["source_shaped_reconstruction_detected"] is True
    assert second["reconstruction_cluster_count"] == 1
    assert second["stats"]["openai_target_count"] > 0


def test_estimated_calls_saved_present_for_clustered_claude_repair():
    plan = run_repair_routing({"python_preflight": _preflight()}, 4, script_draft=_script(), max_cluster_size=4)

    assert "estimated_model_calls_saved" in plan
    assert plan["reconstruction_cluster_count"] == 1


def test_soft_budget_telemetry_records_reason(monkeypatch):
    call_tracker.reset()
    monkeypatch.setattr("app.config.settings.failed_path_max_openai_repair_calls", 0)
    reason = _soft_repair_budget_exceeded("openai")
    warnings: list[str] = []
    gate_summary: dict = {}
    telemetry: dict = {}

    assert reason is not None
    _record_soft_budget_stop("openai", reason, warnings, gate_summary, telemetry)

    assert telemetry["repair_budget_exceeded"] is True
    assert telemetry["repair_budget_exceeded_kind"] == "openai"
    assert "FAILED_PATH_MAX_OPENAI_REPAIR_CALLS" in telemetry["repair_budget_exceeded_reason"]
    assert gate_summary["repair_budget"]["openai_budget_exceeded"] is True
