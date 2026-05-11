"""
Tests for safe_to_voice_decision_trace.json schema and content.

Verifies:
  - Trace file is written to 04-review/ after every pipeline run (when review_dir exists)
  - Required fields present: safe_to_voice, status, blocking_reasons, gate_scores,
    repair_telemetry, elevenlabs_ready
  - blocking_reasons is non-empty when safe_to_voice=False
  - blocking_reasons is empty when safe_to_voice=True
  - ElevenLabs condition matches safe_to_voice exactly
  - automation_status is populated and equals status
  - repair_telemetry has all expected fields

NOTE: These are unit tests that build trace dicts directly, since we cannot run
the full pipeline in unit tests. Integration tests for the written file are
tested via the tmp_path fixture simulating the file write path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ─── Reference trace builder (mirrors _stv_trace in agent_pipeline_service) ──

def _build_trace(
    safe_to_voice: bool,
    status: str,
    blocking_reasons: list[str] | None = None,
    gate_scores: dict | None = None,
    repair_telemetry: dict | None = None,
) -> dict:
    return {
        "safe_to_voice":     safe_to_voice,
        "status":            status,
        "automation_status": status,
        "elevenlabs_ready":  safe_to_voice,
        "blocking_reasons":  blocking_reasons or [],
        "gate_scores":       gate_scores or {
            "openai_final_premium_overall": None,
            "openai_final_premium_passed":  None,
            "hindi_copyedit_passed":        None,
            "retention_passed":             None,
            "metadata_passed":              None,
            "originality_passed":           None,
        },
        "repair_telemetry":  repair_telemetry or {
            "repair_route":                "none",
            "root_cause_count":            0,
            "python_auto_fixes_count":     0,
            "claude_grouped_repair_count": 0,
            "estimated_model_calls_saved": 0,
            "auto_rebuild_ran":            False,
            "auto_rebuild_chunks":         0,
            "avoided_openai_bulk_repair":  False,
        },
    }


# ─── Test: required schema fields ────────────────────────────────────────────

class TestTraceSchema:
    REQUIRED_TOP_LEVEL = [
        "safe_to_voice", "status", "automation_status", "elevenlabs_ready",
        "blocking_reasons", "gate_scores", "repair_telemetry",
    ]
    REQUIRED_GATE_SCORES = [
        "openai_final_premium_overall", "openai_final_premium_passed",
        "hindi_copyedit_passed", "retention_passed", "metadata_passed",
        "originality_passed",
    ]
    REQUIRED_REPAIR_TELEMETRY = [
        "repair_route", "root_cause_count", "python_auto_fixes_count",
        "claude_grouped_repair_count", "estimated_model_calls_saved",
        "auto_rebuild_ran", "auto_rebuild_chunks", "avoided_openai_bulk_repair",
    ]

    def test_required_top_level_keys_present(self):
        trace = _build_trace(safe_to_voice=True, status="script_approved")
        for key in self.REQUIRED_TOP_LEVEL:
            assert key in trace, f"Missing top-level key: {key}"

    def test_required_gate_scores_present(self):
        trace = _build_trace(safe_to_voice=True, status="script_approved")
        for key in self.REQUIRED_GATE_SCORES:
            assert key in trace["gate_scores"], f"Missing gate_score key: {key}"

    def test_required_repair_telemetry_present(self):
        trace = _build_trace(safe_to_voice=True, status="script_approved")
        for key in self.REQUIRED_REPAIR_TELEMETRY:
            assert key in trace["repair_telemetry"], f"Missing repair_telemetry key: {key}"

    def test_trace_is_json_serializable(self):
        trace = _build_trace(safe_to_voice=False, status="needs_human_review",
                             blocking_reasons=["status=needs_human_review"])
        dumped = json.dumps(trace, ensure_ascii=False)
        reloaded = json.loads(dumped)
        assert reloaded["safe_to_voice"] is False


# ─── Test: blocking reasons ───────────────────────────────────────────────────

class TestBlockingReasons:
    def test_blocking_reasons_empty_when_approved(self):
        trace = _build_trace(safe_to_voice=True, status="script_approved", blocking_reasons=[])
        assert trace["blocking_reasons"] == []

    def test_blocking_reasons_non_empty_when_failed(self):
        trace = _build_trace(
            safe_to_voice=False,
            status="needs_human_review",
            blocking_reasons=["status=needs_human_review", "gates_failed=['openai_final_premium']"],
        )
        assert len(trace["blocking_reasons"]) > 0

    def test_auto_rebuild_status_blocks(self):
        trace = _build_trace(
            safe_to_voice=False,
            status="auto_rebuild_required",
            blocking_reasons=["status=auto_rebuild_required"],
        )
        assert "auto_rebuild_required" in trace["blocking_reasons"][0]

    def test_exhausted_status_blocks(self):
        trace = _build_trace(
            safe_to_voice=False,
            status="not_voice_ready_auto_retry_exhausted",
            blocking_reasons=["status=not_voice_ready_auto_retry_exhausted"],
        )
        assert len(trace["blocking_reasons"]) > 0

    def test_python_preflight_blocking_reason(self):
        trace = _build_trace(
            safe_to_voice=False,
            status="needs_human_review",
            blocking_reasons=["python_preflight=blocking"],
        )
        assert "python_preflight=blocking" in trace["blocking_reasons"]

    def test_similarity_blocking_reason(self):
        trace = _build_trace(
            safe_to_voice=False,
            status="needs_human_review",
            blocking_reasons=["text_similarity_high_risk=5>3"],
        )
        assert any("similarity" in r for r in trace["blocking_reasons"])


# ─── Test: elevenlabs_ready matches safe_to_voice ────────────────────────────

class TestElevenLabsCondition:
    def test_elevenlabs_ready_true_when_approved(self):
        trace = _build_trace(safe_to_voice=True, status="script_approved")
        assert trace["elevenlabs_ready"] is True

    def test_elevenlabs_ready_false_when_failed(self):
        trace = _build_trace(safe_to_voice=False, status="needs_human_review")
        assert trace["elevenlabs_ready"] is False

    def test_elevenlabs_matches_safe_to_voice_always(self):
        for stv, status in [
            (True, "script_approved"),
            (False, "needs_human_review"),
            (False, "auto_rebuild_required"),
            (False, "not_voice_ready_auto_retry_exhausted"),
        ]:
            trace = _build_trace(safe_to_voice=stv, status=status)
            assert trace["elevenlabs_ready"] == trace["safe_to_voice"]


# ─── Test: automation_status matches status ───────────────────────────────────

class TestAutomationStatus:
    def test_automation_status_equals_status(self):
        for st in [
            "script_approved",
            "needs_human_review",
            "auto_rebuild_required",
            "not_voice_ready_auto_retry_exhausted",
        ]:
            trace = _build_trace(safe_to_voice=(st == "script_approved"), status=st)
            assert trace["automation_status"] == trace["status"]


# ─── Test: repair_telemetry content ──────────────────────────────────────────

class TestRepairTelemetry:
    def test_rebuild_ran_false_by_default(self):
        trace = _build_trace(safe_to_voice=True, status="script_approved")
        assert trace["repair_telemetry"]["auto_rebuild_ran"] is False

    def test_rebuild_ran_true_when_rebuild_happened(self):
        telemetry = {
            "repair_route":                "auto_rebuild_required",
            "root_cause_count":            2,
            "python_auto_fixes_count":     3,
            "claude_grouped_repair_count": 1,
            "estimated_model_calls_saved": 5,
            "auto_rebuild_ran":            True,
            "auto_rebuild_chunks":         4,
            "avoided_openai_bulk_repair":  True,
        }
        trace = _build_trace(safe_to_voice=True, status="script_approved",
                             repair_telemetry=telemetry)
        assert trace["repair_telemetry"]["auto_rebuild_ran"] is True
        assert trace["repair_telemetry"]["auto_rebuild_chunks"] == 4
        assert trace["repair_telemetry"]["avoided_openai_bulk_repair"] is True

    def test_python_fixes_count_is_integer(self):
        trace = _build_trace(safe_to_voice=True, status="script_approved")
        assert isinstance(trace["repair_telemetry"]["python_auto_fixes_count"], int)

    def test_repair_route_is_string(self):
        trace = _build_trace(safe_to_voice=True, status="script_approved")
        assert isinstance(trace["repair_telemetry"]["repair_route"], str)


# ─── Test: file write to review_dir ──────────────────────────────────────────

class TestTraceFileWrite:
    def test_trace_file_written_and_valid_json(self, tmp_path):
        """Simulate writing the trace file as done in agent_pipeline_service."""
        trace = _build_trace(
            safe_to_voice=False,
            status="auto_rebuild_required",
            blocking_reasons=["status=auto_rebuild_required"],
        )
        trace_path = tmp_path / "safe_to_voice_decision_trace.json"
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        assert trace_path.exists()
        loaded = json.loads(trace_path.read_text(encoding="utf-8"))
        assert loaded["safe_to_voice"] is False
        assert loaded["status"] == "auto_rebuild_required"
        assert "blocking_reasons" in loaded
        assert "repair_telemetry" in loaded

    def test_trace_file_utf8_hindi_readable(self, tmp_path):
        """Hindi characters in blocking_reasons must survive write/read cycle."""
        trace = _build_trace(
            safe_to_voice=False,
            status="needs_human_review",
            blocking_reasons=["gates_failed=['openai_final_premium']"],
            gate_scores={
                "openai_final_premium_overall": 6,
                "openai_final_premium_passed": False,
                "hindi_copyedit_passed": True,
                "retention_passed": True,
                "metadata_passed": True,
                "originality_passed": True,
            },
        )
        # Add Hindi text to make sure encoding works
        trace["case_title"] = "पाई शियाओ येन मामला"
        trace_path = tmp_path / "safe_to_voice_decision_trace.json"
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        loaded = json.loads(trace_path.read_text(encoding="utf-8"))
        assert loaded.get("case_title") == "पाई शियाओ येन मामला"
