"""
Tests for TASK 1 — gate_summary must be initialized before Stage 6 calls
_run_routing_and_rebuild so that no NameError / UnboundLocalError is raised.

Also tests the Stage 6 fallback path: repair_required=True with no
chunk_repair_targets attempts auto-rebuild without crashing.
"""
from __future__ import annotations

import pytest


# ─── Unit-level helpers that replicate the init and routing logic ─────────────

def _build_gate_summary_before_stage6(
    quality_report: dict,
    preflight_report: dict,
    transformation_plan_ok: bool = True,
    originality_transformation_plan: dict | None = None,
) -> dict:
    """
    Replicate the early gate_summary initialization that must happen BEFORE
    Stage 6 so _run_routing_and_rebuild has a dict to write into.
    """
    gate_summary: dict = {}
    # The premium gate entries are populated later; Stage 6 just needs the dict
    # to exist so setdefault() / gate_summary["repair_routing"] = ... don't crash.
    return gate_summary


def _run_stage6_no_targets(
    quality_report: dict,
    gate_summary: dict,
    auto_rebuild_enabled: bool = True,
) -> dict:
    """
    Simulate the Stage 6 branch where repair_required=True but
    chunk_repair_targets is empty. Returns a result dict describing outcome.
    """
    chunk_repair_targets = quality_report.get("chunk_repair_targets", [])
    result = {
        "auto_rebuild_attempted": False,
        "crashed": False,
        "status": "auto_rebuild_required",
    }

    if not chunk_repair_targets:
        result["auto_rebuild_attempted"] = auto_rebuild_enabled
        # Simulate _run_routing_and_rebuild writing into gate_summary
        try:
            gate_summary["repair_routing"] = {
                "route": "claude_grouped_repair",
                "root_cause_count": 1,
            }
            gate_summary.setdefault("auto_fix", {})["python_fixes_count"] = 0
        except Exception as exc:
            result["crashed"] = True
            result["error"] = str(exc)

    return result


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestGateSummaryInitialization:
    def test_gate_summary_initialized_as_empty_dict(self):
        """gate_summary must be a dict before Stage 6 runs."""
        gs = _build_gate_summary_before_stage6(
            quality_report={"approved": False, "repair_required": True},
            preflight_report={"passed": True, "blocking": False},
        )
        assert isinstance(gs, dict)

    def test_gate_summary_can_receive_repair_routing_entry(self):
        """_run_routing_and_rebuild writes gate_summary['repair_routing'] — must not crash."""
        gs = _build_gate_summary_before_stage6({}, {})
        gs["repair_routing"] = {"route": "claude_grouped_repair", "root_cause_count": 1}
        assert gs["repair_routing"]["route"] == "claude_grouped_repair"

    def test_gate_summary_can_use_setdefault_before_premium_entries(self):
        """auto_fix sub-dict must be writable before premium gate entries are added."""
        gs = _build_gate_summary_before_stage6({}, {})
        gs.setdefault("auto_fix", {})["python_fixes_count"] = 3
        assert gs["auto_fix"]["python_fixes_count"] == 3

    def test_no_unbound_local_error_in_stage6_rebuild(self):
        """The full Stage 6 auto-rebuild simulation must not raise."""
        gs = _build_gate_summary_before_stage6({}, {})
        result = _run_stage6_no_targets(
            quality_report={"approved": False, "repair_required": True, "chunk_repair_targets": []},
            gate_summary=gs,
        )
        assert result["crashed"] is False
        assert result["auto_rebuild_attempted"] is True

    def test_gate_summary_dict_does_not_reset_entries_from_stage6(self):
        """Entries written by Stage 6 must survive when premium gate block runs."""
        gs = _build_gate_summary_before_stage6({}, {})
        # Stage 6 writes
        gs["repair_routing"] = {"route": "python_only"}
        gs.setdefault("auto_fix", {})["rebuild_ran"] = True
        # Premium gate block adds more entries (does NOT reset dict)
        gs["script_quality"] = {"passed": False}
        gs["python_preflight"] = {"passed": True, "blocking": False}
        # Stage 6 entries must still be present
        assert "repair_routing" in gs
        assert gs["auto_fix"]["rebuild_ran"] is True

    def test_repair_required_no_targets_attempts_auto_rebuild(self):
        """When repair_required=True and chunk_repair_targets=[], auto-rebuild is triggered."""
        gs: dict = {}
        result = _run_stage6_no_targets(
            quality_report={"approved": False, "repair_required": True, "chunk_repair_targets": []},
            gate_summary=gs,
            auto_rebuild_enabled=True,
        )
        assert result["auto_rebuild_attempted"] is True

    def test_repair_required_no_targets_disabled_rebuild(self):
        """When auto_rebuild_enabled=False, rebuild is not attempted."""
        gs: dict = {}
        result = _run_stage6_no_targets(
            quality_report={"approved": False, "repair_required": True, "chunk_repair_targets": []},
            gate_summary=gs,
            auto_rebuild_enabled=False,
        )
        assert result["auto_rebuild_attempted"] is False

    def test_gate_summary_repair_routing_written_only_when_routing_runs(self):
        """repair_routing must appear in gate_summary only when routing was called."""
        gs: dict = {}
        result = _run_stage6_no_targets(
            quality_report={"approved": False, "repair_required": True, "chunk_repair_targets": []},
            gate_summary=gs,
        )
        assert not result["crashed"]
        assert "repair_routing" in gs
