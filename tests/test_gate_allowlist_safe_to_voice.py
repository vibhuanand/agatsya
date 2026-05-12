"""
Tests for TASK 1 — all_gates_passed must use the explicit REQUIRED_SAFE_TO_VOICE_GATES
allowlist rather than iterating over every entry in gate_summary.

Verifies:
- telemetry entries (repair_routing, auto_fix, pre_oai_repair, repair_telemetry)
  do NOT block all_gates_passed when all required gates pass
- missing required gate blocks all_gates_passed
- failed required gate blocks all_gates_passed
- python_preflight evaluated via blocking field, not passed field
- REQUIRED_SAFE_TO_VOICE_GATES constant contains expected gate names
"""
from __future__ import annotations

import pytest

from app.services.agent_pipeline_service import (
    REQUIRED_SAFE_TO_VOICE_GATES,
    _gate_passed_for_safe_to_voice,
)


# ─── Helper: replicate the allowlist-based all_gates_passed computation ───────

def _all_gates_passed(gate_summary: dict) -> bool:
    """Mirror of the final all_gates_passed computation in agent_pipeline_service."""
    return all(
        _gate_passed_for_safe_to_voice(name, gate_summary.get(name, {"passed": False}))
        for name in REQUIRED_SAFE_TO_VOICE_GATES
    )


def _clean_gate_summary() -> dict:
    """gate_summary with all required gates passing plus telemetry noise."""
    gs: dict = {
        # Required content gates — all passing
        "originality_transformation": {"passed": True},
        "script_quality":             {"passed": True},
        "python_preflight":           {"passed": False, "blocking": False},  # low-only warns
        "hindi_copyedit":             {"passed": True},
        "originality_safety":         {"passed": True},
        "recreated_dialogue":         {"passed": True},
        "metadata_quality":           {"passed": True},
        "retention_quality":          {"passed": True},
        "openai_final_premium":       {"passed": True},
        # Telemetry entries — no `passed` field; must never affect all_gates_passed
        "repair_routing":      {"route": "claude_grouped_repair", "root_cause_count": 2},
        "auto_fix":            {"python_fixes_count": 3, "rebuild_ran": True},
        "pre_oai_repair":      {"rebuild_ran": False, "route": "python_only"},
        "repair_telemetry":    {"repair_route": "none", "root_cause_count": 0},
        "repair_failures":     {"passed": False, "claude_script_repair_failed": True},
        "automation_status":   "script_approved",   # plain string, not a dict
        "safe_to_voice":       False,               # plain bool
    }
    return gs


# ─── Tests: constant content ──────────────────────────────────────────────────

class TestRequiredGatesConstant:
    def test_constant_is_tuple_of_strings(self):
        assert isinstance(REQUIRED_SAFE_TO_VOICE_GATES, tuple)
        assert all(isinstance(g, str) for g in REQUIRED_SAFE_TO_VOICE_GATES)

    def test_expected_gates_present(self):
        for name in (
            "originality_transformation",
            "script_quality",
            "python_preflight",
            "hindi_copyedit",
            "originality_safety",
            "recreated_dialogue",
            "metadata_quality",
            "retention_quality",
            "openai_final_premium",
        ):
            assert name in REQUIRED_SAFE_TO_VOICE_GATES, (
                f"Expected gate '{name}' in REQUIRED_SAFE_TO_VOICE_GATES"
            )

    def test_telemetry_keys_not_in_allowlist(self):
        """Telemetry/status keys must never appear in the gate allowlist."""
        forbidden = {
            "repair_routing", "auto_fix", "pre_oai_repair",
            "repair_telemetry", "repair_failures", "automation_status",
            "safe_to_voice",
        }
        overlap = forbidden & set(REQUIRED_SAFE_TO_VOICE_GATES)
        assert not overlap, f"Telemetry keys found in gate allowlist: {overlap}"


# ─── Tests: all_gates_passed allowlist behaviour ──────────────────────────────

class TestAllGatesPassedAllowlist:
    def test_all_required_gates_passing_returns_true(self):
        gs = _clean_gate_summary()
        assert _all_gates_passed(gs) is True

    def test_telemetry_repair_routing_does_not_block(self):
        """repair_routing dict has no passed field — must not block."""
        gs = _clean_gate_summary()
        gs["repair_routing"] = {"route": "stop_not_voice_ready"}   # would block if checked
        assert _all_gates_passed(gs) is True

    def test_telemetry_auto_fix_does_not_block(self):
        gs = _clean_gate_summary()
        gs["auto_fix"] = {"python_fixes_count": 0, "rebuild_ran": False}
        assert _all_gates_passed(gs) is True

    def test_telemetry_pre_oai_repair_does_not_block(self):
        gs = _clean_gate_summary()
        gs["pre_oai_repair"] = {"passed": False, "route": "python_only"}
        assert _all_gates_passed(gs) is True

    def test_telemetry_repair_failures_does_not_block_gates(self):
        """repair_failures has passed=False but is NOT a required gate."""
        gs = _clean_gate_summary()
        gs["repair_failures"] = {
            "passed": False,
            "claude_script_repair_failed": True,
        }
        # Still True because repair_failures is not in the allowlist
        assert _all_gates_passed(gs) is True

    def test_failed_required_gate_blocks(self):
        gs = _clean_gate_summary()
        gs["openai_final_premium"] = {"passed": False}
        assert _all_gates_passed(gs) is False

    def test_missing_required_gate_blocks(self):
        """Gate not in gate_summary at all → defaults to passed=False."""
        gs = _clean_gate_summary()
        del gs["hindi_copyedit"]
        assert _all_gates_passed(gs) is False

    def test_python_preflight_non_blocking_passes(self):
        """python_preflight with passed=False but blocking=False should NOT block."""
        gs = _clean_gate_summary()
        gs["python_preflight"] = {"passed": False, "blocking": False}
        assert _all_gates_passed(gs) is True

    def test_python_preflight_blocking_blocks(self):
        gs = _clean_gate_summary()
        gs["python_preflight"] = {"passed": False, "blocking": True}
        assert _all_gates_passed(gs) is False

    def test_all_required_gates_failing_returns_false(self):
        gs = _clean_gate_summary()
        for name in REQUIRED_SAFE_TO_VOICE_GATES:
            if name == "python_preflight":
                gs[name] = {"passed": False, "blocking": True}
            else:
                gs[name] = {"passed": False}
        assert _all_gates_passed(gs) is False

    def test_large_telemetry_noise_does_not_affect_result(self):
        """Many telemetry entries with all sorts of values must not change result."""
        gs = _clean_gate_summary()
        for i in range(20):
            gs[f"_telemetry_noise_{i}"] = {"passed": False, "score": i}
        assert _all_gates_passed(gs) is True


# ─── Tests: final status guard (downgrade when gates fail) ───────────────────

class TestFinalStatusGuard:
    """
    Mirror the final authoritative status guard added to agent_pipeline_service:
    if status="script_approved" but required gates don't all pass, downgrade.
    """

    def _apply_status_guard(self, status: str, gate_summary: dict) -> str:
        all_pass = _all_gates_passed(gate_summary)
        if status == "script_approved" and not all_pass:
            return "not_voice_ready_auto_retry_exhausted"
        return status

    def test_status_preserved_when_all_gates_pass(self):
        result = self._apply_status_guard("script_approved", _clean_gate_summary())
        assert result == "script_approved"

    def test_status_downgraded_when_required_gate_fails(self):
        gs = _clean_gate_summary()
        gs["openai_final_premium"] = {"passed": False}
        result = self._apply_status_guard("script_approved", gs)
        assert result == "not_voice_ready_auto_retry_exhausted"

    def test_non_approved_status_not_affected_by_guard(self):
        gs = _clean_gate_summary()
        gs["openai_final_premium"] = {"passed": False}
        result = self._apply_status_guard("auto_rebuild_required", gs)
        assert result == "auto_rebuild_required"

    def test_telemetry_noise_cannot_trigger_downgrade(self):
        gs = _clean_gate_summary()
        gs["repair_routing"] = {"route": "stop_not_voice_ready"}
        # All required gates still pass — no downgrade
        result = self._apply_status_guard("script_approved", gs)
        assert result == "script_approved"
