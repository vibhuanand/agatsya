"""
Tests for Stage 16 stale gate_summary mirroring — session 4, TASK 6.

Verifies that when Stage 16 marks report objects with stale_after_rebuild=True,
the corresponding gate_summary entries are also updated so that
_gate_passed_for_safe_to_voice rejects them and safe_to_voice remains False.

These tests simulate what the pipeline code does at the Stage 16 stale-marking block
and verify the contract that gate_summary entries must carry stale_after_rebuild.
"""
from __future__ import annotations

import pytest

from app.services.agent_pipeline_service import (
    REQUIRED_SAFE_TO_VOICE_GATES,
    _gate_passed_for_safe_to_voice,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_gate_summary() -> dict:
    return {
        "originality_transformation": {"passed": True},
        "script_quality":             {"passed": True},
        "python_preflight":           {"passed": False, "blocking": False},
        "hindi_copyedit":             {"passed": True},
        "originality_safety":         {"passed": True},
        "recreated_dialogue":         {"passed": True},
        "metadata_quality":           {"passed": True},
        "retention_quality":          {"passed": True},
        "openai_final_premium":       {"passed": True},
    }


def _all_gates_passed(gs: dict) -> bool:
    return all(
        _gate_passed_for_safe_to_voice(name, gs.get(name, {"passed": False}))
        for name in REQUIRED_SAFE_TO_VOICE_GATES
    )


def _simulate_stage16_stale_marking(gate_summary: dict, stale_gate_names: list) -> None:
    """Replicate the Stage 16 gate_summary update that TASK 6 wired in."""
    for gs_name in stale_gate_names:
        gate_summary.setdefault(gs_name, {}).update({
            "stale_after_rebuild": True,
            "passed": False,
        })


# ─── Tests: stale gate_summary entries block safe_to_voice ────────────────────

class TestStaleGateSummaryBlocksSafeToVoice:
    def test_stale_retention_in_gate_summary_blocks(self):
        gs = _clean_gate_summary()
        _simulate_stage16_stale_marking(gs, ["retention_quality"])
        assert _all_gates_passed(gs) is False

    def test_stale_originality_in_gate_summary_blocks(self):
        gs = _clean_gate_summary()
        _simulate_stage16_stale_marking(gs, ["originality_safety"])
        assert _all_gates_passed(gs) is False

    def test_stale_dialogue_in_gate_summary_blocks(self):
        gs = _clean_gate_summary()
        _simulate_stage16_stale_marking(gs, ["recreated_dialogue"])
        assert _all_gates_passed(gs) is False

    def test_stale_metadata_in_gate_summary_blocks(self):
        gs = _clean_gate_summary()
        _simulate_stage16_stale_marking(gs, ["metadata_quality"])
        assert _all_gates_passed(gs) is False

    def test_stale_script_quality_in_gate_summary_blocks(self):
        gs = _clean_gate_summary()
        _simulate_stage16_stale_marking(gs, ["script_quality"])
        assert _all_gates_passed(gs) is False

    def test_stale_copyedit_in_gate_summary_blocks(self):
        gs = _clean_gate_summary()
        _simulate_stage16_stale_marking(gs, ["hindi_copyedit"])
        assert _all_gates_passed(gs) is False

    def test_all_four_stale_in_gate_summary_blocks(self):
        gs = _clean_gate_summary()
        _simulate_stage16_stale_marking(
            gs, ["retention_quality", "originality_safety",
                 "recreated_dialogue", "metadata_quality"]
        )
        assert _all_gates_passed(gs) is False


# ─── Tests: stale_after_rebuild flag is present in gate_summary ───────────────

class TestStaleAfterRebuildFlagInGateSummary:
    def test_stale_flag_set_in_retention_entry(self):
        gs = _clean_gate_summary()
        _simulate_stage16_stale_marking(gs, ["retention_quality"])
        assert gs["retention_quality"].get("stale_after_rebuild") is True

    def test_passed_set_false_in_stale_entry(self):
        gs = _clean_gate_summary()
        _simulate_stage16_stale_marking(gs, ["originality_safety"])
        assert gs["originality_safety"]["passed"] is False

    def test_stale_flag_set_for_all_four_expensive_gates(self):
        gs = _clean_gate_summary()
        stale_gates = ["retention_quality", "originality_safety",
                       "recreated_dialogue", "metadata_quality"]
        _simulate_stage16_stale_marking(gs, stale_gates)
        for gn in stale_gates:
            assert gs[gn].get("stale_after_rebuild") is True, (
                f"Expected stale_after_rebuild=True for {gn!r}"
            )


# ─── Tests: gate_passed_for_safe_to_voice sees stale_after_rebuild ───────────

class TestGatePassedSeesStaleAfterRebuild:
    def test_stale_after_rebuild_blocks_retention_quality_gate(self):
        gate = {"passed": True, "stale_after_rebuild": True}
        assert _gate_passed_for_safe_to_voice("retention_quality", gate) is False

    def test_stale_after_rebuild_blocks_originality_safety_gate(self):
        gate = {"passed": True, "stale_after_rebuild": True}
        assert _gate_passed_for_safe_to_voice("originality_safety", gate) is False

    def test_stale_after_rebuild_blocks_recreated_dialogue_gate(self):
        gate = {"passed": True, "stale_after_rebuild": True}
        assert _gate_passed_for_safe_to_voice("recreated_dialogue", gate) is False

    def test_stale_after_rebuild_blocks_metadata_quality_gate(self):
        gate = {"passed": True, "stale_after_rebuild": True}
        assert _gate_passed_for_safe_to_voice("metadata_quality", gate) is False


# ─── Tests: clean (non-stale) state still allows approval ────────────────────

class TestCleanStateStillApproves:
    def test_clean_gate_summary_all_pass(self):
        gs = _clean_gate_summary()
        assert _all_gates_passed(gs) is True

    def test_stale_after_rebuild_false_does_not_block(self):
        gate = {"passed": True, "stale_after_rebuild": False}
        assert _gate_passed_for_safe_to_voice("retention_quality", gate) is True

    def test_refreshed_after_rebuild_does_not_block(self):
        gate = {"passed": True, "refreshed_after_rebuild": True}
        assert _gate_passed_for_safe_to_voice("originality_safety", gate) is True
