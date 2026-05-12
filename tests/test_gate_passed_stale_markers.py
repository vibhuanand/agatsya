"""
Tests for _gate_passed_for_safe_to_voice with stale markers — session 4, TASK 5.

Verifies that refresh_failed, stale_after_mutation, and stale_after_rebuild
all cause _gate_passed_for_safe_to_voice to return False, even when passed=True
is also set (stale evidence must never count as a passing gate).
"""
from __future__ import annotations

import pytest

from app.services.agent_pipeline_service import (
    REQUIRED_SAFE_TO_VOICE_GATES,
    _gate_passed_for_safe_to_voice,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _non_pf_gates() -> list[str]:
    return [g for g in REQUIRED_SAFE_TO_VOICE_GATES if g != "python_preflight"]


# ─── Tests: refresh_failed blocks all gates ───────────────────────────────────

class TestRefreshFailedBlocksGate:
    def test_refresh_failed_blocks_originality_safety(self):
        gate = {"passed": False, "refresh_failed": True, "reason": "missing input"}
        assert _gate_passed_for_safe_to_voice("originality_safety", gate) is False

    def test_refresh_failed_blocks_hindi_copyedit(self):
        gate = {"passed": True, "refresh_failed": True}
        # even passed=True is overridden by refresh_failed
        assert _gate_passed_for_safe_to_voice("hindi_copyedit", gate) is False

    def test_refresh_failed_blocks_retention_quality(self):
        gate = {"passed": True, "refresh_failed": True}
        assert _gate_passed_for_safe_to_voice("retention_quality", gate) is False

    def test_refresh_failed_blocks_metadata_quality(self):
        gate = {"passed": True, "refresh_failed": True}
        assert _gate_passed_for_safe_to_voice("metadata_quality", gate) is False

    def test_refresh_failed_blocks_recreated_dialogue(self):
        gate = {"passed": True, "refresh_failed": True}
        assert _gate_passed_for_safe_to_voice("recreated_dialogue", gate) is False

    def test_refresh_failed_blocks_openai_final_premium(self):
        gate = {"passed": True, "refresh_failed": True}
        assert _gate_passed_for_safe_to_voice("openai_final_premium", gate) is False

    def test_refresh_failed_blocks_python_preflight_even_when_non_blocking(self):
        gate = {"passed": False, "blocking": False, "refresh_failed": True}
        assert _gate_passed_for_safe_to_voice("python_preflight", gate) is False

    def test_refresh_failed_blocks_all_required_gates(self):
        for name in REQUIRED_SAFE_TO_VOICE_GATES:
            if name == "python_preflight":
                gate = {"passed": False, "blocking": False, "refresh_failed": True}
            else:
                gate = {"passed": True, "refresh_failed": True}
            result = _gate_passed_for_safe_to_voice(name, gate)
            assert result is False, (
                f"Expected False for gate {name!r} with refresh_failed=True, got {result}"
            )


# ─── Tests: stale_after_mutation blocks all gates ─────────────────────────────

class TestStaleAfterMutationBlocksGate:
    def test_stale_after_mutation_blocks_originality_safety(self):
        gate = {"passed": True, "stale_after_mutation": True}
        assert _gate_passed_for_safe_to_voice("originality_safety", gate) is False

    def test_stale_after_mutation_blocks_metadata_quality(self):
        gate = {"passed": True, "stale_after_mutation": True}
        assert _gate_passed_for_safe_to_voice("metadata_quality", gate) is False

    def test_stale_after_mutation_blocks_retention_quality(self):
        gate = {"passed": True, "stale_after_mutation": True}
        assert _gate_passed_for_safe_to_voice("retention_quality", gate) is False

    def test_stale_after_mutation_blocks_all_non_pf_gates(self):
        for name in _non_pf_gates():
            gate = {"passed": True, "stale_after_mutation": True}
            result = _gate_passed_for_safe_to_voice(name, gate)
            assert result is False, (
                f"Expected False for gate {name!r} with stale_after_mutation=True"
            )

    def test_stale_after_mutation_blocks_python_preflight(self):
        gate = {"passed": False, "blocking": False, "stale_after_mutation": True}
        assert _gate_passed_for_safe_to_voice("python_preflight", gate) is False


# ─── Tests: stale_after_rebuild blocks all gates ──────────────────────────────

class TestStaleAfterRebuildBlocksGate:
    def test_stale_after_rebuild_blocks_retention_quality(self):
        gate = {"passed": True, "stale_after_rebuild": True}
        assert _gate_passed_for_safe_to_voice("retention_quality", gate) is False

    def test_stale_after_rebuild_blocks_originality_safety(self):
        gate = {"passed": True, "stale_after_rebuild": True}
        assert _gate_passed_for_safe_to_voice("originality_safety", gate) is False

    def test_stale_after_rebuild_blocks_recreated_dialogue(self):
        gate = {"passed": True, "stale_after_rebuild": True}
        assert _gate_passed_for_safe_to_voice("recreated_dialogue", gate) is False

    def test_stale_after_rebuild_blocks_all_non_pf_gates(self):
        for name in _non_pf_gates():
            gate = {"passed": True, "stale_after_rebuild": True}
            result = _gate_passed_for_safe_to_voice(name, gate)
            assert result is False, (
                f"Expected False for gate {name!r} with stale_after_rebuild=True"
            )

    def test_stale_after_rebuild_blocks_python_preflight(self):
        gate = {"passed": False, "blocking": False, "stale_after_rebuild": True}
        assert _gate_passed_for_safe_to_voice("python_preflight", gate) is False


# ─── Tests: fresh passing gates still work ────────────────────────────────────

class TestFreshPassingGates:
    def test_fresh_passing_gate_returns_true(self):
        for name in _non_pf_gates():
            gate = {"passed": True}
            result = _gate_passed_for_safe_to_voice(name, gate)
            assert result is True, (
                f"Expected True for gate {name!r} with passed=True and no stale markers"
            )

    def test_python_preflight_non_blocking_returns_true(self):
        gate = {"passed": False, "blocking": False}
        assert _gate_passed_for_safe_to_voice("python_preflight", gate) is True

    def test_python_preflight_blocking_returns_false(self):
        gate = {"passed": False, "blocking": True}
        assert _gate_passed_for_safe_to_voice("python_preflight", gate) is False

    def test_fresh_with_refreshed_marker_returns_true(self):
        gate = {"passed": True, "refreshed_after_script_mutation": True}
        assert _gate_passed_for_safe_to_voice("hindi_copyedit", gate) is True


# ─── Tests: stale marker overrides passed=True ────────────────────────────────

class TestStalePrecedenceOverPassed:
    def test_refresh_failed_overrides_passed_true(self):
        """refresh_failed=True wins over passed=True."""
        gate = {"passed": True, "refresh_failed": True}
        assert _gate_passed_for_safe_to_voice("script_quality", gate) is False

    def test_stale_after_mutation_overrides_passed_true(self):
        gate = {"passed": True, "stale_after_mutation": True}
        assert _gate_passed_for_safe_to_voice("originality_transformation", gate) is False

    def test_stale_after_rebuild_overrides_passed_true(self):
        gate = {"passed": True, "stale_after_rebuild": True}
        assert _gate_passed_for_safe_to_voice("metadata_quality", gate) is False

    def test_no_stale_marker_passed_false_returns_false(self):
        gate = {"passed": False}
        assert _gate_passed_for_safe_to_voice("script_quality", gate) is False
