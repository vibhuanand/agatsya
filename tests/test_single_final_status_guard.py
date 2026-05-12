"""
Tests for TASK 4 — final status/safe_to_voice decision is the single
authoritative computation.

Verifies:
- script_approved requires ALL required gates to pass (not just OFP alone)
- telemetry entries cannot approve OR block status
- safe_to_voice requires status=script_approved AND all required gates AND
  no repair failures AND not pf_blocking AND similarity OK AND transformation OK
- intermediate status=script_approved is downgraded when required gates fail
- OpenAI final premium passing alone is not sufficient for script_approved
"""
from __future__ import annotations

import pytest

from app.services.agent_pipeline_service import (
    REQUIRED_SAFE_TO_VOICE_GATES,
    _gate_passed_for_safe_to_voice,
)


# ─── Helpers: mirror of final safe_to_voice computation ──────────────────────

def _all_gates_passed(gate_summary: dict) -> bool:
    return all(
        _gate_passed_for_safe_to_voice(name, gate_summary.get(name, {"passed": False}))
        for name in REQUIRED_SAFE_TO_VOICE_GATES
    )


def _final_status(
    intermediate_status: str,
    gate_summary: dict,
) -> str:
    """Replicate the final authoritative status guard."""
    if intermediate_status == "script_approved" and not _all_gates_passed(gate_summary):
        return "not_voice_ready_auto_retry_exhausted"
    return intermediate_status


def _safe_to_voice(
    status: str,
    gate_summary: dict,
    repair_failures: dict | None = None,
    pf_blocking: bool = False,
    transformation_ok: bool = True,
    similarity_high_risk: int = 0,
    similarity_max_allowed: int = 5,
) -> bool:
    """Replicate the final safe_to_voice computation."""
    all_pass = _all_gates_passed(gate_summary)
    no_repair_failures = all(
        not v for k, v in (repair_failures or {}).items() if k != "passed"
    )
    similarity_ok = similarity_high_risk <= similarity_max_allowed
    return (
        status == "script_approved"
        and all_pass
        and no_repair_failures
        and not pf_blocking
        and transformation_ok
        and similarity_ok
    )


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
        # Telemetry (must be ignored by gate check)
        "repair_routing":             {"route": "python_only"},
        "auto_fix":                   {"python_fixes_count": 2},
        "repair_telemetry":           {"repair_route": "python_only"},
        "repair_failures":            {"passed": False, "openai_repair_failed": False},
    }


def _no_failures() -> dict:
    return {
        "claude_script_repair_failed": False,
        "copyedit_repair_failed":      False,
        "metadata_repair_failed":      False,
        "retention_repair_failed":     False,
        "openai_repair_failed":        False,
    }


# ─── Tests: final status guard ────────────────────────────────────────────────

class TestFinalStatusGuard:
    def test_script_approved_when_all_required_gates_pass(self):
        status = _final_status("script_approved", _clean_gate_summary())
        assert status == "script_approved"

    def test_downgrade_when_script_quality_fails(self):
        gs = _clean_gate_summary()
        gs["script_quality"] = {"passed": False}
        status = _final_status("script_approved", gs)
        assert status == "not_voice_ready_auto_retry_exhausted"

    def test_downgrade_when_ofp_fails(self):
        """OFP alone does not gate status — all required gates must pass."""
        gs = _clean_gate_summary()
        gs["openai_final_premium"] = {"passed": False}
        status = _final_status("script_approved", gs)
        assert status == "not_voice_ready_auto_retry_exhausted"

    def test_ofp_passing_alone_not_sufficient_for_script_approved(self):
        """All other gates failing → downgraded even when OFP passes."""
        gs = {name: {"passed": False} for name in REQUIRED_SAFE_TO_VOICE_GATES}
        gs["openai_final_premium"] = {"passed": True}   # OFP passes
        gs["python_preflight"] = {"passed": False, "blocking": False}
        status = _final_status("script_approved", gs)
        assert status == "not_voice_ready_auto_retry_exhausted"

    def test_telemetry_repair_routing_cannot_block_status(self):
        gs = _clean_gate_summary()
        gs["repair_routing"] = {"route": "stop_not_voice_ready"}  # bad telemetry value
        status = _final_status("script_approved", gs)
        assert status == "script_approved"

    def test_telemetry_repair_failures_cannot_block_status(self):
        """repair_failures.passed=False is telemetry — must not affect status."""
        gs = _clean_gate_summary()
        gs["repair_failures"] = {"passed": False, "openai_repair_failed": True}
        # Status is still approved (repair_failures is not a required gate)
        status = _final_status("script_approved", gs)
        assert status == "script_approved"

    def test_non_approved_intermediate_status_not_touched(self):
        gs = _clean_gate_summary()
        for name in ("script_quality", "hindi_copyedit"):
            gs[name] = {"passed": False}
        status = _final_status("auto_rebuild_required", gs)
        assert status == "auto_rebuild_required"

    def test_missing_required_gate_triggers_downgrade(self):
        gs = _clean_gate_summary()
        del gs["retention_quality"]
        status = _final_status("script_approved", gs)
        assert status == "not_voice_ready_auto_retry_exhausted"

    def test_python_preflight_blocking_triggers_downgrade(self):
        gs = _clean_gate_summary()
        gs["python_preflight"] = {"passed": False, "blocking": True}
        status = _final_status("script_approved", gs)
        assert status == "not_voice_ready_auto_retry_exhausted"


# ─── Tests: safe_to_voice single source ──────────────────────────────────────

class TestSafeToVoiceSingleSource:
    def test_true_when_all_conditions_met(self):
        stv = _safe_to_voice("script_approved", _clean_gate_summary(), _no_failures())
        assert stv is True

    def test_false_when_status_not_approved(self):
        stv = _safe_to_voice(
            "not_voice_ready_auto_retry_exhausted", _clean_gate_summary(), _no_failures()
        )
        assert stv is False

    def test_false_when_required_gate_fails(self):
        gs = _clean_gate_summary()
        gs["hindi_copyedit"] = {"passed": False}
        stv = _safe_to_voice("script_approved", gs, _no_failures())
        assert stv is False

    def test_false_when_repair_failure(self):
        failures = _no_failures()
        failures["openai_repair_failed"] = True
        stv = _safe_to_voice("script_approved", _clean_gate_summary(), failures)
        assert stv is False

    def test_false_when_pf_blocking(self):
        stv = _safe_to_voice(
            "script_approved", _clean_gate_summary(), _no_failures(),
            pf_blocking=True,
        )
        assert stv is False

    def test_false_when_transformation_missing(self):
        stv = _safe_to_voice(
            "script_approved", _clean_gate_summary(), _no_failures(),
            transformation_ok=False,
        )
        assert stv is False

    def test_false_when_similarity_exceeds_limit(self):
        stv = _safe_to_voice(
            "script_approved", _clean_gate_summary(), _no_failures(),
            similarity_high_risk=10, similarity_max_allowed=5,
        )
        assert stv is False

    def test_false_when_ofp_gate_missing(self):
        gs = _clean_gate_summary()
        del gs["openai_final_premium"]
        stv = _safe_to_voice("script_approved", gs, _no_failures())
        assert stv is False

    def test_telemetry_noise_cannot_grant_safe_to_voice(self):
        """Even if gate_summary has telemetry claiming passed=True everywhere,
        the required gate allowlist is the only truth."""
        gs = {
            # Only telemetry keys — no required content gates
            "repair_routing":   {"passed": True, "route": "python_only"},
            "auto_fix":         {"passed": True},
            "repair_telemetry": {"passed": True},
        }
        stv = _safe_to_voice("script_approved", gs, _no_failures())
        # Required gates default to passed=False → all_gates_passed=False
        assert stv is False

    def test_all_repair_failure_types_individually_block(self):
        for key in (
            "claude_script_repair_failed",
            "copyedit_repair_failed",
            "metadata_repair_failed",
            "retention_repair_failed",
            "openai_repair_failed",
        ):
            failures = _no_failures()
            failures[key] = True
            stv = _safe_to_voice("script_approved", _clean_gate_summary(), failures)
            assert stv is False, f"Expected False when {key}=True"
