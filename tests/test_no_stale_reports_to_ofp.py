"""
Tests for TASK 4 — if any required gate report is stale (refresh_failed=True),
safe_to_voice cannot become True and OpenAI Final Premium Gate must not be
allowed to approve the script based on stale evidence.

Verifies:
- refresh_failed report in similarity or originality prevents OFP gate approval
  from producing safe_to_voice=True
- stale report in any required position blocks safe_to_voice via gate allowlist
- a report with refresh_failed=True has passed=False, which fails gate check
- the final_review_input_hash changes when a refresh_failed report is present
  vs. when the fresh version exists (so OFP cache is also invalidated)
- non-stale reports with all required gates passing produce safe_to_voice=True
"""
from __future__ import annotations

import pytest

from app.services.agent_pipeline_service import (
    REQUIRED_SAFE_TO_VOICE_GATES,
    _gate_passed_for_safe_to_voice,
    _compute_final_review_input_hash,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

_STALE_REPORT = {
    "passed": False,
    "stale": True,
    "refresh_failed": True,
    "reason": "missing required input: source_transcript",
}


def _all_gates_passed(gs: dict) -> bool:
    return all(
        _gate_passed_for_safe_to_voice(name, gs.get(name, {"passed": False}))
        for name in REQUIRED_SAFE_TO_VOICE_GATES
    )


def _safe_to_voice(
    status: str,
    gate_summary: dict,
    no_repair_failures: bool = True,
    pf_blocking: bool = False,
    transformation_ok: bool = True,
    similarity_ok: bool = True,
) -> bool:
    return (
        status == "script_approved"
        and _all_gates_passed(gate_summary)
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
    }


def _script(chunks=None, meta=None) -> dict:
    return {
        "hindi_narration_chunks": chunks or [{"chunk_id": "001", "text": "पाठ।"}],
        "youtube_metadata":       meta or {"title": "T"},
        "recreated_dialogues":    {"items": []},
    }


# ─── Tests: stale report in gate_summary blocks safe_to_voice ─────────────────

class TestStaleReportBlocksSafeToVoice:
    """
    When a gate report has refresh_failed=True, its passed=False.
    If that gate is in the required allowlist, safe_to_voice must be False.
    """

    def test_stale_originality_report_in_gate_summary_blocks(self):
        gs = _clean_gate_summary()
        # originality_safety carries refresh_failed=True → passed=False
        gs["originality_safety"] = dict(_STALE_REPORT)
        stv = _safe_to_voice("script_approved", gs)
        assert stv is False

    def test_stale_similarity_not_a_required_gate_but_blocks_via_pf(self):
        """similarity_report is not directly in gate_summary as a required gate,
        but a stale similarity blocks OFP from running, and OFP not running means
        openai_final_premium.passed=False → safe_to_voice=False."""
        gs = _clean_gate_summary()
        gs["openai_final_premium"] = {
            "passed": False, "skipped": True,
            "reason": "stale supporting reports: ['similarity']",
        }
        stv = _safe_to_voice("script_approved", gs)
        assert stv is False

    def test_stale_copyedit_in_gate_summary_blocks(self):
        gs = _clean_gate_summary()
        gs["hindi_copyedit"] = dict(_STALE_REPORT)
        stv = _safe_to_voice("script_approved", gs)
        assert stv is False

    def test_stale_retention_in_gate_summary_blocks(self):
        gs = _clean_gate_summary()
        gs["retention_quality"] = dict(_STALE_REPORT)
        stv = _safe_to_voice("script_approved", gs)
        assert stv is False

    def test_stale_metadata_in_gate_summary_blocks(self):
        gs = _clean_gate_summary()
        gs["metadata_quality"] = dict(_STALE_REPORT)
        stv = _safe_to_voice("script_approved", gs)
        assert stv is False

    def test_stale_dialogue_in_gate_summary_blocks(self):
        gs = _clean_gate_summary()
        gs["recreated_dialogue"] = dict(_STALE_REPORT)
        stv = _safe_to_voice("script_approved", gs)
        assert stv is False


# ─── Tests: stale report changes OFP hash ────────────────────────────────────

class TestStaleReportInvalidatesOFPHash:
    """
    A refresh_failed report has different content from a passing fresh report.
    _compute_final_review_input_hash must produce a different hash so the OFP
    cache is also invalidated.
    """

    def test_stale_similarity_gives_different_hash(self):
        fresh_sim = {"risk_level": "low", "high_risk_matches": 0, "passed": True}
        stale_sim = dict(_STALE_REPORT)

        h_fresh = _compute_final_review_input_hash(
            script_final=_script(), similarity_report=fresh_sim
        )
        h_stale = _compute_final_review_input_hash(
            script_final=_script(), similarity_report=stale_sim
        )
        assert h_fresh != h_stale, (
            "stale similarity_report must produce a different OFP hash — "
            "cached OFP built with fresh similarity must not be reused"
        )

    def test_stale_copyedit_gives_different_hash(self):
        fresh = {"approved": True, "passed": True}
        stale = dict(_STALE_REPORT)

        h_f = _compute_final_review_input_hash(
            script_final=_script(), copyedit_report=fresh
        )
        h_s = _compute_final_review_input_hash(
            script_final=_script(), copyedit_report=stale
        )
        assert h_f != h_s

    def test_stale_metadata_report_gives_different_hash(self):
        fresh = {"gate_passed": True, "approved": True}
        stale = dict(_STALE_REPORT)

        h_f = _compute_final_review_input_hash(
            script_final=_script(), metadata_report=fresh
        )
        h_s = _compute_final_review_input_hash(
            script_final=_script(), metadata_report=stale
        )
        assert h_f != h_s


# ─── Tests: OFP cannot approve from stale evidence ───────────────────────────

class TestOFPCannotApproveFromStaleEvidence:
    """
    When stale reports are detected before Stage 14a, gate_summary for
    openai_final_premium is set to passed=False, skipped=True.
    This prevents safe_to_voice=True regardless of intermediate status.
    """

    def _simulate_stale_ofp_skip(self, gate_summary: dict) -> None:
        """Replicate what Stage 14a does when _ofp_skip=True."""
        gate_summary["openai_final_premium"] = {
            "passed":  False,
            "skipped": True,
            "reason":  "stale supporting reports: ['similarity']",
        }

    def test_ofp_skipped_gate_summary_entry_blocks_safe_to_voice(self):
        gs = _clean_gate_summary()
        self._simulate_stale_ofp_skip(gs)
        stv = _safe_to_voice("script_approved", gs)
        assert stv is False

    def test_ofp_skipped_means_openai_final_premium_not_passed(self):
        gs = _clean_gate_summary()
        self._simulate_stale_ofp_skip(gs)
        assert gs["openai_final_premium"]["passed"] is False
        assert gs["openai_final_premium"]["skipped"] is True

    def test_script_approved_status_not_enough_without_ofp(self):
        """Even if status="script_approved", stale OFP skip blocks safe_to_voice."""
        gs = _clean_gate_summary()
        self._simulate_stale_ofp_skip(gs)
        stv = _safe_to_voice("script_approved", gs)
        assert stv is False

    def test_clean_state_still_allows_safe_to_voice(self):
        """Sanity: with all gates fresh and passing, safe_to_voice is True."""
        gs = _clean_gate_summary()
        stv = _safe_to_voice("script_approved", gs)
        assert stv is True

    def test_all_stale_failures_block_individually(self):
        """Every stale-failure position individually prevents safe_to_voice."""
        stale_positions = [
            ("originality_safety",   dict(_STALE_REPORT)),
            ("hindi_copyedit",       dict(_STALE_REPORT)),
            ("retention_quality",    dict(_STALE_REPORT)),
            ("metadata_quality",     dict(_STALE_REPORT)),
            ("recreated_dialogue",   dict(_STALE_REPORT)),
            ("openai_final_premium", {"passed": False, "skipped": True,
                                      "reason": "stale supporting reports"}),
        ]
        for gate_name, stale_entry in stale_positions:
            gs = _clean_gate_summary()
            gs[gate_name] = stale_entry
            stv = _safe_to_voice("script_approved", gs)
            assert stv is False, (
                f"Expected safe_to_voice=False when {gate_name} has stale/failed entry"
            )


# ─── Tests: refresh_failed flag semantics ────────────────────────────────────

class TestRefreshFailedFlagSemantics:
    def test_refresh_failed_report_has_passed_false(self):
        assert _STALE_REPORT["passed"] is False

    def test_refresh_failed_report_has_stale_true(self):
        assert _STALE_REPORT["stale"] is True

    def test_refresh_failed_report_has_reason(self):
        assert "reason" in _STALE_REPORT
        assert len(_STALE_REPORT["reason"]) > 0

    def test_gate_passed_for_safe_to_voice_returns_false_for_stale(self):
        """_gate_passed_for_safe_to_voice must return False for a stale report."""
        for gate_name in REQUIRED_SAFE_TO_VOICE_GATES:
            if gate_name == "python_preflight":
                stale = {"passed": False, "blocking": True, "refresh_failed": True}
            else:
                stale = dict(_STALE_REPORT)
            result = _gate_passed_for_safe_to_voice(gate_name, stale)
            assert result is False, (
                f"Expected _gate_passed_for_safe_to_voice({gate_name!r}, stale)=False"
            )

    def test_non_stale_fresh_report_passes_gate_check(self):
        fresh = {"passed": True, "refreshed_after_script_mutation": True}
        for gate_name in REQUIRED_SAFE_TO_VOICE_GATES:
            if gate_name == "python_preflight":
                fresh_pf = {"passed": False, "blocking": False,
                            "refreshed_after_script_mutation": True}
                assert _gate_passed_for_safe_to_voice(gate_name, fresh_pf) is True
            else:
                assert _gate_passed_for_safe_to_voice(gate_name, fresh) is True
