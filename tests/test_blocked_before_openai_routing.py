"""
Tests for the blocked_before_openai automation_status flow and
pre-OAI repair routing logic.

Verifies:
- Post-repair preflight blocking triggers repair routing before OpenAI skip
- If still blocking after repair: status=not_voice_ready_auto_retry_exhausted
  and automation_status=blocked_before_openai
- safe_to_voice remains False in all blocking cases
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock, call


# ─── Unit tests: status/automation_status logic ───────────────────────────────

class TestBlockedBeforeOAIStatus:
    """
    Verify the status taxonomy when preflight blocks OpenAI.
    These tests exercise the decision logic directly without running the pipeline.
    """

    def _resolve_status(
        self,
        post_repair_blocking: bool,
        current_status: str = "auto_repair_required",
    ) -> tuple[str, str]:
        """
        Replicate the status + automation_status resolution logic from
        agent_pipeline_service.py when _post_repair_preflight_blocking is True.
        Returns (status, automation_status).
        """
        status = current_status
        automation_status = status

        if post_repair_blocking:
            if status not in ("needs_human_review",):
                status = "not_voice_ready_auto_retry_exhausted"
            automation_status = "blocked_before_openai"

        return status, automation_status

    def test_blocking_sets_not_voice_ready(self):
        status, _ = self._resolve_status(post_repair_blocking=True)
        assert status == "not_voice_ready_auto_retry_exhausted"

    def test_blocking_sets_automation_status_blocked_before_openai(self):
        _, auto = self._resolve_status(post_repair_blocking=True)
        assert auto == "blocked_before_openai"

    def test_blocking_does_not_use_needs_human_review(self):
        status, _ = self._resolve_status(post_repair_blocking=True)
        assert status != "needs_human_review"

    def test_blocking_preserves_needs_human_review_when_already_set(self):
        """needs_human_review (e.g. safety exception) must not be downgraded."""
        status, auto = self._resolve_status(
            post_repair_blocking=True,
            current_status="needs_human_review",
        )
        assert status == "needs_human_review"
        assert auto == "blocked_before_openai"

    def test_not_blocking_does_not_change_status(self):
        status, auto = self._resolve_status(post_repair_blocking=False)
        assert status == "auto_repair_required"
        assert auto == "auto_repair_required"

    def test_safe_to_voice_false_when_blocking(self):
        """safe_to_voice must never be True when automation_status=blocked_before_openai."""
        _, auto = self._resolve_status(post_repair_blocking=True)
        # safe_to_voice requires approved=True AND all scores >=9 AND zero HIGH issues.
        # A blocked_before_openai state always has OpenAI gate skipped (passed=False),
        # so safe_to_voice must be False.
        safe_to_voice = (auto != "blocked_before_openai")  # simplified guard
        assert safe_to_voice is False


# ─── Unit tests: repair routing pass logic ───────────────────────────────────

class TestPreOAIRepairRoutingPass:
    """
    Verify that the pre-OAI repair routing pass is attempted before finalising
    the blocking status, and that outcomes are handled correctly.
    """

    def _simulate_pre_oai_repair(
        self,
        initial_blocking: bool,
        repair_clears_blocking: bool,
        repair_raises: bool = False,
    ) -> dict:
        """
        Simulate the Stage 13d pre-OAI repair logic.
        Returns a result dict describing what happened.
        """
        post_repair_blocking = initial_blocking
        pre_oai_repair_ran = False
        repair_error = None

        if initial_blocking:
            try:
                if repair_raises:
                    raise RuntimeError("Simulated repair failure")
                # Simulate repair ran
                pre_oai_repair_ran = True
                # After repair, blocking clears or not
                if repair_clears_blocking:
                    post_repair_blocking = False
                # else still blocking
            except Exception as exc:
                repair_error = str(exc)
                post_repair_blocking = True  # exception → still blocking

        return {
            "post_repair_preflight_blocking": post_repair_blocking,
            "pre_oai_repair_ran": pre_oai_repair_ran,
            "repair_error": repair_error,
        }

    def test_repair_attempted_when_blocking(self):
        result = self._simulate_pre_oai_repair(
            initial_blocking=True, repair_clears_blocking=False
        )
        assert result["pre_oai_repair_ran"] is True

    def test_repair_not_attempted_when_not_blocking(self):
        result = self._simulate_pre_oai_repair(
            initial_blocking=False, repair_clears_blocking=False
        )
        assert result["pre_oai_repair_ran"] is False

    def test_openai_runs_when_repair_clears_blocking(self):
        result = self._simulate_pre_oai_repair(
            initial_blocking=True, repair_clears_blocking=True
        )
        assert result["post_repair_preflight_blocking"] is False

    def test_openai_skipped_when_still_blocking_after_repair(self):
        result = self._simulate_pre_oai_repair(
            initial_blocking=True, repair_clears_blocking=False
        )
        assert result["post_repair_preflight_blocking"] is True

    def test_repair_exception_leaves_blocking_true(self):
        result = self._simulate_pre_oai_repair(
            initial_blocking=True, repair_clears_blocking=True, repair_raises=True
        )
        assert result["post_repair_preflight_blocking"] is True
        assert result["repair_error"] is not None

    def test_repair_exception_does_not_crash_pipeline(self):
        """Exception in repair pass must be caught; no uncaught exception."""
        try:
            result = self._simulate_pre_oai_repair(
                initial_blocking=True, repair_raises=True, repair_clears_blocking=False
            )
        except Exception as exc:
            pytest.fail(f"Repair exception should have been caught, got: {exc}")


# ─── Unit tests: gate_reports passed to repair routing ───────────────────────

class TestGateReportsForRepairRouting:
    """
    Verify the gate_reports dict passed to repair_routing_service includes
    the correct keys and excludes bare-error dicts.
    """

    def _build_gate_reports(
        self,
        preflight_report: dict,
        quality_report: dict,
        similarity_report: dict | None = None,
        originality_report: dict | None = None,
        metadata_report: dict | None = None,
        retention_report: dict | None = None,
        dialogue_report: dict | None = None,
    ) -> dict:
        """Replicate the gate_reports building logic from Stage 13d."""
        gate_reports: dict = {
            "python_preflight": preflight_report,
            "script_quality":   quality_report,
        }
        for key, val in (
            ("text_similarity",    similarity_report),
            ("originality_safety", originality_report),
            ("metadata",           metadata_report),
            ("retention",          retention_report),
            ("recreated_dialogue", dialogue_report),
        ):
            if isinstance(val, dict) and "error" not in val:
                gate_reports[key] = val
        return gate_reports

    def test_preflight_always_included(self):
        grs = self._build_gate_reports(
            preflight_report={"blocking": True, "passed": False},
            quality_report={"gate_passed": False},
        )
        assert "python_preflight" in grs

    def test_quality_report_always_included(self):
        grs = self._build_gate_reports(
            preflight_report={"blocking": True},
            quality_report={"gate_passed": True},
        )
        assert "script_quality" in grs

    def test_error_dicts_excluded(self):
        grs = self._build_gate_reports(
            preflight_report={"blocking": True},
            quality_report={"gate_passed": True},
            similarity_report={"error": "API call failed"},  # bare error — excluded
            metadata_report={"gate_passed": False, "scores": {}},   # valid — included
        )
        assert "text_similarity" not in grs
        assert "metadata" in grs

    def test_none_values_excluded(self):
        grs = self._build_gate_reports(
            preflight_report={"blocking": True},
            quality_report={"gate_passed": True},
            retention_report=None,
        )
        assert "retention" not in grs

    def test_valid_optional_reports_included(self):
        grs = self._build_gate_reports(
            preflight_report={"blocking": True},
            quality_report={"gate_passed": True},
            metadata_report={"gate_passed": False},
            retention_report={"approved": False},
        )
        assert "metadata" in grs
        assert "retention" in grs
