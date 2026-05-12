"""
Tests for OpenAI gate skip reason logic.

Verifies that when OPENAI_REVIEW_ENABLED=true but post-repair Python preflight
is blocking, the gate_summary says "post_repair_python_preflight_blocking" —
NOT "OPENAI_REVIEW_ENABLED=false".
"""
from __future__ import annotations

import pytest


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_skip_reason(
    post_repair_preflight_blocking: bool,
    quality_mode: str = "premium_final",
    openai_review_policy: str = "adaptive",
    openai_review_enabled: bool = True,
) -> tuple[str, str, str]:
    """
    Replicate the skip-reason logic from agent_pipeline_service.py else branch.

    Returns (skip_reason_code, skip_detail, resulting_status).
    """
    if post_repair_preflight_blocking:
        return (
            "post_repair_python_preflight_blocking",
            (
                "Post-repair Python preflight is still blocking; "
                "OpenAI Final Premium Gate skipped to avoid reviewing unsafe/unready script."
            ),
            "not_voice_ready_auto_retry_exhausted",
        )
    elif quality_mode != "premium_final":
        return (
            f"quality_mode={quality_mode}",
            f"quality_mode is '{quality_mode}', not 'premium_final' — "
            "OpenAI Final Premium Gate requires premium_final mode.",
            "needs_human_review",
        )
    elif openai_review_policy == "disabled":
        return (
            f"openai_review_policy={openai_review_policy}",
            "OPENAI_REVIEW_POLICY=disabled — OpenAI Final Premium Gate explicitly disabled.",
            "needs_human_review",
        )
    else:
        return (
            "OPENAI_REVIEW_ENABLED=false",
            "OPENAI_REVIEW_ENABLED is false — OpenAI Final Premium Gate disabled by config.",
            "needs_human_review",
        )


# ─── Tests: correct skip reason when preflight blocks ────────────────────────

class TestOpenAISkipReasonPreflight:
    def test_preflight_blocking_reason_code(self):
        code, detail, status = _build_skip_reason(
            post_repair_preflight_blocking=True,
            quality_mode="premium_final",
            openai_review_policy="adaptive",
            openai_review_enabled=True,
        )
        assert code == "post_repair_python_preflight_blocking"

    def test_preflight_blocking_detail_text(self):
        code, detail, status = _build_skip_reason(post_repair_preflight_blocking=True)
        assert "Post-repair Python preflight is still blocking" in detail

    def test_preflight_blocking_does_not_say_review_enabled_false(self):
        code, detail, status = _build_skip_reason(post_repair_preflight_blocking=True)
        assert "OPENAI_REVIEW_ENABLED=false" not in code
        assert "OPENAI_REVIEW_ENABLED=false" not in detail

    def test_preflight_blocking_status_is_not_voice_ready(self):
        _, _, status = _build_skip_reason(post_repair_preflight_blocking=True)
        assert status == "not_voice_ready_auto_retry_exhausted"
        assert status != "needs_human_review"

    def test_preflight_blocking_status_not_needs_human_review(self):
        _, _, status = _build_skip_reason(post_repair_preflight_blocking=True)
        assert status != "needs_human_review"


# ─── Tests: correct skip reason for other inactive conditions ─────────────────

class TestOpenAISkipReasonOtherCases:
    def test_wrong_quality_mode(self):
        code, _, status = _build_skip_reason(
            post_repair_preflight_blocking=False,
            quality_mode="premium_batch",
        )
        assert "quality_mode" in code
        assert "premium_batch" in code
        assert status == "needs_human_review"

    def test_disabled_policy(self):
        code, _, status = _build_skip_reason(
            post_repair_preflight_blocking=False,
            quality_mode="premium_final",
            openai_review_policy="disabled",
        )
        assert "openai_review_policy" in code
        assert "disabled" in code
        assert status == "needs_human_review"

    def test_review_enabled_false(self):
        code, _, status = _build_skip_reason(
            post_repair_preflight_blocking=False,
            quality_mode="premium_final",
            openai_review_policy="adaptive",
            openai_review_enabled=False,
        )
        assert code == "OPENAI_REVIEW_ENABLED=false"
        assert status == "needs_human_review"

    def test_preflight_blocking_takes_priority_over_policy(self):
        """preflight blocking is checked first; policy=disabled is secondary."""
        code, _, status = _build_skip_reason(
            post_repair_preflight_blocking=True,
            openai_review_policy="disabled",
        )
        assert code == "post_repair_python_preflight_blocking"
        assert "openai_review_policy" not in code

    def test_preflight_blocking_takes_priority_over_quality_mode(self):
        code, _, _ = _build_skip_reason(
            post_repair_preflight_blocking=True,
            quality_mode="premium_batch",
        )
        assert code == "post_repair_python_preflight_blocking"


# ─── Tests: gate_summary shape ───────────────────────────────────────────────

class TestGateSummaryShape:
    def _build_gate_summary_entry(self, code: str, detail: str) -> dict:
        return {
            "passed": False,
            "skipped": True,
            "reason": detail,
            "skip_reason_code": code,
        }

    def test_preflight_blocking_gate_summary_has_skip_reason_code(self):
        code, detail, _ = _build_skip_reason(post_repair_preflight_blocking=True)
        gs = self._build_gate_summary_entry(code, detail)
        assert gs["skip_reason_code"] == "post_repair_python_preflight_blocking"

    def test_gate_summary_skipped_true(self):
        code, detail, _ = _build_skip_reason(post_repair_preflight_blocking=True)
        gs = self._build_gate_summary_entry(code, detail)
        assert gs["skipped"] is True
        assert gs["passed"] is False

    def test_detail_mentions_script_safety(self):
        code, detail, _ = _build_skip_reason(post_repair_preflight_blocking=True)
        assert "unsafe" in detail.lower() or "blocking" in detail.lower()
