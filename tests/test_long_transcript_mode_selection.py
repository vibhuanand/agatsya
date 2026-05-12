"""
Tests for transcript-length-agnostic fact lock mode selection.

Covers:
- FACT_LOCK_MODE=auto dispatch (short → research_view, long → segmented)
- Auto-switch in agent_pipeline_service for premium + threshold
- prompt_budget_service.should_use_segmented_mode thresholds
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import pytest

# Ensure app.services.fact_lock_service is in sys.modules before any
# @patch("app.services.fact_lock_service.*") decorator is evaluated.
# This is belt-and-suspenders alongside the lazy __getattr__ in __init__.py.
import app.services.fact_lock_service  # noqa: F401

from app.services.prompt_budget_service import (
    classify_transcript_size,
    should_use_segmented_mode,
    estimate_tokens,
)


# ─── classify_transcript_size edge cases ─────────────────────────────────────

class TestClassifyTranscriptSizeEdgeCases:
    def test_exactly_at_long_threshold(self):
        result = classify_transcript_size(30_000)
        assert result == "long"

    def test_one_below_long_threshold(self):
        result = classify_transcript_size(29_999)
        assert result == "medium"

    def test_exactly_at_very_long_threshold(self):
        result = classify_transcript_size(60_000)
        assert result == "very_long"

    def test_one_below_very_long_threshold(self):
        result = classify_transcript_size(59_999)
        assert result == "long"

    def test_zero(self):
        assert classify_transcript_size(0) == "small"

    def test_all_classes_returned(self):
        sizes = [0, 15_000, 30_000, 60_000]
        classes = [classify_transcript_size(s) for s in sizes]
        assert set(classes) == {"small", "medium", "long", "very_long"}


# ─── should_use_segmented_mode ────────────────────────────────────────────────

class TestShouldUseSegmentedMode:
    def test_below_threshold_no_segmented(self):
        assert should_use_segmented_mode(0) is False
        assert should_use_segmented_mode(10_000) is False
        assert should_use_segmented_mode(29_999) is False

    def test_at_threshold_uses_segmented(self):
        assert should_use_segmented_mode(30_000) is True

    def test_above_threshold_uses_segmented(self):
        assert should_use_segmented_mode(30_001) is True
        assert should_use_segmented_mode(200_000) is True


# ─── fact_lock_service auto mode dispatch ────────────────────────────────────

class TestFactLockAutoModeDispatch:
    """Test that run_fact_lock dispatches correctly for FACT_LOCK_MODE=auto."""

    def _make_fact_lock_result(self) -> dict:
        return {
            "case_name": "Test Case",
            "source_summary": "test",
            "verified_people": [],
            "verified_dates": [],
            "verified_locations": [],
            "verified_timeline": [],
            "legal_outcome": {},
            "key_evidence_or_turning_points": [],
            "important_audio_or_call_moments": [],
            "emotional_details": [],
            "recreated_scene_candidates": [],
            "facts_to_verify_externally": [],
            "must_not_say": [],
        }

    @patch("app.services.fact_lock_service._run_research_view_mode")
    @patch("app.services.fact_lock_service.settings")
    def test_auto_mode_small_transcript_uses_research_view(
        self, mock_settings, mock_rv
    ):
        mock_settings.fact_lock_mode = "auto"
        mock_settings.long_transcript_clean_chars_threshold = 30_000
        mock_settings.claude_max_tokens = 12_000
        mock_rv.return_value = self._make_fact_lock_result()

        from app.services.fact_lock_service import run_fact_lock
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            facts_dir = Path(tmpdir)
            result = run_fact_lock(
                case_hint="Test",
                episode_number="001",
                source_url="http://example.com",
                transcript_research_view="short view",
                facts_dir=facts_dir,
                clean_transcript="x" * 5_000,  # 5k chars < 30k threshold
            )

        mock_rv.assert_called_once()
        assert result["case_name"] == "Test Case"

    @patch("app.services.fact_lock_service._run_segmented_mode")
    @patch("app.services.fact_lock_service.settings")
    def test_auto_mode_long_transcript_uses_segmented(
        self, mock_settings, mock_seg
    ):
        mock_settings.fact_lock_mode = "auto"
        mock_settings.long_transcript_clean_chars_threshold = 30_000
        mock_settings.claude_max_tokens = 12_000
        mock_seg.return_value = self._make_fact_lock_result()

        from app.services.fact_lock_service import run_fact_lock
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            facts_dir = Path(tmpdir)
            result = run_fact_lock(
                case_hint="Test",
                episode_number="001",
                source_url="http://example.com",
                transcript_research_view="research view",
                facts_dir=facts_dir,
                clean_transcript="x" * 35_000,  # 35k chars > 30k threshold
            )

        mock_seg.assert_called_once()
        assert result["case_name"] == "Test Case"

    @patch("app.services.fact_lock_service._run_research_view_mode")
    @patch("app.services.fact_lock_service.settings")
    def test_auto_mode_no_clean_transcript_falls_back_to_research_view(
        self, mock_settings, mock_rv
    ):
        mock_settings.fact_lock_mode = "auto"
        mock_settings.long_transcript_clean_chars_threshold = 30_000
        mock_settings.claude_max_tokens = 12_000
        mock_rv.return_value = self._make_fact_lock_result()

        from app.services.fact_lock_service import run_fact_lock
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            facts_dir = Path(tmpdir)
            run_fact_lock(
                case_hint="Test",
                episode_number="001",
                source_url="http://example.com",
                transcript_research_view="view",
                facts_dir=facts_dir,
                clean_transcript="",  # no clean transcript
            )

        mock_rv.assert_called_once()

    @patch("app.services.fact_lock_service._run_segmented_mode")
    @patch("app.services.fact_lock_service.settings")
    def test_explicit_segmented_mode_ignores_threshold(
        self, mock_settings, mock_seg
    ):
        """FACT_LOCK_MODE=segmented should always use segmented, regardless of size."""
        mock_settings.fact_lock_mode = "segmented"
        mock_settings.long_transcript_clean_chars_threshold = 30_000
        mock_settings.claude_max_tokens = 12_000
        mock_seg.return_value = self._make_fact_lock_result()

        from app.services.fact_lock_service import run_fact_lock
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            facts_dir = Path(tmpdir)
            run_fact_lock(
                case_hint="Test",
                episode_number="001",
                source_url="http://example.com",
                transcript_research_view="view",
                facts_dir=facts_dir,
                clean_transcript="x" * 1_000,  # small but mode=segmented
            )

        mock_seg.assert_called_once()


# ─── Token estimation for boundary decisions ─────────────────────────────────

class TestTokenEstimationBoundaryDecisions:
    def test_30k_chars_above_safe_budget_signal(self):
        """30 000 chars should estimate to enough tokens to trigger segmented."""
        tokens = estimate_tokens(30_000)
        # 30000 / 3.5 ≈ 8571 tokens — fine for research_view
        # but should_use_segmented_mode checks chars, not tokens
        assert should_use_segmented_mode(30_000) is True

    def test_1000_char_prompt_well_within_budget(self):
        from app.services.prompt_budget_service import prompt_fits_in_budget
        assert prompt_fits_in_budget(1_000) is True

    def test_very_large_prompt_exceeds_budget(self):
        from app.services.prompt_budget_service import prompt_fits_in_budget
        # 22000 * 3.5 = 77000 chars at limit; 200000 chars should exceed
        assert prompt_fits_in_budget(200_000) is False
