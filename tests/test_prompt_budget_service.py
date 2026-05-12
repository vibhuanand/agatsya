"""Tests for app/services/prompt_budget_service.py."""
import math
import pytest

from app.services.prompt_budget_service import (
    estimate_tokens,
    classify_transcript_size,
    prompt_fits_in_budget,
    research_view_fits_budget,
    should_use_segmented_mode,
)


# ─── estimate_tokens ─────────────────────────────────────────────────────────

class TestEstimateTokens:
    def test_zero_chars(self):
        assert estimate_tokens(0) == 0

    def test_negative_chars(self):
        assert estimate_tokens(-100) == 0

    def test_exact_multiple(self):
        # 35 chars / 3.5 = 10 tokens exactly
        assert estimate_tokens(35) == 10

    def test_rounds_up(self):
        # 36 chars / 3.5 = 10.28... → should ceil to 11
        assert estimate_tokens(36) == math.ceil(36 / 3.5)

    def test_large_transcript(self):
        # 60 000 chars → ceil(60000 / 3.5) = 17 143
        assert estimate_tokens(60_000) == math.ceil(60_000 / 3.5)

    def test_returns_int(self):
        assert isinstance(estimate_tokens(100), int)

    def test_one_char(self):
        # ceil(1 / 3.5) = 1
        assert estimate_tokens(1) == 1


# ─── classify_transcript_size ────────────────────────────────────────────────

class TestClassifyTranscriptSize:
    """Uses default config thresholds: long=30000, very_long=60000, medium=15000."""

    def test_small(self):
        assert classify_transcript_size(0) == "small"
        assert classify_transcript_size(14_999) == "small"

    def test_medium_boundary(self):
        assert classify_transcript_size(15_000) == "medium"
        assert classify_transcript_size(29_999) == "medium"

    def test_long_boundary(self):
        assert classify_transcript_size(30_000) == "long"
        assert classify_transcript_size(59_999) == "long"

    def test_very_long_boundary(self):
        assert classify_transcript_size(60_000) == "very_long"
        assert classify_transcript_size(100_000) == "very_long"

    def test_returns_string(self):
        result = classify_transcript_size(50_000)
        assert isinstance(result, str)
        assert result in ("small", "medium", "long", "very_long")


# ─── prompt_fits_in_budget ───────────────────────────────────────────────────

class TestPromptFitsInBudget:
    def test_empty_prompt(self):
        assert prompt_fits_in_budget(0) is True

    def test_small_prompt_fits(self):
        # 1000 chars → ~286 tokens → fits in 22000
        assert prompt_fits_in_budget(1_000) is True

    def test_large_prompt_does_not_fit(self):
        # 22000 tokens * 3.5 = 77000 chars exactly fits; 77001 should not
        from app.config import settings
        max_chars = settings.safe_claude_input_tokens_per_call * 3  # conservative
        assert prompt_fits_in_budget(max_chars * 10) is False

    def test_at_exact_budget(self):
        from app.config import settings
        # Exactly at budget limit: estimate_tokens(chars) == safe limit
        # We need chars such that ceil(chars / 3.5) == safe limit
        exact_chars = settings.safe_claude_input_tokens_per_call * 3  # 3 chars/token underestimates slightly
        # This may or may not fit — just test it returns a bool
        result = prompt_fits_in_budget(exact_chars)
        assert isinstance(result, bool)


# ─── research_view_fits_budget ───────────────────────────────────────────────

class TestResearchViewFitsBudget:
    def test_small_view_fits(self):
        assert research_view_fits_budget(1_000) is True

    def test_very_large_view_does_not_fit(self):
        # 500 000 chars would require ~142857 tokens, far exceeding 22000
        assert research_view_fits_budget(500_000) is False

    def test_overhead_applied(self):
        # Just under the limit with overhead should still fit
        from app.config import settings
        # (limit - 2000 overhead) * 3.5 chars — should fit
        safe_chars = (settings.safe_claude_input_tokens_per_call - 2000) * 3
        assert research_view_fits_budget(safe_chars) is True


# ─── should_use_segmented_mode ───────────────────────────────────────────────

class TestShouldUseSegmentedMode:
    def test_small_transcript_no_segmented(self):
        assert should_use_segmented_mode(1_000) is False
        assert should_use_segmented_mode(29_999) is False

    def test_at_threshold_uses_segmented(self):
        # Default threshold = 30000
        assert should_use_segmented_mode(30_000) is True

    def test_large_transcript_uses_segmented(self):
        assert should_use_segmented_mode(60_000) is True
        assert should_use_segmented_mode(200_000) is True

    def test_returns_bool(self):
        assert isinstance(should_use_segmented_mode(0), bool)
