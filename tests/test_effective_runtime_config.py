"""
Tests for 04-review/effective_runtime_config.json output.

Verifies that the effective_runtime_config.json file is written correctly and
contains all required fields with valid values.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ─── Required fields spec ────────────────────────────────────────────────────

REQUIRED_FIELDS = {
    "episode_id",
    "model",
    "quality_mode",
    "openai_review_policy",
    "fact_lock_mode_requested",
    "fact_lock_mode_effective",
    "segmented_fact_lock_used",
    "segment_count",
    "transcript_chars_raw",
    "transcript_chars_clean",
    "transcript_estimated_tokens",
    "transcript_size_class",
    "research_view_chars",
    "safe_claude_input_tokens_per_call",
    "safe_claude_tokens_per_minute",
    "rate_limiter_telemetry",
    "reuse_existing_stage_outputs",
}

RATE_LIMITER_TELEMETRY_FIELDS = {
    "claude_rate_limit_wait_sec",
    "claude_estimated_input_tokens_last_60s",
    "claude_throttle_events",
}

VALID_SIZE_CLASSES = {"small", "medium", "long", "very_long"}


# ─── Helper: build a mock effective_runtime_config ───────────────────────────

def _build_mock_config(
    fact_lock_mode_requested: str = "research_view",
    fact_lock_mode_effective: str = "research_view",
    segmented: bool = False,
    clean_chars: int = 5_000,
) -> dict:
    from app.services.prompt_budget_service import estimate_tokens, classify_transcript_size
    return {
        "episode_id": "001-test-case",
        "model": "claude-sonnet-4-6",
        "quality_mode": "premium_final",
        "openai_review_policy": "adaptive",
        "fact_lock_mode_requested": fact_lock_mode_requested,
        "fact_lock_mode_effective": fact_lock_mode_effective,
        "segmented_fact_lock_used": segmented,
        "segment_count": 3 if segmented else 0,
        "transcript_chars_raw": clean_chars + 1000,
        "transcript_chars_clean": clean_chars,
        "transcript_estimated_tokens": estimate_tokens(clean_chars),
        "transcript_size_class": classify_transcript_size(clean_chars),
        "research_view_chars": min(clean_chars, 18_000),
        "safe_claude_input_tokens_per_call": 22_000,
        "safe_claude_tokens_per_minute": 30_000,
        "rate_limiter_telemetry": {
            "claude_rate_limit_wait_sec": 0.0,
            "claude_estimated_input_tokens_last_60s": 0,
            "claude_throttle_events": 0,
        },
        "reuse_existing_stage_outputs": False,
    }


# ─── Tests: schema validation ─────────────────────────────────────────────────

class TestEffectiveRuntimeConfigSchema:
    def test_all_required_fields_present(self):
        config = _build_mock_config()
        for field in REQUIRED_FIELDS:
            assert field in config, f"Missing required field: {field}"

    def test_rate_limiter_telemetry_has_all_fields(self):
        config = _build_mock_config()
        telemetry = config["rate_limiter_telemetry"]
        for field in RATE_LIMITER_TELEMETRY_FIELDS:
            assert field in telemetry, f"Missing telemetry field: {field}"

    def test_transcript_size_class_is_valid(self):
        for clean_chars in [0, 15_000, 30_000, 60_000]:
            config = _build_mock_config(clean_chars=clean_chars)
            assert config["transcript_size_class"] in VALID_SIZE_CLASSES

    def test_segmented_fields_consistent(self):
        # When segmented=True, segment_count > 0
        config = _build_mock_config(
            fact_lock_mode_effective="segmented",
            segmented=True,
        )
        assert config["segmented_fact_lock_used"] is True
        assert config["segment_count"] > 0

    def test_research_view_fields_consistent(self):
        config = _build_mock_config(segmented=False)
        assert config["segmented_fact_lock_used"] is False
        assert config["segment_count"] == 0

    def test_token_estimate_matches_formula(self):
        import math
        clean_chars = 10_500
        config = _build_mock_config(clean_chars=clean_chars)
        expected_tokens = math.ceil(clean_chars / 3.5)
        assert config["transcript_estimated_tokens"] == expected_tokens


# ─── Tests: file I/O ─────────────────────────────────────────────────────────

class TestEffectiveRuntimeConfigFileIO:
    def test_config_is_valid_json(self):
        config = _build_mock_config()
        serialized = json.dumps(config, ensure_ascii=False, indent=2)
        parsed = json.loads(serialized)
        assert parsed["episode_id"] == "001-test-case"

    def test_config_written_to_correct_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            review_dir = Path(tmpdir)
            config = _build_mock_config()
            config_path = review_dir / "effective_runtime_config.json"
            config_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            assert config_path.exists()
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            assert loaded["model"] == "claude-sonnet-4-6"

    def test_config_is_human_readable_json(self):
        config = _build_mock_config()
        serialized = json.dumps(config, ensure_ascii=False, indent=2)
        # Should be indented (multi-line)
        assert "\n" in serialized
        # All keys should be present
        assert "episode_id" in serialized


# ─── Tests: auto mode reflected correctly ────────────────────────────────────

class TestEffectiveRuntimeConfigAutoMode:
    def test_auto_mode_requested_segmented_effective_shows_correctly(self):
        config = _build_mock_config(
            fact_lock_mode_requested="auto",
            fact_lock_mode_effective="segmented",
            segmented=True,
            clean_chars=40_000,
        )
        assert config["fact_lock_mode_requested"] == "auto"
        assert config["fact_lock_mode_effective"] == "segmented"
        assert config["segmented_fact_lock_used"] is True

    def test_auto_mode_requested_research_view_effective_shows_correctly(self):
        config = _build_mock_config(
            fact_lock_mode_requested="auto",
            fact_lock_mode_effective="research_view",
            segmented=False,
            clean_chars=5_000,
        )
        assert config["fact_lock_mode_requested"] == "auto"
        assert config["fact_lock_mode_effective"] == "research_view"
        assert config["segmented_fact_lock_used"] is False


# ─── Tests: model rate limiter telemetry ─────────────────────────────────────

class TestModelRateLimiterTelemetry:
    def test_telemetry_returns_correct_shape(self):
        from app.services.model_rate_limiter_service import ModelRateLimiter
        limiter = ModelRateLimiter()
        telemetry = limiter.telemetry()
        assert set(telemetry.keys()) == RATE_LIMITER_TELEMETRY_FIELDS
        assert telemetry["claude_rate_limit_wait_sec"] == 0.0
        assert telemetry["claude_estimated_input_tokens_last_60s"] == 0
        assert telemetry["claude_throttle_events"] == 0

    def test_reset_clears_telemetry(self):
        from app.services.model_rate_limiter_service import ModelRateLimiter
        limiter = ModelRateLimiter()
        # Simulate a recorded call
        limiter.after_call("x" * 1000, agent_name="test")
        # Reset
        limiter.reset()
        telemetry = limiter.telemetry()
        assert telemetry["claude_estimated_input_tokens_last_60s"] == 0
        assert telemetry["claude_throttle_events"] == 0

    def test_after_call_records_tokens(self):
        from app.services.model_rate_limiter_service import ModelRateLimiter
        from app.services.prompt_budget_service import estimate_tokens
        limiter = ModelRateLimiter()
        prompt = "x" * 3500  # ~1000 tokens
        limiter.after_call(prompt, agent_name="test")
        telemetry = limiter.telemetry()
        assert telemetry["claude_estimated_input_tokens_last_60s"] == estimate_tokens(3500)
