"""
Tests for rate-limit retry logic and RateLimitExhaustedError in claude_client.py.

These tests mock the Anthropic client so no real API calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import pytest
import anthropic

from app.services.claude_client import (
    call_claude_agent,
    RateLimitExhaustedError,
    _RATE_LIMIT_RETRY_DELAYS,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_message(text: str = "{}") -> MagicMock:
    """Return a mock Anthropic message object with .content[0].text and .stop_reason."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.stop_reason = "end_turn"
    return msg


def _rate_limit_error() -> anthropic.RateLimitError:
    """Return a real RateLimitError (no network needed — instantiated directly)."""
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    response.text = "rate limited"
    return anthropic.RateLimitError(
        message="Rate limit exceeded",
        response=response,
        body={"error": {"message": "rate limited"}},
    )


# ─── Tests: successful call ───────────────────────────────────────────────────

class TestCallClaudeAgentSuccess:
    @patch("app.services.claude_client.anthropic.Anthropic")
    def test_returns_text_and_stop_reason(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_message('{"key": "value"}')
        mock_anthropic_cls.return_value = mock_client

        raw, stop = call_claude_agent("test prompt", agent_name="test_agent")

        assert raw == '{"key": "value"}'
        assert stop == "end_turn"

    @patch("app.services.claude_client.anthropic.Anthropic")
    def test_single_attempt_on_success(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_message("{}")
        mock_anthropic_cls.return_value = mock_client

        call_claude_agent("prompt", agent_name="agent")

        # Only one API call should be made
        assert mock_client.messages.create.call_count == 1


# ─── Tests: 429 retry then success ───────────────────────────────────────────

class TestCallClaudeAgentRetrySuccess:
    @patch("app.services.claude_client.time.sleep")
    @patch("app.services.claude_client.anthropic.Anthropic")
    def test_retries_after_429_and_succeeds(self, mock_anthropic_cls, mock_sleep):
        mock_client = MagicMock()
        # First call raises 429, second succeeds
        mock_client.messages.create.side_effect = [
            _rate_limit_error(),
            _make_message('{"ok": true}'),
        ]
        mock_anthropic_cls.return_value = mock_client

        raw, stop = call_claude_agent("prompt", agent_name="retry_agent")

        assert raw == '{"ok": true}'
        assert mock_client.messages.create.call_count == 2
        # sleep was called once (before retry 1)
        mock_sleep.assert_called_once_with(_RATE_LIMIT_RETRY_DELAYS[0])

    @patch("app.services.claude_client.time.sleep")
    @patch("app.services.claude_client.anthropic.Anthropic")
    def test_correct_backoff_delays_on_multiple_429s(self, mock_anthropic_cls, mock_sleep):
        mock_client = MagicMock()
        # Three 429s then success
        mock_client.messages.create.side_effect = [
            _rate_limit_error(),
            _rate_limit_error(),
            _rate_limit_error(),
            _make_message("{}"),
        ]
        mock_anthropic_cls.return_value = mock_client

        raw, _ = call_claude_agent("prompt", agent_name="multi_retry")

        assert raw == "{}"
        assert mock_client.messages.create.call_count == 4
        # Verify backoff delays in order
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_calls == _RATE_LIMIT_RETRY_DELAYS


# ─── Tests: retry exhaustion ─────────────────────────────────────────────────

class TestCallClaudeAgentRateLimitExhausted:
    @patch("app.services.claude_client.time.sleep")
    @patch("app.services.claude_client.anthropic.Anthropic")
    def test_raises_rate_limit_exhausted_after_all_retries(self, mock_anthropic_cls, mock_sleep):
        mock_client = MagicMock()
        # All attempts raise 429 (1 initial + 3 retries = 4 calls)
        mock_client.messages.create.side_effect = _rate_limit_error()
        mock_anthropic_cls.return_value = mock_client

        with pytest.raises(RateLimitExhaustedError) as exc_info:
            call_claude_agent("prompt", agent_name="exhausted_agent")

        exc = exc_info.value
        assert exc.agent_name == "exhausted_agent"
        assert exc.retry_count == len(_RATE_LIMIT_RETRY_DELAYS)
        assert isinstance(exc.last_exception, anthropic.RateLimitError)

    @patch("app.services.claude_client.time.sleep")
    @patch("app.services.claude_client.anthropic.Anthropic")
    def test_total_api_calls_equals_max_attempts(self, mock_anthropic_cls, mock_sleep):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _rate_limit_error()
        mock_anthropic_cls.return_value = mock_client

        with pytest.raises(RateLimitExhaustedError):
            call_claude_agent("prompt", agent_name="count_agent")

        # 1 initial + 3 retries = 4 total attempts
        expected_total = len(_RATE_LIMIT_RETRY_DELAYS) + 1
        assert mock_client.messages.create.call_count == expected_total

    @patch("app.services.claude_client.time.sleep")
    @patch("app.services.claude_client.anthropic.Anthropic")
    def test_all_sleep_delays_called_before_exhaustion(self, mock_anthropic_cls, mock_sleep):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _rate_limit_error()
        mock_anthropic_cls.return_value = mock_client

        with pytest.raises(RateLimitExhaustedError):
            call_claude_agent("prompt", agent_name="sleep_check")

        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        # Should sleep with each delay before each retry
        assert len(sleep_calls) == len(_RATE_LIMIT_RETRY_DELAYS)
        assert sleep_calls == _RATE_LIMIT_RETRY_DELAYS


# ─── RateLimitExhaustedError attributes ──────────────────────────────────────

class TestRateLimitExhaustedError:
    def test_error_message_contains_agent_name(self):
        exc = RateLimitExhaustedError(
            agent_name="my_agent",
            retry_count=3,
            last_exception=MagicMock(spec=anthropic.RateLimitError),
        )
        assert "my_agent" in str(exc)
        assert "3" in str(exc)

    def test_is_runtime_error(self):
        exc = RateLimitExhaustedError(
            agent_name="x", retry_count=3, last_exception=MagicMock()
        )
        assert isinstance(exc, RuntimeError)
