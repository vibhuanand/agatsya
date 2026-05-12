"""
Model Rate Limiter Service — rolling 60-second input-token window guard.

Prevents Claude API 429 errors by tracking estimated input tokens sent in the
last 60 seconds and sleeping before calls that would exceed the budget.

Usage (automatic — called by call_claude_agent):
  rate_limiter.before_call(prompt, agent_name="fact_lock")
  # ... make the API call ...
  rate_limiter.after_call(prompt, agent_name="fact_lock")

Telemetry (surfaced in effective_runtime_config.json and pipeline telemetry):
  rate_limiter.telemetry()  →  {
      "claude_rate_limit_wait_sec": float,
      "claude_estimated_input_tokens_last_60s": int,
      "claude_throttle_events": int,
  }

Settings (via .env):
  SAFE_CLAUDE_TOKENS_PER_MINUTE=30000   — rolling 60-second token ceiling
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

from app.config import settings
from app.services.prompt_budget_service import estimate_tokens

logger = logging.getLogger(__name__)


class ModelRateLimiter:
    """
    Thread-safe rolling 60-second input-token window rate limiter.

    Records (timestamp, estimated_tokens) tuples.  Before each Claude call:
      1. Purge entries older than 60 seconds.
      2. Sum remaining entries.
      3. If sum + new_call_tokens > SAFE_CLAUDE_TOKENS_PER_MINUTE, sleep until
         the oldest entry is more than 60 seconds old (i.e., the window clears).

    after_call() records the actual call for accounting (tokens recorded once).
    before_call() checks the budget and sleeps if needed.
    """

    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._calls: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._total_wait_sec: float = 0.0
        self._throttle_events: int  = 0

    # ─── Public API ───────────────────────────────────────────────────────────

    def before_call(self, prompt: str, agent_name: str = "agent") -> None:
        """
        Check rolling window; sleep if this call would exceed the TPM limit.
        Should be called BEFORE making the Claude API call.
        """
        estimated = estimate_tokens(len(prompt))
        limit     = settings.safe_claude_tokens_per_minute

        with self._lock:
            now = time.monotonic()
            self._purge_old(now)
            window_total = sum(t for _, t in self._calls)

            if window_total + estimated > limit:
                # Sleep until the oldest entry exits the 60-second window
                if self._calls:
                    oldest_ts = self._calls[0][0]
                    sleep_needed = max(0.0, (oldest_ts + 60.0) - now)
                else:
                    sleep_needed = 0.0

                if sleep_needed > 0:
                    self._throttle_events += 1
                    self._total_wait_sec  += sleep_needed
                    logger.info(
                        "ModelRateLimiter: throttling agent '%s' — "
                        "window=%d tokens + new=%d > limit=%d; sleeping %.1fs",
                        agent_name,
                        window_total,
                        estimated,
                        limit,
                        sleep_needed,
                    )
                    # Release lock while sleeping so other threads are not blocked
                    self._lock.release()
                    try:
                        time.sleep(sleep_needed)
                    finally:
                        self._lock.acquire()

    def after_call(self, prompt: str, agent_name: str = "agent") -> None:
        """
        Record a completed call in the rolling window.
        Should be called AFTER a successful Claude API call.
        """
        estimated = estimate_tokens(len(prompt))
        with self._lock:
            self._calls.append((time.monotonic(), estimated))
            logger.debug(
                "ModelRateLimiter: recorded %d tokens for agent '%s'",
                estimated,
                agent_name,
            )

    def telemetry(self) -> dict:
        """
        Return telemetry dict for inclusion in effective_runtime_config.json and
        pipeline telemetry output.
        """
        with self._lock:
            now = time.monotonic()
            self._purge_old(now)
            window_total = sum(t for _, t in self._calls)

        return {
            "claude_rate_limit_wait_sec":              round(self._total_wait_sec, 2),
            "claude_estimated_input_tokens_last_60s":  window_total,
            "claude_throttle_events":                  self._throttle_events,
        }

    def reset(self) -> None:
        """Reset all state (called at the start of each pipeline run)."""
        with self._lock:
            self._calls.clear()
            self._total_wait_sec  = 0.0
            self._throttle_events = 0

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _purge_old(self, now: float) -> None:
        """Remove entries older than 60 seconds (must be called with lock held)."""
        cutoff = now - 60.0
        while self._calls and self._calls[0][0] < cutoff:
            self._calls.popleft()


# Module-level singleton — imported by call_claude_agent
rate_limiter = ModelRateLimiter()
