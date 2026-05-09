"""
Pipeline call tracker — per-run model call counting, stage timing, and budget guard.

Reset at the start of each pipeline run via reset().
Thread-safe: all state mutations use a module-level lock.

Budget enforcement:
  inc_claude()  — checks MAX_TOTAL_MODEL_CALLS before counting a Claude call
  inc_openai()  — checks MAX_TOTAL_MODEL_CALLS before counting an OpenAI call
  note_repair() — checks MAX_REPAIR_CALLS / MAX_OPENAI_REPAIR_CALLS for repair stages

Both inc functions raise BudgetExceededError if any limit would be exceeded.
The pipeline should call these BEFORE making the underlying API call so the
error aborts the pipeline cleanly without wasting tokens.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()

_state: dict[str, Any] = {
    "claude_calls":        0,
    "openai_calls":        0,
    "repair_calls":        0,   # Claude repair calls
    "openai_repair_calls": 0,   # OpenAI repair calls
    "stage_timing_sec":    {},
    "stage_start_times":   {},
    "stage_reuse":         [],
}


class BudgetExceededError(RuntimeError):
    """Raised when a model-call budget limit is exceeded.

    The pipeline should catch this at the top level, mark status=failed,
    and surface the error message to the caller.
    """


# ─── Public API ───────────────────────────────────────────────────────────────

def reset() -> None:
    """Clear all counters and timing state. Call at the start of each pipeline run."""
    with _lock:
        _state["claude_calls"]        = 0
        _state["openai_calls"]        = 0
        _state["repair_calls"]        = 0
        _state["openai_repair_calls"] = 0
        _state["stage_timing_sec"]    = {}
        _state["stage_start_times"]   = {}
        _state["stage_reuse"]         = []


def inc_claude(agent_name: str = "") -> None:
    """
    Increment the Claude call counter. Raises BudgetExceededError if the
    total model-call budget (MAX_TOTAL_MODEL_CALLS) would be exceeded.

    Call this BEFORE making the API call.
    """
    from app.config import settings  # lazy import to avoid circular dependency
    with _lock:
        new_total = _state["claude_calls"] + _state["openai_calls"] + 1
        if new_total > settings.max_total_model_calls:
            raise BudgetExceededError(
                f"MAX_TOTAL_MODEL_CALLS={settings.max_total_model_calls} exceeded "
                f"(would reach {new_total}) before agent '{agent_name}'. "
                "Increase MAX_TOTAL_MODEL_CALLS in .env or reduce pipeline calls."
            )
        _state["claude_calls"] += 1
        logger.debug(
            "[call_tracker] Claude call #%d — agent=%s",
            _state["claude_calls"], agent_name or "unknown",
        )


def inc_openai(agent_name: str = "") -> None:
    """
    Increment the OpenAI call counter. Raises BudgetExceededError if the
    total model-call budget (MAX_TOTAL_MODEL_CALLS) would be exceeded.

    Call this BEFORE making the API call.
    """
    from app.config import settings
    with _lock:
        new_total = _state["claude_calls"] + _state["openai_calls"] + 1
        if new_total > settings.max_total_model_calls:
            raise BudgetExceededError(
                f"MAX_TOTAL_MODEL_CALLS={settings.max_total_model_calls} exceeded "
                f"(would reach {new_total}) before OpenAI agent '{agent_name}'. "
                "Increase MAX_TOTAL_MODEL_CALLS in .env or reduce pipeline calls."
            )
        _state["openai_calls"] += 1
        logger.debug(
            "[call_tracker] OpenAI call #%d — agent=%s",
            _state["openai_calls"], agent_name or "unknown",
        )


def note_repair(kind: str = "claude", agent_name: str = "") -> None:
    """
    Check and record a repair-specific model call BEFORE the call is made.

    kind="claude" : checks MAX_REPAIR_CALLS
    kind="openai" : checks MAX_OPENAI_REPAIR_CALLS

    Raises BudgetExceededError if the relevant repair-call limit is exceeded.
    Does NOT increment the main claude/openai counters — call inc_claude/inc_openai
    separately to also check the total-call budget.
    """
    from app.config import settings
    with _lock:
        if kind == "claude":
            new_count = _state["repair_calls"] + 1
            if new_count > settings.max_repair_calls:
                raise BudgetExceededError(
                    f"MAX_REPAIR_CALLS={settings.max_repair_calls} exceeded "
                    f"(would reach {new_count}) before repair agent '{agent_name}'. "
                    "Increase MAX_REPAIR_CALLS in .env or reduce repair passes."
                )
            _state["repair_calls"] = new_count
        elif kind == "openai":
            new_count = _state["openai_repair_calls"] + 1
            if new_count > settings.max_openai_repair_calls:
                raise BudgetExceededError(
                    f"MAX_OPENAI_REPAIR_CALLS={settings.max_openai_repair_calls} exceeded "
                    f"(would reach {new_count}) before OpenAI repair agent '{agent_name}'. "
                    "Increase MAX_OPENAI_REPAIR_CALLS in .env."
                )
            _state["openai_repair_calls"] = new_count


def stage_start(name: str) -> None:
    """Record the start time for a pipeline stage."""
    with _lock:
        _state["stage_start_times"][name] = time.perf_counter()


def stage_end(name: str) -> None:
    """Record the elapsed time for a pipeline stage and remove its start marker."""
    with _lock:
        start = _state["stage_start_times"].pop(name, None)
        if start is not None:
            elapsed = round(time.perf_counter() - start, 2)
            _state["stage_timing_sec"][name] = elapsed


def mark_reuse(stage_name: str) -> None:
    """Record that a stage was skipped by loading its output from disk."""
    with _lock:
        if stage_name not in _state["stage_reuse"]:
            _state["stage_reuse"].append(stage_name)


def get_snapshot() -> dict[str, Any]:
    """Return a copy of all current telemetry. Safe to call at any point."""
    with _lock:
        return {
            "model_calls": {
                "claude_total":       _state["claude_calls"],
                "openai_total":       _state["openai_calls"],
                "repair_claude":      _state["repair_calls"],
                "repair_openai":      _state["openai_repair_calls"],
                "total":              _state["claude_calls"] + _state["openai_calls"],
            },
            "stage_timing_sec": dict(_state["stage_timing_sec"]),
            "stage_reuse":      list(_state["stage_reuse"]),
        }
