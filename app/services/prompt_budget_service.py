"""
Prompt Budget Service — token estimation and transcript size classification.

Used by:
  - fact_lock_service.py    (auto mode dispatch)
  - model_rate_limiter_service.py   (rolling TPM guard)
  - agent_pipeline_service.py       (effective_runtime_config.json)

Conservative token estimate: ceil(chars / 3.5)
Anthropic models use ~3.5 chars per token on average for mixed Hindi/English content.
Rounding up keeps the estimate safe-side.
"""
from __future__ import annotations

import math

from app.config import settings


# ─── Core estimator ───────────────────────────────────────────────────────────

def estimate_tokens(chars: int) -> int:
    """Conservative token estimate: ceil(chars / 3.5)."""
    if chars <= 0:
        return 0
    return math.ceil(chars / 3.5)


# ─── Transcript size classifier ───────────────────────────────────────────────

# Size bands (based on clean_chars):
#   small     — below half the long threshold (fits in one research_view call with margin)
#   medium    — half the long threshold to the long threshold
#   long      — long threshold to very_long threshold (segmented recommended)
#   very_long — above very_long threshold (segmented required)

def classify_transcript_size(clean_chars: int) -> str:
    """
    Classify transcript into: 'small' | 'medium' | 'long' | 'very_long'.

    Boundaries (configurable via .env):
      small:     < LONG_TRANSCRIPT_CLEAN_CHARS_THRESHOLD / 2    (default < 15 000)
      medium:    < LONG_TRANSCRIPT_CLEAN_CHARS_THRESHOLD         (default < 30 000)
      long:      < VERY_LONG_TRANSCRIPT_CLEAN_CHARS_THRESHOLD    (default < 60 000)
      very_long: >= VERY_LONG_TRANSCRIPT_CLEAN_CHARS_THRESHOLD   (default >= 60 000)
    """
    long_threshold      = settings.long_transcript_clean_chars_threshold      # 30 000
    very_long_threshold = settings.very_long_transcript_clean_chars_threshold  # 60 000
    medium_threshold    = long_threshold // 2                                  # 15 000

    if clean_chars >= very_long_threshold:
        return "very_long"
    if clean_chars >= long_threshold:
        return "long"
    if clean_chars >= medium_threshold:
        return "medium"
    return "small"


# ─── Budget helpers ───────────────────────────────────────────────────────────

def prompt_fits_in_budget(prompt_chars: int) -> bool:
    """
    Return True if the estimated token count for prompt_chars is within the
    safe per-call input token budget (SAFE_CLAUDE_INPUT_TOKENS_PER_CALL).
    """
    return estimate_tokens(prompt_chars) <= settings.safe_claude_input_tokens_per_call


def research_view_fits_budget(transcript_research_view_chars: int) -> bool:
    """
    Return True if the research_view (already trimmed) fits within the safe
    per-call budget.  Adds a 2000-token overhead allowance for the prompt template.
    """
    overhead_tokens = 2000
    view_tokens = estimate_tokens(transcript_research_view_chars)
    return (view_tokens + overhead_tokens) <= settings.safe_claude_input_tokens_per_call


def should_use_segmented_mode(clean_chars: int) -> bool:
    """
    Return True if clean_transcript is large enough that segmented mode is
    recommended over research_view.  Used by fact_lock_service auto dispatch.
    """
    return clean_chars >= settings.long_transcript_clean_chars_threshold
