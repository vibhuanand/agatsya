"""
Shared prompt utilities.

Provides:
  get_channel_rules() — load the Agatsya content bible for embedding in agent prompts.
"""
from __future__ import annotations

from pathlib import Path

_BIBLE_PATH = Path("app/prompts/_agatsya_content_bible.txt")


def get_channel_rules() -> str:
    """
    Return the full Agatsya content bible text.
    Embed in agent prompts via the {channel_rules} placeholder.
    """
    return _BIBLE_PATH.read_text(encoding="utf-8")
