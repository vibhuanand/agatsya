"""
Report Normalization Service — type-safe helpers for gate report fields.

Gate reports from Claude may return required_fixes, issues, warnings, and
chunk_repair_targets as lists of strings, dicts, or mixed types.
Joining them naively with "; ".join(...) raises:
    TypeError: sequence item N: expected str instance, dict found

These helpers make warning formatting and gate_summary population safe
regardless of whether items are str, dict, int, float, bool, or list.

Usage:
    from app.services.report_normalization_service import (
        stringify_report_item,
        stringify_report_list,
        safe_join_report_items,
    )
"""
from __future__ import annotations

import json
from typing import Any

# Keys to try (in priority order) when extracting a human-readable string from a dict.
_PREFERRED_KEYS = (
    "problem",
    "issue",
    "message",
    "required_fix",
    "repair_instruction",
    "suggested_fix",
    "field",
    "issue_type",
    "severity",
    "chunk_id",
    "title",
    "text",
)


def stringify_report_item(item: Any) -> str:
    """
    Convert a single gate report item to a human-readable string.

    - str            → returned as-is
    - int/float/bool → str(item)
    - dict           → extract useful keys in priority order; fallback to compact JSON
    - list           → stringify each element and join with "; "
    - anything else  → str(item)
    """
    if isinstance(item, str):
        return item
    if isinstance(item, (int, float, bool)):
        return str(item)
    if isinstance(item, dict):
        parts: list[str] = []
        for key in _PREFERRED_KEYS:
            if key in item and item[key] is not None and item[key] != "":
                val = item[key]
                if isinstance(val, (dict, list)):
                    val = json.dumps(val, ensure_ascii=False)
                else:
                    val = str(val)
                parts.append(val)
        if parts:
            return " | ".join(parts)
        # No useful keys — compact JSON fallback
        return json.dumps(item, ensure_ascii=False, separators=(",", ":"))
    if isinstance(item, list):
        return "; ".join(stringify_report_item(x) for x in item)
    return str(item)


def stringify_report_list(
    value: Any,
    limit: int | None = None,
) -> list[str]:
    """
    Normalize a gate report field value to a list of strings.

    - None    → []
    - list    → each item stringified via stringify_report_item
    - dict    → [stringify_report_item(value)]
    - str     → [value]
    - other   → [str(value)]

    If limit is given, the returned list is capped at that length.
    """
    if value is None:
        result: list[str] = []
    elif isinstance(value, list):
        result = [stringify_report_item(x) for x in value]
    elif isinstance(value, dict):
        result = [stringify_report_item(value)]
    elif isinstance(value, str):
        result = [value]
    else:
        result = [str(value)]

    if limit is not None:
        result = result[:limit]
    return result


def safe_join_report_items(
    value: Any,
    limit: int = 3,
    sep: str = "; ",
) -> str:
    """
    Safely join gate report items as a human-readable string.

    Never raises TypeError regardless of whether items are str, dict, or list.

    Args:
        value:  The report field value (list, dict, str, or None).
        limit:  Maximum number of items to include (default 3).
        sep:    Separator string (default "; ").

    Returns:
        A joined string, or "" if value is empty/None.
    """
    return sep.join(stringify_report_list(value, limit=limit))
