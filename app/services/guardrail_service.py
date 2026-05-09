"""Asset guardrail evaluation — classifies assets as safe, review, or blocked."""
from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GuardrailStatus(str, Enum):
    AUTO_SAFE = "auto_safe"
    MANUAL_REVIEW = "manual_review"
    BLOCKED = "blocked"


_AUTO_SAFE_TERMS = [
    "city", "location", "court building", "courthouse", "map", "hospital",
    "street", "house", "skyline", "building", "architecture", "landscape",
    "nature", "symbolic", "candle", "memorial",
]

_REVIEW_TERMS = [
    "victim", "family", "child", "children", "person", "people", "portrait",
    "face", "photo", "news", "editorial", "journalist",
]

_BLOCKED_TERMS = [
    "crime scene", "autopsy", "watermark", "podcast screenshot",
    "private social media", "unclear license", "blood", "corpse", "dead body",
    "graphic", "forensic", "murder scene",
]


def _classify_keyword(keyword: str) -> GuardrailStatus:
    kw = keyword.lower()
    if any(t in kw for t in _BLOCKED_TERMS):
        return GuardrailStatus.BLOCKED
    if any(t in kw for t in _REVIEW_TERMS):
        return GuardrailStatus.MANUAL_REVIEW
    if any(t in kw for t in _AUTO_SAFE_TERMS):
        return GuardrailStatus.AUTO_SAFE
    return GuardrailStatus.MANUAL_REVIEW  # default to review if uncertain


def evaluate_candidates(
    candidates_dir: Path,
    review_dir: Path,
) -> dict[str, Any]:
    """
    Walk real-candidates/*.json and classify each asset.
    Writes guardrail_results.json to review_dir.
    Returns summary dict.
    """
    review_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list] = {
        GuardrailStatus.AUTO_SAFE: [],
        GuardrailStatus.MANUAL_REVIEW: [],
        GuardrailStatus.BLOCKED: [],
    }

    for candidate_file in candidates_dir.glob("*.json"):
        try:
            assets = json.loads(candidate_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read %s: %s", candidate_file, exc)
            continue

        for asset in assets:
            kw = asset.get("keyword", "")
            status = _classify_keyword(kw)
            asset["guardrail_status"] = status
            asset["candidate_file"] = candidate_file.name
            results[status].append(asset)

    summary = {
        "total_auto_safe": len(results[GuardrailStatus.AUTO_SAFE]),
        "total_manual_review": len(results[GuardrailStatus.MANUAL_REVIEW]),
        "total_blocked": len(results[GuardrailStatus.BLOCKED]),
        "details": {k: v for k, v in results.items()},
    }

    out_path = review_dir / "guardrail_results.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Guardrail: %d safe, %d review, %d blocked",
        summary["total_auto_safe"],
        summary["total_manual_review"],
        summary["total_blocked"],
    )
    return summary
