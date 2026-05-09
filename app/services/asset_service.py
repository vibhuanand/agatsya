"""Visual asset fetching scaffold — Pexels, Pixabay, Wikimedia."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


# ─── Pexels ──────────────────────────────────────────────────────────────────

def _pexels_search(query: str, per_page: int = 3) -> list[dict[str, Any]]:
    if not settings.pexels_api_key:
        return []
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": settings.pexels_api_key}
    params = {"query": query, "per_page": per_page, "orientation": "landscape"}
    with httpx.Client(timeout=20) as client:
        resp = client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json().get("photos", [])


# ─── Pixabay ─────────────────────────────────────────────────────────────────

def _pixabay_search(query: str, per_page: int = 3) -> list[dict[str, Any]]:
    if not settings.pixabay_api_key:
        return []
    url = "https://pixabay.com/api/"
    params = {
        "key": settings.pixabay_api_key,
        "q": query,
        "image_type": "photo",
        "orientation": "horizontal",
        "safesearch": "true",
        "per_page": per_page,
    }
    with httpx.Client(timeout=20) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json().get("hits", [])


# ─── Guardrail check ─────────────────────────────────────────────────────────

_BLOCKED_TERMS = {
    "crime scene", "autopsy", "victim", "body", "blood", "forensic",
    "murder victim", "dead", "corpse",
}

def _is_query_safe(keyword: str) -> bool:
    kw_lower = keyword.lower()
    return not any(blocked in kw_lower for blocked in _BLOCKED_TERMS)


# ─── Main fetch function ──────────────────────────────────────────────────────

def fetch_assets_for_keywords(
    keywords_txt_path: Path,
    candidates_dir: Path,
) -> list[str]:
    """
    For each keyword in asset_keywords.txt, search Pexels and Pixabay.
    Save metadata JSON per keyword in real-candidates/.
    Returns list of saved metadata paths.
    """
    candidates_dir.mkdir(parents=True, exist_ok=True)

    if not (settings.pexels_api_key or settings.pixabay_api_key):
        logger.warning("No asset API keys set — skipping asset fetch")
        return []

    keywords = [
        line.strip()
        for line in keywords_txt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    saved: list[str] = []
    import json

    for kw in keywords:
        if not _is_query_safe(kw):
            logger.warning("Keyword '%s' blocked by guardrail — skipping", kw)
            continue

        results: list[dict] = []

        try:
            pexels = _pexels_search(kw)
            for p in pexels:
                results.append({
                    "source": "pexels",
                    "keyword": kw,
                    "id": p.get("id"),
                    "url": p.get("url"),
                    "photographer": p.get("photographer"),
                    "src_large": p.get("src", {}).get("large2x"),
                    "license": "Pexels License (free for commercial use)",
                    "requires_review": False,
                })
        except Exception as exc:
            logger.warning("Pexels error for '%s': %s", kw, exc)

        try:
            pixabay = _pixabay_search(kw)
            for p in pixabay:
                results.append({
                    "source": "pixabay",
                    "keyword": kw,
                    "id": p.get("id"),
                    "url": p.get("pageURL"),
                    "user": p.get("user"),
                    "src_large": p.get("largeImageURL"),
                    "license": "Pixabay License (free for commercial use)",
                    "requires_review": False,
                })
        except Exception as exc:
            logger.warning("Pixabay error for '%s': %s", kw, exc)

        if results:
            slug = kw.lower().replace(" ", "_")[:40]
            out_path = candidates_dir / f"{slug}.json"
            out_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            saved.append(str(out_path))
            logger.info("Saved %d asset candidates for '%s'", len(results), kw)

    return saved
