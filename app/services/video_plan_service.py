"""
Second-stage video plan generation.

Called after the script_first package has been reviewed and approved.
Reads hindi_narration_chunks.json, recreated_dialogues.json, and case_summary.json
from the episode folder, then calls Claude to generate:
  - video_scene_plan
  - asset_keywords
  - shorts_plan
  - full youtube_metadata
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import anthropic

from app.config import settings
from app.models import VideoPlanRequest, VideoPlanResponse
from app.services.claude_client import _extract_json, _repair_json  # reuse robust parser

logger = logging.getLogger(__name__)

PROMPT_PATH = Path("app/prompts/video_plan_from_approved_script.txt")


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_video_plan_prompt(
    episode_id: str,
    cost_mode: str,
    case_summary: dict,
    narration_chunks: list,
    recreated_dialogues: dict,
) -> str:
    template = _load_prompt()
    replacements = {
        "{episode_id}": episode_id,
        "{cost_mode}": cost_mode,
        "{case_summary_json}": json.dumps(case_summary, ensure_ascii=False, indent=2),
        "{narration_chunks_json}": json.dumps(narration_chunks, ensure_ascii=False, indent=2),
        "{recreated_dialogues_json}": json.dumps(recreated_dialogues, ensure_ascii=False, indent=2),
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def _extract_asset_keywords(scenes: list[dict]) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for scene in scenes:
        for kw in scene.get("real_asset_keywords", []):
            if kw and kw not in seen:
                seen.add(kw)
                lines.append(kw)
    # Also support top-level asset_keywords array if model returns it that way
    return "\n".join(lines)


def generate_video_plan(req: VideoPlanRequest) -> VideoPlanResponse:
    warnings: list[str] = []

    # Locate episode directory
    episode_dir = settings.episodes_dir / req.episode_id
    if not episode_dir.exists():
        raise FileNotFoundError(
            f"Episode directory not found: {episode_dir}. "
            "Run POST /api/episodes/package first."
        )

    pkg_dir = episode_dir / "02-package"

    # Load required script files
    def _load_json(filename: str) -> Any:
        path = pkg_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Required file missing: {path}. "
                "Run POST /api/episodes/package with package_level=script_first first."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        # Check for deferred placeholders
        if isinstance(data, dict) and data.get("status") == "deferred":
            raise ValueError(
                f"{filename} is a deferred placeholder — "
                "complete script_first package generation first."
            )
        return data

    case_summary = _load_json("case_summary.json")
    narration_chunks = _load_json("hindi_narration_chunks.json")
    recreated_dialogues = _load_json("recreated_dialogues.json")

    # Build prompt and call Claude
    prompt = _build_video_plan_prompt(
        episode_id=req.episode_id,
        cost_mode=req.cost_mode,
        case_summary=case_summary,
        narration_chunks=narration_chunks,
        recreated_dialogues=recreated_dialogues,
    )

    logger.info(
        "Calling Claude (%s) for video plan — episode: %s, cost_mode: %s",
        settings.claude_model,
        req.episode_id,
        req.cost_mode,
    )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model=settings.claude_model,
        max_tokens=settings.claude_max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_response = message.content[0].text
    stop_reason = message.stop_reason

    logger.info(
        "Claude video plan response — %d chars, stop_reason=%s",
        len(raw_response),
        stop_reason,
    )

    if stop_reason == "max_tokens":
        warnings.append(
            "Claude hit max_tokens — video plan may be truncated. "
            "Check _video_plan_raw_response.txt"
        )

    # Save raw response before parsing
    raw_path = pkg_dir / "_video_plan_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    logger.info("Video plan raw response saved → %s", raw_path)

    # Parse JSON
    try:
        plan_dict = _extract_json(raw_response)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse video plan response as JSON: {exc}. "
            f"Check {raw_path}"
        ) from exc

    # Save output files
    files: dict[str, str] = {}

    def save(name: str, content: str | dict | list) -> None:
        p = pkg_dir / name
        if isinstance(content, (dict, list)):
            p.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            p.write_text(content, encoding="utf-8")
        files[name] = str(p)

    save("episode_video_plan.json", plan_dict.get("video_scene_plan", {}))
    save("youtube_metadata.json", plan_dict.get("youtube_metadata", {}))
    save("shorts_plan.json", plan_dict.get("shorts_plan", {}))

    # Asset keywords — support both top-level list and extraction from scenes
    top_level_kws = plan_dict.get("asset_keywords", [])
    if top_level_kws:
        save("asset_keywords.txt", "\n".join(top_level_kws))
    else:
        scenes = plan_dict.get("video_scene_plan", {}).get("scenes", [])
        save("asset_keywords.txt", _extract_asset_keywords(scenes))

    # Overwrite production_package.json to merge the video plan in
    prod_path = pkg_dir / "production_package.json"
    if prod_path.exists():
        try:
            prod = json.loads(prod_path.read_text(encoding="utf-8"))
            prod["video_scene_plan"] = plan_dict.get("video_scene_plan", {})
            prod["youtube_metadata"] = plan_dict.get("youtube_metadata", prod.get("youtube_metadata", {}))
            prod["shorts_plan"] = plan_dict.get("shorts_plan", {})
            prod_path.write_text(json.dumps(prod, ensure_ascii=False, indent=2), encoding="utf-8")
            files["production_package.json"] = str(prod_path)
            logger.info("production_package.json updated with video plan")
        except Exception as exc:
            warnings.append(f"Could not merge video plan into production_package.json: {exc}")

    return VideoPlanResponse(
        episode_id=req.episode_id,
        episode_dir=str(episode_dir),
        files=files,
        warnings=warnings,
    )
