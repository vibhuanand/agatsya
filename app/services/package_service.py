"""Orchestrates episode folder creation and production package persistence."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import settings
from app.models import EpisodeInput, PackageResponse, ProductionPackage
from app.services.claude_client import (
    call_claude,
    parse_package_response,
    build_transcript_research_view,
)
from app.services.openai_review_service import review_package

logger = logging.getLogger(__name__)


# ─── Cost policy definitions ──────────────────────────────────────────────────

_COST_POLICIES: dict[str, dict] = {
    "bootstrap": {
        "cost_mode": "bootstrap",
        "allowed": [
            "single Claude call",
            "ElevenLabs narration audio",
            "template cards",
            "free stock images (Pexels / Pixabay)",
            "symbolic AI still images",
            "captions and on-screen text",
            "waveform / recreated-call screens",
            "slow zoom/pan motion on stills",
            "approved real location images",
        ],
        "blocked": [
            "paid AI video generation (Runway, Kling, Pika)",
            "AI video clips (ai_video background_type)",
            "GPT review pass (not enabled by default)",
            "auto-use of victim / family / minor photos",
            "graphic or exploitative visuals",
            "watermarked or unclear-license images",
        ],
        "limits": {
            "max_ai_video_clips": 0,
            "max_ai_images": 5,
            "max_real_assets": 20,
            "prefer_template_cards": True,
        },
        "notes": [
            "Bootstrap mode is optimised for low-cost early production.",
            "Ideal episode length: 15–22 minutes.",
            "Use template cards, location stills, and captions as primary visuals.",
            "Upgrade to standard or premium once channel is monetised.",
        ],
    },
    "standard": {
        "cost_mode": "standard",
        "allowed": [
            "single Claude call",
            "ElevenLabs narration audio",
            "template cards",
            "free stock images",
            "1–3 AI hero video clips",
            "AI still images for key scenes",
            "approved real images",
        ],
        "blocked": [
            "auto-use of victim / family / minor photos",
            "graphic or exploitative visuals",
            "watermarked or unclear-license images",
        ],
        "limits": {
            "max_ai_video_clips": 3,
            "max_ai_images": 10,
            "max_real_assets": 30,
            "prefer_template_cards": False,
        },
        "notes": [
            "Standard mode allows a few hero AI video scenes for cinematic impact.",
            "Keep AI video usage for pivotal moments only.",
        ],
    },
    "premium": {
        "cost_mode": "premium",
        "allowed": [
            "single Claude call",
            "ElevenLabs narration audio",
            "5–10 AI hero video clips",
            "AI still images throughout",
            "optional GPT review pass",
            "free and licensed real images",
            "full motion graphics treatment",
        ],
        "blocked": [
            "auto-use of victim / family / minor photos",
            "graphic or exploitative visuals",
            "watermarked or unclear-license images",
        ],
        "limits": {
            "max_ai_video_clips": 10,
            "max_ai_images": 20,
            "max_real_assets": 50,
            "prefer_template_cards": False,
        },
        "notes": [
            "Premium mode targets highest production quality.",
            "All safety and dignity rules still apply at every level.",
            "GPT review pass can be enabled in the full pipeline.",
        ],
    },
}


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:40]


def _make_episode_dir(episode_number: str, case_hint: str) -> Path:
    slug = _slugify(case_hint)
    folder_name = f"{episode_number}-{slug}"
    base = settings.episodes_dir / folder_name
    for sub in ["01-input", "02-package", "03-audio", "04-assets/real-candidates",
                 "04-assets/approved", "04-assets/generated", "05-renders", "06-review"]:
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def _extract_full_narration(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"## {c.get('section_title', c.get('chunk_id', ''))}\n{c.get('text', '')}"
        for c in chunks
    )


def _extract_asset_keywords(scenes: list[dict]) -> str:
    seen = set()
    lines = []
    for scene in scenes:
        for kw in scene.get("real_asset_keywords", []):
            if kw and kw not in seen:
                seen.add(kw)
                lines.append(kw)
    return "\n".join(lines)


def _extract_elevenlabs_chunks(chunks: list[dict]) -> list[dict]:
    """Flatten narration chunks into ElevenLabs-ready format."""
    result = []
    for c in chunks:
        result.append({
            "chunk_id": c.get("chunk_id"),
            "voice_id": "{ELEVENLABS_NARRATOR_VOICE_ID}",
            "model_id": "{ELEVENLABS_MODEL_ID}",
            "text": c.get("text", ""),
            "voice_settings": {
                "stability": 0.55,
                "similarity_boost": 0.75,
                "style": 0.4,
                "use_speaker_boost": True,
            },
        })
    return result


def create_package(inp: EpisodeInput) -> PackageResponse:
    warnings: list[str] = []
    episode_dir = _make_episode_dir(inp.episode_number, inp.case_hint)

    # 01-input: save full transcript unchanged
    (episode_dir / "01-input" / "source_transcript.txt").write_text(
        inp.raw_transcript, encoding="utf-8"
    )
    (episode_dir / "01-input" / "input_payload.json").write_text(
        inp.model_dump_json(indent=2), encoding="utf-8"
    )

    # Build the transcript research view (beginning / middle / ending excerpts)
    # and save it so it can be inspected alongside the Claude output.
    transcript_research_view = build_transcript_research_view(inp.raw_transcript)
    (episode_dir / "01-input" / "transcript_research_view.txt").write_text(
        transcript_research_view, encoding="utf-8"
    )
    logger.info(
        "Transcript: full=%d chars → research view=%d chars (budget=%d)",
        len(inp.raw_transcript),
        len(transcript_research_view),
        len(transcript_research_view),
    )

    raw_response_path = episode_dir / "02-package" / "_claude_raw_response.txt"

    # Step 1: Call Claude — raw API call only, no parsing yet
    try:
        raw_response, stop_reason = call_claude(
            youtube_url=inp.youtube_url,
            episode_number=inp.episode_number,
            case_hint=inp.case_hint,
            target_duration_min=inp.target_duration_min,
            transcript_research_view=transcript_research_view,
            style=inp.style,
            cost_mode=inp.cost_mode,
            package_level=inp.package_level,
        )
    except Exception as exc:
        logger.error("Claude API call failed: %s", exc)
        raise

    # Step 2: Save raw response immediately — before any parsing attempt.
    # This ensures the file always exists for debugging even if JSON parse fails.
    raw_response_path.write_text(raw_response, encoding="utf-8")
    logger.info("Raw Claude response saved → %s", raw_response_path)

    if stop_reason == "max_tokens":
        warnings.append(
            f"Claude hit max_tokens limit — package may be truncated. "
            f"Increase CLAUDE_MAX_TOKENS in .env. Raw response: {raw_response_path}"
        )

    # Step 3: Parse JSON from raw response
    try:
        package_dict = parse_package_response(raw_response)
    except ValueError as exc:
        logger.error("JSON parse failed: %s", exc)
        raise ValueError(
            f"{exc}\n"
            f"Raw Claude response saved at: {raw_response_path}"
        ) from exc

    # Step 4: Validate with Pydantic — warn but don't crash on schema mismatch
    try:
        ProductionPackage.model_validate(package_dict)
    except Exception as exc:
        logger.warning("Pydantic validation warning: %s", exc)
        warnings.append(f"Schema validation warning: {exc}")

    # Persist all output files
    pkg_dir = episode_dir / "02-package"
    files: dict[str, str] = {}

    def save(name: str, content: str | dict | list) -> Path:
        p = pkg_dir / name
        if isinstance(content, (dict, list)):
            p.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            p.write_text(content, encoding="utf-8")
        files[name] = str(p)
        return p

    # Always save the raw Claude package under its own name
    save("production_package_claude.json", package_dict)

    # Determine final package: Claude-only or GPT-reviewed
    final_package_dict = package_dict

    if inp.enable_gpt_review:
        logger.info("GPT review pass enabled — calling OpenAI reviewer")
        gpt_raw_path = pkg_dir / "_gpt_review_raw_response.txt"
        gpt_error_path = pkg_dir / "_gpt_review_error.txt"
        try:
            reviewed_dict, gpt_raw = review_package(package_dict)
            # Save raw GPT response for debugging
            gpt_raw_path.write_text(gpt_raw, encoding="utf-8")
            logger.info("GPT raw response saved → %s", gpt_raw_path)
            # Save the GPT-reviewed package
            save("production_package_gpt_reviewed.json", reviewed_dict)
            final_package_dict = reviewed_dict
            logger.info("GPT review complete — using reviewed package as final")
        except Exception as exc:
            # GPT review failure must NOT kill the whole generation
            error_msg = (
                f"GPT review failed — falling back to Claude package.\n"
                f"Error: {type(exc).__name__}: {exc}"
            )
            logger.error(error_msg)
            warnings.append(error_msg)
            gpt_error_path.write_text(error_msg, encoding="utf-8")
            logger.info("GPT error saved → %s", gpt_error_path)

    save("production_package.json", final_package_dict)
    save("case_summary.json", final_package_dict.get("case_summary", {}))

    chunks = final_package_dict.get("hindi_narration_chunks", [])
    save("hindi_narration_full.txt", _extract_full_narration(chunks))
    save("hindi_narration_chunks.json", chunks)
    save("recreated_dialogues.json", final_package_dict.get("recreated_dialogues", {}))
    save("elevenlabs_chunks.json", _extract_elevenlabs_chunks(chunks))
    save("youtube_metadata.json", final_package_dict.get("youtube_metadata", {}))

    if inp.package_level == "script_first":
        # Video plan, asset keywords, and shorts are deferred to /api/episodes/video-plan.
        # Save clearly-labelled placeholders so downstream code can detect the state.
        _DEFERRED_VIDEO_PLAN = {
            "status": "deferred",
            "reason": "Generate after script approval using POST /api/episodes/video-plan",
            "next_step": f"POST /api/episodes/video-plan with episode_id: {final_package_dict.get('episode_id', inp.episode_number)}",
        }
        _DEFERRED_SHORTS = {
            "status": "deferred",
            "reason": "Generate after script approval using POST /api/episodes/video-plan",
        }
        save("episode_video_plan.json", _DEFERRED_VIDEO_PLAN)
        save("shorts_plan.json", _DEFERRED_SHORTS)
        save("asset_keywords.txt", "deferred — run POST /api/episodes/video-plan after script approval")
        logger.info("script_first mode: video plan deferred — script files saved")
    else:
        # full_package or video_plan_only: extract everything from Claude response
        save("episode_video_plan.json", final_package_dict.get("video_scene_plan", {}))
        scenes = final_package_dict.get("video_scene_plan", {}).get("scenes", [])
        save("asset_keywords.txt", _extract_asset_keywords(scenes))
        save("shorts_plan.json", final_package_dict.get("shorts_plan", {}))

    # 06-review files
    review_dir = episode_dir / "06-review"
    guardrail_policy = {
        "auto_safe": ["city/location visuals", "court buildings", "maps",
                      "generic hospital/street/house", "symbolic non-person visuals"],
        "manual_review_required": ["victim photos", "family photos", "children",
                                   "case-specific real people", "news/editorial images"],
        "blocked": ["graphic crime scene", "autopsy", "watermarked image",
                    "podcast screenshots", "private social media", "unclear-license image"],
    }
    (review_dir / "asset_guardrail_policy.json").write_text(
        json.dumps(guardrail_policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    quality_checklist = final_package_dict.get("quality_checklist", [])
    (review_dir / "quality_checklist.json").write_text(
        json.dumps(quality_checklist, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Write cost policy for this episode's cost_mode
    cost_policy = _COST_POLICIES.get(inp.cost_mode, _COST_POLICIES["bootstrap"])
    (review_dir / "estimated_cost_policy.json").write_text(
        json.dumps(cost_policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Cost policy (%s) written → %s", inp.cost_mode, review_dir / "estimated_cost_policy.json")

    return PackageResponse(
        episode_id=final_package_dict.get("episode_id", inp.episode_number),
        folder_name=final_package_dict.get("folder_name", episode_dir.name),
        episode_dir=str(episode_dir),
        files=files,
        warnings=warnings,
    )
