"""Agatsya Automation — FastAPI entry point."""
from __future__ import annotations

import logging
import sys
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.models import (
    EpisodeInput,
    FullPipelineInput,
    PackageResponse,
    FullPipelineResponse,
    VideoPlanRequest,
    VideoPlanResponse,
)
from app.services.agent_pipeline_service import run_agent_pipeline
from app.services.package_service import create_package
from app.services.pipeline_service import run_full_pipeline
from app.services.video_plan_service import generate_video_plan

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("agatsya")

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agatsya Automation",
    description="AI-powered Hindi true-crime YouTube production engine",
    version="0.2.0",
)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": "0.2.0",
        "pipeline": "multi-agent",
        "claude_model": settings.claude_model,
        "voice_enabled": bool(settings.elevenlabs_api_key),
        "pexels_enabled": bool(settings.pexels_api_key),
        "pixabay_enabled": bool(settings.pixabay_api_key),
    }


@app.post("/api/episodes/package", response_model=PackageResponse)
async def create_episode_package(inp: EpisodeInput) -> PackageResponse:
    """
    Run the controlled multi-agent pipeline and produce a script package.

    Pipeline (for package_level=script_first, the default):
      1. Transcript Cleaner  — deterministic Python cleanup
      2. Fact Lock Agent     — Claude: extract verified facts only
      3. Story Blueprint     — Claude: classify story type, plan narrative
      4. Script Writer       — Claude: write full Hindi narration
      5. Quality Critic      — Claude: review against fact_lock and blueprint
      6. Script Repair       — Claude: fix listed issues (one pass max)
      7. Final Review        — Claude: post-repair check

    Does NOT run ElevenLabs, asset fetching, or video rendering.
    Video plan is deferred — call POST /api/episodes/video-plan after approval.

    Response includes:
      status          — "script_approved" | "needs_human_review" | "failed"
      quality_summary — word count, estimated duration, repair flag
      files           — all generated file paths
    """
    try:
        if inp.package_level == "script_first":
            return run_agent_pipeline(inp)
        else:
            # Legacy single-call path (full_package / video_plan_only).
            # Not voice-ready: no multi-gate review, no safe_to_voice guarantee.
            # Force status=needs_human_review and safe_to_voice=False so the caller
            # cannot accidentally treat this output as production-approved.
            pkg = create_package(inp)
            pkg.status = "needs_human_review"
            pkg.safe_to_voice = False
            pkg.warnings = list(pkg.warnings) + [
                "Legacy single-call package path is not voice-ready. "
                "Use package_level=script_first for the full quality-gate pipeline."
            ]
            return pkg
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Package creation failed: %s", exc)
        # Save full traceback to episode review dir for post-mortem debugging
        # (prevents blind failures after expensive paid model calls)
        try:
            tb_text = traceback.format_exc()
            episode_dir = settings.episodes_dir / inp.episode_id
            review_dir = episode_dir / "04-review"
            if episode_dir.exists():
                review_dir.mkdir(parents=True, exist_ok=True)
                (review_dir / "_package_exception_traceback.txt").write_text(
                    f"Package creation failed: {exc}\n\n{tb_text}",
                    encoding="utf-8",
                )
                logger.info(
                    "Exception traceback saved → %s",
                    review_dir / "_package_exception_traceback.txt",
                )
        except Exception as save_exc:
            logger.warning("Could not save exception traceback: %s", save_exc)
        raise HTTPException(status_code=500, detail=f"Package creation failed: {exc}")


@app.post("/api/episodes/full", response_model=FullPipelineResponse)
async def run_full_episode_pipeline(inp: FullPipelineInput) -> FullPipelineResponse:
    """
    Run the full pipeline:
    package → (voice) → (assets) → (render)
    enable_voice / enable_assets / enable_render flags control optional steps.

    NOTE: This endpoint is not production-ready. Use POST /api/episodes/package first,
    verify safe_to_voice=true, then set ENABLE_FULL_PIPELINE=true in .env to enable.
    ElevenLabs must never be run when safe_to_voice=false.
    """
    if not settings.enable_full_pipeline:
        raise HTTPException(
            status_code=422,
            detail=(
                "POST /api/episodes/full is not production-ready. "
                "Use POST /api/episodes/package first, verify safe_to_voice=true, "
                "then set ENABLE_FULL_PIPELINE=true to enable this endpoint."
            ),
        )
    try:
        return run_full_pipeline(inp)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Full pipeline failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Full pipeline failed: {exc}")


@app.post("/api/episodes/video-plan", response_model=VideoPlanResponse)
async def create_video_plan(req: VideoPlanRequest) -> VideoPlanResponse:
    """
    Second-stage: generate video_scene_plan, asset_keywords, shorts_plan,
    and full YouTube metadata from an approved script.

    Call this AFTER reviewing and approving the script from /api/episodes/package.

    Input: { "episode_id": "001-meika-jordan", "cost_mode": "bootstrap" }
    """
    try:
        return generate_video_plan(req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Video plan generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video plan generation failed: {exc}")


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})
