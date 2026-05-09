"""Full pipeline orchestrator — runs package → voice → assets → render."""
from __future__ import annotations

import logging
from pathlib import Path

from app.models import FullPipelineInput, FullPipelineResponse
from app.services.package_service import create_package
from app.services.elevenlabs_service import generate_audio_for_chunks
from app.services.asset_service import fetch_assets_for_keywords
from app.services.guardrail_service import evaluate_candidates
from app.services.renderer_service import render_draft

logger = logging.getLogger(__name__)


def run_full_pipeline(inp: FullPipelineInput) -> FullPipelineResponse:
    pipeline_warnings: list[str] = []

    # Step 1: package (always runs)
    pkg = create_package(inp)
    episode_dir = Path(pkg.episode_dir)

    # Step 2: voice generation (optional)
    voice_files: list[str] = []
    if inp.enable_voice:
        chunks_path = episode_dir / "02-package" / "elevenlabs_chunks.json"
        audio_dir = episode_dir / "03-audio"
        try:
            voice_files = generate_audio_for_chunks(chunks_path, audio_dir)
            logger.info("Voice generation complete: %d files", len(voice_files))
        except Exception as exc:
            msg = f"Voice generation failed: {exc}"
            logger.error(msg)
            pipeline_warnings.append(msg)

    # Step 3: asset fetch (optional)
    asset_files: list[str] = []
    if inp.enable_assets:
        kw_path = episode_dir / "02-package" / "asset_keywords.txt"
        candidates_dir = episode_dir / "04-assets" / "real-candidates"
        try:
            asset_files = fetch_assets_for_keywords(kw_path, candidates_dir)
            # Run guardrails
            review_dir = episode_dir / "06-review"
            evaluate_candidates(candidates_dir, review_dir)
        except Exception as exc:
            msg = f"Asset fetch failed: {exc}"
            logger.error(msg)
            pipeline_warnings.append(msg)

    # Step 4: render (optional)
    render_files: list[str] = []
    if inp.enable_render:
        try:
            render_path = render_draft(
                episode_id=pkg.episode_id,
                episode_dir=episode_dir,
            )
            if render_path:
                render_files = [render_path]
        except Exception as exc:
            msg = f"Render failed: {exc}"
            logger.error(msg)
            pipeline_warnings.append(msg)

    return FullPipelineResponse(
        **pkg.model_dump(),
        voice_files=voice_files,
        asset_files=asset_files,
        render_files=render_files,
        pipeline_warnings=pipeline_warnings,
    )
