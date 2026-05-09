"""
Draft video renderer scaffold.

MVP produces a simple slideshow-style draft video using FFmpeg:
- One frame per scene (black card with on-screen text)
- Narration audio stitched together if available
- Outputs draft_{episode_id}.mp4 to 05-renders/

Full production rendering (motion graphics, AI visuals, real assets) is out of scope for MVP.
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

FFMPEG_CMD = "ffmpeg"


def _ffmpeg_available() -> bool:
    try:
        subprocess.run([FFMPEG_CMD, "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _make_title_card_image(text: str, out_path: Path) -> None:
    """Use FFmpeg lavfi to generate a dark title card PNG."""
    safe_text = text.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")[:80]
    cmd = [
        FFMPEG_CMD, "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x0d0d0d:s=1920x1080:r=1",
        "-vf", (
            f"drawtext=fontsize=48:fontcolor=white"
            f":x=(w-text_w)/2:y=(h-text_h)/2"
            f":text='{safe_text}'"
        ),
        "-frames:v", "1",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def render_draft(
    episode_id: str,
    episode_dir: Path,
) -> str | None:
    """
    Render a minimal draft video.
    Returns path to output video or None if FFmpeg not available.
    """
    if not _ffmpeg_available():
        logger.warning("FFmpeg not found — skipping render")
        return None

    renders_dir = episode_dir / "05-renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # Load video plan
    plan_path = episode_dir / "02-package" / "episode_video_plan.json"
    if not plan_path.exists():
        logger.warning("No video plan found at %s", plan_path)
        return None

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    scenes = plan.get("scenes", [])

    if not scenes:
        logger.warning("No scenes in video plan")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        image_inputs: list[str] = []

        for i, scene in enumerate(scenes):
            label = scene.get("on_screen_text") or scene.get("scene_id", f"Scene {i+1}")
            duration = max(scene.get("duration_sec", 5), 3)
            img_path = tmp / f"frame_{i:04d}.png"

            try:
                _make_title_card_image(label, img_path)
            except subprocess.CalledProcessError as exc:
                logger.warning("Could not create frame %d: %s", i, exc)
                continue

            image_inputs.extend([
                "-loop", "1",
                "-t", str(duration),
                "-i", str(img_path),
            ])

        if not image_inputs:
            logger.warning("No frames generated — aborting render")
            return None

        n = len(image_inputs) // 4  # each input uses 4 args
        filter_parts = "".join(f"[{i}:v]" for i in range(n))
        filter_complex = f"{filter_parts}concat=n={n}:v=1:a=0[outv]"

        out_path = renders_dir / f"draft_{episode_id}.mp4"
        cmd = [
            FFMPEG_CMD, "-y",
            *image_inputs,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", "24",
            str(out_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info("Draft render saved → %s", out_path)
            return str(out_path)
        except subprocess.CalledProcessError as exc:
            logger.error("FFmpeg render failed: %s", exc.stderr.decode())
            return None
