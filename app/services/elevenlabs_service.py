"""ElevenLabs voice generation service."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"


def _get_headers() -> dict[str, str]:
    return {
        "xi-api-key": settings.elevenlabs_api_key,
        "Content-Type": "application/json",
    }


def _chunk_path(audio_dir: Path, chunk_id: str) -> Path:
    return audio_dir / f"{chunk_id}.mp3"


def generate_audio_for_chunks(
    chunks_json_path: Path,
    audio_dir: Path,
) -> list[str]:
    """
    Read elevenlabs_chunks.json and generate audio for each chunk.
    Returns list of generated file paths.
    Returns empty list with warning if API key is not set.
    """
    if not settings.elevenlabs_api_key:
        logger.warning("ELEVENLABS_API_KEY not set — skipping voice generation")
        return []

    audio_dir.mkdir(parents=True, exist_ok=True)
    chunks = json.loads(chunks_json_path.read_text(encoding="utf-8"))
    generated: list[str] = []

    voice_id = settings.elevenlabs_narrator_voice_id
    if not voice_id:
        logger.error("ELEVENLABS_NARRATOR_VOICE_ID not set")
        return []

    model_id = settings.elevenlabs_model_id

    with httpx.Client(timeout=120) as client:
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            text = chunk.get("text", "").strip()
            if not text:
                logger.warning("Empty text for chunk %s — skipping", chunk_id)
                continue

            out_path = _chunk_path(audio_dir, chunk_id)
            if out_path.exists():
                logger.info("Chunk %s already exists — skipping", chunk_id)
                generated.append(str(out_path))
                continue

            payload = {
                "text": text,
                "model_id": model_id,
                "voice_settings": chunk.get("voice_settings", {
                    "stability": 0.55,
                    "similarity_boost": 0.75,
                    "style": 0.4,
                    "use_speaker_boost": True,
                }),
            }

            url = f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}"
            try:
                resp = client.post(url, headers=_get_headers(), json=payload)
                resp.raise_for_status()
                out_path.write_bytes(resp.content)
                generated.append(str(out_path))
                logger.info("Generated audio for chunk %s → %s", chunk_id, out_path)
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "ElevenLabs error for chunk %s: %s %s",
                    chunk_id,
                    exc.response.status_code,
                    exc.response.text,
                )
            except Exception as exc:
                logger.error("Unexpected error for chunk %s: %s", chunk_id, exc)

    return generated


def list_voices() -> list[dict]:
    """Fetch available ElevenLabs voices — useful for setup."""
    if not settings.elevenlabs_api_key:
        return []
    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{ELEVENLABS_BASE}/voices", headers=_get_headers())
        resp.raise_for_status()
        return resp.json().get("voices", [])
