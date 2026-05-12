"""
Stage manifest — tracks per-episode input fingerprint, key settings, and prompt hashes.

Saved to: app/storage/episodes/<episode>/stage_manifest.json

Used to guard REUSE_EXISTING_STAGE_OUTPUTS: only reuse stage outputs if
the input hash, critical settings, and relevant prompt file hashes match.

If any of the above differ (different transcript, different hinglish_level,
or a prompt file was edited since last run), the stage is forced to re-run.

Fields tracked:
  - manifest_version: "2" (v2 adds prompt_hashes)
  - input_hash: SHA-256 of the FULL raw transcript (no truncation)
  - cost_mode
  - hinglish_level
  - target_duration_min
  - prompt_hashes: dict mapping stage_name → SHA-256[:16] of prompt file
  - created_at: ISO timestamp of first run
  - updated_at: ISO timestamp of last run
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MANIFEST_VERSION = "2"

# Maps each pipeline stage to the prompt file it uses.
# Only stages that are meaningful to guard against prompt edits are listed.
_STAGE_PROMPT_MAP: dict[str, str] = {
    "fact_lock":                     "fact_lock_agent.txt",
    "story_blueprint":               "story_blueprint_agent.txt",
    "retention_blueprint":           "retention_blueprint_agent.txt",
    "script_outline":                "script_outline_agent.txt",
    "script_writer":                 "narration_chunk_writer_agent.txt",
    "script_quality":                "script_quality_critic_agent.txt",
    "openai_final_premium":          "openai_final_premium_gate.txt",
    # Reusable gate stages — prompt edits must force re-run
    "hindi_copyedit":                "hindi_copyedit_gate_agent.txt",
    "originality_safety":            "originality_safety_gate_agent.txt",
    "recreated_dialogue":            "recreated_dialogue_quality_gate_agent.txt",
    "metadata_quality":              "metadata_quality_gate_agent.txt",
    "retention_quality":             "retention_quality_gate_agent.txt",
    "openai_premium_hindi_editor":   "openai_premium_hindi_editor_gate.txt",
    "openai_originality_youtube_risk": "openai_originality_youtube_risk_gate.txt",
    # Originality transformation — before script outline
    "originality_transformation":    "originality_transformation_agent.txt",
    # Repair stages — tracked in manifest so prompt drift is visible in audit
    "metadata_repair":               "metadata_repair_agent.txt",
    "targeted_chunk_repair":         "targeted_chunk_repair_agent.txt",
    "openai_targeted_chunk_repair":  "openai_targeted_chunk_repair_agent.txt",
    "premium_section_rebuild":       "premium_section_rebuild_agent.txt",
}


def _hash_input(raw_transcript: str) -> str:
    """SHA-256 of the full raw transcript.

    No truncation — detects changes anywhere in the text, including the
    middle and end of long transcripts.
    """
    return hashlib.sha256(raw_transcript.encode("utf-8")).hexdigest()


def _hash_prompt(path: Path) -> Optional[str]:
    """SHA-256 of a prompt file's raw bytes (first 16 hex chars).

    Returns None if the file is not found — callers treat missing files
    as 'unchanged' so a missing prompt never blocks reuse.
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except FileNotFoundError:
        return None


def _compute_prompt_hashes(prompts_dir: Path) -> dict[str, str]:
    """Hash all tracked prompt files. Missing files are silently skipped."""
    hashes: dict[str, str] = {}
    for stage_name, filename in _STAGE_PROMPT_MAP.items():
        h = _hash_prompt(prompts_dir / filename)
        if h is not None:
            hashes[stage_name] = h
    return hashes


def load_manifest(episode_dir: Path) -> dict:
    path = episode_dir / "stage_manifest.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_manifest(
    episode_dir: Path,
    raw_transcript: str,
    cost_mode: str,
    hinglish_level: int,
    target_duration_min: int,
    prompts_dir: Optional[Path] = None,
) -> dict:
    """Save (or overwrite) the stage manifest for this episode.

    If prompts_dir is provided, prompt hashes are recorded so future
    runs can detect prompt edits and force stage re-runs.
    """
    existing = load_manifest(episode_dir)
    now = datetime.now(timezone.utc).isoformat()
    manifest: dict = {
        "manifest_version": MANIFEST_VERSION,
        "input_hash":       _hash_input(raw_transcript),
        "cost_mode":        cost_mode,
        "hinglish_level":   hinglish_level,
        "target_duration_min": target_duration_min,
        "created_at":       existing.get("created_at", now),
        "updated_at":       now,
    }
    if prompts_dir is not None:
        manifest["prompt_hashes"] = _compute_prompt_hashes(prompts_dir)
    path = episode_dir / "stage_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def inputs_changed(
    episode_dir: Path,
    raw_transcript: str,
    cost_mode: str,
    hinglish_level: int,
    target_duration_min: int,
) -> bool:
    """Returns True if transcript or key settings changed since the manifest was saved.

    Returns False when no manifest exists (first run — nothing to compare against).
    """
    manifest = load_manifest(episode_dir)
    if not manifest:
        return False   # no prior manifest — first run
    return (
        manifest.get("input_hash") != _hash_input(raw_transcript)
        or manifest.get("cost_mode") != cost_mode
        or manifest.get("hinglish_level") != hinglish_level
        or manifest.get("target_duration_min") != target_duration_min
    )


def prompt_changed(episode_dir: Path, stage_name: str, prompts_dir: Path) -> bool:
    """Returns True if the prompt file for this stage changed since the manifest was saved.

    Decision table:
      - No manifest at all         → False (first run, don't block)
      - stage_name not in manifest → True  (no prior hash — force rerun to be safe)
      - stage_name not in map      → False (unknown/untracked stage — don't block)
      - prompt file not found      → False (missing file — don't block reuse)
      - hash mismatch              → True  (prompt was edited — force rerun)
      - hash matches               → False (unchanged — allow reuse)
    """
    manifest = load_manifest(episode_dir)
    if not manifest:
        return False   # first run — no manifest to compare against

    saved_hashes = manifest.get("prompt_hashes", {})
    if stage_name not in saved_hashes:
        # Hash was never recorded (e.g. old v1 manifest or newly tracked stage)
        # Force rerun so we don't silently reuse output from an unknown prompt state
        return True

    filename = _STAGE_PROMPT_MAP.get(stage_name)
    if not filename:
        return False   # stage not in the tracking map — don't block

    current_hash = _hash_prompt(prompts_dir / filename)
    if current_hash is None:
        return False   # prompt file not found — don't block reuse

    return saved_hashes[stage_name] != current_hash
