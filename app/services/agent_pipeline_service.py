"""
Controlled Multi-Agent Pipeline Orchestrator

Python is the factory manager. Claude agents are specialized workers.
No agent calls another agent. No agent publishes anything.
Each agent does one job, saves its raw output, and returns parsed JSON.

Pipeline:
  clean_transcript
    → fact_lock          (research_view or segmented)
    → story_blueprint
    → script_draft
    → script_quality_review
    → [one repair pass if needed]
    → [final review if repair ran]
    → text similarity check      (Python-only, premium)
    → originality safety gate    (Claude, premium)
    → recreated dialogue gate    (Claude, premium, skip if no scenes)
    → metadata quality gate      (Claude, premium)
    → gate_summary + safe_to_voice
    → final script package
    → backward-compat copy to 02-package/

Idempotency (REUSE_EXISTING_STAGE_OUTPUTS=true):
    Any stage whose output file already exists is skipped, loading from disk instead.
    Useful for re-running a failed later stage without re-paying for earlier ones.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import settings
from app.models import EpisodeInput, PackageResponse, QualitySummary
from app.schemas import (
    FactLock, StoryBlueprint, ScriptDraft, ScriptQualityReport,
    MetadataQualityReport, RetentionQualityReport, OpenAIFinalPremiumReport,
    normalize_fact_lock_payload,
)
from app.services.transcript_cleaner_service import clean_transcript
from app.services.claude_client import build_transcript_research_view
from app.services.fact_lock_service import run_fact_lock
from app.services.story_blueprint_service import run_story_blueprint
from app.services.case_glossary_service import build_case_glossary
from app.services.python_preflight_service import run_python_preflight
from app.services.script_writer_service import (
    run_script_writer,
    _extract_full_narration,
    _extract_elevenlabs_chunks,
    _slugify,
)
from app.services.script_review_service import run_script_review
from app.services.targeted_chunk_repair_service import (
    run_targeted_chunk_repair,
    promote_draft_as_final,
)
from app.services.package_service import _COST_POLICIES
from app.services.text_similarity_service import run_text_similarity_check
from app.services.hindi_text_lint_service import run_hindi_text_lint
from app.services.hindi_copyedit_gate_service import (
    run_hindi_copyedit_gate,
    run_copyedit_repair,
)
from app.services.originality_safety_gate_service import run_originality_safety_gate
from app.services.recreated_dialogue_quality_gate_service import run_recreated_dialogue_quality_gate
from app.services.metadata_quality_gate_service import run_metadata_quality_gate
from app.services.metadata_repair_service import run_metadata_repair
from app.services.openai_premium_hindi_editor_gate_service import (
    run_openai_premium_hindi_editor_gate,
)
from app.services.openai_originality_youtube_risk_gate_service import (
    run_openai_originality_youtube_risk_gate,
)
from app.services.openai_targeted_chunk_repair_service import (
    run_openai_targeted_chunk_repair,
)
from app.services.openai_final_premium_gate_service import run_openai_final_premium_gate
from app.services import stage_manifest_service
from app.services.retention_blueprint_service import run_retention_blueprint
from app.services.retention_quality_gate_service import (
    run_retention_quality_gate,
    run_retention_repair,
)
from app.services import call_tracker
from app.services.call_tracker import BudgetExceededError

logger = logging.getLogger(__name__)

# Prompts directory — used for prompt hash guards in stage reuse
_PROMPTS_DIR = Path("app/prompts")

# Per-run reuse flag — set in run_agent_pipeline() based on inputs_changed().
# Using threading.local so concurrent requests each get their own flag.
_tls = threading.local()


def _run_reuse_allowed() -> bool:
    """True when stage reuse is permitted for the current pipeline run."""
    return getattr(_tls, "reuse_ok", True)


# ─── Schema validation helpers ────────────────────────────────────────────────

def _validate_fact_lock(data: dict, facts_dir: Path) -> None:
    """
    Validate fact_lock against FactLock schema. Saves error file and raises on failure.

    Normalization runs before this call (normalize_fact_lock_payload), so any
    remaining ValidationError indicates a genuine schema mismatch — likely the
    prompt schema and Pydantic schema have drifted apart.
    """
    try:
        FactLock.model_validate(data)
    except ValidationError as exc:
        err_path = facts_dir / "_fact_lock_validation_error.txt"
        err_path.write_text(str(exc), encoding="utf-8")
        raise ValueError(
            "Fact Lock schema mismatch — pipeline stopped. "
            "Normalization ran but output still does not match FactLock schema. "
            "Check that fact_lock_agent.txt and app/schemas.py FactLock agree.\n"
            f"Error saved at: {err_path}\n"
            f"Validation errors:\n{exc}"
        ) from exc


def _validate_story_blueprint(data: dict, facts_dir: Path) -> None:
    try:
        StoryBlueprint.model_validate(data)
    except ValidationError as exc:
        err_path = facts_dir / "_story_blueprint_validation_error.txt"
        err_path.write_text(str(exc), encoding="utf-8")
        raise ValueError(
            f"Story Blueprint schema validation failed — pipeline stopped.\n"
            f"Error saved at: {err_path}\n"
            f"Details: {exc}"
        ) from exc


def _validate_script_draft(data: dict, script_dir: Path) -> None:
    try:
        ScriptDraft.model_validate(data)
    except ValidationError as exc:
        err_path = script_dir / "_script_draft_validation_error.txt"
        err_path.write_text(str(exc), encoding="utf-8")
        raise ValueError(
            f"Script Draft schema validation failed — pipeline stopped.\n"
            f"Error saved at: {err_path}\n"
            f"Details: {exc}"
        ) from exc


def _validate_quality_report(data: dict, review_dir: Path) -> None:
    """
    Validate quality report against ScriptQualityReport schema.
    Fatal — if the critic output is malformed, do not proceed to repair or promotion.
    A bad quality report means the critic agent may have hallucinated scores or approval;
    proceeding blindly could promote a weak script.
    """
    try:
        ScriptQualityReport.model_validate(data)
    except ValidationError as exc:
        err_path = review_dir / "_script_quality_validation_error.txt"
        err_path.write_text(str(exc), encoding="utf-8")
        raise ValueError(
            f"Script Quality Report schema validation failed — pipeline stopped.\n"
            f"Error saved at: {err_path}\n"
            f"Details: {exc}"
        ) from exc


def _validate_metadata_report(data: dict, review_dir: Path) -> None:
    try:
        MetadataQualityReport.model_validate(data)
    except ValidationError as exc:
        err_path = review_dir / "_metadata_quality_validation_error.txt"
        err_path.write_text(str(exc), encoding="utf-8")
        raise ValueError(
            f"Metadata Quality Report schema validation failed.\n"
            f"Error saved at: {err_path}\nDetails: {exc}"
        ) from exc


def _validate_retention_report(data: dict, review_dir: Path) -> None:
    try:
        RetentionQualityReport.model_validate(data)
    except ValidationError as exc:
        err_path = review_dir / "_retention_quality_validation_error.txt"
        err_path.write_text(str(exc), encoding="utf-8")
        raise ValueError(
            f"Retention Quality Report schema validation failed.\n"
            f"Error saved at: {err_path}\nDetails: {exc}"
        ) from exc


def _validate_openai_final_premium_report(data: dict, review_dir: Path) -> None:
    try:
        OpenAIFinalPremiumReport.model_validate(data)
    except ValidationError as exc:
        err_path = review_dir / "_openai_final_premium_validation_error.txt"
        err_path.write_text(str(exc), encoding="utf-8")
        raise ValueError(
            f"OpenAI Final Premium Report schema validation failed.\n"
            f"Error saved at: {err_path}\nDetails: {exc}"
        ) from exc


# ─── Idempotent file loaders ─────────────────────────────────────────────────

def _try_load_existing_json(
    path: Path,
    stage_name: str,
    episode_dir: Path | None = None,
    prompt_check: str | None = None,
) -> dict | None:
    """Load a JSON stage output if reuse is enabled and file exists.

    Prompt hash guard: if episode_dir and prompt_check are provided, and the
    prompt file for that stage changed since the manifest was saved, the stage
    is forced to re-run regardless of REUSE_EXISTING_STAGE_OUTPUTS=true.
    This prevents silently reusing stale output after prompt edits.
    """
    if not settings.reuse_existing_stage_outputs or not _run_reuse_allowed():
        return None
    # Prompt hash guard
    if episode_dir is not None and prompt_check is not None:
        if stage_manifest_service.prompt_changed(episode_dir, prompt_check, _PROMPTS_DIR):
            logger.info(
                "Stage '%s' prompt '%s' changed since last run — forcing re-run",
                stage_name, prompt_check,
            )
            return None
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.info("Stage '%s' SKIPPED — reusing existing output: %s", stage_name, path)
            call_tracker.mark_reuse(stage_name)
            return data
        except Exception as exc:
            logger.warning("Could not load existing %s: %s — re-running stage", path, exc)
    return None


def _try_load_existing_text(path: Path, stage_name: str) -> str | None:
    """Load a plain-text stage output if reuse is enabled and file exists."""
    if not settings.reuse_existing_stage_outputs or not _run_reuse_allowed():
        return None
    if path.exists():
        try:
            data = path.read_text(encoding="utf-8")
            logger.info("Stage '%s' SKIPPED — reusing existing output: %s", stage_name, path)
            call_tracker.mark_reuse(stage_name)
            return data
        except Exception as exc:
            logger.warning("Could not load existing %s: %s — re-running stage", path, exc)
    return None


# Backward-compat alias — forwards all kwargs including episode_dir / prompt_check
def _try_load_existing(path: Path, stage_name: str, **kwargs: Any) -> dict | None:
    """Alias for _try_load_existing_json for backward compatibility."""
    return _try_load_existing_json(path, stage_name, **kwargs)


def _gate_passed_for_safe_to_voice(name: str, gate: dict) -> bool:
    """Return whether a gate entry should be treated as passing for safe_to_voice.

    python_preflight uses the blocking field (not passed) as its blocking signal.
    passed=False for any issue (including low-only warnings), but blocking=False
    when only low issues exist. Low-only warnings must not block safe_to_voice —
    only medium/high issues (blocking=True) should.

    All other gates use their passed field directly.
    """
    if name == "python_preflight":
        return not gate.get("blocking", True)
    return gate.get("passed", False)


def _reload_latest_gate_reports(
    review_dir: Path,
    lint_report: dict,
    copyedit_report: dict,
    quality_report: dict,
    retention_report: dict,
    similarity_report: dict,
    originality_report: dict,
    dialogue_report: dict,
    metadata_report: dict,
) -> tuple[dict, dict, dict, dict, dict, dict, dict, dict]:
    """Reload gate reports from disk, preferring saved files over in-memory copies.

    After repair stages, on-disk files may be newer than in-memory variables
    (e.g. copyedit repair rewrites hindi_copyedit_report.json, metadata repair
    rewrites metadata_quality_gate_report.json). Calling this before the OpenAI
    Final Premium Gate ensures it receives the latest evidence from every gate.

    Falls back to the supplied in-memory dict if the file does not exist or
    cannot be parsed.
    """
    def _load(filename: str, fallback: dict) -> dict:
        p = review_dir / filename
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return fallback

    # Quality report: prefer the post-repair final version, fall back to initial
    quality = _load("final_script_quality_report.json", {}) or \
              _load("script_quality_report.json", quality_report)

    return (
        _load("hindi_text_lint_report.json",           lint_report),
        _load("hindi_copyedit_report.json",            copyedit_report),
        quality,
        _load("retention_quality_report.json",         retention_report),
        _load("text_similarity_report.json",           similarity_report),
        _load("originality_safety_gate_report.json",   originality_report),
        _load("recreated_dialogue_gate_report.json",   dialogue_report),
        _load("metadata_quality_gate_report.json",     metadata_report),
    )


# ─── Folder setup ─────────────────────────────────────────────────────────────

def _make_episode_dir(episode_number: str, case_hint: str) -> Path:
    slug = _slugify(case_hint)
    folder_name = f"{episode_number}-{slug}"
    base = settings.episodes_dir / folder_name
    for sub in [
        "01-input",
        "02-facts",
        "02-package",
        "03-script",
        "03-audio",
        "04-assets/real-candidates",
        "04-assets/approved",
        "04-assets/generated",
        "04-review",
        "05-renders",
        "06-review",
    ]:
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


# ─── Backward-compat copy to 02-package/ ─────────────────────────────────────

def _write_backward_compat_package(
    episode_dir: Path,
    script_final: dict,
    episode_id: str,
) -> dict[str, str]:
    pkg_dir = episode_dir / "02-package"
    files: dict[str, str] = {}

    def _save(name: str, content: Any) -> None:
        p = pkg_dir / name
        if isinstance(content, (dict, list)):
            p.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            p.write_text(str(content), encoding="utf-8")
        files[name] = str(p)

    _save("production_package.json", script_final)
    _save("production_package_claude.json", script_final)
    _save("case_summary.json", script_final.get("case_summary", {}))

    chunks = script_final.get("hindi_narration_chunks", [])
    (pkg_dir / "hindi_narration_full.txt").write_text(
        _extract_full_narration(chunks), encoding="utf-8"
    )
    files["hindi_narration_full.txt"] = str(pkg_dir / "hindi_narration_full.txt")

    _save("hindi_narration_chunks.json", chunks)
    _save("recreated_dialogues.json", script_final.get("recreated_dialogues", {}))
    _save("elevenlabs_chunks.json", _extract_elevenlabs_chunks(chunks))
    _save("youtube_metadata.json", script_final.get("youtube_metadata", {}))

    # Deferred placeholders
    _save("episode_video_plan.json", {
        "status": "deferred",
        "reason": "Generate after script approval using POST /api/episodes/video-plan",
        "next_step": f"POST /api/episodes/video-plan with episode_id: {episode_id}",
    })
    _save("shorts_plan.json", {
        "status": "deferred",
        "reason": "Generate after script approval using POST /api/episodes/video-plan",
    })
    (pkg_dir / "asset_keywords.txt").write_text(
        "deferred — run POST /api/episodes/video-plan after script approval",
        encoding="utf-8",
    )
    files["asset_keywords.txt"] = str(pkg_dir / "asset_keywords.txt")

    logger.info("Backward-compat package written → %s", pkg_dir)
    return files


# ─── Review files ─────────────────────────────────────────────────────────────

def _write_review_files(
    episode_dir: Path,
    cost_mode: str,
    script_final: dict,
) -> None:
    review_dir = episode_dir / "06-review"

    guardrail_policy = {
        "auto_safe": [
            "city/location visuals", "court buildings", "maps",
            "generic hospital/street/house", "symbolic non-person visuals",
        ],
        "manual_review_required": [
            "victim photos", "family photos", "children",
            "case-specific real people", "news/editorial images",
        ],
        "blocked": [
            "graphic crime scene", "autopsy", "watermarked image",
            "podcast screenshots", "private social media", "unclear-license image",
        ],
    }
    (review_dir / "asset_guardrail_policy.json").write_text(
        json.dumps(guardrail_policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cost_policy = _COST_POLICIES.get(cost_mode, _COST_POLICIES["bootstrap"])
    (review_dir / "estimated_cost_policy.json").write_text(
        json.dumps(cost_policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    checklist = script_final.get("quality_checklist", [])
    (review_dir / "quality_checklist.json").write_text(
        json.dumps(checklist, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ─── Pipeline files index ─────────────────────────────────────────────────────

def _collect_pipeline_files(episode_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for folder in ["01-input", "02-facts", "02-package", "03-script", "04-review", "06-review"]:
        d = episode_dir / folder
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.is_file():
                    files[f"{folder}/{f.name}"] = str(f)
    return files


# ─── Main pipeline entry point ────────────────────────────────────────────────

def run_agent_pipeline(inp: EpisodeInput) -> PackageResponse:
    """
    Controlled multi-agent pipeline for script_first package generation.

    Stages:
       0. Save input
       1. Transcript Cleaner           (Python, no Claude)
       2. Fact Lock Agent              (Claude — research_view, segmented, or auto-segmented for long premium transcripts)
       3. Story Blueprint Agent        (Claude)
      3.5. Retention Blueprint Agent   (Claude — premium only; non-fatal)
       4. Hindi Script Writer          (Claude — uses retention blueprint when available)
       5. Script Quality Critic        (Claude + Python score validation)
       6. Script Repair Agent          (Claude, only if needed — max 1 pass)
       7. Final Quality Review         (Claude, only if repair ran)
      [Premium gates — only when cost_mode=premium]
       8. Hindi Text Lint              (Python-only)
       9. Hindi Copyedit Gate          (Claude)
       9a. Copyedit Targeted Repair    (Claude, only if gate failed — max 1 pass)
       9b. Re-run Lint + Copyedit Gate (Claude, after repair)
      9.5. Retention Quality Gate      (Claude — premium only; scores 8 dimensions)
      9.6. Retention Targeted Repair   (Claude — only if gate failed with targets)
      9.7. Recheck Retention Gate      (Claude — only if repair ran)
      10. Text Similarity Check              (Python-only)
      11. Originality Safety Gate            (Claude)
      12. Recreated Dialogue Gate            (Claude — skipped when no scenes)
      13. Metadata Quality Gate              (Claude)
      13a. Metadata Repair                  (Claude — if gate failed, one pass)
      13b. Recheck Metadata Gate            (Claude — after repair)
      14. OpenAI Premium Hindi Editor Gate   (OpenAI — if openai_review_enabled)
      15. OpenAI Originality/YT Risk Gate    (OpenAI — if openai_review_enabled)
      16. OpenAI Targeted Chunk Repair       (OpenAI — if openai_repair_enabled, one pass max)
      16a. Recheck OpenAI Hindi Editor Gate  (OpenAI — only if 14 failed and repair ran)
      16b. Recheck OpenAI Originality Gate   (OpenAI — only if 15 failed and repair ran)
      17. Backward-compat copy               (Python)
    """
    warnings: list[str] = []
    episode_dir = _make_episode_dir(inp.episode_number, inp.case_hint)
    slug = _slugify(inp.case_hint)
    episode_id = f"{inp.episode_number}-{slug}"

    # Reset call tracker for this pipeline run
    call_tracker.reset()

    logger.info("=" * 60)
    logger.info(
        "AGENT PIPELINE START — episode: %s  case: %s  mode: fact_lock=%s  reuse=%s",
        episode_id, inp.case_hint,
        settings.fact_lock_mode,
        settings.reuse_existing_stage_outputs,
    )
    logger.info(
        "Quality mode: %s  OpenAI policy: %s  skip_gates: %s  "
        "budget: total=%d repair=%d oai_repair=%d",
        settings.quality_mode,
        settings.openai_review_policy,
        settings.skip_final_gates,
        settings.max_total_model_calls,
        settings.max_repair_calls,
        settings.max_openai_repair_calls,
    )
    logger.info("=" * 60)

    if settings.skip_final_gates:
        logger.warning(
            "SKIP_FINAL_GATES=true — all premium quality gates will be bypassed. "
            "Output is NOT voice-ready. For debugging only."
        )

    # ── Stage 0: Save input ───────────────────────────────────────────────────
    input_dir = episode_dir / "01-input"
    (input_dir / "source_transcript.txt").write_text(inp.raw_transcript, encoding="utf-8")
    (input_dir / "input_payload.json").write_text(inp.model_dump_json(indent=2), encoding="utf-8")

    # Check whether inputs changed versus the PREVIOUS manifest (before overwriting it).
    # If they have, disable stage reuse entirely for this run so stale cached outputs
    # are never silently returned.
    _inputs_stale = (
        settings.reuse_existing_stage_outputs
        and stage_manifest_service.inputs_changed(
            episode_dir=episode_dir,
            raw_transcript=inp.raw_transcript,
            cost_mode=inp.cost_mode,
            hinglish_level=inp.hinglish_level,
            target_duration_min=inp.target_duration_min,
        )
    )
    if _inputs_stale:
        _tls.reuse_ok = False
        _stale_msg = (
            "Input or settings changed since last run — stage reuse disabled for this run. "
            "All stages will re-execute to avoid returning stale results."
        )
        logger.warning(_stale_msg)
        warnings.append(_stale_msg)
    else:
        _tls.reuse_ok = True

    # Stage manifest is saved at END of a successful run (see bottom of this function).
    # Saving here would cause the next run to see a "fresh" manifest even if the pipeline
    # crashed midway and stage output files are stale or missing.

    # ── Stage 1: Clean transcript ─────────────────────────────────────────────
    logger.info("Stage 1 — Transcript Cleaner")
    clean_txt_path = input_dir / "clean_transcript.txt"

    existing_clean = _try_load_existing_text(clean_txt_path, "transcript_cleaner")
    if existing_clean is not None:
        clean = existing_clean
    else:
        clean = clean_transcript(
            inp.raw_transcript,
            report_path=input_dir / "transcript_cleanup_report.json",
        )
        clean_txt_path.write_text(clean, encoding="utf-8")
    logger.info("Clean transcript: %d chars", len(clean))

    # Build research view from cleaned transcript
    research_view_path = input_dir / "transcript_research_view.txt"
    existing_rv = _try_load_existing_text(research_view_path, "transcript_research_view")
    if existing_rv is not None:
        research_view = existing_rv
    else:
        research_view = build_transcript_research_view(clean)
        research_view_path.write_text(research_view, encoding="utf-8")
    logger.info("Research view: %d chars", len(research_view))

    facts_dir = episode_dir / "02-facts"

    # ── Stage 2: Fact Lock ────────────────────────────────────────────────────
    call_tracker.stage_start("fact_lock")
    logger.info("Stage 2 — Fact Lock Agent")

    # Task 1: Auto-enable segmented fact extraction for long premium transcripts.
    # If cost_mode=premium AND transcript exceeds threshold AND mode is not already segmented,
    # override to segmented for better coverage on long episodes.
    effective_fact_lock_mode: str | None = None
    if (
        inp.cost_mode == "premium"
        and settings.fact_lock_mode.lower() != "segmented"
        and len(clean) >= settings.premium_segmented_fact_lock_threshold
    ):
        effective_fact_lock_mode = "segmented"
        logger.info(
            "Stage 2 — Auto-switching to segmented fact extraction "
            "(premium + transcript %d chars >= threshold %d)",
            len(clean),
            settings.premium_segmented_fact_lock_threshold,
        )

    existing_fl = _try_load_existing(
        facts_dir / "fact_lock.json", "fact_lock",
        episode_dir=episode_dir, prompt_check="fact_lock",
    )
    if existing_fl is not None:
        fact_lock = existing_fl
    else:
        fact_lock = run_fact_lock(
            case_hint=inp.case_hint,
            episode_number=inp.episode_number,
            source_url=inp.youtube_url,
            transcript_research_view=research_view,
            facts_dir=facts_dir,
            clean_transcript=clean,          # passed for segmented mode
            override_mode=effective_fact_lock_mode,
        )

    # Normalize before validation — converts any plain strings in structured-list
    # fields to the expected object shape. Re-save so downstream agents and the
    # idempotent loader both get the normalized version.
    fact_lock = normalize_fact_lock_payload(fact_lock)
    (facts_dir / "fact_lock.json").write_text(
        json.dumps(fact_lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _validate_fact_lock(fact_lock, facts_dir)
    call_tracker.stage_end("fact_lock")
    logger.info(
        "Fact lock: %d people, %d dates, %d timeline events",
        len(fact_lock.get("verified_people", [])),
        len(fact_lock.get("verified_dates", [])),
        len(fact_lock.get("verified_timeline", [])),
    )

    # ── Stage 3: Story Blueprint ──────────────────────────────────────────────
    call_tracker.stage_start("story_blueprint")
    logger.info("Stage 3 — Story Blueprint Agent")
    existing_bp = _try_load_existing(
        facts_dir / "story_blueprint.json", "story_blueprint",
        episode_dir=episode_dir, prompt_check="story_blueprint",
    )
    if existing_bp is not None:
        blueprint = existing_bp
    else:
        blueprint = run_story_blueprint(
            case_hint=inp.case_hint,
            fact_lock=fact_lock,
            facts_dir=facts_dir,
        )

    _validate_story_blueprint(blueprint, facts_dir)
    call_tracker.stage_end("story_blueprint")
    logger.info(
        "Blueprint: type='%s', sections=%d",
        blueprint.get("primary_story_type", "unknown"),
        len(blueprint.get("narrative_sections", [])),
    )

    script_dir = episode_dir / "03-script"
    review_dir = episode_dir / "04-review"

    # ── Stage 3.25: Case Glossary (Python, zero model cost) ─────────────────
    logger.info("Stage 3.25 — Case Glossary (Python)")
    case_glossary = build_case_glossary(
        fact_lock=fact_lock,
        blueprint=blueprint,
        facts_dir=facts_dir,
    )
    logger.info(
        "Case glossary: %d preferred terms, %d forbidden terms",
        len(case_glossary.get("preferred_terms", {})),
        len(case_glossary.get("do_not_use", [])),
    )

    # ── Stage 3.5: Retention Blueprint (premium only, non-fatal) ─────────────
    retention_blueprint: dict = {}
    retention_report: dict = {}   # populated in Stage 9.5 when retention_blueprint is present
    if inp.cost_mode == "premium":
        logger.info("Stage 3.5 — Retention Blueprint Agent")
        existing_rb = _try_load_existing(
            facts_dir / "retention_blueprint.json", "retention_blueprint",
            episode_dir=episode_dir, prompt_check="retention_blueprint",
        )
        if existing_rb is not None:
            retention_blueprint = existing_rb
        else:
            try:
                retention_blueprint = run_retention_blueprint(
                    fact_lock=fact_lock,
                    blueprint=blueprint,
                    target_duration_min=inp.target_duration_min,
                    case_hint=inp.case_hint,
                    hinglish_level=inp.hinglish_level,
                    facts_dir=facts_dir,
                )
                logger.info(
                    "Retention blueprint: %d re-engagement moments, %d shorts candidates",
                    len(retention_blueprint.get("re_engagement_moments", [])),
                    len(retention_blueprint.get("shorts_candidates", [])),
                )
            except Exception as exc:
                logger.error("Retention blueprint failed (non-fatal): %s", exc)
                warnings.append(
                    f"Retention Blueprint Agent failed: {exc}. "
                    "Script outline will use standard narrative structure."
                )
                retention_blueprint = {}

        # Task 7: Write shorts_strategy.json to 02-package/ from retention blueprint
        shorts_candidates = retention_blueprint.get("shorts_candidates", [])
        if shorts_candidates:
            pkg_dir = episode_dir / "02-package"
            pkg_dir.mkdir(parents=True, exist_ok=True)
            shorts_strategy = {
                "source": "retention_blueprint",
                "shorts_candidates": shorts_candidates,
                "title_thumbnail_angles": retention_blueprint.get("title_thumbnail_angles", []),
                "subscriber_conversion_moment": retention_blueprint.get(
                    "subscriber_conversion_moment", ""
                ),
            }
            (pkg_dir / "shorts_strategy.json").write_text(
                json.dumps(shorts_strategy, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(
                "Stage 3.5 — shorts_strategy.json written → 02-package/ (%d candidates)",
                len(shorts_candidates),
            )

    # ── Stage 4: Script Writer ────────────────────────────────────────────────
    call_tracker.stage_start("script_writer")
    logger.info("Stage 4 — Hindi Script Writer Agent")
    existing_draft = _try_load_existing(
        script_dir / "script_draft.json", "script_writer",
        episode_dir=episode_dir, prompt_check="script_writer",
    )
    if existing_draft is not None:
        script_draft = existing_draft
    else:
        script_draft = run_script_writer(
            case_hint=inp.case_hint,
            episode_number=inp.episode_number,
            episode_id=episode_id,
            target_duration_min=inp.target_duration_min,
            cost_mode=inp.cost_mode,
            style=inp.style,
            fact_lock=fact_lock,
            blueprint=blueprint,
            script_dir=script_dir,
            hinglish_level=inp.hinglish_level,
            retention_blueprint=retention_blueprint if retention_blueprint else None,
            case_glossary=case_glossary,
        )

    _validate_script_draft(script_draft, script_dir)
    call_tracker.stage_end("script_writer")
    logger.info(
        "Script draft: %d narration chunks",
        len(script_draft.get("hindi_narration_chunks", [])),
    )

    # ── Stage 5: Script Quality Review ───────────────────────────────────────
    call_tracker.stage_start("quality_review")
    logger.info("Stage 5 — Script Quality Critic Agent")
    existing_qr = _try_load_existing(
        review_dir / "script_quality_report.json", "quality_critic",
        episode_dir=episode_dir, prompt_check="script_quality",
    )
    if existing_qr is not None:
        quality_report = existing_qr
    else:
        quality_report = run_script_review(
            target_duration_min=inp.target_duration_min,
            cost_mode=inp.cost_mode,
            fact_lock=fact_lock,
            blueprint=blueprint,
            script_draft=script_draft,
            review_dir=review_dir,
            is_final_review=False,
            hinglish_level=inp.hinglish_level,
            case_glossary=case_glossary,
        )

    _validate_quality_report(quality_report, review_dir)
    call_tracker.stage_end("quality_review")

    approved = quality_report.get("approved", False)
    repair_required = quality_report.get("repair_required", False)
    scores = quality_report.get("scores", {})
    logger.info(
        "Quality review: approved=%s, repair_required=%s | scores: %s",
        approved, repair_required, scores,
    )

    # ── Stage 5.5: Python Preflight (zero model cost) ───────────────────────
    logger.info("Stage 5.5 — Python Preflight Gate")
    preflight_report = run_python_preflight(
        script_draft=script_draft,
        fact_lock=fact_lock,
        case_glossary=case_glossary,
        review_dir=review_dir,
        target_duration_min=inp.target_duration_min,
        hinglish_level=inp.hinglish_level,
    )
    # Carry preflight blocking + metadata targets forward so Stage 13a can use them.
    _preflight_blocking = preflight_report.get("blocking", False)
    _preflight_meta_targets = preflight_report.get("metadata_repair_targets", [])
    _ran_any_repair = False  # set True when chunk OR metadata repair runs

    if not preflight_report.get("passed", False):
        pf_targets = preflight_report.get("chunk_repair_targets", [])
        pf_metadata = preflight_report.get("metadata_issues", [])
        if pf_targets:
            existing_targets = quality_report.get("chunk_repair_targets", [])
            seen = {
                (t.get("chunk_id", ""), t.get("issue_type", ""), t.get("problem", ""))
                for t in existing_targets
            }
            for target in pf_targets:
                key = (
                    target.get("chunk_id", ""),
                    target.get("issue_type", ""),
                    target.get("problem", ""),
                )
                if key not in seen:
                    existing_targets.append(target)
                    seen.add(key)
            quality_report["chunk_repair_targets"] = existing_targets
            quality_report["approved"] = False
            quality_report["repair_required"] = True
            approved = False
            repair_required = True
        if pf_metadata:
            quality_report.setdefault("youtube_metadata_issues", [])
            quality_report["youtube_metadata_issues"].extend(
                i.get("problem", str(i)) for i in pf_metadata
            )
        warnings.append(
            "Python preflight found deterministic issues; repair targets were merged before final gates. "
            "See 04-review/python_preflight_report.json."
        )
        (review_dir / "script_quality_report.json").write_text(
            json.dumps(quality_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Stage 6: Targeted Chunk Repair (one pass max) ─────────────────────────
    repair_has_failures = False   # default: no repair ran, no failures
    if not approved and repair_required:
        logger.info("Stage 6 — Targeted Chunk Repair")
        try:
            chunk_repair_targets = quality_report.get("chunk_repair_targets", [])
            if not chunk_repair_targets:
                warnings.append(
                    "Repair required but critic did not provide chunk_repair_targets. "
                    "Human review required before proceeding to audio generation."
                )
                logger.warning(
                    "Stage 6 — repair_required=true but no chunk_repair_targets were provided. "
                    "Skipping automated repair and forcing needs_human_review."
                )
                script_final = promote_draft_as_final(script_draft, script_dir)
                status = "needs_human_review"
                quality_report["approved"] = False
                quality_report["repair_required"] = True
            else:
                # Budget check is now per-chunk inside run_targeted_chunk_repair
                script_final, repair_report = run_targeted_chunk_repair(
                    fact_lock=fact_lock,
                    blueprint=blueprint,
                    script_draft=script_draft,
                    quality_report=quality_report,
                    hinglish_level=inp.hinglish_level,
                    script_dir=script_dir,
                    review_dir=review_dir,
                )

                _ran_any_repair = True
                repair_has_failures = repair_report.get("has_failures", False)
                repair_failed_count = repair_report.get("chunks_failed", 0)

                if repair_has_failures:
                    warnings.append(
                        f"{repair_failed_count} chunk repair(s) failed — original content kept. "
                        "Human review required before proceeding to audio generation. "
                        "See 04-review/script_repair_report.json for details."
                    )
                    logger.warning(
                        "Stage 6 — %d chunk repair(s) failed. Status forced to needs_human_review.",
                        repair_failed_count,
                    )

                # Stage 7: Final quality review after repair
                logger.info("Stage 7 — Final Quality Review (post-repair)")
                final_quality = run_script_review(
                    target_duration_min=inp.target_duration_min,
                    cost_mode=inp.cost_mode,
                    fact_lock=fact_lock,
                    blueprint=blueprint,
                    script_draft=script_final,
                    review_dir=review_dir,
                    is_final_review=True,
                    hinglish_level=inp.hinglish_level,
                    case_glossary=case_glossary,
                )
                quality_report = final_quality

                # Approve only if critic approved AND every targeted repair succeeded
                if final_quality.get("approved", False) and not repair_has_failures:
                    status = "script_approved"
                    logger.info("Final review: APPROVED after repair (no repair failures)")
                else:
                    status = "needs_human_review"
                    if not final_quality.get("approved", False):
                        warnings.append(
                            "Script was repaired but final quality review did not approve. "
                            "Human review required before proceeding to audio generation. "
                            "See 04-review/final_script_quality_report.json for details."
                        )
                    logger.warning("Final review: NOT APPROVED after repair — needs_human_review")
        except Exception as exc:
            logger.error("Script repair failed: %s", exc)
            warnings.append(
                f"Script repair failed: {exc}. "
                "Draft promoted as final. Human review required."
            )
            script_final = promote_draft_as_final(script_draft, script_dir)
            status = "needs_human_review"
    else:
        script_final = promote_draft_as_final(script_draft, script_dir)
        status = "script_approved" if approved else "needs_human_review"
        if not approved:
            warnings.append(
                "Script quality review did not approve and repair was not flagged. "
                "Human review recommended. See 04-review/script_quality_report.json."
            )

    # ── Premium quality gates (cost_mode=premium only) ───────────────────────
    gate_summary: dict[str, dict] = {}
    safe_to_voice = False

    # Always record script_quality gate result
    gate_summary["script_quality"] = {
        "passed": quality_report.get("approved", False),
        "scores": quality_report.get("scores", {}),
        "cost_mode": quality_report.get("cost_mode", inp.cost_mode),
    }

    # Always record Python preflight result (initial run at Stage 5.5).
    # Updated below if post-repair recheck runs.
    _pf_counts = preflight_report.get("severity_counts", {})
    gate_summary["python_preflight"] = {
        "passed":    preflight_report.get("passed", False),
        "blocking":  preflight_report.get("blocking", False),
        "high":      _pf_counts.get("high", 0),
        "medium":    _pf_counts.get("medium", 0),
        "low":       _pf_counts.get("low", 0),
        "report":    "python_preflight_report.json",
        "rechecked": False,
    }

    if inp.cost_mode == "premium" and settings.skip_final_gates:
        # ── SKIP_FINAL_GATES mode — bypass all premium gates ──────────────────
        logger.warning(
            "SKIP_FINAL_GATES=true — skipping all premium quality gates (stages 8–16). "
            "This episode is NOT voice-ready."
        )
        gate_summary.update({
            "hindi_copyedit":                  {"passed": True, "skipped": True, "reason": "SKIP_FINAL_GATES"},
            "retention_quality":               {"passed": True, "skipped": True, "reason": "SKIP_FINAL_GATES"},
            "originality_safety":              {"passed": True, "skipped": True, "reason": "SKIP_FINAL_GATES"},
            "recreated_dialogue":              {"passed": True, "skipped": True, "reason": "SKIP_FINAL_GATES"},
            "metadata_quality":                {"passed": True, "skipped": True, "reason": "SKIP_FINAL_GATES"},
            "openai_final_premium":            {"passed": True, "skipped": True, "reason": "SKIP_FINAL_GATES"},
            "openai_premium_hindi_editor":     {"passed": True, "skipped": True, "reason": "SKIP_FINAL_GATES"},
            "openai_originality_youtube_risk": {"passed": True, "skipped": True, "reason": "SKIP_FINAL_GATES"},
            "repair_failures": {
                "claude_script_repair_failed":  False,
                "copyedit_repair_failed":       False,
                "metadata_repair_failed":       False,
                "retention_repair_failed":      False,
                "openai_repair_failed":         False,
                "passed":                       True,
            },
        })
        safe_to_voice = False   # Never voice-ready in skip-gates mode
        gate_summary["safe_to_voice"] = False  # type: ignore[assignment]
        warnings.append(
            "SKIP_FINAL_GATES=true — all premium quality gates were bypassed. "
            "safe_to_voice is permanently False in this mode. "
            "Remove SKIP_FINAL_GATES from .env before audio production."
        )

    elif inp.cost_mode == "premium":
        # ── Stage 8: Hindi Text Lint (Python-only) ────────────────────────────
        logger.info("Stage 8 — Hindi Text Lint (Python)")
        lint_report = run_hindi_text_lint(script_final, hinglish_level=inp.hinglish_level)
        lint_path = review_dir / "hindi_text_lint_report.json"
        lint_path.write_text(
            json.dumps(lint_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "Hindi lint: %d issues (high=%d) risk=%s",
            lint_report.get("total_issues", 0),
            lint_report.get("high_issues", 0),
            lint_report.get("risk_level", "?"),
        )

        # ── Stage 9: Hindi Copyedit Gate ──────────────────────────────────────
        logger.info("Stage 9 — Hindi Copyedit Gate")
        existing_ce = _try_load_existing(
            review_dir / "hindi_copyedit_report.json", "hindi_copyedit",
            episode_dir=episode_dir, prompt_check="hindi_copyedit",
        )
        if existing_ce is not None:
            copyedit_report = existing_ce
        else:
            try:
                copyedit_report = run_hindi_copyedit_gate(
                    script_draft=script_final,
                    fact_lock=fact_lock,
                    blueprint=blueprint,
                    hinglish_level=inp.hinglish_level,
                    lint_report=lint_report,
                    review_dir=review_dir,
                )
            except Exception as exc:
                logger.error("Hindi copyedit gate failed: %s", exc)
                warnings.append(
                    f"Hindi copyedit gate failed: {exc}. Manual review required."
                )
                copyedit_report = {
                    "approved": False, "score": 0, "error": str(exc),
                    "chunk_repair_targets": [],
                }

        # ── Stage 9a: Copyedit Targeted Repair (if gate failed) ───────────────
        copyedit_repair_has_failures = False
        if not copyedit_report.get("approved", False):
            ce_targets = copyedit_report.get("chunk_repair_targets", [])
            if ce_targets:
                logger.info("Stage 9a — Copyedit Targeted Repair (%d targets)", len(ce_targets))
                try:
                    script_final, copyedit_repair_report = run_copyedit_repair(
                        script_draft=script_final,
                        fact_lock=fact_lock,
                        blueprint=blueprint,
                        copyedit_targets=ce_targets,
                        hinglish_level=inp.hinglish_level,
                        script_dir=script_dir,
                        review_dir=review_dir,
                    )
                    _ran_any_repair = True
                    copyedit_repair_has_failures = copyedit_repair_report.get("has_failures", False)
                    if copyedit_repair_has_failures:
                        failed_n = copyedit_repair_report.get("chunks_failed", 0)
                        warnings.append(
                            f"Copyedit repair: {failed_n} chunk(s) failed — "
                            "original content kept. Manual review required."
                        )

                    # ── Stage 9b: Re-run lint + copyedit gate (once) ──────────
                    logger.info("Stage 9b — Re-run Hindi Lint + Copyedit Gate (post-repair)")
                    lint_report = run_hindi_text_lint(
                        script_final, hinglish_level=inp.hinglish_level
                    )
                    lint_path.write_text(
                        json.dumps(lint_report, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    copyedit_report = run_hindi_copyedit_gate(
                        script_draft=script_final,
                        fact_lock=fact_lock,
                        blueprint=blueprint,
                        hinglish_level=inp.hinglish_level,
                        lint_report=lint_report,
                        review_dir=review_dir,
                        is_recheck=True,
                    )
                    logger.info(
                        "Copyedit re-check: approved=%s score=%s",
                        copyedit_report.get("approved", False),
                        copyedit_report.get("score", "?"),
                    )
                except Exception as exc:
                    logger.error("Copyedit repair/recheck failed: %s", exc)
                    warnings.append(
                        f"Copyedit targeted repair failed: {exc}. Original chunks kept."
                    )
                    copyedit_repair_has_failures = True
            else:
                # Gate failed but no targets to repair — human must fix manually
                warnings.append(
                    "Hindi copyedit gate FAILED but no chunk_repair_targets provided. "
                    "Manual review required before audio generation."
                )

        gate_summary["hindi_copyedit"] = {
            "passed":                      copyedit_report.get("approved", False),
            "score":                       copyedit_report.get("score", 0),
            "grammar_score":               copyedit_report.get("grammar_score", 0),
            "matra_nasalization_score":    copyedit_report.get("matra_nasalization_score", 0),
            "sentence_flow_score":         copyedit_report.get("sentence_flow_score", 0),
            "legal_language_clarity_score":copyedit_report.get("legal_language_clarity_score", 0),
            "hinglish_consistency_score":  copyedit_report.get("hinglish_level_consistency_score", 0),
            "high_severity_issues":        sum(
                1 for i in copyedit_report.get("issues", [])
                if i.get("severity") == "high"
            ),
            "repair_had_failures":         copyedit_repair_has_failures,
        }

        if not copyedit_report.get("approved", False):
            status = "needs_human_review"
            if not any("copyedit" in w.lower() for w in warnings):
                warnings.append(
                    "Hindi copyedit gate FAILED. "
                    "See 04-review/hindi_copyedit_report.json for grammar and style issues."
                )

        # ── Stage 10: Text Similarity (Python-only) ───────────────────────────
        logger.info("Stage 10 — Text Similarity Check (Python)")
        clean_transcript_text = (episode_dir / "01-input" / "clean_transcript.txt").read_text(
            encoding="utf-8"
        )
        similarity_report = run_text_similarity_check(
            source_transcript=clean_transcript_text,
            script_draft=script_final,
        )
        sim_path = review_dir / "text_similarity_report.json"
        sim_path.write_text(
            json.dumps(similarity_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "Text similarity: risk=%s matches=%d",
            similarity_report.get("risk_level", "?"),
            similarity_report.get("total_match_count", 0),
        )

        # ── Stage 11: Originality & Safety Gate ───────────────────────────────
        logger.info("Stage 11 — Originality Safety Gate")
        existing_orig = _try_load_existing(
            review_dir / "originality_safety_gate_report.json", "originality_gate",
            episode_dir=episode_dir, prompt_check="originality_safety",
        )
        if existing_orig is not None:
            originality_report = existing_orig
        else:
            try:
                originality_report = run_originality_safety_gate(
                    script_draft=script_final,
                    source_transcript=clean_transcript_text,
                    similarity_report=similarity_report,
                    review_dir=review_dir,
                )
            except Exception as exc:
                logger.error("Originality gate failed: %s", exc)
                warnings.append(
                    f"Originality safety gate failed: {exc}. Manual review required."
                )
                originality_report = {"gate_passed": False, "error": str(exc)}

        gate_summary["originality_safety"] = {
            "passed":         originality_report.get("gate_passed", False),
            "scores":         originality_report.get("scores", {}),
            "required_fixes": originality_report.get("required_fixes", []),
        }
        if not originality_report.get("gate_passed", False):
            required_fixes = originality_report.get("required_fixes", [])
            if required_fixes:
                warnings.append(
                    "Originality/safety gate FAILED. Required fixes: "
                    + "; ".join(required_fixes[:3])
                    + (f" (+{len(required_fixes)-3} more)" if len(required_fixes) > 3 else "")
                )
            status = "needs_human_review"

        # ── Stage 12: Recreated Dialogue Quality Gate ─────────────────────────
        logger.info("Stage 12 — Recreated Dialogue Quality Gate")
        existing_dlg = _try_load_existing(
            review_dir / "recreated_dialogue_gate_report.json", "dialogue_gate",
            episode_dir=episode_dir, prompt_check="recreated_dialogue",
        )
        if existing_dlg is not None:
            dialogue_report = existing_dlg
        else:
            try:
                dialogue_report = run_recreated_dialogue_quality_gate(
                    script_draft=script_final,
                    fact_lock=fact_lock,
                    review_dir=review_dir,
                )
            except Exception as exc:
                logger.error("Dialogue gate failed: %s", exc)
                warnings.append(
                    f"Recreated dialogue quality gate failed: {exc}. Manual review required."
                )
                dialogue_report = {"gate_passed": False, "error": str(exc)}

        gate_summary["recreated_dialogue"] = {
            "passed":         dialogue_report.get("gate_passed", False),
            "no_scenes":      dialogue_report.get("no_recreated_scenes", False),
            "scores":         dialogue_report.get("scores", {}),
            "required_fixes": dialogue_report.get("required_fixes", []),
        }
        if not dialogue_report.get("gate_passed", False):
            required_fixes = dialogue_report.get("required_fixes", [])
            if required_fixes:
                warnings.append(
                    "Recreated dialogue gate FAILED. Required fixes: "
                    + "; ".join(required_fixes[:3])
                    + (f" (+{len(required_fixes)-3} more)" if len(required_fixes) > 3 else "")
                )
            status = "needs_human_review"

        # ── Stage 13: Metadata Quality Gate ───────────────────────────────────
        logger.info("Stage 13 — Metadata Quality Gate")
        existing_meta = _try_load_existing(
            review_dir / "metadata_quality_gate_report.json", "metadata_gate",
            episode_dir=episode_dir, prompt_check="metadata_quality",
        )
        if existing_meta is not None:
            metadata_report = existing_meta
        else:
            try:
                metadata_report = run_metadata_quality_gate(
                    script_draft=script_final,
                    fact_lock=fact_lock,
                    review_dir=review_dir,
                )
            except Exception as exc:
                logger.error("Metadata gate failed: %s", exc)
                warnings.append(
                    f"Metadata quality gate failed: {exc}. Manual review required."
                )
                metadata_report = {"gate_passed": False, "error": str(exc)}

        # Validate metadata gate report schema (non-fatal)
        try:
            _validate_metadata_report(metadata_report, review_dir)
        except ValueError as exc:
            logger.warning("Metadata Quality Report schema validation (non-fatal): %s", exc)
            warnings.append(
                f"Metadata Quality Report schema mismatch (non-fatal). "
                "Results may be unreliable — see 04-review/_metadata_quality_validation_error.txt. "
                "Status set to needs_human_review."
            )
            status = "needs_human_review"

        # ── Stage 13a: Metadata Repair (if gate failed OR preflight blocked) ──
        # Trigger repair when:
        #   - Claude's metadata quality gate failed (gate_passed=False), OR
        #   - Python preflight found blocking metadata issues (preflight.blocking=True
        #     AND metadata_repair_targets is non-empty).
        # This ensures Python-detected metadata problems are fixed before the
        # expensive OpenAI Final Premium Gate is called.
        _preflight_forces_meta_repair = (
            _preflight_blocking
            and bool(_preflight_meta_targets)
            and metadata_report.get("gate_passed", False)  # gate passed but preflight blocked
        )
        if _preflight_forces_meta_repair:
            # Inject preflight targets as required_fixes so metadata repair agent sees them
            existing_req = list(metadata_report.get("required_fixes", []))
            for tgt in _preflight_meta_targets:
                problem = tgt.get("problem", "")
                if problem and problem not in existing_req:
                    existing_req.append(problem)
            metadata_report["required_fixes"] = existing_req
            metadata_report["gate_passed"] = False  # force repair
            logger.info(
                "Stage 13a — Python preflight blocking metadata issues will trigger metadata repair "
                "(%d preflight targets injected)", len(_preflight_meta_targets)
            )
            warnings.append(
                "Python preflight found blocking metadata issues — "
                "metadata repair triggered before OpenAI Final Premium Gate. "
                "See 04-review/python_preflight_report.json."
            )

        metadata_repair_has_failures = False
        if not metadata_report.get("gate_passed", False):
            meta_required_fixes = metadata_report.get("required_fixes", [])
            if meta_required_fixes and not _preflight_forces_meta_repair:
                warnings.append(
                    "Metadata quality gate FAILED. Required fixes: "
                    + "; ".join(meta_required_fixes[:3])
                    + (f" (+{len(meta_required_fixes)-3} more)" if len(meta_required_fixes) > 3 else "")
                )

            logger.info("Stage 13a — Metadata Repair (targeted fix of youtube_metadata)")
            try:
                script_final, _meta_repair_report = run_metadata_repair(
                    script_draft=script_final,
                    fact_lock=fact_lock,
                    gate_report=metadata_report,
                    script_dir=script_dir,
                    review_dir=review_dir,
                )
                _ran_any_repair = True

                # ── Stage 13b: Recheck Metadata Gate ─────────────────────────
                logger.info("Stage 13b — Recheck Metadata Quality Gate (post-repair)")
                try:
                    metadata_report = run_metadata_quality_gate(
                        script_draft=script_final,
                        fact_lock=fact_lock,
                        review_dir=review_dir,
                    )
                    logger.info(
                        "Metadata recheck: passed=%s | scores=%s",
                        metadata_report.get("gate_passed", False),
                        metadata_report.get("scores", {}),
                    )
                    if not metadata_report.get("gate_passed", False):
                        status = "needs_human_review"
                        warnings.append(
                            "Metadata recheck FAILED after repair. "
                            "Manual review required — "
                            "see 04-review/metadata_quality_gate_report.json."
                        )
                except Exception as exc:
                    logger.error("Metadata recheck failed: %s", exc)
                    warnings.append(
                        f"Metadata quality gate recheck failed: {exc}. Manual review required."
                    )
                    metadata_repair_has_failures = True

            except Exception as exc:
                logger.error("Metadata repair failed: %s", exc)
                warnings.append(
                    f"Metadata repair failed: {exc}. "
                    "Manual correction of youtube_metadata required."
                )
                metadata_repair_has_failures = True
                status = "needs_human_review"
        else:
            # Gate passed — no repair needed
            pass

        gate_summary["metadata_quality"] = {
            "passed":         metadata_report.get("gate_passed", False),
            "scores":         metadata_report.get("scores", {}),
            "required_fixes": metadata_report.get("required_fixes", []),
            "repair_ran":     not metadata_report.get("gate_passed", True),
        }
        if not metadata_report.get("gate_passed", False) and not metadata_repair_has_failures:
            status = "needs_human_review"

        # ── Stage 9.5: Retention Quality Gate ────────────────────────────────
        retention_repair_has_failures = False
        if retention_blueprint:
            logger.info("Stage 9.5 — Retention Quality Gate")
            existing_rq = _try_load_existing(
                review_dir / "retention_quality_report.json", "retention_quality",
                episode_dir=episode_dir, prompt_check="retention_quality",
            )
            if existing_rq is not None:
                retention_report = existing_rq
            else:
                try:
                    retention_report = run_retention_quality_gate(
                        script_draft=script_final,
                        retention_blueprint=retention_blueprint,
                        blueprint=blueprint,
                        target_duration_min=inp.target_duration_min,
                        review_dir=review_dir,
                    )
                except Exception as exc:
                    logger.error("Retention quality gate failed: %s", exc)
                    warnings.append(
                        f"Retention quality gate failed: {exc}. Manual review required."
                    )
                    retention_report = {"approved": False, "error": str(exc), "chunk_repair_targets": []}

            # Validate retention gate report schema (non-fatal)
            try:
                _validate_retention_report(retention_report, review_dir)
            except ValueError as exc:
                logger.warning("Retention Quality Report schema validation (non-fatal): %s", exc)
                warnings.append(
                    f"Retention Quality Report schema mismatch (non-fatal). "
                    "Results may be unreliable — see 04-review/_retention_quality_validation_error.txt. "
                    "Status set to needs_human_review."
                )
                status = "needs_human_review"

            gate_summary["retention_quality"] = {
                "passed":                  retention_report.get("approved", False),
                "overall_retention_score": retention_report.get("overall_retention_score", 0),
                "opening_hook_score":      retention_report.get("opening_hook_score", 0),
                "curiosity_gap_score":     retention_report.get("curiosity_gap_score", 0),
                "pacing_score":            retention_report.get("pacing_score", 0),
                "emotional_arc_score":     retention_report.get("emotional_arc_score", 0),
                "ending_payoff_score":     retention_report.get("ending_payoff_score", 0),
                "high_severity_issues":    sum(
                    1 for i in retention_report.get("issues", [])
                    if i.get("severity") == "high"
                ),
            }

            if not retention_report.get("approved", False):
                status = "needs_human_review"
                rt_targets = retention_report.get("chunk_repair_targets", [])

                if rt_targets:
                    # ── Stage 9.6: Retention Targeted Repair ─────────────────
                    logger.info(
                        "Stage 9.6 — Retention Targeted Repair (%d target(s))", len(rt_targets)
                    )
                    try:
                        script_final, retention_repair_report = run_retention_repair(
                            script_draft=script_final,
                            fact_lock=fact_lock,
                            blueprint=blueprint,
                            retention_blueprint=retention_blueprint,
                            repair_targets=rt_targets,
                            hinglish_level=inp.hinglish_level,
                            script_dir=script_dir,
                            review_dir=review_dir,
                        )
                        _ran_any_repair = True
                        retention_repair_has_failures = retention_repair_report.get(
                            "has_failures", False
                        )
                        if retention_repair_has_failures:
                            failed_n = retention_repair_report.get("chunks_failed", 0)
                            warnings.append(
                                f"Retention repair: {failed_n} chunk(s) failed — "
                                "original content kept. Manual review required. "
                                "See 04-review/retention_repair_report.json."
                            )

                        # ── Stage 9.7: Recheck Retention Gate ────────────────
                        logger.info("Stage 9.7 — Recheck Retention Quality Gate")
                        try:
                            retention_recheck = run_retention_quality_gate(
                                script_draft=script_final,
                                retention_blueprint=retention_blueprint,
                                blueprint=blueprint,
                                target_duration_min=inp.target_duration_min,
                                review_dir=review_dir,
                                is_recheck=True,
                            )
                            retention_report = retention_recheck
                            rq_now_passed = retention_recheck.get("approved", False)
                            gate_summary["retention_quality"] = {
                                "passed":                  rq_now_passed,
                                "overall_retention_score": retention_recheck.get("overall_retention_score", 0),
                                "opening_hook_score":      retention_recheck.get("opening_hook_score", 0),
                                "curiosity_gap_score":     retention_recheck.get("curiosity_gap_score", 0),
                                "pacing_score":            retention_recheck.get("pacing_score", 0),
                                "emotional_arc_score":     retention_recheck.get("emotional_arc_score", 0),
                                "ending_payoff_score":     retention_recheck.get("ending_payoff_score", 0),
                                "high_severity_issues":    sum(
                                    1 for i in retention_recheck.get("issues", [])
                                    if i.get("severity") == "high"
                                ),
                                "recheck": True,
                            }
                            if rq_now_passed:
                                logger.info("Retention quality recheck: PASSED")
                            else:
                                status = "needs_human_review"
                                logger.warning(
                                    "Retention quality recheck: still FAILED "
                                    "(overall=%s)",
                                    retention_recheck.get("overall_retention_score", 0),
                                )
                                warnings.append(
                                    "Retention quality recheck FAILED after repair. "
                                    "Manual review required — "
                                    "see 04-review/retention_quality_report.json."
                                )
                        except Exception as exc:
                            logger.error("Retention quality recheck failed: %s", exc)
                            warnings.append(
                                f"Retention quality recheck failed: {exc}. Manual review required."
                            )
                            retention_repair_has_failures = True

                    except Exception as exc:
                        logger.error("Retention repair failed: %s", exc)
                        warnings.append(
                            f"Retention targeted repair failed: {exc}. "
                            "Original chunks kept. Manual review required."
                        )
                        retention_repair_has_failures = True
                else:
                    warnings.append(
                        "Retention quality gate FAILED but no chunk_repair_targets provided. "
                        "Manual review required — see 04-review/retention_quality_report.json."
                    )
        else:
            # No retention blueprint — skip retention gate (not blocking)
            gate_summary["retention_quality"] = {
                "passed": True,
                "skipped": True,
                "reason": "No retention blueprint — standard mode or blueprint generation failed",
            }

        openai_repair_has_failures = False   # default: no OpenAI repair ran

        # ── Stages 14–16: OpenAI independent review gates ─────────────────────
        # Gating logic (new combined final premium gate):
        #   policy=adaptive  → run combined Final Premium Gate (single call, all dimensions)
        #                       if passes: skip legacy Stage 14/15; if fails with targets: repair once
        #   policy=always    → run Final Premium Gate + legacy Stage 14 + legacy Stage 15 (all 3)
        #                       all 3 must pass for safe_to_voice=True
        #   policy=disabled  → skip OpenAI calls, but output is NOT voice-ready
        #   quality_mode != premium_final, or openai_review_enabled=false → not voice-ready

        # ── Stage 13c: Post-repair Python Preflight recheck ──────────────────
        # Runs only when chunk repair OR metadata repair has executed.
        # If still blocking → skip OpenAI Final Premium Gate entirely.
        _post_repair_preflight_blocking = False
        if _ran_any_repair:
            logger.info(
                "Stage 13c — Post-repair Python Preflight recheck (ran_any_repair=True)"
            )
            try:
                _post_pf_report = run_python_preflight(
                    script_draft=script_final,
                    fact_lock=fact_lock,
                    case_glossary=case_glossary,
                    review_dir=review_dir,
                    target_duration_min=inp.target_duration_min,
                    hinglish_level=inp.hinglish_level,
                    label="_after_repair",
                )
                _post_repair_preflight_blocking = _post_pf_report.get("blocking", False)
                _post_pf_counts = _post_pf_report.get("severity_counts", {})
                gate_summary["python_preflight"] = {
                    "passed":    _post_pf_report.get("passed", False),
                    "blocking":  _post_repair_preflight_blocking,
                    "high":      _post_pf_counts.get("high", 0),
                    "medium":    _post_pf_counts.get("medium", 0),
                    "low":       _post_pf_counts.get("low", 0),
                    "report":    "python_preflight_report_after_repair.json",
                    "rechecked": True,
                }
                if _post_repair_preflight_blocking:
                    status = "needs_human_review"
                    high_n = _post_pf_counts.get("high", 0)
                    med_n  = _post_pf_counts.get("medium", 0)
                    warnings.append(
                        f"Post-repair Python preflight still BLOCKING "
                        f"(high={high_n}, medium={med_n}). "
                        "OpenAI Final Premium Gate skipped. "
                        "Fix blocking issues manually before re-running. "
                        "See 04-review/python_preflight_report_after_repair.json."
                    )
                    logger.warning(
                        "Stage 13c — Post-repair preflight BLOCKING "
                        "(high=%d, medium=%d) — skipping OFP gate.",
                        high_n, med_n,
                    )
            except Exception as exc:
                # Exception means the safety gate could not run — treat as blocking.
                # A script that cannot be safety-checked must not be voice-ready.
                logger.error("Post-repair Python preflight recheck failed: %s", exc)
                _post_repair_preflight_blocking = True
                status = "needs_human_review"
                gate_summary["python_preflight"] = {
                    "passed":    False,
                    "blocking":  True,
                    "high":      0,
                    "medium":    0,
                    "low":       0,
                    "report":    "python_preflight_report_after_repair.json",
                    "rechecked": True,
                    "error":     str(exc),
                }
                warnings.append(
                    f"Post-repair Python preflight recheck raised an exception: {exc}. "
                    "Treated as blocking — OpenAI Final Premium Gate skipped. "
                    "Manual review required."
                )

        _openai_gates_active = (
            settings.openai_review_enabled
            and settings.quality_mode == "premium_final"
            and settings.openai_review_policy != "disabled"
            and not _post_repair_preflight_blocking  # skip OFP if still blocking
        )

        if _openai_gates_active:
            if not settings.openai_api_key:
                logger.warning(
                    "OpenAI gates enabled but OPENAI_API_KEY is missing — "
                    "skipping OpenAI gates and setting status=needs_human_review"
                )
                warnings.append(
                    "OpenAI review enabled but OPENAI_API_KEY is missing. "
                    "safe_to_voice=False — configure OPENAI_API_KEY and rerun. "
                    "Troubleshooting: add OPENAI_API_KEY=sk-... to your .env file."
                )
                gate_summary["openai_final_premium"] = {
                    "passed": False, "skipped": False,
                    "reason": "OPENAI_API_KEY missing",
                }
                gate_summary["openai_premium_hindi_editor"] = {
                    "passed": True, "skipped": True,
                    "reason": "OPENAI_API_KEY missing — skipped",
                }
                gate_summary["openai_originality_youtube_risk"] = {
                    "passed": True, "skipped": True,
                    "reason": "OPENAI_API_KEY missing — skipped",
                }
                status = "needs_human_review"
            else:
                # ── Stage 14a: OpenAI Final Premium Gate (combined) ───────────

                # Task 5: Reload all gate reports from disk before calling the
                # Final Premium Gate so it receives the latest repaired versions,
                # not stale in-memory copies from before repair stages ran.
                (lint_report, copyedit_report, quality_report, retention_report,
                 similarity_report, originality_report, dialogue_report,
                 metadata_report) = _reload_latest_gate_reports(
                    review_dir,
                    lint_report, copyedit_report, quality_report,
                    retention_report, similarity_report, originality_report,
                    dialogue_report, metadata_report,
                )

                call_tracker.stage_start("openai_final_premium")
                logger.info("Stage 14a — OpenAI Final Premium Gate (combined)")
                existing_ofp = _try_load_existing_json(
                    review_dir / "openai_final_premium_report.json",
                    "openai_final_premium",
                    episode_dir=episode_dir, prompt_check="openai_final_premium",
                )
                if existing_ofp is not None:
                    ofp_report = existing_ofp
                else:
                    try:
                        ofp_report = run_openai_final_premium_gate(
                            script_draft=script_final,
                            fact_lock=fact_lock,
                            blueprint=blueprint,
                            hinglish_level=inp.hinglish_level,
                            lint_report=lint_report,
                            copyedit_report=copyedit_report,
                            quality_report=quality_report,
                            retention_report=retention_report,
                            similarity_report=similarity_report,
                            originality_report=originality_report,
                            dialogue_report=dialogue_report,
                            metadata_report=metadata_report,
                            review_dir=review_dir,
                        )
                    except Exception as exc:
                        logger.error("OpenAI Final Premium Gate failed: %s", exc)
                        warnings.append(
                            f"OpenAI Final Premium Gate failed: {exc}. "
                            "safe_to_voice=False — manual review required."
                        )
                        ofp_report = {
                            "approved": False, "safe_to_voice": False,
                            "error": str(exc), "overall_score": 0,
                            "chunk_repair_targets": [],
                        }

                call_tracker.stage_end("openai_final_premium")

                # Validate schema — BLOCKING: a malformed report must not be trusted
                try:
                    _validate_openai_final_premium_report(ofp_report, review_dir)
                except ValueError as exc:
                    logger.error(
                        "OpenAI Final Premium Report schema validation FAILED (blocking): %s", exc
                    )
                    warnings.append(
                        "OpenAI Final Premium Report schema mismatch — report structure invalid. "
                        "See 04-review/_openai_final_premium_validation_error.txt. "
                        "safe_to_voice=False, status=needs_human_review. "
                        "Re-run with REUSE_EXISTING_STAGE_OUTPUTS=false to regenerate."
                    )
                    # Force the report into a safe failure state so gate_passed=False below
                    ofp_report["approved"] = False
                    ofp_report["safe_to_voice"] = False
                    status = "needs_human_review"

                ofp_gate_passed = (
                    ofp_report.get("approved", False)
                    and ofp_report.get("safe_to_voice", False)
                )
                gate_summary["openai_final_premium"] = {
                    "passed":                    ofp_gate_passed,
                    "approved":                  ofp_report.get("approved", False),
                    "safe_to_voice":             ofp_report.get("safe_to_voice", False),
                    "overall_score":             ofp_report.get("overall_score", 0),
                    "hindi_quality_score":       ofp_report.get("hindi_quality_score", 0),
                    "retention_score":           ofp_report.get("retention_score", 0),
                    "originality_score":         ofp_report.get("originality_score", 0),
                    "youtube_safety_score":      ofp_report.get("youtube_safety_score", 0),
                    "metadata_score":            ofp_report.get("metadata_score", 0),
                    "recreated_dialogue_score":  ofp_report.get("recreated_dialogue_score", 10),
                    "high_severity_issues":      sum(
                        1 for i in ofp_report.get("issues", [])
                        if isinstance(i, dict) and i.get("severity") == "high"
                    ),
                    "chunk_repair_targets_count": len(ofp_report.get("chunk_repair_targets", [])),
                    "recommendation":            ofp_report.get("recommendation", "needs_human_review"),
                }

                if not ofp_gate_passed:
                    status = "needs_human_review"
                    high_n = gate_summary["openai_final_premium"]["high_severity_issues"]
                    warnings.append(
                        f"OpenAI Final Premium Gate FAILED (overall_score="
                        f"{ofp_report.get('overall_score', 0)}, high_issues={high_n}, "
                        f"recommendation={ofp_report.get('recommendation', '?')}). "
                        "See 04-review/openai_final_premium_report.json."
                    )

                if settings.openai_review_policy == "adaptive":
                    # Adaptive: Final Premium Gate is the only OpenAI gate
                    # If it fails with chunk_repair_targets → repair once then recheck
                    # If it still fails → needs_human_review
                    gate_summary["openai_premium_hindi_editor"] = {
                        "passed": True, "skipped": True,
                        "reason": "OPENAI_REVIEW_POLICY=adaptive — Final Premium Gate used instead",
                    }
                    gate_summary["openai_originality_youtube_risk"] = {
                        "passed": True, "skipped": True,
                        "reason": "OPENAI_REVIEW_POLICY=adaptive — Final Premium Gate used instead",
                    }

                    # ── Stage 16: OpenAI Targeted Chunk Repair (adaptive) ─────
                    if not ofp_gate_passed and settings.openai_repair_enabled:
                        ofp_targets = ofp_report.get("chunk_repair_targets", [])
                        if ofp_targets:
                            if len(ofp_targets) > settings.openai_repair_max_chunks:
                                logger.warning(
                                    "Stage 16 — Too many OpenAI repair targets (%d > %d). "
                                    "Skipping repair, setting needs_human_review.",
                                    len(ofp_targets), settings.openai_repair_max_chunks,
                                )
                                warnings.append(
                                    f"Too many OpenAI Final Premium repair targets "
                                    f"({len(ofp_targets)} > {settings.openai_repair_max_chunks}). "
                                    "Manual review required — do not run ElevenLabs."
                                )
                                openai_repair_has_failures = True
                                status = "needs_human_review"
                            else:
                                logger.info(
                                    "Stage 16 — OpenAI Targeted Chunk Repair (%d target(s))",
                                    len(ofp_targets),
                                )
                                try:
                                    script_final, oai_repair_report = (
                                        run_openai_targeted_chunk_repair(
                                            script_draft=script_final,
                                            repair_targets=ofp_targets,
                                            fact_lock=fact_lock,
                                            blueprint=blueprint,
                                            hinglish_level=inp.hinglish_level,
                                            script_dir=script_dir,
                                            review_dir=review_dir,
                                        )
                                    )
                                    openai_repair_has_failures = oai_repair_report.get(
                                        "has_failures", False
                                    )
                                    if openai_repair_has_failures:
                                        failed_n = oai_repair_report.get("chunks_failed", 0)
                                        warnings.append(
                                            f"OpenAI targeted repair: {failed_n} chunk(s) failed — "
                                            "original content kept. Manual review required."
                                        )

                                    # ── Stage 16b: Python preflight (pre-recheck guard) ──
                                    # Run cheap deterministic check BEFORE the expensive
                                    # OpenAI recheck. If OAI repair introduced new safety
                                    # issues, block here and skip the OFP recheck entirely.
                                    logger.info(
                                        "Stage 16b — Python Preflight recheck "
                                        "(pre-OFP-recheck safety guard)"
                                    )
                                    _post_oai_pf_blocking = False
                                    try:
                                        _post_oai_pf = run_python_preflight(
                                            script_draft=script_final,
                                            fact_lock=fact_lock,
                                            case_glossary=case_glossary,
                                            review_dir=review_dir,
                                            target_duration_min=inp.target_duration_min,
                                            hinglish_level=inp.hinglish_level,
                                            label="_after_openai_repair",
                                        )
                                        _post_oai_pf_blocking = _post_oai_pf.get(
                                            "blocking", False
                                        )
                                        if _post_oai_pf_blocking:
                                            _po_counts = _post_oai_pf.get("severity_counts", {})
                                            status = "needs_human_review"
                                            gate_summary["python_preflight"].update({
                                                "passed":    False,
                                                "blocking":  True,
                                                "high":      _po_counts.get("high", 0),
                                                "medium":    _po_counts.get("medium", 0),
                                                "low":       _po_counts.get("low", 0),
                                                "report":    "python_preflight_report_after_openai_repair.json",
                                                "rechecked": True,
                                            })
                                            warnings.append(
                                                "Post-OpenAI-repair Python preflight is BLOCKING. "
                                                "safe_to_voice=False. OFP recheck skipped. "
                                                "See 04-review/python_preflight_report_after_openai_repair.json."
                                            )
                                            logger.warning(
                                                "Stage 16b — Python preflight BLOCKING after OAI "
                                                "repair (high=%d, medium=%d). "
                                                "Skipping Stage 16a OFP recheck.",
                                                _po_counts.get("high", 0),
                                                _po_counts.get("medium", 0),
                                            )
                                    except Exception as exc:
                                        # Exception means the safety gate could not run.
                                        # Treat as blocking — must not call OFP recheck.
                                        logger.error(
                                            "Python preflight after OAI repair failed: %s", exc
                                        )
                                        _post_oai_pf_blocking = True
                                        status = "needs_human_review"
                                        gate_summary["python_preflight"].update({
                                            "passed":    False,
                                            "blocking":  True,
                                            "report":    "python_preflight_report_after_openai_repair.json",
                                            "rechecked": True,
                                            "error":     str(exc),
                                        })
                                        warnings.append(
                                            f"Python preflight after OpenAI repair raised an exception: {exc}. "
                                            "Treated as blocking — OFP recheck skipped. "
                                            "Manual review required."
                                        )

                                    if not _post_oai_pf_blocking:
                                        # ── Task 6: Refresh deterministic checks ──
                                        # Run only when preflight is clean — avoids
                                        # burning expensive OFP recheck on a bad script.
                                        logger.info(
                                            "Stage 16 post-repair — "
                                            "Refreshing Hindi lint + text similarity"
                                        )
                                        lint_report = run_hindi_text_lint(
                                            script_final,
                                            hinglish_level=inp.hinglish_level,
                                        )
                                        (review_dir / "hindi_text_lint_report.json").write_text(
                                            json.dumps(
                                                lint_report, ensure_ascii=False, indent=2
                                            ),
                                            encoding="utf-8",
                                        )
                                        _sim_transcript = (
                                            episode_dir / "01-input" / "clean_transcript.txt"
                                        ).read_text(encoding="utf-8")
                                        similarity_report = run_text_similarity_check(
                                            source_transcript=_sim_transcript,
                                            script_draft=script_final,
                                        )
                                        (review_dir / "text_similarity_report.json").write_text(
                                            json.dumps(
                                                similarity_report, ensure_ascii=False, indent=2
                                            ),
                                            encoding="utf-8",
                                        )
                                        logger.info(
                                            "Post-repair lint: %d issues (high=%d) | "
                                            "similarity risk=%s",
                                            lint_report.get("total_issues", 0),
                                            lint_report.get("high_issues", 0),
                                            similarity_report.get("risk_level", "?"),
                                        )

                                        # Task 5: Reload all latest gate reports
                                        # (captures copyedit, metadata, etc. repairs too)
                                        (lint_report, copyedit_report, quality_report,
                                         retention_report, similarity_report,
                                         originality_report, dialogue_report,
                                         metadata_report) = _reload_latest_gate_reports(
                                            review_dir,
                                            lint_report, copyedit_report, quality_report,
                                            retention_report, similarity_report,
                                            originality_report, dialogue_report,
                                            metadata_report,
                                        )

                                        # ── Stage 16a: Recheck (post-repair) ──────
                                        # Only runs when Stage 16b Python preflight is clean.
                                        # Saved to a separate file so the first-pass
                                        # report is preserved as a reference.
                                        logger.info(
                                            "Stage 16a — Recheck OpenAI Final Premium Gate "
                                            "(post-repair)"
                                        )
                                        try:
                                            ofp_recheck = run_openai_final_premium_gate(
                                                script_draft=script_final,
                                                fact_lock=fact_lock,
                                                blueprint=blueprint,
                                                hinglish_level=inp.hinglish_level,
                                                lint_report=lint_report,
                                                copyedit_report=copyedit_report,
                                                quality_report=quality_report,
                                                retention_report=retention_report,
                                                similarity_report=similarity_report,
                                                originality_report=originality_report,
                                                dialogue_report=dialogue_report,
                                                metadata_report=metadata_report,
                                                review_dir=review_dir,
                                                label="_after_repair",
                                            )
                                            ofp_report = ofp_recheck
                                            ofp_now_passed = (
                                                ofp_recheck.get("approved", False)
                                                and ofp_recheck.get("safe_to_voice", False)
                                            )
                                            gate_summary["openai_final_premium"].update({
                                                "passed":        ofp_now_passed,
                                                "approved":      ofp_recheck.get("approved", False),
                                                "safe_to_voice": ofp_recheck.get("safe_to_voice", False),
                                                "overall_score": ofp_recheck.get("overall_score", 0),
                                                "recheck":       True,
                                                "recheck_report": "openai_final_premium_report_after_repair.json",
                                            })
                                            if ofp_now_passed:
                                                logger.info(
                                                    "OpenAI Final Premium Gate recheck: PASSED"
                                                )
                                            else:
                                                status = "needs_human_review"
                                                logger.warning(
                                                    "OpenAI Final Premium Gate recheck: still FAILED"
                                                )
                                                warnings.append(
                                                    f"OpenAI Final Premium Gate recheck FAILED "
                                                    f"(overall_score="
                                                    f"{ofp_recheck.get('overall_score', 0)}). "
                                                    "Manual review required. "
                                                    "See 04-review/"
                                                    "openai_final_premium_report_after_repair.json."
                                                )
                                        except Exception as exc:
                                            logger.error(
                                                "OpenAI Final Premium Gate recheck failed: %s",
                                                exc,
                                            )
                                            warnings.append(
                                                f"OpenAI Final Premium Gate recheck failed: {exc}. "
                                                "Manual review required."
                                            )
                                            openai_repair_has_failures = True

                                except Exception as exc:
                                    logger.error("OpenAI targeted repair failed: %s", exc)
                                    warnings.append(
                                        f"OpenAI targeted chunk repair failed: {exc}. "
                                        "Manual review required — do not run ElevenLabs."
                                    )
                                    openai_repair_has_failures = True

                elif settings.openai_review_policy == "always":
                    # Always: run Final Premium Gate AND legacy Stage 14 + Stage 15
                    # All three must pass for safe_to_voice=True
                    logger.info(
                        "OPENAI_REVIEW_POLICY=always — also running legacy Stage 14 + Stage 15"
                    )

                    # Stage 14: OpenAI Premium Hindi Editor Gate
                    call_tracker.stage_start("openai_hindi_editor")
                    logger.info("Stage 14 — OpenAI Premium Hindi Editor Gate")
                    existing_ohe = _try_load_existing_json(
                        review_dir / "openai_premium_hindi_editor_report.json",
                        "openai_hindi_editor",
                        episode_dir=episode_dir,
                        prompt_check="openai_premium_hindi_editor",
                    )
                    if existing_ohe is not None:
                        ohe_report = existing_ohe
                    else:
                        try:
                            ohe_report = run_openai_premium_hindi_editor_gate(
                                script_draft=script_final,
                                fact_lock=fact_lock,
                                blueprint=blueprint,
                                hinglish_level=inp.hinglish_level,
                                lint_report=lint_report,
                                copyedit_report=copyedit_report,
                                quality_report=quality_report,
                                review_dir=review_dir,
                            )
                        except Exception as exc:
                            logger.error("OpenAI Hindi editor gate failed: %s", exc)
                            warnings.append(
                                f"OpenAI Premium Hindi Editor Gate failed: {exc}. "
                                "safe_to_voice=False — manual review required."
                            )
                            ohe_report = {
                                "approved": False, "safe_to_voice": False,
                                "error": str(exc), "overall_score": 0,
                            }

                    ohe_gate_passed = (
                        ohe_report.get("approved", False)
                        and ohe_report.get("safe_to_voice", False)
                    )
                    gate_summary["openai_premium_hindi_editor"] = {
                        "passed":               ohe_gate_passed,
                        "approved":             ohe_report.get("approved", False),
                        "safe_to_voice":        ohe_report.get("safe_to_voice", False),
                        "overall_score":        ohe_report.get("overall_score", 0),
                        "grammar_score":        ohe_report.get("grammar_score", 0),
                        "matra_score":          ohe_report.get("matra_nasalization_score", 0),
                        "high_severity_issues": sum(
                            1 for i in ohe_report.get("issues", [])
                            if i.get("severity") == "high"
                        ),
                        "chunk_repair_targets_count": len(ohe_report.get("chunk_repair_targets", [])),
                    }
                    call_tracker.stage_end("openai_hindi_editor")
                    if not ohe_gate_passed:
                        status = "needs_human_review"
                        high_n = gate_summary["openai_premium_hindi_editor"]["high_severity_issues"]
                        warnings.append(
                            f"OpenAI Hindi editor gate FAILED (overall_score="
                            f"{ohe_report.get('overall_score', 0)}, high_issues={high_n}). "
                            "See 04-review/openai_premium_hindi_editor_report.json."
                        )

                    # Stage 15: OpenAI Originality & YouTube Risk Gate
                    call_tracker.stage_start("openai_originality_risk")
                    logger.info("Stage 15 — OpenAI Originality / YouTube Risk Gate")
                    existing_oyr = _try_load_existing_json(
                        review_dir / "openai_originality_youtube_risk_report.json",
                        "openai_originality_risk",
                        episode_dir=episode_dir,
                        prompt_check="openai_originality_youtube_risk",
                    )
                    if existing_oyr is not None:
                        oyr_report = existing_oyr
                    else:
                        try:
                            clean_transcript_for_oyr = (
                                episode_dir / "01-input" / "clean_transcript.txt"
                            ).read_text(encoding="utf-8")
                            oyr_report = run_openai_originality_youtube_risk_gate(
                                script_draft=script_final,
                                source_transcript=clean_transcript_for_oyr,
                                fact_lock=fact_lock,
                                blueprint=blueprint,
                                claude_originality_report=originality_report,
                                claude_metadata_report=metadata_report,
                                claude_dialogue_report=dialogue_report,
                                review_dir=review_dir,
                            )
                        except Exception as exc:
                            logger.error("OpenAI originality/risk gate failed: %s", exc)
                            warnings.append(
                                f"OpenAI Originality/YouTube Risk Gate failed: {exc}. "
                                "safe_to_voice=False — manual review required."
                            )
                            oyr_report = {
                                "approved": False, "safe_to_voice": False,
                                "error": str(exc),
                            }

                    oyr_gate_passed = (
                        oyr_report.get("approved", False)
                        and oyr_report.get("safe_to_voice", False)
                    )
                    gate_summary["openai_originality_youtube_risk"] = {
                        "passed":               oyr_gate_passed,
                        "approved":             oyr_report.get("approved", False),
                        "safe_to_voice":        oyr_report.get("safe_to_voice", False),
                        "copying_risk":         oyr_report.get("copying_risk_score", "?"),
                        "transformative_value": oyr_report.get("transformative_value_score", "?"),
                        "youtube_ad_safety":    oyr_report.get("youtube_ad_safety_score", "?"),
                        "metadata_safety":      oyr_report.get("metadata_safety_score", "?"),
                        "high_severity_issues": sum(
                            1 for i in oyr_report.get("issues", [])
                            if i.get("severity") == "high"
                        ),
                        "required_fixes_count": len(oyr_report.get("required_fixes", [])),
                    }
                    call_tracker.stage_end("openai_originality_risk")
                    if not oyr_gate_passed:
                        status = "needs_human_review"
                        fixes = oyr_report.get("required_fixes", [])
                        warnings.append(
                            "OpenAI originality/YT risk gate FAILED. Required fixes: "
                            + ("; ".join(fixes[:3]) if fixes else "see report")
                            + (f" (+{len(fixes)-3} more)" if len(fixes) > 3 else "")
                            + " — See 04-review/openai_originality_youtube_risk_report.json."
                        )

                    # Stage 16: OpenAI Targeted Chunk Repair (always mode)
                    if settings.openai_repair_enabled:
                        ohe_targets = (
                            [] if ohe_gate_passed
                            else ohe_report.get("chunk_repair_targets", [])
                        )
                        oyr_targets = (
                            [] if oyr_gate_passed
                            else oyr_report.get("chunk_repair_targets", [])
                        )
                        ofp_targets_always = (
                            [] if ofp_gate_passed
                            else ofp_report.get("chunk_repair_targets", [])
                        )

                        # Merge all targets from all 3 gates
                        combined_targets: dict[str, dict] = {}
                        for t in ohe_targets + oyr_targets + ofp_targets_always:
                            cid = t.get("chunk_id", "")
                            if not cid:
                                continue
                            if cid in combined_targets:
                                combined_targets[cid]["repair_instruction"] = (
                                    combined_targets[cid].get("repair_instruction", "")
                                    + " | "
                                    + t.get("repair_instruction", "")
                                )
                            else:
                                combined_targets[cid] = dict(t)
                        all_oai_targets = list(combined_targets.values())

                        if all_oai_targets:
                            if len(all_oai_targets) > settings.openai_repair_max_chunks:
                                logger.warning(
                                    "Stage 16 — Too many OpenAI repair targets (%d > %d). "
                                    "Skipping repair, setting needs_human_review.",
                                    len(all_oai_targets), settings.openai_repair_max_chunks,
                                )
                                warnings.append(
                                    f"Too many OpenAI repair targets "
                                    f"({len(all_oai_targets)} > {settings.openai_repair_max_chunks}). "
                                    "Manual review required — do not run ElevenLabs."
                                )
                                openai_repair_has_failures = True
                                status = "needs_human_review"
                            else:
                                logger.info(
                                    "Stage 16 — OpenAI Targeted Chunk Repair (%d target(s))",
                                    len(all_oai_targets),
                                )
                                try:
                                    script_final, oai_repair_report = (
                                        run_openai_targeted_chunk_repair(
                                            script_draft=script_final,
                                            repair_targets=all_oai_targets,
                                            fact_lock=fact_lock,
                                            blueprint=blueprint,
                                            hinglish_level=inp.hinglish_level,
                                            script_dir=script_dir,
                                            review_dir=review_dir,
                                        )
                                    )
                                    openai_repair_has_failures = oai_repair_report.get(
                                        "has_failures", False
                                    )
                                    if openai_repair_has_failures:
                                        failed_n = oai_repair_report.get("chunks_failed", 0)
                                        warnings.append(
                                            f"OpenAI targeted repair: {failed_n} chunk(s) failed — "
                                            "original content kept. Manual review required."
                                        )

                                    # Recheck all failed gates after repair
                                    if not ohe_gate_passed:
                                        logger.info("Stage 16a — Recheck OpenAI Hindi Editor Gate")
                                        try:
                                            ohe_recheck = run_openai_premium_hindi_editor_gate(
                                                script_draft=script_final,
                                                fact_lock=fact_lock,
                                                blueprint=blueprint,
                                                hinglish_level=inp.hinglish_level,
                                                lint_report=lint_report,
                                                copyedit_report=copyedit_report,
                                                quality_report=quality_report,
                                                review_dir=review_dir,
                                            )
                                            ohe_now_passed = (
                                                ohe_recheck.get("approved", False)
                                                and ohe_recheck.get("safe_to_voice", False)
                                            )
                                            gate_summary["openai_premium_hindi_editor"].update({
                                                "passed":        ohe_now_passed,
                                                "overall_score": ohe_recheck.get("overall_score", 0),
                                                "recheck":       True,
                                            })
                                            if not ohe_now_passed:
                                                status = "needs_human_review"
                                                warnings.append(
                                                    f"OpenAI Hindi editor recheck FAILED "
                                                    f"(overall_score={ohe_recheck.get('overall_score', 0)}). "
                                                    "Manual review required."
                                                )
                                        except Exception as exc:
                                            logger.error(
                                                "OpenAI Hindi editor recheck failed: %s", exc
                                            )
                                            warnings.append(
                                                f"OpenAI Hindi editor recheck failed: {exc}."
                                            )
                                            openai_repair_has_failures = True

                                    if not oyr_gate_passed:
                                        logger.info(
                                            "Stage 16b — Recheck OpenAI Originality/YT Risk Gate"
                                        )
                                        try:
                                            clean_for_recheck = (
                                                episode_dir / "01-input" / "clean_transcript.txt"
                                            ).read_text(encoding="utf-8")
                                            oyr_recheck = run_openai_originality_youtube_risk_gate(
                                                script_draft=script_final,
                                                source_transcript=clean_for_recheck,
                                                fact_lock=fact_lock,
                                                blueprint=blueprint,
                                                claude_originality_report=originality_report,
                                                claude_metadata_report=metadata_report,
                                                claude_dialogue_report=dialogue_report,
                                                review_dir=review_dir,
                                            )
                                            oyr_now_passed = (
                                                oyr_recheck.get("approved", False)
                                                and oyr_recheck.get("safe_to_voice", False)
                                            )
                                            gate_summary["openai_originality_youtube_risk"].update({
                                                "passed":  oyr_now_passed,
                                                "recheck": True,
                                            })
                                            if not oyr_now_passed:
                                                status = "needs_human_review"
                                                fixes_r = oyr_recheck.get("required_fixes", [])
                                                warnings.append(
                                                    "OpenAI originality/YT risk recheck FAILED. "
                                                    "Required fixes: "
                                                    + ("; ".join(fixes_r[:3]) if fixes_r else "see report")
                                                    + (f" (+{len(fixes_r)-3} more)" if len(fixes_r) > 3 else "")
                                                )
                                        except Exception as exc:
                                            logger.error(
                                                "OpenAI originality/YT risk recheck failed: %s", exc
                                            )
                                            warnings.append(
                                                f"OpenAI originality/YT risk recheck failed: {exc}."
                                            )
                                            openai_repair_has_failures = True

                                    # ── Stage 16c: Python preflight after OAI repair (always mode) ──
                                    logger.info(
                                        "Stage 16c — Python Preflight recheck after OpenAI repair"
                                    )
                                    try:
                                        _post_oai_pf_a = run_python_preflight(
                                            script_draft=script_final,
                                            fact_lock=fact_lock,
                                            case_glossary=case_glossary,
                                            review_dir=review_dir,
                                            target_duration_min=inp.target_duration_min,
                                            hinglish_level=inp.hinglish_level,
                                            label="_after_openai_repair",
                                        )
                                        if _post_oai_pf_a.get("blocking", False):
                                            _poa_counts = _post_oai_pf_a.get("severity_counts", {})
                                            status = "needs_human_review"
                                            gate_summary["python_preflight"].update({
                                                "passed":    False,
                                                "blocking":  True,
                                                "high":      _poa_counts.get("high", 0),
                                                "medium":    _poa_counts.get("medium", 0),
                                                "low":       _poa_counts.get("low", 0),
                                                "report":    "python_preflight_report_after_openai_repair.json",
                                                "rechecked": True,
                                            })
                                            warnings.append(
                                                "Post-OpenAI-repair Python preflight is BLOCKING. "
                                                "safe_to_voice=False. "
                                                "See 04-review/python_preflight_report_after_openai_repair.json."
                                            )
                                            logger.warning(
                                                "Stage 16c — Python preflight BLOCKING after OAI repair "
                                                "(high=%d, medium=%d).",
                                                _poa_counts.get("high", 0),
                                                _poa_counts.get("medium", 0),
                                            )
                                    except Exception as exc:
                                        # Exception means the safety gate could not run.
                                        # Treat as blocking — a script that cannot be
                                        # safety-checked must not be voice-ready.
                                        logger.error(
                                            "Python preflight after OAI repair failed: %s", exc
                                        )
                                        status = "needs_human_review"
                                        gate_summary["python_preflight"].update({
                                            "passed":    False,
                                            "blocking":  True,
                                            "report":    "python_preflight_report_after_openai_repair.json",
                                            "rechecked": True,
                                            "error":     str(exc),
                                        })
                                        warnings.append(
                                            f"Python preflight after OpenAI repair raised an exception: {exc}. "
                                            "Treated as blocking — manual review required."
                                        )

                                except Exception as exc:
                                    logger.error("OpenAI targeted repair failed: %s", exc)
                                    warnings.append(
                                        f"OpenAI targeted chunk repair failed: {exc}. "
                                        "Manual review required — do not run ElevenLabs."
                                    )
                                    openai_repair_has_failures = True

        else:
            # OpenAI gates inactive — quality_mode, policy, or legacy flag disabled them
            _skip_reason = (
                f"quality_mode={settings.quality_mode}"
                if settings.quality_mode != "premium_final"
                else f"openai_review_policy={settings.openai_review_policy}"
                if settings.openai_review_policy == "disabled"
                else "OPENAI_REVIEW_ENABLED=false"
            )
            logger.info(
                "OpenAI gates skipped (%s) — forcing needs_human_review", _skip_reason
            )
            gate_summary["openai_final_premium"] = {
                "passed": False, "skipped": True,
                "reason": (
                    f"{_skip_reason} — Final Premium Gate did not run, "
                    "so safe_to_voice must remain false"
                ),
            }
            gate_summary["openai_premium_hindi_editor"] = {
                "passed": True, "skipped": True,
                "reason": f"{_skip_reason} — legacy gate skipped",
            }
            gate_summary["openai_originality_youtube_risk"] = {
                "passed": True, "skipped": True,
                "reason": f"{_skip_reason} — legacy gate skipped",
            }
            status = "needs_human_review"
            warnings.append(
                f"OpenAI Final Premium Gate skipped ({_skip_reason}). "
                "safe_to_voice=False — do not run ElevenLabs until the final premium gate passes."
            )

        # ── Final gate summary + safe_to_voice ────────────────────────────────
        # all_gates_passed checks every content gate entry currently in gate_summary.
        # repair_failures is added AFTER this check so it does not interfere.
        # safe_to_voice additionally requires no_repair_failures as a belt-and-suspenders check.
        #
        # python_preflight is evaluated via blocking (not passed) because passed=False
        # for any issue including low-only warnings, but low warnings must not block
        # safe_to_voice. See _gate_passed_for_safe_to_voice for the full decision table.
        all_gates_passed = all(
            _gate_passed_for_safe_to_voice(name, gate)
            for name, gate in gate_summary.items()
        )

        no_repair_failures = (
            (not repair_has_failures)
            and (not copyedit_repair_has_failures)
            and (not metadata_repair_has_failures)
            and (not retention_repair_has_failures)
            and (not openai_repair_has_failures)
        )

        # Record all repair failure state in gate_summary for auditability
        gate_summary["repair_failures"] = {
            "claude_script_repair_failed":  repair_has_failures,
            "copyedit_repair_failed":       copyedit_repair_has_failures,
            "metadata_repair_failed":       metadata_repair_has_failures,
            "retention_repair_failed":      retention_repair_has_failures,
            "openai_repair_failed":         openai_repair_has_failures,
            "passed":                       no_repair_failures,
        }

        _pf_gate_ok = not gate_summary.get("python_preflight", {}).get("blocking", True)
        safe_to_voice = (
            (status == "script_approved")
            and all_gates_passed
            and no_repair_failures
            and _pf_gate_ok  # python_preflight must not be blocking
        )

        # Stamp safe_to_voice into gate_summary for single-field API inspection
        gate_summary["safe_to_voice"] = safe_to_voice  # type: ignore[assignment]

        if copyedit_repair_has_failures and not any("Copyedit repair had failures" in w for w in warnings):
            warnings.append(
                "Copyedit repair had failures — original content kept in affected chunks. "
                "Do not run ElevenLabs. Manual review required. "
                "See 04-review/hindi_copyedit_repair_report.json."
            )
            status = "needs_human_review"

        logger.info(
            "All gates — script=%s copyedit=%s retention=%s originality=%s dialogue=%s "
            "metadata=%s oai_final=%s oai_hindi=%s oai_origin=%s repair_ok=%s all_passed=%s safe_to_voice=%s",
            gate_summary.get("script_quality", {}).get("passed", False),
            gate_summary.get("hindi_copyedit", {}).get("passed", False),
            gate_summary.get("retention_quality", {}).get("passed", False),
            gate_summary.get("originality_safety", {}).get("passed", False),
            gate_summary.get("recreated_dialogue", {}).get("passed", False),
            gate_summary.get("metadata_quality", {}).get("passed", False),
            gate_summary.get("openai_final_premium", {}).get("passed", "N/A"),
            gate_summary.get("openai_premium_hindi_editor", {}).get("passed", "N/A"),
            gate_summary.get("openai_originality_youtube_risk", {}).get("passed", "N/A"),
            no_repair_failures,
            all_gates_passed,
            safe_to_voice,
        )

        if safe_to_voice:
            logger.info(
                "ALL GATES PASSED (Claude + OpenAI) + ZERO REPAIR FAILURES — safe_to_voice=True"
            )
        else:
            if not all_gates_passed:
                warnings.append(
                    "One or more premium quality gates failed. "
                    "safe_to_voice=False — do NOT proceed to ElevenLabs until all gates pass. "
                    "See 04-review/*_gate_report.json and 04-review/hindi_copyedit_report.json."
                )
            elif not no_repair_failures:
                # Gates all passed but a repair layer had failures — warn explicitly
                failed_layers = []
                if repair_has_failures:
                    failed_layers.append("Claude script repair")
                if copyedit_repair_has_failures:
                    failed_layers.append("copyedit repair")
                if metadata_repair_has_failures:
                    failed_layers.append("metadata repair")
                if retention_repair_has_failures:
                    failed_layers.append("retention repair")
                if openai_repair_has_failures:
                    failed_layers.append("OpenAI targeted repair")
                warnings.append(
                    "All quality gates passed but repair failures prevent safe_to_voice=True "
                    f"({', '.join(failed_layers)}). "
                    "Fix the affected chunks manually before audio generation."
                )

    # ── Stage 14: Backward-compat copy to 02-package/ ────────────────────────
    # folder_name is always the actual episode directory name (e.g. "001-meika-jordan"),
    # not the slug that script agents may have stored. Set it BEFORE writing package files
    # so every downstream file that copies script_final carries the correct folder_name.
    canonical_folder_name = episode_dir.name
    script_final["folder_name"] = canonical_folder_name

    logger.info("Stage 14 — Writing backward-compat 02-package/ files")
    pkg_files = _write_backward_compat_package(
        episode_dir=episode_dir,
        script_final=script_final,
        episode_id=episode_id,
    )

    # Mirror script_writer raw response to 02-package/ for backward compat
    src_raw = script_dir / "_script_writer_raw_response.txt"
    if src_raw.exists():
        (episode_dir / "02-package" / "_claude_raw_response.txt").write_text(
            src_raw.read_text(encoding="utf-8"), encoding="utf-8"
        )

    _write_review_files(episode_dir, inp.cost_mode, script_final)

    # ── Save stage manifest (after all stages complete) ───────────────────────
    # Saved here so a mid-run crash leaves the previous manifest intact.
    # The next run will then see inputs_changed()=True and disable reuse.
    stage_manifest_service.save_manifest(
        episode_dir=episode_dir,
        raw_transcript=inp.raw_transcript,
        cost_mode=inp.cost_mode,
        hinglish_level=inp.hinglish_level,
        target_duration_min=inp.target_duration_min,
        prompts_dir=_PROMPTS_DIR,
    )

    # ── Collect all output files ───────────────────────────────────────────────
    all_files = _collect_pipeline_files(episode_dir)
    all_files.update(pkg_files)

    # ── Build quality summary ──────────────────────────────────────────────────
    q_summary = QualitySummary(
        approved=quality_report.get("approved", False),
        scores=quality_report.get("scores"),
        estimated_word_count=quality_report.get(
            "python_word_count",
            quality_report.get("estimated_word_count", 0),
        ),
        estimated_duration_min=quality_report.get(
            "python_duration_min",
            quality_report.get("estimated_duration_min", 0.0),
        ),
        repair_required=quality_report.get("repair_required", False),
    )

    telemetry = call_tracker.get_snapshot()

    logger.info(
        "PIPELINE COMPLETE — episode: %s  status: %s  duration: %.1f min  "
        "safe_to_voice: %s  scores: %s  model_calls: %s",
        episode_id, status, q_summary.estimated_duration_min,
        safe_to_voice,
        quality_report.get("scores", {}),
        telemetry.get("model_calls", {}),
    )

    return PackageResponse(
        episode_id=script_final.get("episode_id", episode_id),
        folder_name=canonical_folder_name,
        episode_dir=str(episode_dir),
        status=status,
        files=all_files,
        quality_summary=q_summary,
        gate_summary=gate_summary if gate_summary else None,
        safe_to_voice=safe_to_voice,
        warnings=warnings,
        telemetry=telemetry,
    )
