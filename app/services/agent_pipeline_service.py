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

import dataclasses
import hashlib
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
from app.services.claude_client import build_transcript_research_view, RateLimitExhaustedError
from app.services.prompt_budget_service import estimate_tokens, classify_transcript_size
from app.services.model_rate_limiter_service import rate_limiter as _model_rate_limiter
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
from app.services.originality_transformation_service import run_originality_transformation
from app.services import stage_manifest_service
from app.services.retention_blueprint_service import run_retention_blueprint
from app.services.retention_quality_gate_service import (
    run_retention_quality_gate,
    run_retention_repair,
)
from app.services import call_tracker
from app.services.call_tracker import BudgetExceededError
from app.services.report_normalization_service import (
    safe_join_report_items,
    stringify_report_list,
)
from app.services.repair_routing_service import run_repair_routing
from app.services.deterministic_auto_fix_service import run_deterministic_auto_fix
from app.services.premium_section_rebuild_service import run_premium_section_rebuild

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


def _compute_script_hash(script: dict) -> str:
    """Return a short SHA-256 hex digest of the script's narration chunks.

    Kept for backward-compatibility.  New code should use
    _compute_final_review_input_hash which covers the full OFP input set.
    """
    chunks = script.get("hindi_narration_chunks", [])
    payload = json.dumps(chunks, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclasses.dataclass
class ArtifactState:
    """Tracks which final-review artifacts have been mutated since gates last ran.

    Any stage that modifies script narration, youtube_metadata, or recreated_dialogues
    must call the appropriate mark_*_mutated() method so that downstream helpers
    (_finalize_reports_before_openai, Stage 14a) know to refresh gate reports.
    """

    script_mutated: bool = False
    metadata_mutated: bool = False
    dialogue_mutated: bool = False
    final_review_inputs_mutated: bool = False
    mutation_sources: list = dataclasses.field(default_factory=list)

    def mark_script_mutated(self, source: str) -> None:
        self.script_mutated = True
        self.final_review_inputs_mutated = True
        if source not in self.mutation_sources:
            self.mutation_sources.append(source)

    def mark_metadata_mutated(self, source: str) -> None:
        self.metadata_mutated = True
        self.final_review_inputs_mutated = True
        if source not in self.mutation_sources:
            self.mutation_sources.append(source)

    def mark_dialogue_mutated(self, source: str) -> None:
        self.dialogue_mutated = True
        self.final_review_inputs_mutated = True
        if source not in self.mutation_sources:
            self.mutation_sources.append(source)

    def to_dict(self) -> dict:
        return {
            "script_mutated":               self.script_mutated,
            "metadata_mutated":             self.metadata_mutated,
            "dialogue_mutated":             self.dialogue_mutated,
            "final_review_inputs_mutated":  self.final_review_inputs_mutated,
            "mutation_sources":             list(self.mutation_sources),
        }


def _compute_final_review_input_hash(
    *,
    script_final: dict,
    hinglish_level: int = 2,
    target_duration_min: int = 10,
    lint_report: dict | None = None,
    similarity_report: dict | None = None,
    copyedit_report: dict | None = None,
    quality_report: dict | None = None,
    retention_report: dict | None = None,
    originality_report: dict | None = None,
    dialogue_report: dict | None = None,
    metadata_report: dict | None = None,
    fact_lock: dict | None = None,
    blueprint: dict | None = None,
    retention_blueprint: dict | None = None,
    originality_transformation_plan: dict | None = None,
) -> str:
    """Return a 16-char hex digest that covers everything OpenAI Final Premium Gate reviews.

    The hash includes:
    - hindi_narration_chunks         (script text changed after repair/rebuild)
    - youtube_metadata               (metadata changed after metadata repair)
    - recreated_dialogues            (dialogue changed after targeted repair)
    - hinglish_level, target_duration_min
    - Compact identity hashes of: fact_lock, blueprint, retention_blueprint,
      originality_transformation_plan
    - Content hashes of each supporting gate report passed into OFP:
      lint, similarity, copyedit, quality, retention, originality, dialogue, metadata

    Saved as ``final_review_input_hash`` in openai_final_premium_report.json.
    OFP is reused only when this hash exactly matches the stored value.
    """

    def _report_sig(r: dict | None) -> str:
        """Return a short signature for a gate report using full canonical JSON.

        Uses the complete report dict (sorted keys) so that any content change —
        including required_fixes text, high_risk_matches, issue lists, etc. —
        produces a different hash and forces OFP to rerun.

        A missing or refresh_failed report returns the sentinel "none" which
        can never collide with a valid report hash.
        """
        if not r or r.get("refresh_failed"):
            return "none"
        return hashlib.sha256(
            json.dumps(r, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:8]

    def _plan_sig(p: dict | None) -> str:
        if not p:
            return "none"
        return hashlib.sha256(
            json.dumps(p, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]

    payload = {
        # Script content
        "narration_chunks":    script_final.get("hindi_narration_chunks", []),
        "youtube_metadata":    script_final.get("youtube_metadata", {}),
        "recreated_dialogues": script_final.get("recreated_dialogues", {}),
        # Run parameters
        "hinglish_level":      hinglish_level,
        "target_duration_min": target_duration_min,
        # Blueprint/plan identity (compact)
        "fact_lock_sig":       _plan_sig(fact_lock),
        "blueprint_sig":       _plan_sig(blueprint),
        "retention_bp_sig":    _plan_sig(retention_blueprint),
        "transform_plan_sig":  _plan_sig(originality_transformation_plan),
        # Supporting gate report signatures
        "lint_sig":            _report_sig(lint_report),
        "similarity_sig":      _report_sig(similarity_report),
        "copyedit_sig":        _report_sig(copyedit_report),
        "quality_sig":         _report_sig(quality_report),
        "retention_sig":       _report_sig(retention_report),
        "originality_sig":     _report_sig(originality_report),
        "dialogue_sig":        _report_sig(dialogue_report),
        "metadata_sig":        _report_sig(metadata_report),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _gate_passed_for_safe_to_voice(name: str, gate: dict) -> bool:
    """Return whether a gate entry should be treated as passing for safe_to_voice.

    Stale markers (refresh_failed, stale_after_mutation, stale_after_rebuild) always
    return False regardless of the passed field — stale evidence cannot approve a script.

    python_preflight uses the blocking field (not passed) as its blocking signal.
    passed=False for any issue (including low-only warnings), but blocking=False
    when only low issues exist. Low-only warnings must not block safe_to_voice —
    only medium/high issues (blocking=True) should.

    All other gates use their passed field directly.
    """
    # Stale evidence must never count as a passing gate
    if (gate.get("refresh_failed")
            or gate.get("stale_after_mutation")
            or gate.get("stale_after_rebuild")):
        return False
    if name == "python_preflight":
        return not gate.get("blocking", True)
    return gate.get("passed", False)


# Explicit allowlist of gates that must all pass for safe_to_voice=True.
#
# gate_summary also holds telemetry entries (repair_routing, auto_fix,
# pre_oai_repair, repair_telemetry, repair_failures, automation_status,
# safe_to_voice) that do NOT have a `passed` field and must NEVER affect
# all_gates_passed.  Iterating over gate_summary.items() directly would
# cause those entries to falsely block approval.
REQUIRED_SAFE_TO_VOICE_GATES: tuple[str, ...] = (
    "originality_transformation",
    "script_quality",
    "python_preflight",
    "hindi_copyedit",
    "originality_safety",
    "recreated_dialogue",
    "metadata_quality",
    "retention_quality",
    "openai_final_premium",
)


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


def _refresh_reports_after_script_mutation(
    *,
    script_final: dict,
    fact_lock: dict,
    blueprint: dict,
    review_dir: Path,
    gate_summary: dict,
    warnings: list[str],
    # Current in-memory report objects — returned unchanged on exception
    lint_report: dict,
    similarity_report: dict,
    quality_report: dict,
    copyedit_report: dict,
    # Required for similarity + originality gates
    source_transcript: str = "",
    # Optional inputs used only when those gates are requested
    hinglish_level: int = 2,
    case_glossary: dict | None = None,
    retention_report: dict | None = None,
    retention_blueprint: dict | None = None,
    originality_report: dict | None = None,
    dialogue_report: dict | None = None,
    metadata_report: dict | None = None,
    target_duration_min: int = 10,
    cost_mode: str = "standard",
    is_final_review: bool = True,
    # Feature flags — caller decides which expensive gates to rerun
    rerun_lint: bool = True,
    rerun_similarity: bool = True,
    rerun_quality: bool = False,
    rerun_copyedit: bool = False,
    rerun_retention: bool = False,
    rerun_originality: bool = False,
    rerun_dialogue: bool = False,
    rerun_metadata: bool = False,
) -> dict:
    """Refresh gate reports after any flow that mutates script_final or narration chunks.

    Must be called after every path that mutates script text or package content:
    - deterministic auto-fix that changed narration/metadata/recreated dialogue
    - premium_section_rebuild
    - pre-OAI Claude repair (Stage 13d)
    - OpenAI targeted repair that mutated narration or metadata
    - metadata repair
    - retention repair that changed narration

    Returns a dict with refreshed report objects:
    {
        "lint_report":        <dict>,
        "similarity_report":  <dict>,
        "quality_report":     <dict>,
        "copyedit_report":    <dict>,
        "retention_report":   <dict | None>,
        "originality_report": <dict | None>,
        "dialogue_report":    <dict | None>,
        "metadata_report":    <dict | None>,
    }

    Each refreshed report carries "refreshed_after_script_mutation": True.
    Reports that were not rerun are returned as-is from the caller.

    When a required input is missing and the gate cannot run, a structured
    failure marker is returned:
    {
        "passed":         False,
        "stale":          True,
        "refresh_failed": True,
        "reason":         "missing required input: <name>",
    }

    On any other exception the original in-memory dict is kept and a warning
    is appended — the report is NOT silently promoted as fresh.
    """
    _MARKER = "refreshed_after_script_mutation"
    _STALE_FAILURE = lambda reason: {  # noqa: E731
        "passed": False,
        "stale": True,
        "refresh_failed": True,
        "reason": reason,
    }

    result: dict[str, dict | None] = {
        "lint_report":        lint_report,
        "similarity_report":  similarity_report,
        "quality_report":     quality_report,
        "copyedit_report":    copyedit_report,
        "retention_report":   retention_report,
        "originality_report": originality_report,
        "dialogue_report":    dialogue_report,
        "metadata_report":    metadata_report,
    }

    # ── Lint (Python-only, no API call) ───────────────────────────────────────
    if rerun_lint:
        try:
            fresh_lint = run_hindi_text_lint(
                script_draft=script_final,
                hinglish_level=hinglish_level,
            )
            fresh_lint[_MARKER] = True
            (review_dir / "hindi_text_lint_report.json").write_text(
                json.dumps(fresh_lint, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result["lint_report"] = fresh_lint
        except Exception as _exc:
            warnings.append(f"hindi_text_lint refresh failed after script mutation: {_exc}")
            result["lint_report"] = _STALE_FAILURE(
                f"hindi_text_lint refresh exception: {_exc}"
            )

    # ── Text similarity (Python-only, needs source_transcript) ───────────────
    if rerun_similarity:
        if not source_transcript:
            _fail = _STALE_FAILURE("missing required input: source_transcript")
            result["similarity_report"] = _fail
            warnings.append(
                "text_similarity refresh skipped after script mutation: "
                "source_transcript not supplied — similarity_report is stale."
            )
        else:
            try:
                fresh_sim = run_text_similarity_check(
                    source_transcript=source_transcript,
                    script_draft=script_final,
                )
                fresh_sim[_MARKER] = True
                (review_dir / "text_similarity_report.json").write_text(
                    json.dumps(fresh_sim, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                result["similarity_report"] = fresh_sim
            except Exception as _exc:
                warnings.append(f"text_similarity refresh failed after script mutation: {_exc}")
                result["similarity_report"] = _STALE_FAILURE(
                    f"text_similarity refresh exception: {_exc}"
                )

    # ── Script quality (expensive Claude call, opt-in) ────────────────────────
    if rerun_quality:
        try:
            fresh_quality = run_script_review(
                target_duration_min=target_duration_min,
                cost_mode=cost_mode,
                fact_lock=fact_lock,
                blueprint=blueprint,
                script_draft=script_final,
                review_dir=review_dir,
                is_final_review=is_final_review,
                hinglish_level=hinglish_level,
                case_glossary=case_glossary,
            )
            fresh_quality[_MARKER] = True
            (review_dir / "final_script_quality_report.json").write_text(
                json.dumps(fresh_quality, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result["quality_report"] = fresh_quality
            gate_summary["script_quality"] = {
                "passed": fresh_quality.get("approved", False),
                "scores": fresh_quality.get("scores", {}),
                _MARKER:  True,
            }
        except Exception as _exc:
            warnings.append(f"script_quality refresh failed after script mutation: {_exc}")
            result["quality_report"] = _STALE_FAILURE(
                f"script_quality refresh exception: {_exc}"
            )

    # ── Hindi copyedit (expensive Claude call, opt-in) ────────────────────────
    if rerun_copyedit:
        try:
            fresh_copy = run_hindi_copyedit_gate(
                script_draft=script_final,
                fact_lock=fact_lock,
                blueprint=blueprint,
                hinglish_level=hinglish_level,
                lint_report=result.get("lint_report") or lint_report,
                review_dir=review_dir,
            )
            fresh_copy[_MARKER] = True
            (review_dir / "hindi_copyedit_report.json").write_text(
                json.dumps(fresh_copy, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result["copyedit_report"] = fresh_copy
            gate_summary["hindi_copyedit"] = {
                "passed": fresh_copy.get("approved", False),
                _MARKER:  True,
            }
        except Exception as _exc:
            warnings.append(f"hindi_copyedit refresh failed after script mutation: {_exc}")
            result["copyedit_report"] = _STALE_FAILURE(
                f"hindi_copyedit refresh exception: {_exc}"
            )

    # ── Retention quality (needs retention_blueprint + target_duration_min) ───
    if rerun_retention:
        if retention_blueprint is None:
            _fail = _STALE_FAILURE("missing required input: retention_blueprint")
            result["retention_report"] = _fail
            warnings.append(
                "retention_quality refresh skipped after script mutation: "
                "retention_blueprint not supplied — retention_report is stale."
            )
        else:
            try:
                fresh_ret = run_retention_quality_gate(
                    script_draft=script_final,
                    retention_blueprint=retention_blueprint,
                    blueprint=blueprint,
                    target_duration_min=target_duration_min,
                    review_dir=review_dir,
                )
                fresh_ret[_MARKER] = True
                (review_dir / "retention_quality_report.json").write_text(
                    json.dumps(fresh_ret, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                result["retention_report"] = fresh_ret
                gate_summary["retention_quality"] = {
                    "passed": fresh_ret.get("approved", False),
                    _MARKER:  True,
                }
            except Exception as _exc:
                warnings.append(
                    f"retention_quality refresh failed after script mutation: {_exc}"
                )
                result["retention_report"] = _STALE_FAILURE(
                    f"retention_quality refresh exception: {_exc}"
                )

    # ── Originality safety (needs source_transcript + similarity_report) ──────
    if rerun_originality:
        _curr_sim = result.get("similarity_report") or similarity_report
        if not source_transcript:
            _fail = _STALE_FAILURE("missing required input: source_transcript")
            result["originality_report"] = _fail
            warnings.append(
                "originality_safety refresh skipped after script mutation: "
                "source_transcript not supplied — originality_report is stale."
            )
        elif _curr_sim and _curr_sim.get("refresh_failed"):
            _fail = _STALE_FAILURE(
                "dependency refresh_failed: similarity_report is stale — "
                "originality_safety requires a fresh similarity_report"
            )
            result["originality_report"] = _fail
            warnings.append(
                "originality_safety refresh skipped: similarity_report refresh failed — "
                "originality_report is stale."
            )
        else:
            try:
                fresh_orig = run_originality_safety_gate(
                    script_draft=script_final,
                    source_transcript=source_transcript,
                    similarity_report=_curr_sim or {},
                    review_dir=review_dir,
                )
                fresh_orig[_MARKER] = True
                (review_dir / "originality_safety_gate_report.json").write_text(
                    json.dumps(fresh_orig, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                result["originality_report"] = fresh_orig
                gate_summary["originality_safety"] = {
                    "passed": fresh_orig.get("approved", fresh_orig.get("gate_passed", False)),
                    _MARKER:  True,
                }
            except Exception as _exc:
                warnings.append(
                    f"originality_safety refresh failed after script mutation: {_exc}"
                )
                result["originality_report"] = _STALE_FAILURE(
                    f"originality_safety refresh exception: {_exc}"
                )

    # ── Recreated dialogue (needs fact_lock only) ─────────────────────────────
    if rerun_dialogue:
        try:
            fresh_dial = run_recreated_dialogue_quality_gate(
                script_draft=script_final,
                fact_lock=fact_lock,
                review_dir=review_dir,
            )
            fresh_dial[_MARKER] = True
            (review_dir / "recreated_dialogue_gate_report.json").write_text(
                json.dumps(fresh_dial, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result["dialogue_report"] = fresh_dial
            gate_summary["recreated_dialogue"] = {
                "passed": fresh_dial.get("approved", fresh_dial.get("gate_passed", False)),
                _MARKER:  True,
            }
        except Exception as _exc:
            warnings.append(
                f"recreated_dialogue refresh failed after script mutation: {_exc}"
            )
            result["dialogue_report"] = _STALE_FAILURE(
                f"recreated_dialogue refresh exception: {_exc}"
            )

    # ── Metadata quality (needs fact_lock only) ───────────────────────────────
    if rerun_metadata:
        try:
            fresh_meta = run_metadata_quality_gate(
                script_draft=script_final,
                fact_lock=fact_lock,
                review_dir=review_dir,
            )
            fresh_meta[_MARKER] = True
            (review_dir / "metadata_quality_gate_report.json").write_text(
                json.dumps(fresh_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result["metadata_report"] = fresh_meta
            gate_summary["metadata_quality"] = {
                "passed": fresh_meta.get("approved", fresh_meta.get("gate_passed", False)),
                _MARKER:  True,
            }
        except Exception as _exc:
            warnings.append(
                f"metadata_quality refresh failed after script mutation: {_exc}"
            )
            result["metadata_report"] = _STALE_FAILURE(
                f"metadata_quality refresh exception: {_exc}"
            )

    return result


def _finalize_reports_before_openai(
    *,
    artifact_state: "ArtifactState",
    script_final: dict,
    fact_lock: dict,
    blueprint: dict,
    review_dir: "Path",
    gate_summary: dict,
    warnings: list,
    lint_report: dict,
    similarity_report: dict,
    copyedit_report: dict,
    quality_report: dict,
    retention_report: dict | None = None,
    originality_report: dict | None = None,
    dialogue_report: dict | None = None,
    metadata_report: dict | None = None,
    source_transcript: str = "",
    hinglish_level: int = 2,
    target_duration_min: int = 10,
    retention_blueprint: dict | None = None,
    case_glossary: dict | None = None,
    cost_mode: str = "standard",
) -> dict:
    """Mandatory gate-report freshness check run directly before Stage 14a OFP.

    When artifact_state reports that any final-review artifact was mutated since
    the last gate run, this function determines which reports are now stale and
    attempts to refresh cheap gates (lint, similarity).  Expensive Claude gates
    (quality, copyedit, retention, originality, dialogue, metadata) are marked
    stale_after_mutation if they cannot be refreshed without extra API cost.

    Returns a dict with:
      refresh_ok         bool   — True if no required refresh failed
      refreshed_reports  dict   — name→report for successfully refreshed reports
      failed_refreshes   list   — names of reports that failed to refresh
      stale_reports      list   — names of reports marked stale (not refreshed)
      blocking           bool   — True when a required refresh failed (OFP must be skipped)
    """
    result: dict = {
        "refresh_ok":        True,
        "refreshed_reports": {},
        "failed_refreshes":  [],
        "stale_reports":     [],
        "blocking":          False,
    }

    if not artifact_state.final_review_inputs_mutated:
        # Nothing changed — reports are still fresh
        return result

    # Determine which reports may now be stale
    stale_candidates: list[str] = []
    if artifact_state.script_mutated or artifact_state.dialogue_mutated:
        stale_candidates.extend(["lint", "similarity", "originality"])
    if artifact_state.metadata_mutated:
        stale_candidates.append("metadata")
    if artifact_state.script_mutated:
        stale_candidates.extend(["retention", "dialogue"])

    # Always attempt a cheap refresh (lint + similarity) when script mutated
    rerun_lint = artifact_state.script_mutated or artifact_state.dialogue_mutated
    rerun_similarity = artifact_state.script_mutated or artifact_state.dialogue_mutated

    refresh_result = _refresh_reports_after_script_mutation(
        script_final=script_final,
        fact_lock=fact_lock,
        blueprint=blueprint,
        review_dir=review_dir,
        gate_summary=gate_summary,
        warnings=warnings,
        lint_report=lint_report,
        similarity_report=similarity_report,
        quality_report=quality_report,
        copyedit_report=copyedit_report,
        source_transcript=source_transcript,
        hinglish_level=hinglish_level,
        target_duration_min=target_duration_min,
        retention_report=retention_report,
        retention_blueprint=retention_blueprint,
        originality_report=originality_report,
        dialogue_report=dialogue_report,
        metadata_report=metadata_report,
        case_glossary=case_glossary,
        cost_mode=cost_mode,
        rerun_lint=rerun_lint,
        rerun_similarity=rerun_similarity,
        rerun_quality=False,   # expensive — not refreshed here
        rerun_copyedit=False,  # expensive — not refreshed here
        rerun_retention=False, # expensive — not refreshed here
        rerun_originality=False,
        rerun_dialogue=False,
        rerun_metadata=False,
    )

    # Collect refreshed reports
    for name, key in (
        ("lint",        "lint_report"),
        ("similarity",  "similarity_report"),
        ("originality", "originality_report"),
        ("metadata",    "metadata_report"),
        ("retention",   "retention_report"),
        ("dialogue",    "dialogue_report"),
    ):
        rpt = refresh_result.get(key)
        if rpt and rpt.get("refreshed_after_script_mutation"):
            result["refreshed_reports"][name] = rpt

    # Check for refresh failures — these block OFP
    failed: list[str] = []
    for name, key in (
        ("similarity",  "similarity_report"),
        ("originality", "originality_report"),
    ):
        rpt = refresh_result.get(key)
        if isinstance(rpt, dict) and rpt.get("refresh_failed"):
            failed.append(name)

    if failed:
        result["failed_refreshes"] = failed
        result["refresh_ok"]       = False
        result["blocking"]         = True
        warnings.append(
            f"Pre-OFP report refresh failed for: {failed}. "
            "OpenAI Final Premium Gate cannot run with stale evidence."
        )

    # Mark expensive stale gates in gate_summary
    expensive_stale = [c for c in stale_candidates if c not in result["refreshed_reports"] and c not in failed]
    for gate_name in expensive_stale:
        gs_key = {
            "retention":   "retention_quality",
            "originality": "originality_safety",
            "dialogue":    "recreated_dialogue",
            "metadata":    "metadata_quality",
        }.get(gate_name, gate_name)
        if gs_key in gate_summary:
            gate_summary[gs_key].setdefault("stale_after_mutation", True)
    result["stale_reports"] = expensive_stale

    return result


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


def _current_repair_counts() -> dict[str, int]:
    calls = call_tracker.get_snapshot().get("model_calls", {})
    return {
        "claude": int(calls.get("repair_claude", 0) or 0),
        "openai": int(calls.get("repair_openai", 0) or 0),
    }


def _soft_repair_budget_exceeded(kind: str) -> str | None:
    counts = _current_repair_counts()
    if kind == "claude" and counts["claude"] >= settings.failed_path_max_claude_repair_calls:
        return (
            f"FAILED_PATH_MAX_CLAUDE_REPAIR_CALLS="
            f"{settings.failed_path_max_claude_repair_calls} reached"
        )
    if kind == "openai" and counts["openai"] >= settings.failed_path_max_openai_repair_calls:
        return (
            f"FAILED_PATH_MAX_OPENAI_REPAIR_CALLS="
            f"{settings.failed_path_max_openai_repair_calls} reached"
        )
    return None


def _record_soft_budget_stop(
    kind: str,
    reason: str,
    warnings: list[str],
    gate_summary: dict,
    repair_cost_telemetry: dict[str, Any] | None = None,
) -> None:
    msg = (
        f"Soft {kind} repair budget exceeded ({reason}); stopping automated repair "
        "to control cost. safe_to_voice=false."
    )
    warnings.append(msg)
    gate_summary.setdefault("repair_budget", {})[f"{kind}_budget_exceeded"] = True
    gate_summary["repair_budget"][f"{kind}_budget_reason"] = reason
    if repair_cost_telemetry is not None:
        repair_cost_telemetry["repair_budget_exceeded"] = True
        repair_cost_telemetry["repair_budget_exceeded_kind"] = kind
        repair_cost_telemetry["repair_budget_exceeded_reason"] = reason


def _retention_score(report: dict) -> float:
    try:
        return float(report.get("overall_retention_score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _retention_failure_is_localized(report: dict) -> bool:
    """True when retention failure can still be safely repaired as targeted chunks."""
    targets = report.get("chunk_repair_targets", []) or []
    unique_chunks = {t.get("chunk_id", "") for t in targets if t.get("chunk_id")}
    if not unique_chunks:
        return False
    if len(unique_chunks) > min(settings.openai_repair_max_chunks, 4):
        return False
    # Very low overall retention usually means broad structure failure, not a
    # localized hook/transition issue worth spending OpenAI repair on.
    return _retention_score(report) >= 7.0


def _blocking_chunk_ids_from_reports(*reports: dict) -> set[str]:
    chunk_ids: set[str] = set()
    for report in reports:
        if not isinstance(report, dict):
            continue
        for key in ("chunk_repair_targets", "issues"):
            for item in report.get(key, []) or []:
                if isinstance(item, dict) and item.get("chunk_id"):
                    chunk_ids.add(str(item["chunk_id"]))
    return chunk_ids


def _filter_openai_targets_to_blockers(targets: list[dict], *reports: dict) -> list[dict]:
    blockers = _blocking_chunk_ids_from_reports(*reports)
    if not blockers:
        return targets
    return [t for t in targets if t.get("chunk_id") in blockers]


def _preflight_blocker_trace(
    report: dict,
    *,
    python_fix_attempted: bool,
    claude_repair_attempted: bool,
    openai_repair_attempted: bool,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for issue in (report.get("issues", []) or []) + (report.get("metadata_issues", []) or []):
        if issue.get("severity") not in {"critical", "high", "medium"}:
            continue
        location = (
            issue.get("chunk_id")
            or issue.get("field")
            or ("youtube_metadata" if issue in report.get("metadata_issues", []) else "unknown")
        )
        blockers.append({
            "issue_type": issue.get("type", "unknown"),
            "severity": issue.get("severity", "unknown"),
            "location": location,
            "problem": issue.get("problem", ""),
            "python_fix_attempted": python_fix_attempted,
            "claude_repair_attempted": claude_repair_attempted,
            "openai_repair_attempted": openai_repair_attempted,
            "recommended_next_action": (
                "Manual review or focused prompt/schema adjustment required before another premium run."
            ),
        })
    return blockers


# ─── Shared repair-routing + rebuild helper ───────────────────────────────────

def _run_routing_and_rebuild(
    script_draft: dict,
    gate_reports: dict[str, dict],
    fact_lock: dict,
    blueprint: dict,
    retention_blueprint: dict,
    originality_transformation_plan: dict,
    case_glossary: dict,
    hinglish_level: int,
    case_hint: str,
    review_dir: Path,
    script_dir: Path,
    warnings: list[str],
    gate_summary: dict,
    root_cause_repair_attempts: dict[str, int] | None = None,
) -> tuple[dict, dict, bool]:
    """
    Step 1: Run repair_routing_service to classify gate failures into root causes.
    Step 2: Run deterministic_auto_fix_service (zero AI cost).
    Step 3: Run premium_section_rebuild_service (one Claude call per root-cause group).

    Returns
    -------
    (updated_script_draft, routing_plan, rebuild_ran)

    ``rebuild_ran`` is True iff Claude section rebuild actually executed.
    The caller is responsible for re-running gate checks after this.
    """
    # ── Step 1: Repair routing ────────────────────────────────────────────────
    routing_plan: dict = {"route": "stop_not_voice_ready"}
    try:
        routing_plan = run_repair_routing(
            all_gate_reports=gate_reports,
            openai_repair_max_chunks=settings.openai_repair_max_chunks,
            review_dir=review_dir,
            script_draft=script_draft,
            max_cluster_size=settings.max_openai_cluster_repair_chunks,
            previous_root_cause_attempts=root_cause_repair_attempts or {},
        )
        _route = routing_plan.get("route", "stop_not_voice_ready")
        logger.info(
            "Repair routing: route=%s root_causes=%d python_fixes=%d claude_targets=%d",
            _route,
            len(routing_plan.get("root_causes", [])),
            len(routing_plan.get("python_fixes", [])),
            len(routing_plan.get("claude_repair_targets", [])),
        )
        gate_summary["repair_routing"] = {
            "route":               _route,
            "root_cause_count":    len(routing_plan.get("root_causes", [])),
            "python_fixes_count":  len(routing_plan.get("python_fixes", [])),
            "claude_targets_count": len(routing_plan.get("claude_repair_targets", [])),
            "reconstruction_cluster_count": routing_plan.get("reconstruction_cluster_count", 0),
            "source_shaped_reconstruction_detected": routing_plan.get(
                "source_shaped_reconstruction_detected", False
            ),
            "claude_repair_skipped_due_previous_failure": routing_plan.get(
                "claude_repair_skipped_due_previous_failure", False
            ),
            "estimated_model_calls_saved": routing_plan.get("estimated_model_calls_saved", 0),
            "repeat_root_cause_detected": routing_plan.get("repeat_root_cause_detected", []),
        }
    except Exception as _exc:
        logger.error("Repair routing failed: %s", _exc)
        warnings.append(f"Repair routing failed: {_exc}.")
        gate_summary["repair_routing"] = {"route": "stop_not_voice_ready", "error": str(_exc)}
        return script_draft, routing_plan, False

    if routing_plan.get("route") == "stop_not_voice_ready":
        warnings.append(
            "Repair routing: unrecoverable issues detected — manual review required."
        )
        return script_draft, routing_plan, False

    # ── Step 2: Deterministic auto-fix ────────────────────────────────────────
    _af_report: dict = {}
    try:
        script_draft, _af_report = run_deterministic_auto_fix(
            script_draft=script_draft,
            routing_plan=routing_plan,
            case_hint=case_hint,
            review_dir=review_dir,
        )
        _fix_n = _af_report.get("total_fixes_applied", 0)
        logger.info("Deterministic auto-fix: %d fix(es) applied", _fix_n)
        gate_summary.setdefault("auto_fix", {})["python_fixes_count"] = _fix_n
    except Exception as _exc:
        logger.error("Deterministic auto-fix failed: %s", _exc)
        warnings.append(f"Deterministic auto-fix failed: {_exc}.")

    # ── Step 2b: Route English-quote targets to Claude rebuild ─────────────────
    # deterministic_auto_fix_service flags long verbatim English/source quotes but
    # cannot safely replace complex dialogue with regex.  Convert those targets to
    # Claude rebuild targets so premium_section_rebuild_service gets them.
    _eq_targets = _af_report.get("english_quote_repair_targets", [])
    if _eq_targets:
        _existing_cids: set[str] = {
            cid
            for _t in routing_plan.get("claude_repair_targets", [])
            for cid in _t.get("chunk_ids", [])
        }
        _new_eq_added = 0
        for _eq in _eq_targets:
            _cid = _eq.get("chunk_id", "")
            if not _cid or _cid in _existing_cids:
                continue   # already queued or invalid
            routing_plan.setdefault("claude_repair_targets", []).append({
                "area":                   "hindi_quality",
                "root_cause_id":          f"eq_{_cid}",
                "repair_instruction":     _eq.get("repair_instruction", ""),
                "affected_targets":       [_eq],
                "chunk_ids":              [_cid],
                "preferred_repair_owner": "claude",
                "reason":                 (
                    "Verbatim English/source quote detected — requires Hindi "
                    "translation or paraphrase, not blind regex replacement"
                ),
                "issue_type":             "exact_english_quote_copy",
            })
            _existing_cids.add(_cid)
            _new_eq_added += 1
        if _new_eq_added:
            logger.info(
                "Repair routing: appended %d English-quote chunk(s) to Claude rebuild targets",
                _new_eq_added,
            )
            gate_summary.setdefault("auto_fix", {})["english_quote_targets_added"] = _new_eq_added

    # ── Step 3: Claude grouped rebuild ────────────────────────────────────────
    rebuild_ran = False
    if settings.auto_rebuild_enabled and routing_plan.get("claude_repair_targets"):
        _budget_reason = _soft_repair_budget_exceeded("claude")
        if _budget_reason:
            _record_soft_budget_stop("claude", _budget_reason, warnings, gate_summary)
            routing_plan.setdefault("notes", []).append(_budget_reason)
            routing_plan["route"] = "stop_not_voice_ready"
            return script_draft, routing_plan, False
        try:
            script_draft, _rebuild_report = run_premium_section_rebuild(
                script_draft=script_draft,
                routing_plan=routing_plan,
                fact_lock=fact_lock,
                blueprint=blueprint,
                retention_blueprint=retention_blueprint,
                originality_transformation_plan=originality_transformation_plan,
                case_glossary=case_glossary,
                hinglish_level=hinglish_level,
                review_dir=review_dir,
                script_dir=script_dir,
            )
            _rebuilt_n = _rebuild_report.get("rebuilt_count", 0)
            # rebuild_ran is True only when Claude actually rebuilt ≥1 chunk.
            # skipped=True means the service returned early (no targets found,
            # cap exceeded, etc.) — that does not count as a rebuild.
            rebuild_ran = (
                _rebuilt_n > 0
                and not _rebuild_report.get("skipped", False)
            )
            logger.info(
                "Premium section rebuild: %d chunk(s) rebuilt (rebuild_ran=%s)",
                _rebuilt_n, rebuild_ran,
            )
            gate_summary.setdefault("auto_fix", {}).update({
                "rebuild_ran":    rebuild_ran,
                "rebuild_chunks": _rebuilt_n,
            })
            if rebuild_ran and root_cause_repair_attempts is not None:
                for target in routing_plan.get("claude_repair_targets", []):
                    key = target.get("root_cause_key") or target.get("root_cause_id")
                    if key:
                        root_cause_repair_attempts[key] = root_cause_repair_attempts.get(key, 0) + 1
        except Exception as _exc:
            logger.error("Premium section rebuild failed: %s", _exc)
            warnings.append(f"Premium section rebuild failed: {_exc}.")

    return script_draft, routing_plan, rebuild_ran


# ─── Main pipeline entry point ────────────────────────────────────────────────

def run_agent_pipeline(inp: EpisodeInput) -> PackageResponse:
    """
    Public entry point. Wraps _run_agent_pipeline_inner with graceful
    RateLimitExhaustedError handling (status=rate_limited_retry_later).
    """
    try:
        return _run_agent_pipeline_inner(inp)
    except RateLimitExhaustedError as exc:
        slug   = _slugify(inp.case_hint)
        ep_id  = f"{inp.episode_number}-{slug}"
        ep_dir = settings.episodes_dir / ep_id
        logger.error(
            "PIPELINE ABORTED — RateLimitExhaustedError after %d retries "
            "for agent '%s' — episode: %s",
            exc.retry_count,
            exc.agent_name,
            ep_id,
        )
        return PackageResponse(
            episode_id=ep_id,
            folder_name=ep_id,
            episode_dir=str(ep_dir),
            status="rate_limited_retry_later",
            files={},
            quality_summary=None,
            gate_summary=None,
            safe_to_voice=False,
            warnings=[
                f"Pipeline aborted: Anthropic rate limit (429) exhausted after "
                f"{exc.retry_count} retries for agent '{exc.agent_name}'. "
                "Re-run when API capacity recovers."
            ],
            telemetry={
                "error_type":  "provider_rate_limit",
                "agent":       exc.agent_name,
                "retry_count": exc.retry_count,
                "message":     str(exc),
                "stage":       exc.agent_name,
            },
        )


def _run_agent_pipeline_inner(inp: EpisodeInput) -> PackageResponse:
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

    # Reset call tracker and rate limiter for this pipeline run
    call_tracker.reset()
    _model_rate_limiter.reset()

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

    # ── Effective runtime config trace ────────────────────────────────────────
    # Written on every run so operators can audit what settings were actually used.
    # Includes model, fact_lock_mode requested vs effective, transcript size classification,
    # token estimates, segmented fact lock details, and rate limiter telemetry snapshot.
    _clean_chars  = len(clean)
    _effective_fl = effective_fact_lock_mode or settings.fact_lock_mode
    _segmented_used = _effective_fl == "segmented"
    _seg_count = 0
    if _segmented_used:
        _seg_idx_path = facts_dir / "fact_lock_segment_index.json"
        if _seg_idx_path.exists():
            try:
                _seg_idx = json.loads(_seg_idx_path.read_text(encoding="utf-8"))
                _seg_count = _seg_idx.get("total_segments", 0)
            except Exception:
                pass
    _effective_runtime_config = {
        "episode_id":                    episode_id,
        "model":                         settings.claude_model,
        "quality_mode":                  settings.quality_mode,
        "openai_review_policy":          settings.openai_review_policy,
        "fact_lock_mode_requested":      settings.fact_lock_mode,
        "fact_lock_mode_effective":      _effective_fl,
        "segmented_fact_lock_used":      _segmented_used,
        "segment_count":                 _seg_count,
        "transcript_chars_raw":          len(inp.raw_transcript),
        "transcript_chars_clean":        _clean_chars,
        "transcript_estimated_tokens":   estimate_tokens(_clean_chars),
        "transcript_size_class":         classify_transcript_size(_clean_chars),
        "research_view_chars":           len(research_view),
        "safe_claude_input_tokens_per_call": settings.safe_claude_input_tokens_per_call,
        "safe_claude_tokens_per_minute": settings.safe_claude_tokens_per_minute,
        "rate_limiter_telemetry":        _model_rate_limiter.telemetry(),
        "reuse_existing_stage_outputs":  settings.reuse_existing_stage_outputs,
    }
    (review_dir / "effective_runtime_config.json").write_text(
        json.dumps(_effective_runtime_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Effective runtime config saved → %s/effective_runtime_config.json  "
        "(fact_lock=%s, transcript=%s, %d chars)",
        review_dir,
        _effective_fl,
        _effective_runtime_config["transcript_size_class"],
        _clean_chars,
    )

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

    # ── Stage 3.75: Originality Transformation Plan ───────────────────────────
    # Runs after fact_lock + story_blueprint, before script_outline / script_writer.
    # Non-fatal: failure is logged + warned, but safe_to_voice is blocked.
    originality_transformation_plan: dict = {}
    _transformation_plan_ok = False   # True when plan exists and is non-empty

    if settings.originality_transformation_enabled and settings.quality_mode != "premium_batch":
        logger.info("Stage 3.75 — Originality Transformation Planner")
        existing_otp = _try_load_existing(
            facts_dir / "originality_transformation_plan.json",
            "originality_transformation",
            episode_dir=episode_dir,
            prompt_check="originality_transformation",
        )
        if existing_otp is not None:
            originality_transformation_plan = existing_otp
            _transformation_plan_ok = bool(originality_transformation_plan)
            logger.info(
                "Stage 3.75 — Loaded existing transformation plan "
                "(source_risk=%s, original_sections=%d)",
                originality_transformation_plan.get("source_dependency_risk", "?"),
                len(originality_transformation_plan.get("original_story_structure", [])),
            )
        else:
            try:
                call_tracker.stage_start("originality_transformation")
                originality_transformation_plan = run_originality_transformation(
                    case_hint=inp.case_hint,
                    target_duration_min=inp.target_duration_min,
                    hinglish_level=inp.hinglish_level,
                    fact_lock=fact_lock,
                    blueprint=blueprint,
                    source_transcript=clean,
                    facts_dir=facts_dir,
                )
                _transformation_plan_ok = bool(originality_transformation_plan)
                call_tracker.stage_end("originality_transformation")
                logger.info(
                    "Stage 3.75 — Transformation plan ready "
                    "(source_risk=%s, original_sections=%d, instructions=%d)",
                    originality_transformation_plan.get("source_dependency_risk", "?"),
                    len(originality_transformation_plan.get("original_story_structure", [])),
                    len(originality_transformation_plan.get("writer_instructions", [])),
                )
            except Exception as exc:
                logger.error("Originality transformation plan failed (non-fatal): %s", exc)
                warnings.append(
                    f"Originality Transformation Planner failed: {exc}. "
                    "Script will be written without transformation guidance. "
                    "safe_to_voice will be False. "
                    "See 02-facts/_originality_transformation_raw_response.txt for details."
                )
                originality_transformation_plan = {}
                _transformation_plan_ok = False
    else:
        # Feature disabled or premium_batch mode — treat as not required
        _transformation_plan_ok = True
        logger.info(
            "Stage 3.75 — Skipped (originality_transformation_enabled=%s quality_mode=%s)",
            settings.originality_transformation_enabled,
            settings.quality_mode,
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
            originality_transformation_plan=originality_transformation_plan or None,
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
        source_transcript=clean,
    )
    # Carry preflight blocking + metadata targets forward so Stage 13a can use them.
    _preflight_blocking = preflight_report.get("blocking", False)
    _preflight_meta_targets = preflight_report.get("metadata_repair_targets", [])
    _ran_any_repair = False  # set True when chunk OR metadata repair runs
    artifact_state = ArtifactState()  # tracks script/metadata/dialogue mutations
    if preflight_report.get("metadata_python_fixes_applied"):
        artifact_state.mark_metadata_mutated("stage5_5_metadata_python_autofix")
    root_cause_repair_attempts: dict[str, int] = {}
    repair_cost_telemetry: dict[str, Any] = {
        "root_cause_repair_attempts": root_cause_repair_attempts,
        "claude_repair_skipped_due_previous_failure": False,
        "openai_cluster_repair_ran": False,
        "openai_cluster_repair_chunks": [],
        "python_blocked_before_openai_final": False,
        "estimated_claude_calls_saved": 0,
        "repeat_root_cause_detected": [],
        "source_shaped_reconstruction_detected": preflight_report.get(
            "source_shaped_reconstruction_detected", False
        ),
        "reconstruction_cluster_count": len(
            preflight_report.get("reconstruction_cluster_candidates", [])
        ),
        "remaining_root_causes": [],
        "retention_blocked_before_openai": False,
        "retention_repair_attempted": False,
        "retention_score_after_repair": None,
        "repair_budget_exceeded": False,
        "repair_budget_exceeded_kind": None,
        "repair_budget_exceeded_reason": None,
    }
    post_openai_preflight_blockers: list[dict[str, Any]] = []

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

    # ── Gate summary — initialized here so Stage 6 repair routing can record
    # telemetry.  Premium gate entries are added below after Stage 6 completes.
    # Initializing early prevents NameError/UnboundLocalError when Stage 6 calls
    # _run_routing_and_rebuild(gate_summary=gate_summary).
    gate_summary: dict[str, dict] = {}

    # ── Stage 6: Targeted Chunk Repair (one pass max) ─────────────────────────
    repair_has_failures = False   # default: no repair ran, no failures
    if not approved and repair_required:
        logger.info("Stage 6 — Targeted Chunk Repair")
        try:
            chunk_repair_targets = quality_report.get("chunk_repair_targets", [])
            if not chunk_repair_targets:
                warnings.append(
                    "Repair required but critic did not provide chunk_repair_targets. "
                    "Attempting auto-rebuild via premium_section_rebuild_service."
                )
                logger.warning(
                    "Stage 6 — repair_required=true but no chunk_repair_targets. "
                    "Routing to auto-rebuild."
                )
                script_final = promote_draft_as_final(script_draft, script_dir)
                status = "auto_rebuild_required"
                quality_report["approved"] = False
                quality_report["repair_required"] = True
                # Attempt grouped rebuild when critic provided no specific targets
                try:
                    _s6_rr_reports = {"script_quality": quality_report}
                    script_final, _s6_routing, _s6_rebuilt = _run_routing_and_rebuild(
                        script_draft=script_final,
                        gate_reports=_s6_rr_reports,
                        fact_lock=fact_lock,
                        blueprint=blueprint,
                        retention_blueprint=retention_blueprint,
                        originality_transformation_plan=originality_transformation_plan or {},
                        case_glossary=case_glossary,
                        hinglish_level=inp.hinglish_level,
                        case_hint=getattr(inp, "case_hint", ""),
                        review_dir=review_dir,
                        script_dir=script_dir,
                        warnings=warnings,
                        gate_summary=gate_summary,
                        root_cause_repair_attempts=root_cause_repair_attempts,
                    )
                    if _s6_routing.get("route") == "stop_not_voice_ready":
                        status = "not_voice_ready_auto_retry_exhausted"
                    # Re-run script review after rebuild
                    if _s6_rebuilt:
                        quality_report = run_script_review(
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
                        if quality_report.get("approved", False):
                            status = "script_approved"
                        else:
                            status = "not_voice_ready_auto_retry_exhausted"
                except Exception as _s6_exc:
                    logger.error("Stage 6 auto-rebuild failed: %s", _s6_exc)
                    status = "not_voice_ready_auto_retry_exhausted"
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
                artifact_state.mark_script_mutated("stage6_script_repair")
                repair_has_failures = repair_report.get("has_failures", False)
                repair_failed_count = repair_report.get("chunks_failed", 0)
                # If Stage 6 repaired deterministic source-copy/reconstruction
                # targets that came from Python preflight, count that as the one
                # allowed Claude attempt for those root causes. If the same root
                # cause survives the post-repair preflight, routing will escalate
                # to OpenAI cluster repair instead of spending Claude again.
                if preflight_report.get("source_shaped_reconstruction_detected"):
                    _initial_source_routing = run_repair_routing(
                        all_gate_reports={"python_preflight": preflight_report},
                        openai_repair_max_chunks=settings.openai_repair_max_chunks,
                        review_dir=None,
                        script_draft=script_draft,
                        max_cluster_size=settings.max_openai_cluster_repair_chunks,
                        previous_root_cause_attempts=root_cause_repair_attempts,
                    )
                    for _cluster in _initial_source_routing.get(
                        "source_copy_reconstruction_clusters", []
                    ):
                        _key = _cluster.get("root_cause_key")
                        if _key:
                            root_cause_repair_attempts[_key] = (
                                root_cause_repair_attempts.get(_key, 0) + 1
                            )

                if repair_has_failures:
                    warnings.append(
                        f"{repair_failed_count} chunk repair(s) failed — original content kept. "
                        "See 04-review/script_repair_report.json for details."
                    )
                    logger.warning(
                        "Stage 6 — %d chunk repair(s) failed.",
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
                    status = "not_voice_ready_auto_retry_exhausted"
                    if not final_quality.get("approved", False):
                        warnings.append(
                            "Script was repaired but final quality review did not approve. "
                            "See 04-review/final_script_quality_report.json for details."
                        )
                    logger.warning("Final review: NOT APPROVED after repair — not_voice_ready_auto_retry_exhausted")
        except Exception as exc:
            logger.error("Script repair failed: %s", exc)
            warnings.append(
                f"Script repair failed: {exc}. "
                "Draft promoted as final. Automated retry exhausted — safe_to_voice=false."
            )
            script_final = promote_draft_as_final(script_draft, script_dir)
            status = "not_voice_ready_auto_retry_exhausted"
    else:
        script_final = promote_draft_as_final(script_draft, script_dir)
        status = "script_approved" if approved else "auto_rebuild_required"
        if not approved:
            warnings.append(
                "Script quality review did not approve and repair was not required. "
                "Auto-rebuild will be attempted in Stage 16. "
                "See 04-review/script_quality_report.json."
            )

    # ── Premium quality gates (cost_mode=premium only) ───────────────────────
    # gate_summary was initialized before Stage 6 — do not reset it here.
    safe_to_voice = False

    # Record originality_transformation gate result (set in Stage 3.75)
    gate_summary["originality_transformation"] = {
        "passed":   _transformation_plan_ok,
        "skipped":  not settings.originality_transformation_enabled
                    or settings.quality_mode == "premium_batch",
        "plan_exists": bool(originality_transformation_plan),
        "source_risk": originality_transformation_plan.get("source_dependency_risk", "unknown")
                        if originality_transformation_plan else "unknown",
    }

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

    retention_blocked_before_openai = False
    retention_repair_attempted = False
    retention_score_after_repair: float | None = None

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
            # originality_transformation gate result was already written above (Stage 3.75)
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
                    f"Hindi copyedit gate call failed: {exc}. "
                    "No repair targets available — gate marked failed."
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
                    artifact_state.mark_script_mutated("stage9a_copyedit_repair")
                    copyedit_repair_has_failures = copyedit_repair_report.get("has_failures", False)
                    if copyedit_repair_has_failures:
                        failed_n = copyedit_repair_report.get("chunks_failed", 0)
                        warnings.append(
                            f"Copyedit repair: {failed_n} chunk(s) failed — "
                            "original content kept. Automated retry exhausted — safe_to_voice=false."
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
                    "No repair targets available — gate marked failed."
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
            status = "not_voice_ready_auto_retry_exhausted"
            if not any("copyedit" in w.lower() for w in warnings):
                warnings.append(
                    "Hindi copyedit gate FAILED after repair attempt. "
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
                    f"Originality safety gate call failed: {exc}. "
                    "Gate marked failed — auto-rebuild will be attempted by OFP gate."
                )
                originality_report = {"gate_passed": False, "error": str(exc)}

        gate_summary["originality_safety"] = {
            "passed":         originality_report.get("gate_passed", False),
            "scores":         originality_report.get("scores", {}),
            "required_fixes": stringify_report_list(originality_report.get("required_fixes", [])),
        }
        if not originality_report.get("gate_passed", False):
            required_fixes = originality_report.get("required_fixes", [])
            if required_fixes:
                warnings.append(
                    "Originality/safety gate FAILED. Required fixes: "
                    + safe_join_report_items(required_fixes, limit=3)
                    + (f" (+{len(required_fixes)-3} more)" if len(required_fixes) > 3 else "")
                    + " — OpenAI Final Premium Gate will attempt auto-rebuild."
                )
            # Auto-rebuild path: OFP gate (Stage 14/16) will pick up these issues
            # and route them through repair_routing_service + premium_section_rebuild.
            status = "auto_rebuild_required"

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
                    f"Recreated dialogue quality gate call failed: {exc}. "
                    "Gate marked failed — auto-rebuild will be attempted by OFP gate."
                )
                dialogue_report = {"gate_passed": False, "error": str(exc)}

        gate_summary["recreated_dialogue"] = {
            "passed":         dialogue_report.get("gate_passed", False),
            "no_scenes":      dialogue_report.get("no_recreated_scenes", False),
            "scores":         dialogue_report.get("scores", {}),
            "required_fixes": stringify_report_list(dialogue_report.get("required_fixes", [])),
        }
        if not dialogue_report.get("gate_passed", False):
            required_fixes = dialogue_report.get("required_fixes", [])
            if required_fixes:
                warnings.append(
                    "Recreated dialogue gate FAILED. Attempting deterministic auto-fix "
                    "(disclaimer insertion). Required fixes: "
                    + safe_join_report_items(required_fixes, limit=3)
                    + (f" (+{len(required_fixes)-3} more)" if len(required_fixes) > 3 else "")
                )
            # Run deterministic fix first — inserts missing "फिर से रचा गया संवाद" labels
            try:
                script_final, _dlg_fix_report = run_deterministic_auto_fix(
                    script_draft=script_final,
                    case_hint=getattr(inp, "case_hint", ""),
                    review_dir=review_dir,
                )
                _dlg_fixes = _dlg_fix_report.get("total_fixes_applied", 0)
                if _dlg_fixes:
                    logger.info(
                        "Stage 12 deterministic auto-fix: %d fix(es) applied",
                        _dlg_fixes,
                    )
                    # Inspect each change to set mutation flags accurately.
                    # context field patterns:
                    #   "chunk:<id>"              → narration / script mutated
                    #   "recreated_dialogue:<id>" → dialogue mutated
                    #   "title_options", "recommended_title", "description",
                    #   "tag", "thumbnail_text", "pinned_comment" → metadata
                    #   "folder_name"             → slug only (no OFP-relevant mutation)
                    _METADATA_CONTEXTS = frozenset({
                        "title_options", "recommended_title", "description",
                        "tag", "thumbnail_text", "pinned_comment",
                        "tags",
                    })
                    for _chg in _dlg_fix_report.get("changes", []):
                        _ctx = _chg.get("context", "")
                        if _ctx.startswith("chunk:"):
                            _ran_any_repair = True
                            artifact_state.mark_script_mutated(
                                "stage12_deterministic_auto_fix_narration"
                            )
                        elif _ctx.startswith("recreated_dialogue:"):
                            _ran_any_repair = True
                            artifact_state.mark_dialogue_mutated(
                                "stage12_deterministic_auto_fix_dialogue"
                            )
                        elif _ctx in _METADATA_CONTEXTS:
                            _ran_any_repair = True
                            artifact_state.mark_metadata_mutated(
                                "stage12_deterministic_auto_fix_metadata"
                            )
                        # "folder_name" context: slug-only change; no OFP-input mutation
            except Exception as _dlg_exc:
                logger.error("Stage 12 deterministic auto-fix failed: %s", _dlg_exc)
            # OFP gate will recheck and repair remaining issues
            status = "auto_repair_required"

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
                    f"Metadata quality gate call failed: {exc}. "
                    "Gate marked failed — auto-rebuild will be attempted by OFP gate."
                )
                metadata_report = {"gate_passed": False, "error": str(exc)}

        # Validate metadata gate report schema (non-fatal, with structured fallback)
        try:
            _validate_metadata_report(metadata_report, review_dir)
        except ValueError as exc:
            logger.warning("Metadata Quality Report schema validation failed — applying fallback report: %s", exc)
            # Build a structured fallback so downstream gates always have a usable dict.
            # The fallback marks the gate as failed and surfaces the validation error
            # as a required_fix. Operators can inspect _metadata_quality_validation_error.txt
            # for the full Pydantic details.
            _meta_fallback: dict = {
                "gate_passed": False,
                "scores": {},
                "required_fixes": [
                    "Metadata quality schema validation failed — Claude returned unexpected field names or types. "
                    "Re-run with REUSE_EXISTING_STAGE_OUTPUTS=false to get a fresh gate result.",
                ],
                "high_severity_issues": 1,
                "validation_error": str(exc)[:500],
                "_fallback": True,
            }
            # Overwrite the malformed report on disk with the structured fallback
            try:
                (review_dir / "metadata_quality_gate_report.json").write_text(
                    json.dumps(_meta_fallback, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as _wexc:
                logger.warning("Could not overwrite metadata_quality_gate_report.json with fallback: %s", _wexc)
            metadata_report = _meta_fallback
            warnings.append(
                "Metadata Quality Report schema mismatch — structured fallback applied "
                "(gate_passed=False, high_severity_issues=1). "
                "See 04-review/_metadata_quality_validation_error.txt. "
                "Re-run with REUSE_EXISTING_STAGE_OUTPUTS=false to retry."
            )
            if status not in ("needs_human_review", "not_voice_ready_auto_retry_exhausted"):
                status = "not_voice_ready_auto_retry_exhausted"

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
                    + safe_join_report_items(meta_required_fixes, limit=3)
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
                artifact_state.mark_metadata_mutated("stage13a_metadata_repair")

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
                        status = "not_voice_ready_auto_retry_exhausted"
                        warnings.append(
                            "Metadata recheck FAILED after repair. "
                            "See 04-review/metadata_quality_gate_report.json."
                        )
                except Exception as exc:
                    logger.error("Metadata recheck failed: %s", exc)
                    warnings.append(
                        f"Metadata quality gate recheck failed: {exc}."
                    )
                    metadata_repair_has_failures = True
                    status = "not_voice_ready_auto_retry_exhausted"

            except Exception as exc:
                logger.error("Metadata repair failed: %s", exc)
                warnings.append(
                    f"Metadata repair failed: {exc}. "
                    "Metadata may still have quality issues."
                )
                metadata_repair_has_failures = True
                status = "not_voice_ready_auto_retry_exhausted"
        else:
            # Gate passed — no repair needed
            pass

        gate_summary["metadata_quality"] = {
            "passed":         metadata_report.get("gate_passed", False),
            "scores":         metadata_report.get("scores", {}),
            "required_fixes": stringify_report_list(metadata_report.get("required_fixes", [])),
            "repair_ran":     not metadata_report.get("gate_passed", True),
        }
        if not metadata_report.get("gate_passed", False) and not metadata_repair_has_failures:
            status = "not_voice_ready_auto_retry_exhausted"

        # ── Stage 9.5: Retention Quality Gate ────────────────────────────────
        retention_blocked_before_openai = False
        retention_repair_attempted = False
        retention_score_after_repair: float | None = None
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
                        f"Retention quality gate call failed: {exc}. "
                        "Gate marked failed — auto-rebuild will be attempted by OFP gate."
                    )
                    retention_report = {"approved": False, "error": str(exc), "chunk_repair_targets": []}

            # Validate retention gate report schema (non-fatal)
            try:
                _validate_retention_report(retention_report, review_dir)
            except ValueError as exc:
                logger.warning("Retention Quality Report schema validation (non-fatal): %s", exc)
                warnings.append(
                    "Retention Quality Report schema mismatch — gate results unreliable. "
                    "See 04-review/_retention_quality_validation_error.txt. "
                    "Pipeline cannot auto-repair without a valid gate report; "
                    "re-run with REUSE_EXISTING_STAGE_OUTPUTS=false."
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
                status = "auto_rebuild_required"   # repair will run below
                rt_targets = retention_report.get("chunk_repair_targets", [])

                if rt_targets:
                    retention_repair_attempted = True
                    repair_cost_telemetry["retention_repair_attempted"] = True
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
                        artifact_state.mark_script_mutated("stage9_6_retention_repair")
                        retention_repair_has_failures = retention_repair_report.get(
                            "has_failures", False
                        )
                        if retention_repair_has_failures:
                            failed_n = retention_repair_report.get("chunks_failed", 0)
                            warnings.append(
                                f"Retention repair: {failed_n} chunk(s) failed — "
                                "original content kept. "
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
                            retention_score_after_repair = _retention_score(retention_recheck)
                            repair_cost_telemetry["retention_score_after_repair"] = retention_score_after_repair
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
                                status = "not_voice_ready_auto_retry_exhausted"
                                if not _retention_failure_is_localized(retention_recheck):
                                    retention_blocked_before_openai = True
                                    repair_cost_telemetry["retention_blocked_before_openai"] = True
                                    warnings.append(
                                        "Retention failure remains broad after one repair pass; "
                                        "OpenAI Final Premium Gate will be skipped to avoid expensive "
                                        "non-localized repair. See 04-review/retention_quality_report.json."
                                    )
                                logger.warning(
                                    "Retention quality recheck: still FAILED "
                                    "(overall=%s)",
                                    retention_recheck.get("overall_retention_score", 0),
                                )
                                warnings.append(
                                    "Retention quality recheck FAILED after repair. "
                                    "See 04-review/retention_quality_report.json."
                                )
                        except Exception as exc:
                            logger.error("Retention quality recheck failed: %s", exc)
                            warnings.append(
                                f"Retention quality recheck failed: {exc}."
                            )
                            retention_repair_has_failures = True
                            status = "not_voice_ready_auto_retry_exhausted"

                    except Exception as exc:
                        logger.error("Retention repair failed: %s", exc)
                        warnings.append(
                            f"Retention targeted repair failed: {exc}. "
                            "Original chunks kept."
                        )
                        retention_repair_has_failures = True
                        status = "not_voice_ready_auto_retry_exhausted"
                else:
                    retention_blocked_before_openai = True
                    repair_cost_telemetry["retention_blocked_before_openai"] = True
                    warnings.append(
                        "Retention quality gate FAILED but no chunk_repair_targets provided. "
                        "OpenAI Final Premium Gate will be skipped because the failure is not localized. "
                        "See 04-review/retention_quality_report.json."
                    )
                    status = "not_voice_ready_auto_retry_exhausted"
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
                    source_transcript=clean_transcript_text,
                )
                _post_repair_preflight_blocking = _post_pf_report.get("blocking", False)
                if _post_pf_report.get("metadata_python_fixes_applied"):
                    artifact_state.mark_metadata_mutated("stage13c_metadata_python_autofix")
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
                    _pf_high = _post_pf_counts.get("high", 0)
                    _pf_med  = _post_pf_counts.get("medium", 0)
                    logger.warning(
                        "Stage 13c — Post-repair preflight BLOCKING "
                        "(high=%d, medium=%d) — attempting pre-OAI repair pass.",
                        _pf_high, _pf_med,
                    )
                    # ── Stage 13d: Pre-OAI repair pass ───────────────────────
                    # Before giving up and skipping OpenAI, run one deterministic
                    # auto-fix pass + repair-routing + Claude rebuild on the
                    # issues the preflight report surfaced. Then re-check preflight.
                    # Do NOT re-run fact_lock / blueprint / originality_transformation.
                    try:
                        _pre_oai_gate_reports: dict[str, dict] = {
                            "python_preflight": _post_pf_report,
                            "script_quality":   quality_report,
                        }
                        # Add optional gates only when they are real dicts (not
                        # bare error dicts) to avoid misleading the router.
                        for _gr_key, _gr_val in (
                            ("text_similarity",   similarity_report),
                            ("originality_safety", originality_report),
                            ("retention",         retention_report),
                            ("metadata",          metadata_report),
                            ("recreated_dialogue", dialogue_report),
                        ):
                            if isinstance(_gr_val, dict) and "error" not in _gr_val:
                                _pre_oai_gate_reports[_gr_key] = _gr_val

                        logger.info(
                            "Stage 13d — pre-OAI repair: routing with %d gate reports",
                            len(_pre_oai_gate_reports),
                        )
                        script_final, _pre_oai_routing, _pre_oai_rebuilt = (
                            _run_routing_and_rebuild(
                                script_draft=script_final,
                                gate_reports=_pre_oai_gate_reports,
                                fact_lock=fact_lock,
                                blueprint=blueprint,
                                retention_blueprint=retention_blueprint,
                                originality_transformation_plan=originality_transformation_plan,
                                case_glossary=case_glossary,
                                hinglish_level=inp.hinglish_level,
                                case_hint=inp.case_hint,
                                review_dir=review_dir,
                                script_dir=script_dir,
                                warnings=warnings,
                                gate_summary=gate_summary,
                                root_cause_repair_attempts=root_cause_repair_attempts,
                            )
                        )
                        gate_summary.setdefault("pre_oai_repair", {})[
                            "ran"
                        ] = True
                        gate_summary["pre_oai_repair"]["rebuild_ran"] = _pre_oai_rebuilt
                        gate_summary["pre_oai_repair"]["route"] = _pre_oai_routing.get(
                            "route", "unknown"
                        )
                        artifact_state.mark_script_mutated("stage13d_pre_oai_repair")

                        # ── Refresh cheap gate reports after Stage 13d mutation ────
                        # Stage 13d changed script_final (repair/rebuild).  Refresh
                        # lint and text-similarity immediately so Stage 14a OFP
                        # receives evidence based on the post-repair script — not the
                        # stale pre-repair versions that are still in memory.
                        logger.info(
                            "Stage 13d — refreshing lint + similarity after pre-OAI repair"
                        )
                        _13d_refresh = _refresh_reports_after_script_mutation(
                            script_final=script_final,
                            fact_lock=fact_lock,
                            blueprint=blueprint,
                            review_dir=review_dir,
                            gate_summary=gate_summary,
                            warnings=warnings,
                            lint_report=lint_report,
                            similarity_report=similarity_report,
                            quality_report=quality_report,
                            copyedit_report=copyedit_report,
                            source_transcript=clean_transcript_text,
                            hinglish_level=inp.hinglish_level,
                            target_duration_min=inp.target_duration_min,
                            retention_blueprint=retention_blueprint if retention_blueprint else None,
                            retention_report=retention_report,
                            originality_report=originality_report,
                            dialogue_report=dialogue_report,
                            metadata_report=metadata_report,
                            cost_mode=inp.cost_mode,
                            case_glossary=case_glossary,
                            # Only cheap Python-only gates here — Claude gates are
                            # too expensive to rerun at every repair pass.
                            rerun_lint=True,
                            rerun_similarity=True,
                            rerun_quality=False,
                            rerun_copyedit=False,
                            rerun_retention=False,
                            rerun_originality=False,
                            rerun_dialogue=False,
                            rerun_metadata=False,
                        )
                        lint_report       = _13d_refresh["lint_report"]
                        similarity_report = _13d_refresh["similarity_report"]
                        gate_summary.setdefault("pre_oai_repair", {})[
                            "reports_refreshed"
                        ] = ["lint", "similarity"]

                        # Re-run Python preflight after repair attempt
                        logger.info("Stage 13d — re-running Python preflight after pre-OAI repair")
                        _post_pf_report2 = run_python_preflight(
                            script_draft=script_final,
                            fact_lock=fact_lock,
                            case_glossary=case_glossary,
                            review_dir=review_dir,
                            target_duration_min=inp.target_duration_min,
                            hinglish_level=inp.hinglish_level,
                            label="_after_pre_oai_repair",
                            source_transcript=clean_transcript_text,
                        )
                        _post_repair_preflight_blocking = _post_pf_report2.get(
                            "blocking", False
                        )
                        if _post_pf_report2.get("metadata_python_fixes_applied"):
                            artifact_state.mark_metadata_mutated("stage13d_metadata_python_autofix")
                        _post_pf_counts2 = _post_pf_report2.get("severity_counts", {})
                        gate_summary["python_preflight"].update({
                            "passed":       _post_pf_report2.get("passed", False),
                            "blocking":     _post_repair_preflight_blocking,
                            "high":         _post_pf_counts2.get("high", 0),
                            "medium":       _post_pf_counts2.get("medium", 0),
                            "low":          _post_pf_counts2.get("low", 0),
                            "report":       "python_preflight_report_after_pre_oai_repair.json",
                            "pre_oai_repair_ran": True,
                        })
                        if _post_repair_preflight_blocking:
                            _pf_high = _post_pf_counts2.get("high", 0)
                            _pf_med  = _post_pf_counts2.get("medium", 0)
                            logger.warning(
                                "Stage 13d — preflight still BLOCKING after pre-OAI repair "
                                "(high=%d, medium=%d) — evaluating OpenAI cluster rescue.",
                                _pf_high, _pf_med,
                            )
                            _cluster_reports = {
                                "python_preflight": _post_pf_report2,
                                "script_quality": quality_report,
                                "text_similarity": similarity_report,
                            }
                            _cluster_routing = run_repair_routing(
                                all_gate_reports=_cluster_reports,
                                openai_repair_max_chunks=settings.max_openai_cluster_repair_chunks,
                                review_dir=review_dir,
                                script_draft=script_final,
                                max_cluster_size=settings.max_openai_cluster_repair_chunks,
                                previous_root_cause_attempts=root_cause_repair_attempts,
                            )
                            repair_cost_telemetry["claude_repair_skipped_due_previous_failure"] = (
                                _cluster_routing.get("claude_repair_skipped_due_previous_failure", False)
                            )
                            repair_cost_telemetry["repeat_root_cause_detected"] = _cluster_routing.get(
                                "repeat_root_cause_detected", []
                            )
                            repair_cost_telemetry["reconstruction_cluster_count"] = _cluster_routing.get(
                                "reconstruction_cluster_count", 0
                            )
                            repair_cost_telemetry["source_shaped_reconstruction_detected"] = (
                                repair_cost_telemetry["source_shaped_reconstruction_detected"]
                                or _cluster_routing.get("source_shaped_reconstruction_detected", False)
                            )
                            _cluster_targets = _cluster_routing.get("openai_repair_targets", [])
                            _cluster_chunk_ids = list(dict.fromkeys(
                                t.get("chunk_id", "") for t in _cluster_targets if t.get("chunk_id")
                            ))
                            if (
                                settings.openai_cluster_repair_enabled
                                and settings.openai_repair_enabled
                                and bool(settings.openai_api_key)
                                and _cluster_targets
                                and len(_cluster_chunk_ids) <= settings.max_openai_cluster_repair_chunks
                            ):
                                _budget_reason = _soft_repair_budget_exceeded("openai")
                                if _budget_reason:
                                    _record_soft_budget_stop(
                                        "openai",
                                        _budget_reason,
                                        warnings,
                                        gate_summary,
                                        repair_cost_telemetry,
                                    )
                                    openai_repair_has_failures = True
                                    status = "not_voice_ready_auto_retry_exhausted"
                                    raise RuntimeError(_budget_reason)
                                logger.info(
                                    "Stage 13e — OpenAI cluster repair rescue (%d chunks)",
                                    len(_cluster_chunk_ids),
                                )
                                script_final, _cluster_repair_report = run_openai_targeted_chunk_repair(
                                    script_draft=script_final,
                                    repair_targets=_cluster_targets,
                                    fact_lock=fact_lock,
                                    blueprint=blueprint,
                                    hinglish_level=inp.hinglish_level,
                                    script_dir=script_dir,
                                    review_dir=review_dir,
                                )
                                artifact_state.mark_script_mutated("stage13e_openai_cluster_repair")
                                _ran_any_repair = True
                                repair_cost_telemetry["openai_cluster_repair_ran"] = True
                                repair_cost_telemetry["openai_cluster_repair_chunks"] = _cluster_chunk_ids
                                openai_repair_has_failures = _cluster_repair_report.get("has_failures", False)
                                if openai_repair_has_failures:
                                    warnings.append(
                                        "OpenAI cluster repair had chunk failures. "
                                        "Automated retry exhausted — safe_to_voice=false."
                                    )

                                _post_pf_report2 = run_python_preflight(
                                    script_draft=script_final,
                                    fact_lock=fact_lock,
                                    case_glossary=case_glossary,
                                    review_dir=review_dir,
                                    target_duration_min=inp.target_duration_min,
                                    hinglish_level=inp.hinglish_level,
                                    label="_after_openai_cluster_repair",
                                    source_transcript=clean_transcript_text,
                                )
                                _post_repair_preflight_blocking = _post_pf_report2.get("blocking", False)
                                _post_pf_counts2 = _post_pf_report2.get("severity_counts", {})
                                gate_summary["python_preflight"].update({
                                    "passed": _post_pf_report2.get("passed", False),
                                    "blocking": _post_repair_preflight_blocking,
                                    "high": _post_pf_counts2.get("high", 0),
                                    "medium": _post_pf_counts2.get("medium", 0),
                                    "low": _post_pf_counts2.get("low", 0),
                                    "report": "python_preflight_report_after_openai_cluster_repair.json",
                                    "openai_cluster_repair_ran": True,
                                })
                                _13e_refresh = _refresh_reports_after_script_mutation(
                                    script_final=script_final,
                                    fact_lock=fact_lock,
                                    blueprint=blueprint,
                                    review_dir=review_dir,
                                    gate_summary=gate_summary,
                                    warnings=warnings,
                                    lint_report=lint_report,
                                    similarity_report=similarity_report,
                                    quality_report=quality_report,
                                    copyedit_report=copyedit_report,
                                    source_transcript=clean_transcript_text,
                                    hinglish_level=inp.hinglish_level,
                                    target_duration_min=inp.target_duration_min,
                                    retention_blueprint=retention_blueprint if retention_blueprint else None,
                                    retention_report=retention_report,
                                    originality_report=originality_report,
                                    dialogue_report=dialogue_report,
                                    metadata_report=metadata_report,
                                    cost_mode=inp.cost_mode,
                                    case_glossary=case_glossary,
                                    rerun_lint=True,
                                    rerun_similarity=True,
                                    rerun_quality=False,
                                    rerun_copyedit=False,
                                    rerun_retention=False,
                                    rerun_originality=False,
                                    rerun_dialogue=False,
                                    rerun_metadata=False,
                                )
                                lint_report = _13e_refresh["lint_report"]
                                similarity_report = _13e_refresh["similarity_report"]
                            else:
                                repair_cost_telemetry["python_blocked_before_openai_final"] = True
                                repair_cost_telemetry["remaining_root_causes"] = [
                                    t.get("root_cause_key", t.get("chunk_id", ""))
                                    for t in _cluster_targets
                                ]
                        else:
                            logger.info(
                                "Stage 13d — pre-OAI repair cleared preflight blocking — "
                                "OpenAI gate will now run."
                            )
                    except Exception as _pre_oai_exc:
                        logger.error(
                            "Stage 13d — pre-OAI repair pass failed: %s — "
                            "treating preflight as still blocking.",
                            _pre_oai_exc,
                        )
                        warnings.append(
                            f"Pre-OAI repair pass failed: {_pre_oai_exc}. "
                            "OpenAI Final Premium Gate skipped."
                        )
                        _post_repair_preflight_blocking = True

                if _post_repair_preflight_blocking:
                    status = "not_voice_ready_auto_retry_exhausted"
                    high_n = gate_summary.get("python_preflight", {}).get("high", _post_pf_counts.get("high", 0))
                    med_n  = gate_summary.get("python_preflight", {}).get("medium", _post_pf_counts.get("medium", 0))
                    warnings.append(
                        f"Post-repair Python preflight still BLOCKING "
                        f"(high={high_n}, medium={med_n}). "
                        "OpenAI Final Premium Gate skipped. "
                        "Re-run the pipeline after addressing blocking issues. "
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
                status = "needs_human_review"   # KEEP: safety check exception is truly unrecoverable
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
            and not retention_blocked_before_openai
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

                # Mandatory freshness check: if artifact_state shows any mutation
                # since the last gate run, attempt a cheap refresh (lint + similarity)
                # and mark expensive gates stale. If refresh_failed, block OFP.
                _finalize_result = _finalize_reports_before_openai(
                    artifact_state=artifact_state,
                    script_final=script_final,
                    fact_lock=fact_lock,
                    blueprint=blueprint,
                    review_dir=review_dir,
                    gate_summary=gate_summary,
                    warnings=warnings,
                    lint_report=lint_report,
                    similarity_report=similarity_report,
                    copyedit_report=copyedit_report,
                    quality_report=quality_report,
                    retention_report=retention_report or {},
                    originality_report=originality_report or {},
                    dialogue_report=dialogue_report or {},
                    metadata_report=metadata_report or {},
                    source_transcript=(
                        # Prefer the already-cleaned in-memory string (Stage 10
                        # defines clean_transcript_text by reading the file).
                        # Fall back to reading the file directly (resume runs),
                        # then to raw_transcript, then empty (triggers stale failure
                        # in the finalization helper rather than silently using blank).
                        (clean_transcript_text if clean_transcript_text else None)
                        or (
                            (episode_dir / "01-input" / "clean_transcript.txt").read_text(
                                encoding="utf-8"
                            )
                            if (episode_dir / "01-input" / "clean_transcript.txt").exists()
                            else None
                        )
                        or inp.raw_transcript
                        or ""
                    ),
                    hinglish_level=inp.hinglish_level,
                    target_duration_min=inp.target_duration_min,
                    retention_blueprint=retention_blueprint,
                    case_glossary=case_glossary,
                    cost_mode=inp.cost_mode,
                )
                # Use any freshly-refreshed reports from finalization
                for _rep_name, _rep_obj in _finalize_result["refreshed_reports"].items():
                    if _rep_name == "lint":
                        lint_report = _rep_obj
                    elif _rep_name == "similarity":
                        similarity_report = _rep_obj
                    elif _rep_name == "originality":
                        originality_report = _rep_obj
                    elif _rep_name == "metadata":
                        metadata_report = _rep_obj
                    elif _rep_name == "retention":
                        retention_report = _rep_obj
                    elif _rep_name == "dialogue":
                        dialogue_report = _rep_obj

                if _finalize_result["blocking"]:
                    _failed = _finalize_result["failed_refreshes"]
                    logger.warning(
                        "Stage 14a — pre-OFP finalization BLOCKING: refresh failed for %s",
                        _failed,
                    )
                    gate_summary["openai_final_premium"] = {
                        "passed":  False,
                        "skipped": True,
                        "reason":  f"required_gate_refresh_failed_after_script_mutation: {_failed}",
                    }
                    status = "not_voice_ready_auto_retry_exhausted"
                    gate_summary["automation_status"] = "blocked_before_openai"

                # Before calling OFP, ensure in-memory reports are fresh.
                # Stage 13d already refreshed lint + similarity via
                # _refresh_reports_after_script_mutation.  For reports that were
                # mutated by earlier repair stages (copyedit, metadata, retention),
                # prefer the in-memory copy if it already carries a freshness
                # marker; otherwise fall back to the disk file so we never pass a
                # pre-repair in-memory object.  Expensive Claude gates (quality,
                # copyedit, retention, originality, dialogue) are NOT re-called
                # here — that would double the API cost.
                def _prefer_fresh(in_mem: dict, filename: str, fallback: dict) -> dict:
                    if in_mem.get("refreshed_after_script_mutation") or \
                       in_mem.get("refreshed_after_rebuild"):
                        return in_mem
                    p = review_dir / filename
                    if p.exists():
                        try:
                            return json.loads(p.read_text(encoding="utf-8"))
                        except Exception:
                            pass
                    return fallback

                lint_report       = _prefer_fresh(lint_report,       "hindi_text_lint_report.json",          lint_report)
                similarity_report = _prefer_fresh(similarity_report,  "text_similarity_report.json",          similarity_report)
                copyedit_report   = _prefer_fresh(copyedit_report,    "hindi_copyedit_report.json",           copyedit_report)
                quality_report    = _prefer_fresh(quality_report,     "final_script_quality_report.json",     quality_report)
                retention_report  = _prefer_fresh(retention_report,   "retention_quality_report.json",        retention_report)
                originality_report = _prefer_fresh(originality_report, "originality_safety_gate_report.json", originality_report)
                dialogue_report   = _prefer_fresh(dialogue_report,    "recreated_dialogue_gate_report.json",  dialogue_report)
                metadata_report   = _prefer_fresh(metadata_report,    "metadata_quality_gate_report.json",    metadata_report)

                # Abort OFP if any required report has refresh_failed=True —
                # we must not approve a script based on stale evidence.
                _stale_reports = [
                    name for name, rpt in (
                        ("similarity", similarity_report),
                        ("originality", originality_report),
                    )
                    if isinstance(rpt, dict) and rpt.get("refresh_failed")
                ]
                if _stale_reports:
                    warnings.append(
                        f"OpenAI Final Premium Gate skipped: required gate report(s) "
                        f"are stale after script mutation ({', '.join(_stale_reports)}). "
                        "Re-run the pipeline to refresh all gate reports before OFP."
                    )
                    logger.warning(
                        "Stage 14a — OFP skipped: stale reports detected: %s",
                        _stale_reports,
                    )
                    gate_summary["openai_final_premium"] = {
                        "passed":  False,
                        "skipped": True,
                        "reason":  f"stale supporting reports: {_stale_reports}",
                    }
                    status = "not_voice_ready_auto_retry_exhausted"

                # _ofp_skip: set when stale reports or finalization blocking prevents OFP
                _ofp_skip = bool(_stale_reports) or _finalize_result["blocking"]

                call_tracker.stage_start("openai_final_premium")
                logger.info("Stage 14a — OpenAI Final Premium Gate (combined)")
                # Final-review-input-hash guard: reuse the cached OFP report only
                # when ALL inputs that OFP reviews are identical to the stored run.
                # This covers narration chunks, youtube_metadata, recreated_dialogues,
                # run parameters, and all supporting gate reports — not just narration.
                _current_ofp_hash = _compute_final_review_input_hash(
                    script_final=script_final,
                    hinglish_level=inp.hinglish_level,
                    target_duration_min=inp.target_duration_min,
                    lint_report=lint_report,
                    similarity_report=similarity_report,
                    copyedit_report=copyedit_report,
                    quality_report=quality_report,
                    retention_report=retention_report,
                    originality_report=originality_report,
                    dialogue_report=dialogue_report,
                    metadata_report=metadata_report,
                    fact_lock=fact_lock,
                    blueprint=blueprint,
                    retention_blueprint=retention_blueprint,
                    originality_transformation_plan=originality_transformation_plan,
                )
                existing_ofp = _try_load_existing_json(
                    review_dir / "openai_final_premium_report.json",
                    "openai_final_premium",
                    episode_dir=episode_dir, prompt_check="openai_final_premium",
                )
                if existing_ofp is not None:
                    _stored_hash = existing_ofp.get("final_review_input_hash", "")
                    if _stored_hash and _stored_hash == _current_ofp_hash:
                        ofp_report = existing_ofp
                        logger.info(
                            "Stage 14a — OFP report reused (final_review_input_hash=%s matches)",
                            _current_ofp_hash,
                        )
                    else:
                        logger.info(
                            "Stage 14a — OFP report DISCARDED "
                            "(stored hash %r ≠ current %r) — rerunning",
                            _stored_hash, _current_ofp_hash,
                        )
                        existing_ofp = None  # force rerun below
                if _ofp_skip:
                    # Stale-report guard already set gate_summary["openai_final_premium"]
                    # and status above — synthesize a failed ofp_report so downstream
                    # code that reads ofp_report has a safe default.
                    ofp_report = {
                        "approved": False, "safe_to_voice": False,
                        "overall_score": 0, "chunk_repair_targets": [],
                        "skipped": True, "reason": "stale supporting reports",
                    }
                elif existing_ofp is None:
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
                            transformation_plan=originality_transformation_plan or None,
                        )
                        ofp_report["final_review_input_hash"] = _current_ofp_hash
                    except Exception as exc:
                        logger.error("OpenAI Final Premium Gate failed: %s", exc)
                        warnings.append(
                            f"OpenAI Final Premium Gate call failed: {exc}. "
                            "safe_to_voice=False — re-run with REUSE_EXISTING_STAGE_OUTPUTS=false to retry."
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
                    status = "auto_rebuild_required"   # repair/rebuild will run below
                    high_n = gate_summary["openai_final_premium"]["high_severity_issues"]
                    warnings.append(
                        f"OpenAI Final Premium Gate FAILED (overall_score="
                        f"{ofp_report.get('overall_score', 0)}, high_issues={high_n}, "
                        f"recommendation={ofp_report.get('recommendation', '?')}). "
                        "Auto-rebuild will be attempted. "
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
                        ofp_targets = _filter_openai_targets_to_blockers(
                            ofp_report.get("chunk_repair_targets", []),
                            gate_summary.get("python_preflight", {}),
                            quality_report,
                            retention_report,
                            metadata_report,
                            similarity_report,
                        )
                        if ofp_targets:
                            if len(ofp_targets) > settings.openai_repair_max_chunks:
                                logger.warning(
                                    "Stage 16 — %d targets exceed openai_repair_max_chunks=%d. "
                                    "Routing to auto-rebuild flow (adaptive mode).",
                                    len(ofp_targets), settings.openai_repair_max_chunks,
                                )
                                status = "auto_rebuild_required"
                                _rr_reports = {
                                    "openai_final_premium": ofp_report,
                                    "copyedit":             copyedit_report,
                                    "script_quality":       quality_report,
                                    "retention":            retention_report,
                                    "originality_safety":   originality_report,
                                    "recreated_dialogue":   dialogue_report,
                                    "metadata":             metadata_report,
                                }
                                script_final, _routing_plan, _rebuild_ran = (
                                    _run_routing_and_rebuild(
                                        script_draft=script_final,
                                        gate_reports=_rr_reports,
                                        fact_lock=fact_lock,
                                        blueprint=blueprint,
                                        retention_blueprint=retention_blueprint,
                                        originality_transformation_plan=originality_transformation_plan or {},
                                        case_glossary=case_glossary,
                                        hinglish_level=inp.hinglish_level,
                                        case_hint=getattr(inp, "case_hint", ""),
                                        review_dir=review_dir,
                                        script_dir=script_dir,
                                        warnings=warnings,
                                        gate_summary=gate_summary,
                                        root_cause_repair_attempts=root_cause_repair_attempts,
                                    )
                                )
                                if _routing_plan.get("route") == "stop_not_voice_ready":
                                    status = "not_voice_ready_auto_retry_exhausted"
                                    openai_repair_has_failures = True
                                else:
                                    # ── Post-rebuild gate refresh ──────────────
                                    # Refresh cheap deterministic gates and reload
                                    # all on-disk reports BEFORE calling OFP recheck,
                                    # so the recheck never sees stale pre-rebuild data.
                                    try:
                                        lint_report = run_hindi_text_lint(
                                            script_final,
                                            hinglish_level=inp.hinglish_level,
                                        )
                                        (review_dir / "hindi_text_lint_report.json").write_text(
                                            json.dumps(lint_report, ensure_ascii=False, indent=2),
                                            encoding="utf-8",
                                        )
                                        _sim_t = (
                                            episode_dir / "01-input" / "clean_transcript.txt"
                                        ).read_text(encoding="utf-8")
                                        similarity_report = run_text_similarity_check(
                                            source_transcript=_sim_t,
                                            script_draft=script_final,
                                        )
                                        (review_dir / "text_similarity_report.json").write_text(
                                            json.dumps(similarity_report, ensure_ascii=False, indent=2),
                                            encoding="utf-8",
                                        )

                                        # Python preflight guard — if rebuild introduced
                                        # new safety/blocking issues, skip OFP recheck.
                                        _rb_pf_blocking = False
                                        try:
                                            _rb_pf = run_python_preflight(
                                                script_draft=script_final,
                                                fact_lock=fact_lock,
                                                case_glossary=case_glossary,
                                                review_dir=review_dir,
                                                target_duration_min=inp.target_duration_min,
                                                hinglish_level=inp.hinglish_level,
                                                label="_after_auto_rebuild",
                                                source_transcript=_sim_t,
                                            )
                                            _rb_pf_blocking = _rb_pf.get("blocking", False)
                                            _rb_pf_counts = _rb_pf.get("severity_counts", {})
                                            gate_summary["python_preflight"].update({
                                                "passed":    _rb_pf.get("passed", False),
                                                "blocking":  _rb_pf_blocking,
                                                "high":      _rb_pf_counts.get("high", 0),
                                                "medium":    _rb_pf_counts.get("medium", 0),
                                                "low":       _rb_pf_counts.get("low", 0),
                                                "report":    "python_preflight_report_after_auto_rebuild.json",
                                                "rechecked": True,
                                            })
                                            if _rb_pf_blocking:
                                                logger.warning(
                                                    "Stage 16 — Post-rebuild Python preflight BLOCKING "
                                                    "(high=%d, medium=%d) — skipping OFP recheck.",
                                                    _rb_pf_counts.get("high", 0),
                                                    _rb_pf_counts.get("medium", 0),
                                                )
                                                status = "not_voice_ready_auto_retry_exhausted"
                                                openai_repair_has_failures = True
                                                warnings.append(
                                                    "Post-rebuild Python preflight BLOCKING "
                                                    f"(high={_rb_pf_counts.get('high', 0)}, "
                                                    f"medium={_rb_pf_counts.get('medium', 0)}) — "
                                                    "OFP recheck skipped. Automated retry exhausted."
                                                )
                                        except Exception as _rb_pf_exc:
                                            logger.error(
                                                "Post-rebuild Python preflight failed: %s — "
                                                "treating as blocking.",
                                                _rb_pf_exc,
                                            )
                                            _rb_pf_blocking = True
                                            status = "not_voice_ready_auto_retry_exhausted"
                                            openai_repair_has_failures = True

                                        if not _rb_pf_blocking:
                                            # ── Post-rebuild gate regeneration ────────
                                            # Pre-rebuild reports are stale. Regenerate
                                            # the two most OFP-critical content gates
                                            # (script_quality and hindi_copyedit) so the
                                            # recheck sees evidence from the rebuilt script.
                                            # lint and text_similarity are already fresh
                                            # (rerun above). Expensive gates that are not
                                            # rerun are tagged stale_after_rebuild=True so
                                            # OFP has explicit context about provenance.
                                            logger.info(
                                                "Stage 16 — regenerating script_quality + "
                                                "hindi_copyedit after rebuild"
                                            )
                                            try:
                                                quality_report = run_script_review(
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
                                                quality_report["refreshed_after_rebuild"] = True
                                                (review_dir / "final_script_quality_report.json").write_text(
                                                    json.dumps(quality_report, ensure_ascii=False, indent=2),
                                                    encoding="utf-8",
                                                )
                                                gate_summary["script_quality"] = {
                                                    "passed": quality_report.get("approved", False),
                                                    "scores": quality_report.get("scores", {}),
                                                    "refreshed_after_rebuild": True,
                                                }
                                            except Exception as _sq16_exc:
                                                logger.warning(
                                                    "Stage 16 post-rebuild script review failed: %s "
                                                    "— using pre-rebuild quality report",
                                                    _sq16_exc,
                                                )
                                                quality_report["stale_after_rebuild"] = True

                                            try:
                                                copyedit_report = run_hindi_copyedit_gate(
                                                    script_draft=script_final,
                                                    fact_lock=fact_lock,
                                                    blueprint=blueprint,
                                                    hinglish_level=inp.hinglish_level,
                                                    lint_report=lint_report,
                                                    review_dir=review_dir,
                                                )
                                                copyedit_report["refreshed_after_rebuild"] = True
                                                (review_dir / "hindi_copyedit_report.json").write_text(
                                                    json.dumps(copyedit_report, ensure_ascii=False, indent=2),
                                                    encoding="utf-8",
                                                )
                                            except Exception as _ce16_exc:
                                                logger.warning(
                                                    "Stage 16 post-rebuild copyedit gate failed: %s "
                                                    "— using pre-rebuild report",
                                                    _ce16_exc,
                                                )
                                                copyedit_report["stale_after_rebuild"] = True

                                            # Mark expensive Claude gates as potentially
                                            # stale — not rerun to avoid extra API cost,
                                            # but OFP receives explicit provenance signal.
                                            lint_report["refreshed_after_rebuild"] = True
                                            similarity_report["refreshed_after_rebuild"] = True
                                            _stale_gate_name_map = {
                                                "retention_quality":  retention_report,
                                                "originality_safety": originality_report,
                                                "recreated_dialogue": dialogue_report,
                                                "metadata_quality":   metadata_report,
                                            }
                                            for _gs_name, _sr in _stale_gate_name_map.items():
                                                if isinstance(_sr, dict):
                                                    _sr.setdefault("stale_after_rebuild", True)
                                                    # Mirror stale marker into gate_summary so
                                                    # _gate_passed_for_safe_to_voice sees it
                                                    gate_summary.setdefault(_gs_name, {}).update({
                                                        "stale_after_rebuild": True,
                                                        "passed": False,
                                                    })
                                            # Also update quality/copyedit if they failed to refresh
                                            if quality_report.get("stale_after_rebuild"):
                                                gate_summary.setdefault("script_quality", {}).update({
                                                    "stale_after_rebuild": True,
                                                    "passed": False,
                                                })
                                            if copyedit_report.get("stale_after_rebuild"):
                                                gate_summary.setdefault("hindi_copyedit", {}).update({
                                                    "stale_after_rebuild": True,
                                                    "passed": False,
                                                })
                                            ofp_rebuild_recheck = run_openai_final_premium_gate(
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
                                                label="_after_auto_rebuild",
                                                transformation_plan=originality_transformation_plan or None,
                                            )
                                            ofp_report = ofp_rebuild_recheck
                                            _ofp_rebuild_passed = (
                                                ofp_rebuild_recheck.get("approved", False)
                                                and ofp_rebuild_recheck.get("safe_to_voice", False)
                                            )
                                            gate_summary["openai_final_premium"].update({
                                                "passed":        _ofp_rebuild_passed,
                                                "approved":      ofp_rebuild_recheck.get("approved", False),
                                                "safe_to_voice": ofp_rebuild_recheck.get("safe_to_voice", False),
                                                "overall_score": ofp_rebuild_recheck.get("overall_score", 0),
                                                "recheck":       True,
                                                "recheck_label": "_after_auto_rebuild",
                                            })
                                            # script_approved only when OFP AND the
                                            # freshly-regenerated quality report both
                                            # approve. Final safe_to_voice at the bottom
                                            # of the pipeline is the authoritative check.
                                            _sq_approved_post = quality_report.get("approved", False)
                                            if _ofp_rebuild_passed and _sq_approved_post:
                                                status = "script_approved"
                                                logger.info(
                                                    "Stage 16 adaptive — OFP recheck + script_quality "
                                                    "both approved after auto-rebuild: PASSED"
                                                )
                                            elif _ofp_rebuild_passed and not _sq_approved_post:
                                                status = "not_voice_ready_auto_retry_exhausted"
                                                openai_repair_has_failures = True
                                                warnings.append(
                                                    "OFP recheck PASSED but refreshed script quality "
                                                    "review did not approve — not voice-ready. "
                                                    "Automated retry exhausted."
                                                )
                                                logger.warning(
                                                    "Stage 16 adaptive — OFP passed but script_quality "
                                                    "did not approve after rebuild"
                                                )
                                            else:
                                                status = "not_voice_ready_auto_retry_exhausted"
                                                openai_repair_has_failures = True
                                                warnings.append(
                                                    f"OFP recheck after auto-rebuild FAILED "
                                                    f"(overall_score="
                                                    f"{ofp_rebuild_recheck.get('overall_score', 0)}) — "
                                                    "automated retry exhausted — safe_to_voice=false."
                                                )
                                    except Exception as exc:
                                        logger.error(
                                            "OFP recheck after auto-rebuild failed: %s", exc
                                        )
                                        status = "not_voice_ready_auto_retry_exhausted"
                                        openai_repair_has_failures = True
                                        warnings.append(
                                            f"OFP recheck after auto-rebuild failed: {exc}. "
                                            "Automated retry exhausted — safe_to_voice=false."
                                        )
                            else:
                                _budget_reason = _soft_repair_budget_exceeded("openai")
                                if _budget_reason:
                                    _record_soft_budget_stop(
                                        "openai",
                                        _budget_reason,
                                        warnings,
                                        gate_summary,
                                        repair_cost_telemetry,
                                    )
                                    status = "not_voice_ready_auto_retry_exhausted"
                                    openai_repair_has_failures = True
                                    ofp_targets = []
                                if not ofp_targets:
                                    logger.info(
                                        "Stage 16 — OpenAI targeted repair has no current blocker targets."
                                    )
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
                                            "original content kept. "
                                            "Automated retry exhausted — safe_to_voice=false."
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
                                            source_transcript=clean_transcript_text,
                                        )
                                        _post_oai_pf_blocking = _post_oai_pf.get(
                                            "blocking", False
                                        )
                                        if _post_oai_pf_blocking:
                                            _po_counts = _post_oai_pf.get("severity_counts", {})
                                            post_openai_preflight_blockers = _preflight_blocker_trace(
                                                _post_oai_pf,
                                                python_fix_attempted=bool(
                                                    _post_oai_pf.get("metadata_python_fixes_applied")
                                                ),
                                                claude_repair_attempted=True,
                                                openai_repair_attempted=True,
                                            )
                                            status = "not_voice_ready_auto_retry_exhausted"
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
                                                "Post-OpenAI-repair Python preflight is BLOCKING — "
                                                "auto-retry exhausted. safe_to_voice=False. OFP recheck skipped. "
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
                                                transformation_plan=originality_transformation_plan or None,
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
                                                status = "not_voice_ready_auto_retry_exhausted"
                                                logger.warning(
                                                    "OpenAI Final Premium Gate recheck: still FAILED"
                                                )
                                                warnings.append(
                                                    f"OpenAI Final Premium Gate recheck FAILED "
                                                    f"(overall_score="
                                                    f"{ofp_recheck.get('overall_score', 0)}) — "
                                                    "automated retry exhausted — safe_to_voice=false. "
                                                    "See 04-review/"
                                                    "openai_final_premium_report_after_repair.json."
                                                )
                                        except Exception as exc:
                                            logger.error(
                                                "OpenAI Final Premium Gate recheck failed: %s",
                                                exc,
                                            )
                                            warnings.append(
                                                f"OpenAI Final Premium Gate recheck call failed: {exc}. "
                                                "Automated retry exhausted — safe_to_voice=false."
                                            )
                                            openai_repair_has_failures = True

                                except Exception as exc:
                                    logger.error("OpenAI targeted repair failed: %s", exc)
                                    warnings.append(
                                        f"OpenAI targeted chunk repair call failed: {exc}. "
                                        "Automated retry exhausted — do not run ElevenLabs."
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
                                f"OpenAI Premium Hindi Editor Gate call failed: {exc}. "
                                "Gate marked failed — auto-rebuild will be attempted in Stage 16."
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
                        status = "auto_rebuild_required"   # Stage 16 repair will run
                        high_n = gate_summary["openai_premium_hindi_editor"]["high_severity_issues"]
                        warnings.append(
                            f"OpenAI Hindi editor gate FAILED (overall_score="
                            f"{ohe_report.get('overall_score', 0)}, high_issues={high_n}). "
                            "Auto-rebuild will be attempted. "
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
                                f"OpenAI Originality/YouTube Risk Gate call failed: {exc}. "
                                "Gate marked failed — auto-rebuild will be attempted in Stage 16."
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
                        status = "auto_rebuild_required"   # Stage 16 repair will run
                        fixes = oyr_report.get("required_fixes", [])
                        warnings.append(
                            "OpenAI originality/YT risk gate FAILED. Required fixes: "
                            + (safe_join_report_items(fixes, limit=3) if fixes else "see report")
                            + (f" (+{len(fixes)-3} more)" if len(fixes) > 3 else "")
                            + " — Auto-rebuild will be attempted. "
                            "See 04-review/openai_originality_youtube_risk_report.json."
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
                        all_oai_targets = _filter_openai_targets_to_blockers(
                            all_oai_targets,
                            gate_summary.get("python_preflight", {}),
                            quality_report,
                            retention_report,
                            metadata_report,
                            similarity_report,
                        )

                        if all_oai_targets:
                            if len(all_oai_targets) > settings.openai_repair_max_chunks:
                                logger.warning(
                                    "Stage 16 — %d targets exceed openai_repair_max_chunks=%d. "
                                    "Routing to auto-rebuild flow (always mode).",
                                    len(all_oai_targets), settings.openai_repair_max_chunks,
                                )
                                status = "auto_rebuild_required"
                                _rr_reports_always = {
                                    "openai_final_premium":           ofp_report,
                                    "openai_premium_hindi_editor":    ohe_report,
                                    "openai_originality_youtube_risk": oyr_report,
                                    "copyedit":                       copyedit_report,
                                    "script_quality":                 quality_report,
                                    "retention":                      retention_report,
                                    "originality_safety":             originality_report,
                                    "recreated_dialogue":             dialogue_report,
                                    "metadata":                       metadata_report,
                                }
                                script_final, _routing_plan_always, _rebuild_ran_always = (
                                    _run_routing_and_rebuild(
                                        script_draft=script_final,
                                        gate_reports=_rr_reports_always,
                                        fact_lock=fact_lock,
                                        blueprint=blueprint,
                                        retention_blueprint=retention_blueprint,
                                        originality_transformation_plan=originality_transformation_plan or {},
                                        case_glossary=case_glossary,
                                        hinglish_level=inp.hinglish_level,
                                        case_hint=getattr(inp, "case_hint", ""),
                                        review_dir=review_dir,
                                        script_dir=script_dir,
                                        warnings=warnings,
                                        gate_summary=gate_summary,
                                        root_cause_repair_attempts=root_cause_repair_attempts,
                                    )
                                )
                                if _routing_plan_always.get("route") == "stop_not_voice_ready":
                                    status = "not_voice_ready_auto_retry_exhausted"
                                    openai_repair_has_failures = True
                                else:
                                    # Refresh lint then recheck all failing gates once
                                    try:
                                        lint_report = run_hindi_text_lint(
                                            script_final,
                                            hinglish_level=inp.hinglish_level,
                                        )
                                    except Exception as _exc:
                                        logger.warning("Lint refresh after rebuild failed: %s", _exc)

                                    if not ohe_gate_passed:
                                        try:
                                            ohe_rb_recheck = run_openai_premium_hindi_editor_gate(
                                                script_draft=script_final,
                                                fact_lock=fact_lock,
                                                blueprint=blueprint,
                                                hinglish_level=inp.hinglish_level,
                                                lint_report=lint_report,
                                                copyedit_report=copyedit_report,
                                                quality_report=quality_report,
                                                review_dir=review_dir,
                                            )
                                            ohe_rb_passed = (
                                                ohe_rb_recheck.get("approved", False)
                                                and ohe_rb_recheck.get("safe_to_voice", False)
                                            )
                                            gate_summary["openai_premium_hindi_editor"].update({
                                                "passed":        ohe_rb_passed,
                                                "overall_score": ohe_rb_recheck.get("overall_score", 0),
                                                "recheck":       True,
                                                "recheck_label": "_after_auto_rebuild",
                                            })
                                            if not ohe_rb_passed:
                                                warnings.append(
                                                    f"OHE recheck after auto-rebuild FAILED "
                                                    f"(overall_score={ohe_rb_recheck.get('overall_score', 0)})."
                                                )
                                        except Exception as exc:
                                            logger.error(
                                                "OHE recheck after rebuild failed: %s", exc
                                            )
                                            warnings.append(
                                                f"OpenAI Hindi editor recheck after rebuild failed: {exc}."
                                            )
                                            openai_repair_has_failures = True

                                    if not oyr_gate_passed:
                                        try:
                                            clean_for_rb_recheck = (
                                                episode_dir / "01-input" / "clean_transcript.txt"
                                            ).read_text(encoding="utf-8")
                                            oyr_rb_recheck = run_openai_originality_youtube_risk_gate(
                                                script_draft=script_final,
                                                source_transcript=clean_for_rb_recheck,
                                                fact_lock=fact_lock,
                                                blueprint=blueprint,
                                                claude_originality_report=originality_report,
                                                claude_metadata_report=metadata_report,
                                                claude_dialogue_report=dialogue_report,
                                                review_dir=review_dir,
                                            )
                                            oyr_rb_passed = (
                                                oyr_rb_recheck.get("approved", False)
                                                and oyr_rb_recheck.get("safe_to_voice", False)
                                            )
                                            gate_summary["openai_originality_youtube_risk"].update({
                                                "passed":        oyr_rb_passed,
                                                "recheck":       True,
                                                "recheck_label": "_after_auto_rebuild",
                                            })
                                            if not oyr_rb_passed:
                                                warnings.append(
                                                    "OYR recheck after auto-rebuild FAILED."
                                                )
                                        except Exception as exc:
                                            logger.error(
                                                "OYR recheck after rebuild failed: %s", exc
                                            )
                                            warnings.append(
                                                f"OYR recheck after rebuild failed: {exc}."
                                            )
                                            openai_repair_has_failures = True

                                    # Final OFP recheck
                                    try:
                                        ofp_rb_recheck = run_openai_final_premium_gate(
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
                                            label="_after_auto_rebuild",
                                            transformation_plan=originality_transformation_plan or None,
                                        )
                                        ofp_report = ofp_rb_recheck
                                        _ofp_rb_passed = (
                                            ofp_rb_recheck.get("approved", False)
                                            and ofp_rb_recheck.get("safe_to_voice", False)
                                        )
                                        gate_summary["openai_final_premium"].update({
                                            "passed":        _ofp_rb_passed,
                                            "approved":      ofp_rb_recheck.get("approved", False),
                                            "safe_to_voice": ofp_rb_recheck.get("safe_to_voice", False),
                                            "overall_score": ofp_rb_recheck.get("overall_score", 0),
                                            "recheck":       True,
                                            "recheck_label": "_after_auto_rebuild",
                                        })
                                        if _ofp_rb_passed:
                                            status = "script_approved"
                                            logger.info(
                                                "Stage 16 always — OFP recheck after auto-rebuild: PASSED"
                                            )
                                        else:
                                            status = "not_voice_ready_auto_retry_exhausted"
                                            openai_repair_has_failures = True
                                            warnings.append(
                                                f"OFP recheck after auto-rebuild FAILED "
                                                f"(overall_score="
                                                f"{ofp_rb_recheck.get('overall_score', 0)}) — "
                                                "automated retry exhausted — safe_to_voice=false."
                                            )
                                    except Exception as exc:
                                        logger.error(
                                            "OFP recheck after auto-rebuild failed: %s", exc
                                        )
                                        status = "not_voice_ready_auto_retry_exhausted"
                                        openai_repair_has_failures = True
                                        warnings.append(
                                            f"OFP recheck after auto-rebuild (always mode) call failed: {exc}. "
                                            "Automated retry exhausted — safe_to_voice=false."
                                        )
                            else:
                                _budget_reason = _soft_repair_budget_exceeded("openai")
                                if _budget_reason:
                                    _record_soft_budget_stop(
                                        "openai",
                                        _budget_reason,
                                        warnings,
                                        gate_summary,
                                        repair_cost_telemetry,
                                    )
                                    status = "not_voice_ready_auto_retry_exhausted"
                                    openai_repair_has_failures = True
                                    all_oai_targets = []
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
                                            "original content kept. "
                                            "Automated retry exhausted — safe_to_voice=false."
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
                                                status = "not_voice_ready_auto_retry_exhausted"
                                                warnings.append(
                                                    f"OpenAI Hindi editor recheck FAILED after targeted repair "
                                                    f"(overall_score={ohe_recheck.get('overall_score', 0)}) — "
                                                    "automated retry exhausted — safe_to_voice=false."
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
                                                status = "not_voice_ready_auto_retry_exhausted"
                                                fixes_r = oyr_recheck.get("required_fixes", [])
                                                warnings.append(
                                                    "OpenAI originality/YT risk recheck FAILED after targeted repair. "
                                                    "Required fixes: "
                                                    + (safe_join_report_items(fixes_r, limit=3) if fixes_r else "see report")
                                                    + (f" (+{len(fixes_r)-3} more)" if len(fixes_r) > 3 else "")
                                                    + " — automated retry exhausted — safe_to_voice=false."
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
                                            source_transcript=clean_transcript_text,
                                        )
                                        if _post_oai_pf_a.get("blocking", False):
                                            _poa_counts = _post_oai_pf_a.get("severity_counts", {})
                                            post_openai_preflight_blockers = _preflight_blocker_trace(
                                                _post_oai_pf_a,
                                                python_fix_attempted=bool(
                                                    _post_oai_pf_a.get("metadata_python_fixes_applied")
                                                ),
                                                claude_repair_attempted=True,
                                                openai_repair_attempted=True,
                                            )
                                            status = "not_voice_ready_auto_retry_exhausted"
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
                                                "Post-OpenAI-repair Python preflight is BLOCKING — "
                                                "auto-retry exhausted. safe_to_voice=False. "
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
                                        f"OpenAI targeted chunk repair call failed: {exc}. "
                                        "Automated retry exhausted — do not run ElevenLabs."
                                    )
                                    openai_repair_has_failures = True

        else:
            # OpenAI gates inactive — determine exact reason so gate_summary is accurate
            if _post_repair_preflight_blocking:
                _skip_reason = "post_repair_python_preflight_blocking"
                _skip_detail = (
                    "Post-repair Python preflight is still blocking; "
                    "OpenAI Final Premium Gate skipped to avoid reviewing unsafe/unready script."
                )
                # Status was already set to not_voice_ready_auto_retry_exhausted at
                # the preflight check; reinforce it and mark automation_status specifically.
                if status not in ("needs_human_review",):
                    status = "not_voice_ready_auto_retry_exhausted"
                gate_summary["automation_status"] = "blocked_before_openai"
                warnings.append(
                    "Blocked before OpenAI because deterministic Python preflight still has "
                    "blocking issues. See 04-review/python_preflight_report_after_repair.json."
                )
                logger.warning("OpenAI gates skipped: %s", _skip_reason)
            elif retention_blocked_before_openai:
                _skip_reason = "retention_blocked_before_openai"
                _skip_detail = (
                    "Retention quality remains broadly below threshold after one repair pass; "
                    "OpenAI Final Premium Gate skipped to avoid broad expensive repair."
                )
                status = "not_voice_ready_auto_retry_exhausted"
                gate_summary["automation_status"] = "blocked_before_openai"
                logger.warning("OpenAI gates skipped: %s", _skip_reason)
            elif settings.quality_mode != "premium_final":
                _skip_reason = f"quality_mode={settings.quality_mode}"
                _skip_detail = (
                    f"quality_mode is '{settings.quality_mode}', not 'premium_final' — "
                    "OpenAI Final Premium Gate requires premium_final mode."
                )
                status = "needs_human_review"
                logger.info("OpenAI gates skipped (%s) — forcing needs_human_review", _skip_reason)
            elif settings.openai_review_policy == "disabled":
                _skip_reason = f"openai_review_policy={settings.openai_review_policy}"
                _skip_detail = (
                    "OPENAI_REVIEW_POLICY=disabled — OpenAI Final Premium Gate explicitly disabled."
                )
                status = "needs_human_review"
                logger.info("OpenAI gates skipped (%s) — forcing needs_human_review", _skip_reason)
            else:
                # openai_review_enabled=false is the only remaining case
                _skip_reason = "OPENAI_REVIEW_ENABLED=false"
                _skip_detail = (
                    "OPENAI_REVIEW_ENABLED is false — OpenAI Final Premium Gate disabled by config."
                )
                status = "needs_human_review"
                logger.info("OpenAI gates skipped (%s) — forcing needs_human_review", _skip_reason)

            gate_summary["openai_final_premium"] = {
                "passed": False, "skipped": True,
                "reason": _skip_detail,
                "skip_reason_code": _skip_reason,
            }
            gate_summary["openai_premium_hindi_editor"] = {
                "passed": True, "skipped": True,
                "reason": f"{_skip_reason} — legacy gate skipped",
            }
            gate_summary["openai_originality_youtube_risk"] = {
                "passed": True, "skipped": True,
                "reason": f"{_skip_reason} — legacy gate skipped",
            }
            if _skip_reason not in {"post_repair_python_preflight_blocking", "retention_blocked_before_openai"}:
                warnings.append(
                    f"OpenAI Final Premium Gate skipped ({_skip_reason}). "
                    "safe_to_voice=False — do not run ElevenLabs until the final premium gate passes."
                )

        # ── Finalize status from repair failure flags ──────────────────────────
        # Any repair failure that hasn't already updated status must be reflected
        # here — BEFORE all_gates_passed / safe_to_voice / trace are computed.
        # This ensures the trace always captures the final status value.
        if copyedit_repair_has_failures:
            if not any("Copyedit repair had failures" in w for w in warnings):
                warnings.append(
                    "Copyedit repair had failures — original content kept in affected chunks. "
                    "Do not run ElevenLabs. Automated retry exhausted — safe_to_voice=false. "
                    "See 04-review/hindi_copyedit_repair_report.json."
                )
            if status not in ("needs_human_review", "not_voice_ready_auto_retry_exhausted"):
                status = "not_voice_ready_auto_retry_exhausted"

        # Belt-and-suspenders: any remaining repair failure flag that slipped through
        # inline status updates also gets a terminal status here.
        if not (
            (not repair_has_failures)
            and (not copyedit_repair_has_failures)
            and (not metadata_repair_has_failures)
            and (not retention_repair_has_failures)
            and (not openai_repair_has_failures)
        ) and status not in ("needs_human_review", "not_voice_ready_auto_retry_exhausted"):
            status = "not_voice_ready_auto_retry_exhausted"

        # ── Final gate summary + safe_to_voice ────────────────────────────────
        # all_gates_passed checks ONLY the REQUIRED_SAFE_TO_VOICE_GATES allowlist.
        # gate_summary also contains telemetry entries (repair_routing, auto_fix,
        # pre_oai_repair, repair_telemetry, repair_failures, automation_status)
        # that do not carry a `passed` field; iterating over all items would
        # cause those entries to falsely block approval.
        #
        # python_preflight is evaluated via blocking (not passed) because passed=False
        # for any issue including low-only warnings, but low warnings must not block
        # safe_to_voice. See _gate_passed_for_safe_to_voice for the full decision table.
        all_gates_passed = all(
            _gate_passed_for_safe_to_voice(name, gate_summary.get(name, {"passed": False}))
            for name in REQUIRED_SAFE_TO_VOICE_GATES
        )

        # ── Final authoritative status guard ──────────────────────────────────
        # Intermediate branches (Stage 6, Stage 16 OFP recheck) may set
        # status="script_approved" before all required gates have been evaluated.
        # Guard here: if the required gate allowlist does not fully pass, downgrade
        # status so safe_to_voice cannot be True from an intermediate approval alone.
        if status == "script_approved" and not all_gates_passed:
            status = "not_voice_ready_auto_retry_exhausted"
            warnings.append(
                "Final gate recheck: status downgraded from script_approved — "
                "not all required gates passed. Check gate_summary for details."
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

        # Originality transformation plan must exist (unless feature disabled/batch mode)
        _transformation_gate_ok = gate_summary.get(
            "originality_transformation", {}
        ).get("passed", True)

        # High-risk source similarity check (verbatim long phrases from source transcript)
        _sim_high_risk = similarity_report.get("high_risk_matches", 0)
        _sim_max_allowed = settings.source_similarity_max_high_risk_matches
        _similarity_ok = _sim_high_risk <= _sim_max_allowed
        if not _similarity_ok:
            logger.warning(
                "safe_to_voice blocked: high_risk_matches=%d exceeds "
                "SOURCE_SIMILARITY_MAX_HIGH_RISK_MATCHES=%d",
                _sim_high_risk, _sim_max_allowed,
            )
            if not any("high_risk_matches" in w for w in warnings):
                warnings.append(
                    f"Text similarity check found {_sim_high_risk} high-risk verbatim match(es) "
                    f"(allowed: {_sim_max_allowed}). "
                    "safe_to_voice=False — reduce source-verbatim content before audio generation. "
                    "See 04-review/text_similarity_report.json."
                )

        safe_to_voice = (
            (status == "script_approved")
            and all_gates_passed
            and no_repair_failures
            and _pf_gate_ok             # python_preflight must not be blocking
            and _transformation_gate_ok  # originality_transformation_plan must exist
            and _similarity_ok           # high-risk source copy must be within limit
        )

        if not _transformation_gate_ok and not any("ransformation" in w for w in warnings):
            warnings.append(
                "Originality Transformation Plan missing or failed. "
                "safe_to_voice=False — run the pipeline again or check "
                "02-facts/_originality_transformation_raw_response.txt."
            )

        # Stamp safe_to_voice into gate_summary for single-field API inspection
        gate_summary["safe_to_voice"] = safe_to_voice  # type: ignore[assignment]

        # ── Automation status + repair telemetry ──────────────────────────────
        gate_summary["automation_status"] = status
        _rr_tel = gate_summary.get("repair_routing", {})
        _af_tel = gate_summary.get("auto_fix", {})
        repair_cost_telemetry["root_cause_repair_attempts"] = dict(root_cause_repair_attempts)
        repair_cost_telemetry["estimated_claude_calls_saved"] = max(
            repair_cost_telemetry.get("estimated_claude_calls_saved", 0),
            _rr_tel.get("estimated_model_calls_saved", 0),
        )
        repair_cost_telemetry["python_blocked_before_openai_final"] = (
            repair_cost_telemetry.get("python_blocked_before_openai_final", False)
            or _post_repair_preflight_blocking
            or retention_blocked_before_openai
        )
        gate_summary["repair_telemetry"] = {
            "repair_route":                _rr_tel.get("route", "none"),
            "root_cause_count":            _rr_tel.get("root_cause_count", 0),
            "python_auto_fixes_count":     _af_tel.get("python_fixes_count", 0),
            "claude_grouped_repair_count": _rr_tel.get("claude_targets_count", 0),
            "estimated_model_calls_saved": _rr_tel.get("estimated_model_calls_saved", 0),
            "auto_rebuild_ran":            _af_tel.get("rebuild_ran", False),
            "auto_rebuild_chunks":         _af_tel.get("rebuild_chunks", 0),
            "avoided_openai_bulk_repair":  bool(
                _rr_tel.get("route") in (
                    "python_only", "claude_grouped_repair", "auto_rebuild_required"
                )
            ),
            **repair_cost_telemetry,
        }

        # ── Safe-to-voice decision trace (04-review/) ─────────────────────────
        _stv_blocking: list[str] = []
        if status not in ("script_approved",):
            _stv_blocking.append(f"status={status}")
        if not all_gates_passed:
            _failing_gate_names = [
                _gn for _gn, _gs in gate_summary.items()
                if isinstance(_gs, dict) and _gs.get("passed") is False
            ]
            _stv_blocking.append(f"gates_failed={_failing_gate_names}")
        if not no_repair_failures:
            _stv_blocking.append("repair_failures=True")
        if not _pf_gate_ok:
            _stv_blocking.append("python_preflight=blocking")
        if not _transformation_gate_ok:
            _stv_blocking.append("originality_transformation_plan=missing")
        if not _similarity_ok:
            _stv_blocking.append(
                f"text_similarity_high_risk={_sim_high_risk}>{_sim_max_allowed}"
            )
        # Collect unresolved high-severity issues still failing after all repairs
        _unresolved: list[str] = []
        for _gn, _gs in gate_summary.items():
            if not isinstance(_gs, dict):
                continue
            if _gs.get("passed") is False and not _gs.get("skipped"):
                _unresolved.append(_gn)
        _repair_tel = gate_summary.get("repair_telemetry", {})
        _pf_gs = gate_summary.get("python_preflight", {})
        _ofp_gs = gate_summary.get("openai_final_premium", {})
        _stv_trace = {
            "safe_to_voice":            safe_to_voice,
            "status":                   status,
            "automation_status":        gate_summary.get("automation_status", status),
            "elevenlabs_ready":         safe_to_voice,
            "elevenlabs_allowed":       safe_to_voice,   # explicit alias for downstream consumers
            "blocking_reasons":         _stv_blocking,
            "unresolved_issues":        _unresolved,     # gates still failing after all repairs
            # ── OpenAI gate diagnostics (Task 8) ────────────────────────────
            "openai_review_enabled":             settings.openai_review_enabled,
            "openai_gate_requested":             (
                settings.openai_review_enabled
                and settings.quality_mode == "premium_final"
                and settings.openai_review_policy != "disabled"
            ),
            "openai_gate_ran":                   _openai_gates_active,
            "openai_gate_skipped_reason":        (
                _ofp_gs.get("skip_reason_code")
                if _ofp_gs.get("skipped") else None
            ),
            # ── Python preflight diagnostics (Task 8) ───────────────────────
            "post_repair_python_preflight_blocking": _post_repair_preflight_blocking,
            "post_openai_preflight_blockers": post_openai_preflight_blockers,
            "python_preflight_high_count":       _pf_gs.get("high", 0),
            "python_preflight_medium_count":     _pf_gs.get("medium", 0),
            "python_preflight_low_count":        _pf_gs.get("low", 0),
            "python_preflight_rechecked":        _pf_gs.get("rechecked", False),
            "pre_oai_repair_ran":                gate_summary.get(
                "pre_oai_repair", {}
            ).get("ran", False),
            # ── Top-level shortcuts (also present inside repair_telemetry) ───
            "repair_route":             _repair_tel.get("repair_route", "none"),
            "python_auto_fixes_count":  _repair_tel.get("python_auto_fixes_count", 0),
            "auto_rebuild_ran":         _repair_tel.get("auto_rebuild_ran", False),
            "gate_scores": {
                "openai_final_premium_overall": _ofp_gs.get("overall_score"),
                "openai_final_premium_passed":  _ofp_gs.get("passed"),
                "hindi_copyedit_passed": gate_summary.get(
                    "hindi_copyedit", {}
                ).get("passed"),
                "retention_passed": gate_summary.get(
                    "retention_quality", {}
                ).get("passed"),
                "metadata_passed": gate_summary.get(
                    "metadata_quality", {}
                ).get("passed"),
                "originality_passed": gate_summary.get(
                    "originality_safety", {}
                ).get("passed"),
            },
            "repair_telemetry": _repair_tel,
        }
        try:
            (review_dir / "safe_to_voice_decision_trace.json").write_text(
                json.dumps(_stv_trace, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "safe_to_voice_decision_trace.json written (safe_to_voice=%s, status=%s)",
                safe_to_voice, status,
            )
        except Exception as _exc:
            logger.warning("Could not write safe_to_voice_decision_trace.json: %s", _exc)

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
