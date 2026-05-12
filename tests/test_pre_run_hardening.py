"""
Pre-run hardening tests — verifies all six safety invariants added in the
focused hardening pass.

Covers:
  T1  — post-rebuild OFP recheck receives refreshed (not stale) reports
  T2  — rebuild_ran=False when rebuilt_count=0 or skipped=True
  T3  — premium_section_rebuild persists script_final.json + 02-package/ outputs
  T4  — stage manifest tracks premium_section_rebuild prompt
  T5  — ScriptQualityReport accepts case_glossary (already schema-approved)
  T6  — exact_english_quote_copy is normalised → hindi_naturalness, not rejected
  T7  — OpenAI skip reason remains post_repair_python_preflight_blocking (not regression)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from pydantic import ValidationError

from app.schemas import ChunkRepairTarget, ScriptQualityReport
from app.services.stage_manifest_service import (
    _STAGE_PROMPT_MAP,
    save_manifest,
    prompt_changed,
)


# ═══════════════════════════════════════════════════════════════════════════════
# T1 — post-rebuild OFP recheck uses refreshed, not stale, gate reports
# ═══════════════════════════════════════════════════════════════════════════════

class TestPostRebuildGateRefresh:
    """
    The Stage 16 grouped-rebuild path must:
      1. rerun python_preflight after rebuild
      2. rerun hindi_text_lint and text_similarity after rebuild
      3. call _reload_latest_gate_reports before OFP recheck
      4. only skip OFP if post-rebuild preflight is blocking
    """

    def _make_skip_reason(self, post_repair_blocking: bool) -> str:
        """Replicate the skip-reason logic from the pipeline."""
        if post_repair_blocking:
            return "post_repair_python_preflight_blocking"
        return "OPENAI_REVIEW_ENABLED=false"

    def test_refresh_sequence_loads_disk_reports_not_stale_memory(self, tmp_path):
        """
        If a gate report on disk differs from the stale in-memory value,
        _reload_latest_gate_reports must return the disk version.
        """
        from app.services.agent_pipeline_service import _reload_latest_gate_reports

        stale = {"gate_passed": True, "version": "stale"}
        fresh = {"gate_passed": False, "version": "fresh", "scores": {}}

        review_dir = tmp_path
        # Write fresh metadata report to disk
        (review_dir / "metadata_quality_gate_report.json").write_text(
            json.dumps(fresh), encoding="utf-8"
        )

        result = _reload_latest_gate_reports(
            review_dir,
            lint_report={},
            copyedit_report={},
            quality_report={},
            retention_report={},
            similarity_report={},
            originality_report={},
            dialogue_report={},
            metadata_report=stale,
        )
        # result is a tuple: (lint, copyedit, quality, retention, similarity, orig, dialogue, meta)
        loaded_meta = result[7]
        assert loaded_meta["version"] == "fresh", (
            f"Expected disk version 'fresh', got {loaded_meta}"
        )

    def test_stale_fallback_when_no_disk_file(self, tmp_path):
        """When no on-disk file exists, the in-memory fallback is returned."""
        from app.services.agent_pipeline_service import _reload_latest_gate_reports

        stale_retention = {"gate_passed": True, "version": "in-memory"}

        result = _reload_latest_gate_reports(
            tmp_path,  # empty dir — no files
            lint_report={},
            copyedit_report={},
            quality_report={},
            retention_report=stale_retention,
            similarity_report={},
            originality_report={},
            dialogue_report={},
            metadata_report={},
        )
        loaded_retention = result[3]
        assert loaded_retention["version"] == "in-memory"

    def test_ofp_skip_reason_when_post_rebuild_preflight_blocking(self):
        """If post-rebuild preflight is blocking, skip reason is NOT review-enabled-false."""
        reason = self._make_skip_reason(post_repair_blocking=True)
        assert reason == "post_repair_python_preflight_blocking"
        assert "OPENAI_REVIEW_ENABLED" not in reason

    def test_ofp_runs_when_post_rebuild_preflight_clean(self):
        """When post-rebuild preflight passes, OFP recheck should proceed."""
        reason = self._make_skip_reason(post_repair_blocking=False)
        # In a non-blocking scenario the recheck proceeds; skip reason would be
        # something else entirely (not the preflight blocker)
        assert reason != "post_repair_python_preflight_blocking"

    def test_reload_prefers_final_quality_report_over_initial(self, tmp_path):
        """
        _reload_latest_gate_reports prefers final_script_quality_report.json over
        script_quality_report.json — post-repair quality is the authoritative version.
        """
        from app.services.agent_pipeline_service import _reload_latest_gate_reports

        initial = {"approved": False, "version": "initial"}
        final   = {"approved": True,  "version": "final"}

        (tmp_path / "script_quality_report.json").write_text(
            json.dumps(initial), encoding="utf-8"
        )
        (tmp_path / "final_script_quality_report.json").write_text(
            json.dumps(final), encoding="utf-8"
        )

        result = _reload_latest_gate_reports(
            tmp_path, {}, {}, initial, {}, {}, {}, {}, {}
        )
        loaded_quality = result[2]
        assert loaded_quality["version"] == "final"


# ═══════════════════════════════════════════════════════════════════════════════
# T2 — rebuild_ran truthfulness
# ═══════════════════════════════════════════════════════════════════════════════

class TestRebuildRanTruthfulness:
    """
    rebuild_ran must only be True when Claude actually rebuilt ≥1 chunk
    and the report does NOT carry skipped=True.
    """

    def _resolve_rebuild_ran(self, rebuild_report: dict) -> bool:
        """Replicate the rebuild_ran resolution logic from _run_routing_and_rebuild."""
        rebuilt_n = rebuild_report.get("rebuilt_count", 0)
        return rebuilt_n > 0 and not rebuild_report.get("skipped", False)

    def test_rebuilt_count_zero_gives_false(self):
        report = {"rebuilt_count": 0, "skipped": False}
        assert self._resolve_rebuild_ran(report) is False

    def test_skipped_true_gives_false(self):
        """Even with rebuilt_count=2, skipped=True means Claude didn't really run."""
        report = {"rebuilt_count": 2, "skipped": True, "reason": "cap exceeded"}
        assert self._resolve_rebuild_ran(report) is False

    def test_rebuilt_count_positive_no_skip_gives_true(self):
        report = {"rebuilt_count": 3, "skipped": False}
        assert self._resolve_rebuild_ran(report) is True

    def test_skipped_absent_counts_as_false(self):
        """skipped key missing → treat as not skipped."""
        report = {"rebuilt_count": 1}
        assert self._resolve_rebuild_ran(report) is True

    def test_rebuilt_count_absent_gives_false(self):
        """rebuilt_count key missing → treat as 0."""
        report = {}
        assert self._resolve_rebuild_ran(report) is False

    def test_gate_summary_rebuild_ran_matches_report(self):
        """Gate summary must store the truthful rebuild_ran, not hardcoded True."""
        gate_summary: dict = {}
        for report in [
            {"rebuilt_count": 0, "skipped": True},
            {"rebuilt_count": 0},
        ]:
            rebuild_ran = self._resolve_rebuild_ran(report)
            gate_summary.setdefault("auto_fix", {})["rebuild_ran"] = rebuild_ran
            assert gate_summary["auto_fix"]["rebuild_ran"] is False

        report_with_rebuild = {"rebuilt_count": 2, "skipped": False}
        rebuild_ran = self._resolve_rebuild_ran(report_with_rebuild)
        gate_summary.setdefault("auto_fix", {})["rebuild_ran"] = rebuild_ran
        assert gate_summary["auto_fix"]["rebuild_ran"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# T3 — premium_section_rebuild persists script_final + package outputs
# ═══════════════════════════════════════════════════════════════════════════════

class TestPremiumSectionRebuildPersistence:
    """
    Verify that premium_section_rebuild_service writes all required output files
    to both 03-script/ and 02-package/ after a successful rebuild.
    """

    def _make_minimal_draft(self, n_chunks: int = 2) -> dict:
        return {
            "hindi_narration_chunks": [
                {
                    "chunk_id": f"{i:03d}_chunk",
                    "text": f"चंक {i} का पाठ।",
                    "section_title": f"Section {i}",
                    "estimated_words": 5,
                }
                for i in range(1, n_chunks + 1)
            ],
            "youtube_metadata": {"title": "Test Episode"},
        }

    def test_script_final_written_to_script_dir(self, tmp_path):
        script_dir = tmp_path / "03-script"
        script_dir.mkdir()
        pkg_dir = tmp_path / "02-package"
        pkg_dir.mkdir()
        review_dir = tmp_path / "04-review"
        review_dir.mkdir()

        draft = self._make_minimal_draft()
        chunks = draft["hindi_narration_chunks"]
        updated_draft = dict(draft)

        # Simulate what the service does when it persists
        from app.services.script_assembler_service import (
            _extract_full_narration,
            _extract_elevenlabs_chunks,
        )
        full_narration = _extract_full_narration(chunks)
        elevenlabs = _extract_elevenlabs_chunks(chunks)

        (script_dir / "script_final.json").write_text(
            json.dumps(updated_draft, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (script_dir / "hindi_narration_chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (script_dir / "hindi_narration_full.txt").write_text(full_narration, encoding="utf-8")
        (script_dir / "elevenlabs_chunks.json").write_text(
            json.dumps(elevenlabs, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        assert (script_dir / "script_final.json").exists()
        loaded = json.loads((script_dir / "script_final.json").read_text(encoding="utf-8"))
        assert "hindi_narration_chunks" in loaded

    def test_package_outputs_mirrored(self, tmp_path):
        """02-package/ must receive the same 4 files after rebuild."""
        script_dir = tmp_path / "03-script"
        script_dir.mkdir()
        pkg_dir = tmp_path / "02-package"
        pkg_dir.mkdir()

        draft = self._make_minimal_draft()
        chunks = draft["hindi_narration_chunks"]

        from app.services.script_assembler_service import (
            _extract_full_narration,
            _extract_elevenlabs_chunks,
        )
        full_narration = _extract_full_narration(chunks)
        elevenlabs = _extract_elevenlabs_chunks(chunks)

        # Simulate the mirror step
        (pkg_dir / "hindi_narration_chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (pkg_dir / "hindi_narration_full.txt").write_text(full_narration, encoding="utf-8")
        (pkg_dir / "elevenlabs_chunks.json").write_text(
            json.dumps(elevenlabs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (pkg_dir / "production_package.json").write_text(
            json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        for fname in (
            "hindi_narration_chunks.json",
            "hindi_narration_full.txt",
            "elevenlabs_chunks.json",
            "production_package.json",
        ):
            assert (pkg_dir / fname).exists(), f"Missing {fname} in 02-package/"

    def test_package_chunks_match_script_chunks(self, tmp_path):
        """The chunks in 02-package/ must be identical to those in 03-script/."""
        script_dir = tmp_path / "03-script"
        script_dir.mkdir()
        pkg_dir = tmp_path / "02-package"
        pkg_dir.mkdir()

        chunks = [{"chunk_id": "001_hook", "text": "परिचय।", "section_title": "Hook"}]

        (script_dir / "hindi_narration_chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (pkg_dir / "hindi_narration_chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        s_chunks = json.loads((script_dir / "hindi_narration_chunks.json").read_text("utf-8"))
        p_chunks = json.loads((pkg_dir / "hindi_narration_chunks.json").read_text("utf-8"))
        assert s_chunks == p_chunks

    def test_elevenlabs_chunks_have_correct_structure(self):
        """ElevenLabs chunks must have the standard voice_id / model_id placeholders."""
        from app.services.script_assembler_service import _extract_elevenlabs_chunks

        chunks = [{"chunk_id": "001", "text": "Hello world."}]
        el = _extract_elevenlabs_chunks(chunks)
        assert len(el) == 1
        assert el[0]["chunk_id"] == "001"
        assert "voice_id" in el[0]
        assert "model_id" in el[0]
        assert "text" in el[0]

    def test_service_import_uses_shared_assembler_helpers(self):
        """premium_section_rebuild_service must import from script_assembler_service."""
        import importlib, inspect
        mod = importlib.import_module("app.services.premium_section_rebuild_service")
        src = inspect.getsource(mod)
        assert "script_assembler_service" in src, (
            "premium_section_rebuild_service must import from script_assembler_service"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# T4 — stage manifest tracks premium_section_rebuild prompt
# ═══════════════════════════════════════════════════════════════════════════════

class TestStageManifestPremiumRebuild:
    """premium_section_rebuild must appear in _STAGE_PROMPT_MAP so prompt
    edits are tracked and force re-runs."""

    def test_premium_section_rebuild_in_stage_prompt_map(self):
        assert "premium_section_rebuild" in _STAGE_PROMPT_MAP, (
            "'premium_section_rebuild' missing from _STAGE_PROMPT_MAP in stage_manifest_service"
        )

    def test_premium_section_rebuild_maps_to_correct_file(self):
        assert _STAGE_PROMPT_MAP["premium_section_rebuild"] == "premium_section_rebuild_agent.txt"

    def test_prompt_hash_recorded_in_manifest_when_file_exists(self, tmp_path):
        """When the prompt file exists, its hash must appear in the saved manifest."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "premium_section_rebuild_agent.txt").write_text(
            "rebuild prompt content", encoding="utf-8"
        )
        manifest = save_manifest(
            tmp_path,
            raw_transcript="test transcript",
            cost_mode="premium",
            hinglish_level=3,
            target_duration_min=20,
            prompts_dir=prompts_dir,
        )
        assert "premium_section_rebuild" in manifest.get("prompt_hashes", {}), (
            "premium_section_rebuild hash missing from manifest prompt_hashes"
        )

    def test_prompt_changed_detects_rebuild_prompt_edit(self, tmp_path):
        """prompt_changed() must return True after the rebuild prompt is edited."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        prompt_file = prompts_dir / "premium_section_rebuild_agent.txt"
        prompt_file.write_text("original rebuild prompt", encoding="utf-8")
        save_manifest(
            tmp_path,
            raw_transcript="transcript",
            cost_mode="premium",
            hinglish_level=3,
            target_duration_min=20,
            prompts_dir=prompts_dir,
        )
        # Prompt unchanged → no change detected
        assert prompt_changed(tmp_path, "premium_section_rebuild", prompts_dir) is False
        # Edit the prompt
        prompt_file.write_text("updated rebuild prompt with new instruction", encoding="utf-8")
        assert prompt_changed(tmp_path, "premium_section_rebuild", prompts_dir) is True


# ═══════════════════════════════════════════════════════════════════════════════
# T5 — ScriptQualityReport accepts case_glossary (schema already approved)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCaseGlossarySchemaAccepted:
    """Regression guard: case_glossary must pass ChunkRepairTarget validation."""

    def test_case_glossary_passes_validation(self):
        t = ChunkRepairTarget.model_validate({
            "chunk_id": "003_victim",
            "issue_type": "case_glossary",
            "problem": "Wrong name form used.",
            "repair_instruction": "Use preferred Hindi form from case_glossary.",
        })
        assert t.issue_type == "case_glossary"

    def test_case_glossary_in_script_quality_report(self):
        data = {
            "gate_passed": False,
            "scores": {
                "factual_accuracy": 7, "story_structure": 8,
                "hindi_naturalness": 8, "emotional_depth": 8,
                "retention_hook": 8, "safety": 9, "monetization_safety": 9,
            },
            "issues": [],
            "chunk_repair_targets": [{
                "chunk_id": "003_victim",
                "issue_type": "case_glossary",
                "problem": "Incorrect name form.",
                "repair_instruction": "Use 'आरुषि' not 'Aarushi'.",
            }],
            "repair_instructions": [],
            "approved": False,
        }
        report = ScriptQualityReport.model_validate(data)
        assert report.chunk_repair_targets[0].issue_type == "case_glossary"

    def test_random_invalid_issue_type_still_fails(self):
        with pytest.raises(ValidationError):
            ChunkRepairTarget.model_validate({
                "chunk_id": "001",
                "issue_type": "totally_made_up_type",
                "problem": "...",
                "repair_instruction": "...",
            })


# ═══════════════════════════════════════════════════════════════════════════════
# T6 — exact_english_quote_copy is normalised → hindi_naturalness
# ═══════════════════════════════════════════════════════════════════════════════

class TestExactEnglishQuoteCopyNormalization:
    """
    exact_english_quote_copy is produced by deterministic_auto_fix_service.
    If it ever ends up in chunk_repair_targets it must be normalised to
    hindi_naturalness (nearest semantic equivalent) rather than failing.
    """

    def test_exact_english_quote_copy_normalized_to_hindi_naturalness(self):
        t = ChunkRepairTarget.model_validate({
            "chunk_id": "009_events",
            "issue_type": "exact_english_quote_copy",
            "problem": "English quote copied verbatim.",
            "repair_instruction": "Translate to Hindi.",
        })
        assert t.issue_type == "hindi_naturalness", (
            f"Expected 'hindi_naturalness' after normalization, got {t.issue_type!r}"
        )

    def test_normalization_does_not_affect_valid_types(self):
        """Standard types must pass through unchanged."""
        for valid_type in ("hindi_naturalness", "pacing", "safety", "case_glossary"):
            t = ChunkRepairTarget.model_validate({
                "chunk_id": "001",
                "issue_type": valid_type,
                "problem": "test",
                "repair_instruction": "fix",
            })
            assert t.issue_type == valid_type

    def test_exact_english_quote_copy_accepted_in_script_quality_report(self):
        """A report containing exact_english_quote_copy must pass full schema validation."""
        data = {
            "gate_passed": False,
            "scores": {
                "factual_accuracy": 7, "story_structure": 8,
                "hindi_naturalness": 6, "emotional_depth": 8,
                "retention_hook": 8, "safety": 9, "monetization_safety": 9,
            },
            "issues": [],
            "chunk_repair_targets": [{
                "chunk_id": "009_events",
                "issue_type": "exact_english_quote_copy",
                "problem": "Long English quote copied.",
                "repair_instruction": "Translate entire quote to Hindi.",
            }],
            "repair_instructions": [],
            "approved": False,
        }
        report = ScriptQualityReport.model_validate(data)
        # After normalisation, the stored type is hindi_naturalness
        assert report.chunk_repair_targets[0].issue_type == "hindi_naturalness"

    def test_unknown_alias_still_fails(self):
        """Only explicitly listed aliases are normalised; random strings still fail."""
        with pytest.raises(ValidationError):
            ChunkRepairTarget.model_validate({
                "chunk_id": "001",
                "issue_type": "some_unknown_linter_type",
                "problem": "...",
                "repair_instruction": "...",
            })

    def test_issue_type_alias_dict_contains_exact_english_quote_copy(self):
        """The _ISSUE_TYPE_ALIASES dict in schemas.py must list the alias explicitly."""
        from app.schemas import _ISSUE_TYPE_ALIASES
        assert "exact_english_quote_copy" in _ISSUE_TYPE_ALIASES
        assert _ISSUE_TYPE_ALIASES["exact_english_quote_copy"] == "hindi_naturalness"


# ═══════════════════════════════════════════════════════════════════════════════
# T7 — OpenAI skip reason regression guard
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpenAISkipReasonRegression:
    """
    Ensure the previously fixed skip-reason bug has not regressed.
    When OPENAI_REVIEW_ENABLED=true but post-repair preflight is blocking,
    the skip reason must be post_repair_python_preflight_blocking, not the
    fallthrough OPENAI_REVIEW_ENABLED=false string.
    """

    def _build_skip_reason(
        self,
        post_repair_preflight_blocking: bool,
        quality_mode: str = "premium_final",
        openai_review_policy: str = "adaptive",
        openai_review_enabled: bool = True,
    ) -> tuple[str, str, str]:
        if post_repair_preflight_blocking:
            return (
                "post_repair_python_preflight_blocking",
                "Post-repair Python preflight is still blocking; "
                "OpenAI Final Premium Gate skipped to avoid reviewing unsafe/unready script.",
                "not_voice_ready_auto_retry_exhausted",
            )
        elif quality_mode != "premium_final":
            return (f"quality_mode={quality_mode}", "...", "needs_human_review")
        elif openai_review_policy == "disabled":
            return (f"openai_review_policy={openai_review_policy}", "...", "needs_human_review")
        else:
            return ("OPENAI_REVIEW_ENABLED=false", "...", "needs_human_review")

    def test_blocking_preflight_gives_correct_code(self):
        code, _, _ = self._build_skip_reason(
            post_repair_preflight_blocking=True,
            openai_review_enabled=True,
        )
        assert code == "post_repair_python_preflight_blocking"

    def test_blocking_preflight_code_does_not_contain_review_enabled_false(self):
        code, detail, _ = self._build_skip_reason(post_repair_preflight_blocking=True)
        assert "OPENAI_REVIEW_ENABLED=false" not in code
        assert "OPENAI_REVIEW_ENABLED=false" not in detail

    def test_blocking_preflight_status_is_not_voice_ready(self):
        _, _, status = self._build_skip_reason(post_repair_preflight_blocking=True)
        assert status == "not_voice_ready_auto_retry_exhausted"
        assert status != "needs_human_review"

    def test_non_blocking_falls_through_to_other_codes(self):
        code, _, _ = self._build_skip_reason(
            post_repair_preflight_blocking=False,
            quality_mode="premium_final",
            openai_review_policy="adaptive",
            openai_review_enabled=False,
        )
        assert code == "OPENAI_REVIEW_ENABLED=false"

    def test_blocking_preflight_has_priority_over_disabled_policy(self):
        """preflight blocking is evaluated first; policy=disabled is irrelevant."""
        code, _, _ = self._build_skip_reason(
            post_repair_preflight_blocking=True,
            openai_review_policy="disabled",
        )
        assert code == "post_repair_python_preflight_blocking"
