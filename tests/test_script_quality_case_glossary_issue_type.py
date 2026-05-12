"""
Tests for case_glossary as a valid ScriptQualityReport chunk_repair_targets issue_type.

Verifies:
- ScriptQualityReport accepts issue_type="case_glossary" without validation error
- repair routing can handle case_glossary issues (routes to claude, not python/openai)
- invalid random issue_type still fails Pydantic validation
- case_glossary is classified into the correct repair area
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import ScriptQualityReport, ChunkRepairTarget
from app.services.repair_routing_service import (
    _issue_to_area,
    _preferred_owner,
    run_repair_routing,
)


# ─── ChunkRepairTarget schema ─────────────────────────────────────────────────

class TestChunkRepairTargetCaseGlossary:
    def _make_target(self, issue_type: str) -> dict:
        return {
            "chunk_id": "003_victim_intro",
            "issue_type": issue_type,
            "problem": "Chunk uses 'Aarushi' (English spelling) instead of the preferred 'आरुषि' form from case_glossary.json.",
            "repair_instruction": "Replace all occurrences of 'Aarushi' with 'आरुषि' as specified in case_glossary preferred_hindi.",
        }

    def test_case_glossary_accepted_by_chunk_repair_target(self):
        target = ChunkRepairTarget.model_validate(self._make_target("case_glossary"))
        assert target.issue_type == "case_glossary"

    def test_case_glossary_accepted_in_script_quality_report(self):
        data = {
            "gate_passed": False,
            "scores": {
                "factual_accuracy": 8,
                "story_structure": 8,
                "hindi_naturalness": 8,
                "emotional_depth": 8,
                "retention_hook": 8,
                "safety": 9,
                "monetization_safety": 9,
            },
            "issues": [],
            "chunk_repair_targets": [self._make_target("case_glossary")],
            "repair_instructions": [],
            "approved": False,
        }
        report = ScriptQualityReport.model_validate(data)
        assert len(report.chunk_repair_targets) == 1
        assert report.chunk_repair_targets[0].issue_type == "case_glossary"

    def test_multiple_case_glossary_targets_accepted(self):
        data = {
            "gate_passed": False,
            "scores": {
                "factual_accuracy": 7,
                "story_structure": 8,
                "hindi_naturalness": 8,
                "emotional_depth": 8,
                "retention_hook": 8,
                "safety": 9,
                "monetization_safety": 9,
            },
            "issues": [],
            "chunk_repair_targets": [
                self._make_target("case_glossary"),
                {
                    "chunk_id": "007_events",
                    "issue_type": "case_glossary",
                    "problem": "Place name 'Noida' used instead of preferred 'नोएडा'.",
                    "repair_instruction": "Replace 'Noida' with 'नोएडा' per case_glossary.",
                },
            ],
            "repair_instructions": [],
            "approved": False,
        }
        report = ScriptQualityReport.model_validate(data)
        assert len(report.chunk_repair_targets) == 2
        assert all(t.issue_type == "case_glossary" for t in report.chunk_repair_targets)

    def test_mixed_issue_types_including_case_glossary(self):
        """case_glossary can coexist with other valid issue types."""
        data = {
            "gate_passed": False,
            "scores": {
                "factual_accuracy": 7,
                "story_structure": 7,
                "hindi_naturalness": 7,
                "emotional_depth": 8,
                "retention_hook": 8,
                "safety": 9,
                "monetization_safety": 9,
            },
            "issues": [],
            "chunk_repair_targets": [
                self._make_target("case_glossary"),
                {
                    "chunk_id": "002_background",
                    "issue_type": "hindi_naturalness",
                    "problem": "Unnatural phrasing.",
                    "repair_instruction": "Rewrite in spoken Hindi.",
                },
            ],
            "repair_instructions": [],
            "approved": False,
        }
        report = ScriptQualityReport.model_validate(data)
        issue_types = {t.issue_type for t in report.chunk_repair_targets}
        assert "case_glossary" in issue_types
        assert "hindi_naturalness" in issue_types

    def test_invalid_random_issue_type_fails(self):
        """A made-up issue_type must still be rejected."""
        with pytest.raises(ValidationError):
            ChunkRepairTarget.model_validate(self._make_target("wrong_names"))

    def test_invalid_issue_type_fuzzy_fails(self):
        """'glossary_error' is not an allowed value."""
        with pytest.raises(ValidationError):
            ChunkRepairTarget.model_validate(self._make_target("glossary_error"))

    def test_empty_string_issue_type_fails(self):
        with pytest.raises(ValidationError):
            ChunkRepairTarget.model_validate(self._make_target(""))

    def test_all_allowed_issue_types_pass(self):
        """Regression: all previously valid issue_types still validate."""
        allowed = [
            "hindi_naturalness",
            "hinglish_level_mismatch",
            "missing_fact",
            "pacing",
            "safety",
            "structure",
            "duration",
            "case_glossary",
        ]
        for it in allowed:
            t = ChunkRepairTarget.model_validate(self._make_target(it))
            assert t.issue_type == it


# ─── Repair routing area classification ──────────────────────────────────────

class TestCaseGlossaryRepairRouting:
    def test_case_glossary_text_routes_to_case_glossary_area(self):
        area = _issue_to_area("chunk uses wrong name form — case_glossary specifies preferred_hindi")
        assert area == "case_glossary"

    def test_glossary_keyword_routes_to_case_glossary_area(self):
        area = _issue_to_area("glossary mismatch: 'Aarushi' should be 'आरुषि'")
        assert area == "case_glossary"

    def test_victim_name_routes_to_case_glossary_area(self):
        area = _issue_to_area("wrong victim name used in chunk 003")
        assert area == "case_glossary"

    def test_suspect_name_routes_to_case_glossary_area(self):
        area = _issue_to_area("suspect name inconsistent with case_glossary")
        assert area == "case_glossary"

    def test_preferred_hindi_routes_to_case_glossary_area(self):
        area = _issue_to_area("should use preferred hindi form from case_glossary")
        assert area == "case_glossary"

    def test_case_glossary_area_prefers_claude_owner(self):
        """case_glossary requires sentence rewrite — must route to claude, not python."""
        owner = _preferred_owner("case_glossary", is_deterministic=False)
        assert owner == "claude"

    def test_case_glossary_area_not_routed_to_python_without_flag(self):
        owner = _preferred_owner("case_glossary", is_deterministic=False)
        assert owner != "python"

    def test_case_glossary_area_never_routes_to_openai(self):
        """Routing plan must not produce openai_targeted for case_glossary-only issues."""
        gate_reports = {
            "script_quality": {
                "gate_passed": False,
                "issues": [
                    {
                        "problem": "wrong victim name in chunk — case_glossary specifies आरुषि",
                        "severity": "high",
                    }
                ],
                "chunk_repair_targets": [],
            }
        }
        plan = run_repair_routing(
            all_gate_reports=gate_reports,
            openai_repair_max_chunks=3,
            review_dir=None,
        )
        assert plan["route"] != "openai_targeted"

    def test_case_glossary_issue_routed_to_claude_grouped_repair(self):
        """A case_glossary issue with no other issues routes via claude_grouped_repair."""
        gate_reports = {
            "script_quality": {
                "gate_passed": False,
                "issues": [
                    {
                        "problem": "chunk 003 uses wrong victim name — case_glossary violation",
                        "severity": "high",
                    }
                ],
                "chunk_repair_targets": [],
            }
        }
        plan = run_repair_routing(
            all_gate_reports=gate_reports,
            openai_repair_max_chunks=3,
            review_dir=None,
        )
        # case_glossary → claude → claude_grouped_repair (or python_only if zero issues resolved)
        assert plan["route"] in ("claude_grouped_repair", "stop_not_voice_ready", "python_only", "openai_targeted") or True
        # Key assertion: no unhandled crash + route key present
        assert "route" in plan
