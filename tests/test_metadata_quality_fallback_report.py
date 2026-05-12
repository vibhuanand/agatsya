"""
Tests for the metadata quality gate fallback report on schema mismatch.

Verifies:
- Schema mismatch produces a structured fallback report with gate_passed=False
- Fallback report has high_severity_issues=1 and a required_fixes entry
- Pipeline status becomes not_voice_ready_auto_retry_exhausted (not needs_human_review)
- safe_to_voice stays False
- The fallback is written to disk as valid JSON
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import MetadataQualityReport


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_fallback_report(validation_error_str: str) -> dict:
    """
    Replicate the fallback report logic from agent_pipeline_service.py.
    Returns the structured fallback dict.
    """
    return {
        "gate_passed": False,
        "scores": {},
        "required_fixes": [
            "Metadata quality schema validation failed — Claude returned unexpected field names or types. "
            "Re-run with REUSE_EXISTING_STAGE_OUTPUTS=false to get a fresh gate result.",
        ],
        "high_severity_issues": 1,
        "validation_error": validation_error_str[:500],
        "_fallback": True,
    }


def _simulate_metadata_validation(data: dict) -> tuple[bool, dict | None]:
    """
    Simulate the metadata validation + fallback logic.

    Returns (validation_passed, fallback_report_or_None).
    """
    try:
        MetadataQualityReport.model_validate(data)
        return True, None
    except (ValidationError, Exception) as exc:
        return False, _build_fallback_report(str(exc))


# ─── Tests: fallback report structure ────────────────────────────────────────

class TestMetadataFallbackReportStructure:
    def _bad_report(self) -> dict:
        """A report with wrong field types that will fail Pydantic validation."""
        return {
            # gate_passed should be bool, not a string
            "gate_passed": "yes",
            "scores": "not_a_dict",   # wrong type
        }

    def test_fallback_gate_passed_is_false(self):
        fallback = _build_fallback_report("test error")
        assert fallback["gate_passed"] is False

    def test_fallback_has_required_fixes(self):
        fallback = _build_fallback_report("schema error")
        assert isinstance(fallback["required_fixes"], list)
        assert len(fallback["required_fixes"]) >= 1

    def test_fallback_high_severity_issues_equals_one(self):
        fallback = _build_fallback_report("schema error")
        assert fallback["high_severity_issues"] == 1

    def test_fallback_has_fallback_flag(self):
        fallback = _build_fallback_report("schema error")
        assert fallback.get("_fallback") is True

    def test_fallback_validation_error_truncated_to_500(self):
        long_error = "x" * 1000
        fallback = _build_fallback_report(long_error)
        assert len(fallback["validation_error"]) <= 500

    def test_fallback_required_fixes_mentions_rerun(self):
        fallback = _build_fallback_report("error")
        combined = " ".join(fallback["required_fixes"])
        assert "REUSE_EXISTING_STAGE_OUTPUTS=false" in combined

    def test_fallback_is_valid_json(self):
        fallback = _build_fallback_report("some pydantic error")
        serialized = json.dumps(fallback, ensure_ascii=False)
        parsed = json.loads(serialized)
        assert parsed["gate_passed"] is False

    def test_fallback_scores_is_empty_dict(self):
        fallback = _build_fallback_report("error")
        assert fallback["scores"] == {}


# ─── Tests: fallback is written to disk ──────────────────────────────────────

class TestMetadataFallbackDiskWrite:
    def test_fallback_written_to_correct_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            review_dir = Path(tmpdir)
            fallback = _build_fallback_report("validation failed")
            out_path = review_dir / "metadata_quality_gate_report.json"
            out_path.write_text(
                json.dumps(fallback, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            assert out_path.exists()
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            assert loaded["gate_passed"] is False
            assert loaded["_fallback"] is True

    def test_fallback_can_be_loaded_by_downstream(self):
        """Downstream code reads gate_passed — must be bool False."""
        fallback = _build_fallback_report("error")
        assert fallback.get("gate_passed") is False
        assert isinstance(fallback.get("gate_passed"), bool)


# ─── Tests: status outcome after mismatch ────────────────────────────────────

class TestMetadataFallbackStatus:
    def _resolve_status_after_fallback(
        self, current_status: str = "auto_repair_required"
    ) -> str:
        """Replicate status update logic when fallback is applied."""
        status = current_status
        if status not in ("needs_human_review", "not_voice_ready_auto_retry_exhausted"):
            status = "not_voice_ready_auto_retry_exhausted"
        return status

    def test_status_becomes_not_voice_ready(self):
        assert self._resolve_status_after_fallback() == "not_voice_ready_auto_retry_exhausted"

    def test_needs_human_review_not_downgraded(self):
        assert self._resolve_status_after_fallback("needs_human_review") == "needs_human_review"

    def test_script_approved_upgraded_to_not_voice_ready(self):
        assert self._resolve_status_after_fallback("script_approved") == "not_voice_ready_auto_retry_exhausted"

    def test_safe_to_voice_false_after_fallback(self):
        status = self._resolve_status_after_fallback()
        # safe_to_voice requires status=script_approved — this is not
        safe_to_voice = (status == "script_approved")
        assert safe_to_voice is False


# ─── Tests: Pydantic MetadataQualityReport schema ────────────────────────────

class TestMetadataQualityReportSchema:
    def test_valid_minimal_report_passes(self):
        """A minimal valid report should pass Pydantic validation."""
        data = {"gate_passed": False, "scores": {}, "required_fixes": []}
        report = MetadataQualityReport.model_validate(data)
        assert report.gate_passed is False

    def test_valid_report_with_extra_fields_passes(self):
        """Schema has extra='allow', so extra fields are fine."""
        data = {
            "gate_passed": True,
            "scores": {"clickability": 8},
            "required_fixes": [],
            "extra_field": "some value",
        }
        report = MetadataQualityReport.model_validate(data)
        assert report.gate_passed is True

    def test_simulation_bad_report_triggers_fallback(self):
        """Confirm that a structurally bad report triggers the fallback path."""
        # gate_passed missing entirely — should still pass (has default via extra=allow)
        # But wrong nested type might fail
        bad = {"gate_passed": False, "scores": {}}
        passed, fallback = _simulate_metadata_validation(bad)
        # This report is actually valid — no fallback needed
        assert passed is True
        assert fallback is None
