"""
Tests for _compute_final_review_input_hash with full-JSON _report_sig — session 4, TASK 7.

Verifies that the new full-canonical-JSON approach to _report_sig detects content
changes that the old compact sig missed:
- required_fixes text changes in quality/copyedit/metadata reports
- chunk_repair_targets additions (not just count, but content)
- Any field change in a gate report produces a different hash
- refresh_failed still returns sentinel "none" (no collision)
- Deterministic: same inputs → same hash
"""
from __future__ import annotations

import pytest

from app.services.agent_pipeline_service import _compute_final_review_input_hash


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _script() -> dict:
    return {
        "hindi_narration_chunks": [{"chunk_id": "001", "text": "पाठ।"}],
        "youtube_metadata": {"title": "T"},
        "recreated_dialogues": {"items": []},
    }


def _base(**overrides) -> str:
    kwargs = dict(script_final=_script())
    kwargs.update(overrides)
    return _compute_final_review_input_hash(**kwargs)


# ─── Tests: required_fixes text changes change hash ───────────────────────────

class TestRequiredFixesTextChangesHash:
    def test_quality_report_required_fixes_text_change(self):
        """Old compact sig only hashed issues count — text change would be missed."""
        r1 = {"approved": False, "required_fixes": ["Fix the hook section."]}
        r2 = {"approved": False, "required_fixes": ["Fix the hook section. Also fix ending."]}
        h1 = _base(quality_report=r1)
        h2 = _base(quality_report=r2)
        assert h1 != h2, (
            "required_fixes text change must change the OFP hash — "
            "old compact sig missed this because it only hashed issues count"
        )

    def test_copyedit_report_required_fixes_text_change(self):
        r1 = {"approved": False, "chunk_repair_targets": [{"chunk_id": "001", "issue": "old"}]}
        r2 = {"approved": False, "chunk_repair_targets": [{"chunk_id": "001", "issue": "new text"}]}
        h1 = _base(copyedit_report=r1)
        h2 = _base(copyedit_report=r2)
        assert h1 != h2, (
            "chunk_repair_targets content change must change hash — "
            "old compact sig only compared count, not content"
        )

    def test_metadata_report_fix_text_change(self):
        r1 = {"gate_passed": False, "required_fixes": ["Update title."]}
        r2 = {"gate_passed": False, "required_fixes": ["Update title. Add keywords."]}
        h1 = _base(metadata_report=r1)
        h2 = _base(metadata_report=r2)
        assert h1 != h2

    def test_same_count_different_content_different_hash(self):
        """Two reports with same issues count but different issue text → different hash."""
        r1 = {"approved": False, "issues": [{"chunk_id": "001", "problem": "old issue A"}]}
        r2 = {"approved": False, "issues": [{"chunk_id": "001", "problem": "new issue B"}]}
        # Both have issues count = 1 — old compact sig would produce same hash
        h1 = _base(quality_report=r1)
        h2 = _base(quality_report=r2)
        assert h1 != h2, (
            "Same issues count but different issue text must still differ — "
            "full JSON sig required"
        )


# ─── Tests: any field change detected ─────────────────────────────────────────

class TestAnyFieldChangeDetected:
    def test_new_field_in_report_changes_hash(self):
        r1 = {"passed": True, "overall_score": 85}
        r2 = {"passed": True, "overall_score": 85, "new_field": "extra_data"}
        h1 = _base(quality_report=r1)
        h2 = _base(quality_report=r2)
        assert h1 != h2

    def test_score_change_detected(self):
        r1 = {"passed": True, "overall_score": 85}
        r2 = {"passed": True, "overall_score": 90}
        h1 = _base(quality_report=r1)
        h2 = _base(quality_report=r2)
        assert h1 != h2

    def test_high_risk_matches_change_detected(self):
        r1 = {"passed": True, "risk_level": "low",  "high_risk_matches": 0}
        r2 = {"passed": True, "risk_level": "high", "high_risk_matches": 12}
        h1 = _base(similarity_report=r1)
        h2 = _base(similarity_report=r2)
        assert h1 != h2

    def test_nested_content_change_detected(self):
        r1 = {"passed": True, "sections": [{"name": "hook", "score": 8}]}
        r2 = {"passed": True, "sections": [{"name": "hook", "score": 9}]}
        h1 = _base(quality_report=r1)
        h2 = _base(quality_report=r2)
        assert h1 != h2


# ─── Tests: refresh_failed still returns sentinel ─────────────────────────────

class TestRefreshFailedSentinel:
    def test_refresh_failed_report_gives_same_hash_as_none(self):
        """refresh_failed reports produce sentinel 'none' — treated as absent."""
        h_none = _base(similarity_report=None)
        h_stale = _base(similarity_report={
            "passed": False, "refresh_failed": True,
            "reason": "missing source_transcript",
        })
        # Both produce "none" sentinel in _report_sig — hash should be the same
        assert h_none == h_stale, (
            "refresh_failed report must produce same hash as None report — "
            "the sentinel 'none' represents absence of valid evidence"
        )

    def test_fresh_vs_stale_report_different_hash(self):
        fresh = {"passed": True, "risk_level": "none", "high_risk_matches": 0}
        stale = {"passed": False, "refresh_failed": True, "reason": "missing input"}
        h_fresh = _base(similarity_report=fresh)
        h_stale = _base(similarity_report=stale)
        assert h_fresh != h_stale


# ─── Tests: determinism ───────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_inputs_produce_same_hash(self):
        r = {"passed": True, "required_fixes": ["fix A", "fix B"], "overall_score": 85}
        h1 = _base(quality_report=r)
        h2 = _base(quality_report=r)
        assert h1 == h2

    def test_complex_report_is_deterministic(self):
        r = {
            "passed": False,
            "chunk_repair_targets": [
                {"chunk_id": "003", "issue": "Long sentence."},
                {"chunk_id": "007", "issue": "Hinglish mismatch."},
            ],
            "overall_score": 62,
            "required_fixes": ["Fix chunk 003 verbosity.", "Fix chunk 007 code-mix."],
        }
        h1 = _base(copyedit_report=r)
        h2 = _base(copyedit_report=r)
        assert h1 == h2


# ─── Tests: hash format ───────────────────────────────────────────────────────

class TestHashFormat:
    def test_returns_16_char_hex(self):
        h = _base()
        assert isinstance(h, str)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)
