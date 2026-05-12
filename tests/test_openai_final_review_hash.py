"""
Tests for _compute_final_review_input_hash — verifies that metadata changes,
recreated_dialogues changes, narration changes, and gate report changes all
produce a different hash so OFP is never reused when evidence changed.
"""
from __future__ import annotations

import pytest

from app.services.agent_pipeline_service import (
    _compute_final_review_input_hash,
    _compute_script_hash,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _script(chunks=None, metadata=None, dialogues=None) -> dict:
    return {
        "hindi_narration_chunks": chunks or [{"chunk_id": "001", "text": "हुक।"}],
        "youtube_metadata":       metadata or {"title": "Ep 1"},
        "recreated_dialogues":    dialogues or {"items": []},
    }


def _base_hash(**overrides) -> str:
    kwargs = dict(
        script_final=_script(),
        hinglish_level=2,
        target_duration_min=10,
    )
    kwargs.update(overrides)
    return _compute_final_review_input_hash(**kwargs)


# ─── Tests: constant format ───────────────────────────────────────────────────

class TestHashFormat:
    def test_returns_16_char_hex(self):
        h = _base_hash()
        assert isinstance(h, str)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic_same_inputs(self):
        assert _base_hash() == _base_hash()


# ─── Tests: script content changes ────────────────────────────────────────────

class TestNarrationChangeChangesHash:
    def test_narration_text_change(self):
        h1 = _compute_final_review_input_hash(script_final=_script(
            chunks=[{"chunk_id": "001", "text": "पहला पाठ।"}]
        ))
        h2 = _compute_final_review_input_hash(script_final=_script(
            chunks=[{"chunk_id": "001", "text": "दूसरा पाठ।"}]
        ))
        assert h1 != h2

    def test_extra_chunk_changes_hash(self):
        h1 = _compute_final_review_input_hash(script_final=_script(
            chunks=[{"chunk_id": "001", "text": "एक।"}]
        ))
        h2 = _compute_final_review_input_hash(script_final=_script(
            chunks=[{"chunk_id": "001", "text": "एक।"}, {"chunk_id": "002", "text": "दो।"}]
        ))
        assert h1 != h2


# ─── Tests: metadata change changes hash ──────────────────────────────────────

class TestMetadataChangeChangesHash:
    def test_title_change(self):
        h1 = _compute_final_review_input_hash(script_final=_script(
            metadata={"title": "Episode 1"}
        ))
        h2 = _compute_final_review_input_hash(script_final=_script(
            metadata={"title": "Episode 2 — Different Title"}
        ))
        assert h1 != h2

    def test_description_change(self):
        h1 = _compute_final_review_input_hash(script_final=_script(
            metadata={"title": "T", "description": "old"}
        ))
        h2 = _compute_final_review_input_hash(script_final=_script(
            metadata={"title": "T", "description": "new"}
        ))
        assert h1 != h2

    def test_narration_unchanged_metadata_changed_still_different(self):
        """OFP reviews metadata — metadata change alone must invalidate hash."""
        chunks = [{"chunk_id": "001", "text": "same text."}]
        h1 = _compute_final_review_input_hash(script_final=_script(
            chunks=chunks, metadata={"title": "A"}
        ))
        h2 = _compute_final_review_input_hash(script_final=_script(
            chunks=chunks, metadata={"title": "B"}
        ))
        assert h1 != h2, (
            "metadata change must change final_review_input_hash "
            "(OFP reviews metadata — cannot reuse old approval)"
        )


# ─── Tests: narration-only hash (_compute_script_hash) does NOT catch metadata ─

class TestScriptHashDoesNotCoverMetadata:
    """Demonstrate why _compute_script_hash alone was insufficient."""

    def test_script_hash_same_when_only_metadata_changes(self):
        chunks = [{"chunk_id": "001", "text": "same narration."}]
        h1 = _compute_script_hash(_script(chunks=chunks, metadata={"title": "A"}))
        h2 = _compute_script_hash(_script(chunks=chunks, metadata={"title": "B DIFFERENT"}))
        # script_hash only hashes narration — same result
        assert h1 == h2

    def test_final_review_hash_differs_for_same_case(self):
        chunks = [{"chunk_id": "001", "text": "same narration."}]
        h1 = _compute_final_review_input_hash(script_final=_script(
            chunks=chunks, metadata={"title": "A"}
        ))
        h2 = _compute_final_review_input_hash(script_final=_script(
            chunks=chunks, metadata={"title": "B DIFFERENT"}
        ))
        assert h1 != h2


# ─── Tests: recreated_dialogues change changes hash ───────────────────────────

class TestDialoguesChangeChangesHash:
    def test_dialogue_item_added(self):
        h1 = _compute_final_review_input_hash(script_final=_script(dialogues={"items": []}))
        h2 = _compute_final_review_input_hash(script_final=_script(
            dialogues={"items": [{"scene_id": "s1", "text": "New dialogue."}]}
        ))
        assert h1 != h2

    def test_dialogue_text_changed(self):
        d1 = {"items": [{"scene_id": "s1", "text": "Original dialogue."}]}
        d2 = {"items": [{"scene_id": "s1", "text": "Changed dialogue."}]}
        h1 = _compute_final_review_input_hash(script_final=_script(dialogues=d1))
        h2 = _compute_final_review_input_hash(script_final=_script(dialogues=d2))
        assert h1 != h2


# ─── Tests: run-parameter changes change hash ─────────────────────────────────

class TestRunParameterChangesHash:
    def test_hinglish_level_change(self):
        h1 = _compute_final_review_input_hash(
            script_final=_script(), hinglish_level=2
        )
        h2 = _compute_final_review_input_hash(
            script_final=_script(), hinglish_level=3
        )
        assert h1 != h2

    def test_target_duration_change(self):
        h1 = _compute_final_review_input_hash(
            script_final=_script(), target_duration_min=10
        )
        h2 = _compute_final_review_input_hash(
            script_final=_script(), target_duration_min=15
        )
        assert h1 != h2


# ─── Tests: gate report changes change hash ───────────────────────────────────

class TestGateReportChangesHash:
    def test_lint_report_change_changes_hash(self):
        h1 = _compute_final_review_input_hash(
            script_final=_script(),
            lint_report={"total_issues": 0, "passed": True},
        )
        h2 = _compute_final_review_input_hash(
            script_final=_script(),
            lint_report={"total_issues": 5, "passed": False},
        )
        assert h1 != h2

    def test_similarity_report_change_changes_hash(self):
        h1 = _compute_final_review_input_hash(
            script_final=_script(),
            similarity_report={"risk_level": "none", "high_risk_matches": 0},
        )
        h2 = _compute_final_review_input_hash(
            script_final=_script(),
            similarity_report={"risk_level": "high", "high_risk_matches": 12},
        )
        assert h1 != h2

    def test_metadata_report_change_changes_hash(self):
        h1 = _compute_final_review_input_hash(
            script_final=_script(),
            metadata_report={"gate_passed": True},
        )
        h2 = _compute_final_review_input_hash(
            script_final=_script(),
            metadata_report={"gate_passed": False, "required_fixes": ["fix title"]},
        )
        assert h1 != h2

    def test_copyedit_report_change_changes_hash(self):
        h1 = _compute_final_review_input_hash(
            script_final=_script(),
            copyedit_report={"approved": True},
        )
        h2 = _compute_final_review_input_hash(
            script_final=_script(),
            copyedit_report={"approved": False, "chunk_repair_targets": [{"chunk_id": "x"}]},
        )
        assert h1 != h2


# ─── Tests: OFP reuse decision using final_review_input_hash ──────────────────

class TestOFPReuseWithFinalReviewHash:
    def _make_ofp_report(self, h: str) -> dict:
        return {
            "approved": True, "safe_to_voice": True,
            "overall_score": 88, "final_review_input_hash": h,
        }

    def _reuse_decision(self, cached: dict, current_hash: str) -> bool:
        stored = cached.get("final_review_input_hash", "")
        return bool(stored and stored == current_hash)

    def test_reuse_when_hash_matches(self):
        h = _base_hash()
        cached = self._make_ofp_report(h)
        assert self._reuse_decision(cached, h) is True

    def test_no_reuse_when_hash_differs(self):
        h_old = _base_hash()
        cached = self._make_ofp_report(h_old)
        h_new = _compute_final_review_input_hash(
            script_final=_script(metadata={"title": "Changed"}),
        )
        assert self._reuse_decision(cached, h_new) is False

    def test_no_reuse_when_hash_absent(self):
        cached = {"approved": True, "safe_to_voice": True}  # old format, no hash
        assert self._reuse_decision(cached, _base_hash()) is False

    def test_old_script_hash_field_not_reused(self):
        """Old OFP reports stored script_hash (narration-only) — must not reuse."""
        h = _base_hash()
        cached = {
            "approved": True,
            "script_hash": h,   # OLD field — not final_review_input_hash
        }
        assert self._reuse_decision(cached, h) is False
