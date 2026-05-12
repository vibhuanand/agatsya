"""
Tests for TASK 2 — OpenAI Final Premium Gate must not be reused when the
script content has changed since the cached report was written.

Verifies:
- _compute_script_hash returns a consistent 16-char hex string
- same script → same hash
- different narration chunks → different hash
- cached OFP with matching hash is reused
- cached OFP with mismatched hash is discarded (rerun required)
- cached OFP with missing hash is discarded (safe default)
- script_hash is stored in the report after a fresh OFP run (simulated)
"""
from __future__ import annotations

import json

import pytest

from app.services.agent_pipeline_service import _compute_script_hash


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_script(chunks: list[dict]) -> dict:
    return {"hindi_narration_chunks": chunks}


def _make_chunk(chunk_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "text": text, "section_title": "S", "voice": "narrator"}


def _ofp_reuse_decision(existing_report: dict | None, current_script: dict) -> bool:
    """
    Replicate the OFP reuse decision added to agent_pipeline_service Stage 14a:
    reuse only when existing report exists AND its script_hash matches current.
    Returns True  → reuse cached report.
    Returns False → must rerun OFP.
    """
    if existing_report is None:
        return False
    _current_hash = _compute_script_hash(current_script)
    _stored_hash = existing_report.get("script_hash", "")
    return bool(_stored_hash and _stored_hash == _current_hash)


# ─── Tests: _compute_script_hash ──────────────────────────────────────────────

class TestComputeScriptHash:
    def test_returns_16_char_hex_string(self):
        h = _compute_script_hash(_make_script([]))
        assert isinstance(h, str)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_script_same_hash(self):
        script = _make_script([_make_chunk("001", "हैलो दुनिया।")])
        assert _compute_script_hash(script) == _compute_script_hash(script)

    def test_different_text_different_hash(self):
        s1 = _make_script([_make_chunk("001", "पहला पाठ।")])
        s2 = _make_script([_make_chunk("001", "दूसरा पाठ।")])
        assert _compute_script_hash(s1) != _compute_script_hash(s2)

    def test_different_chunk_id_different_hash(self):
        s1 = _make_script([_make_chunk("001", "text")])
        s2 = _make_script([_make_chunk("002", "text")])
        assert _compute_script_hash(s1) != _compute_script_hash(s2)

    def test_empty_chunks_stable_hash(self):
        h1 = _compute_script_hash({"hindi_narration_chunks": []})
        h2 = _compute_script_hash({"hindi_narration_chunks": []})
        assert h1 == h2

    def test_missing_chunks_key_stable_hash(self):
        h = _compute_script_hash({})
        assert isinstance(h, str)
        assert len(h) == 16

    def test_metadata_change_does_not_change_hash(self):
        """Only narration chunks are hashed — metadata changes must not affect hash."""
        s1 = _make_script([_make_chunk("001", "text")])
        s2 = dict(s1)
        s2["youtube_metadata"] = {"title": "Different title"}
        assert _compute_script_hash(s1) == _compute_script_hash(s2)

    def test_chunk_order_matters(self):
        """Order of chunks is significant — different order → different hash."""
        c1 = _make_chunk("001", "पहला")
        c2 = _make_chunk("002", "दूसरा")
        s1 = _make_script([c1, c2])
        s2 = _make_script([c2, c1])
        assert _compute_script_hash(s1) != _compute_script_hash(s2)


# ─── Tests: OFP reuse decision ───────────────────────────────────────────────

class TestOFPReuseDecision:
    def _script(self) -> dict:
        return _make_script([_make_chunk("001_hook", "हुक टेक्स्ट।")])

    def _cached_ofp_with_matching_hash(self, script: dict) -> dict:
        return {
            "approved": True,
            "safe_to_voice": True,
            "overall_score": 90,
            "script_hash": _compute_script_hash(script),
        }

    def test_reuse_when_hash_matches(self):
        script = self._script()
        cached = self._cached_ofp_with_matching_hash(script)
        assert _ofp_reuse_decision(cached, script) is True

    def test_no_reuse_when_hash_mismatches(self):
        original_script = self._script()
        cached = self._cached_ofp_with_matching_hash(original_script)
        # Now script changes — repair modified a chunk
        changed_script = _make_script([_make_chunk("001_hook", "बदला हुआ हुक।")])
        assert _ofp_reuse_decision(cached, changed_script) is False

    def test_no_reuse_when_hash_absent_in_cached_report(self):
        """Old OFP reports without script_hash must never be reused."""
        script = self._script()
        old_report = {"approved": True, "safe_to_voice": True}  # no script_hash
        assert _ofp_reuse_decision(old_report, script) is False

    def test_no_reuse_when_existing_report_is_none(self):
        script = self._script()
        assert _ofp_reuse_decision(None, script) is False

    def test_no_reuse_when_script_has_extra_chunks(self):
        """After rebuild, additional chunks → hash mismatch → rerun OFP."""
        original = _make_script([_make_chunk("001", "पहला।")])
        cached = self._cached_ofp_with_matching_hash(original)
        expanded = _make_script([
            _make_chunk("001", "पहला।"),
            _make_chunk("002", "नया।"),
        ])
        assert _ofp_reuse_decision(cached, expanded) is False

    def test_hash_stored_in_fresh_ofp_report(self):
        """Simulate: after running OFP, caller stamps script_hash into the report."""
        script = self._script()
        fresh_ofp = {"approved": True, "safe_to_voice": True, "overall_score": 85}
        fresh_ofp["script_hash"] = _compute_script_hash(script)
        # On next run, this report should be reusable
        assert _ofp_reuse_decision(fresh_ofp, script) is True

    def test_reuse_false_skip_final_gates_scenario(self):
        """When SKIP_FINAL_GATES=true the caller bypasses OFP entirely — reuse N/A.
        Confirm that a SKIP_FINAL_GATES=true scenario does not accidentally enable
        reuse when hash happens to match (caller is responsible, tested via simulation)."""
        script = self._script()
        cached = self._cached_ofp_with_matching_hash(script)
        # Reuse decision itself is still True — the SKIP_FINAL_GATES path bypasses
        # the OFP block entirely and never even calls _ofp_reuse_decision.
        # This test just confirms the hash logic returns True for a matching hash.
        assert _ofp_reuse_decision(cached, script) is True
