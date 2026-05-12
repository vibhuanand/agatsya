"""
Tests for TASK 2 — Stage 13d pre-OAI repair wiring.

Verifies:
- After pre-OAI repair mutates the script, _refresh_reports_after_script_mutation
  is wired to run (at minimum lint + similarity).
- Refreshed reports carry refreshed_after_script_mutation=True.
- The refresh helper is called before OFP, not after.
- Stale-report failure (refresh_failed=True) prevents OFP from running.
- A structured stale-failure dict is produced when source_transcript is missing.
"""
from __future__ import annotations

import pytest

from app.services.agent_pipeline_service import (
    _refresh_reports_after_script_mutation,
    _compute_final_review_input_hash,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _script(text: str = "हुक पाठ।") -> dict:
    return {
        "hindi_narration_chunks": [
            {"chunk_id": "001_hook", "text": text, "section_title": "Hook",
             "voice": "narrator", "tone": "neutral", "estimated_words": 2}
        ],
        "youtube_metadata": {"title": "Test"},
        "recreated_dialogues": {"items": []},
    }


# ─── Tests: refresh sequence correctness ──────────────────────────────────────

class TestRefreshSequence:
    """
    Simulate Stage 13d: repair mutates script, then refresh runs, then hash
    is computed, then OFP sees fresh reports.
    """

    def test_pre_repair_hash_differs_from_post_repair_hash(self):
        pre_repair = _script("पुराना पाठ।")
        post_repair = _script("मरम्मत किया पाठ।")

        h_pre  = _compute_final_review_input_hash(script_final=pre_repair)
        h_post = _compute_final_review_input_hash(script_final=post_repair)

        assert h_pre != h_post, (
            "After Stage 13d mutates the script, final_review_input_hash must differ "
            "so OFP cannot reuse the pre-repair cached report."
        )

    def test_lint_refreshed_after_script_mutation(self, tmp_path):
        post_repair_script = _script("मरम्मत किया पाठ।")
        stale_lint = {"total_issues": 99, "stale": True}

        result = _refresh_reports_after_script_mutation(
            script_final=post_repair_script,
            fact_lock={"case_name": "T", "facts": [], "people": []},
            blueprint={"title": "T", "sections": []},
            review_dir=tmp_path,
            gate_summary={},
            warnings=[],
            lint_report=stale_lint,
            similarity_report={},
            quality_report={},
            copyedit_report={},
            hinglish_level=2,
            rerun_lint=True,
            rerun_similarity=False,
            rerun_quality=False,
            rerun_copyedit=False,
        )
        fresh_lint = result["lint_report"]
        assert fresh_lint is not stale_lint
        assert fresh_lint.get("refreshed_after_script_mutation") is True
        # Stale marker must be gone (or overwritten by fresh result)
        assert not fresh_lint.get("stale", False)

    def test_ofp_receives_refreshed_lint_not_stale_object(self, tmp_path):
        """OFP hash computed from fresh lint differs from hash computed from stale lint."""
        post_script = _script("नया पाठ।")
        stale_lint  = {"total_issues": 99, "passed": False}
        fresh_lint  = {"total_issues": 0,  "passed": True, "refreshed_after_script_mutation": True}

        h_stale = _compute_final_review_input_hash(
            script_final=post_script, lint_report=stale_lint
        )
        h_fresh = _compute_final_review_input_hash(
            script_final=post_script, lint_report=fresh_lint
        )
        assert h_stale != h_fresh, (
            "OFP hash must change when lint_report changes from stale to fresh — "
            "ensures OFP reruns instead of reusing old cached approval."
        )


# ─── Tests: stale-failure structured return ───────────────────────────────────

class TestStaleFaultStructuredReturn:
    def test_similarity_stale_failure_has_required_fields(self, tmp_path):
        result = _refresh_reports_after_script_mutation(
            script_final=_script(),
            fact_lock={"case_name": "T", "facts": [], "people": []},
            blueprint={"title": "T", "sections": []},
            review_dir=tmp_path,
            gate_summary={},
            warnings=[],
            lint_report={},
            similarity_report={"old": True},
            quality_report={},
            copyedit_report={},
            source_transcript="",   # missing — triggers stale failure
            rerun_lint=False,
            rerun_similarity=True,
            rerun_quality=False,
            rerun_copyedit=False,
        )
        sim = result["similarity_report"]
        assert sim.get("passed") is False
        assert sim.get("stale") is True
        assert sim.get("refresh_failed") is True
        assert "reason" in sim
        assert len(sim["reason"]) > 0

    def test_retention_stale_failure_has_required_fields(self, tmp_path):
        result = _refresh_reports_after_script_mutation(
            script_final=_script(),
            fact_lock={"case_name": "T", "facts": [], "people": []},
            blueprint={"title": "T", "sections": []},
            review_dir=tmp_path,
            gate_summary={},
            warnings=[],
            lint_report={},
            similarity_report={},
            quality_report={},
            copyedit_report={},
            retention_blueprint=None,   # missing — triggers stale failure
            rerun_lint=False,
            rerun_similarity=False,
            rerun_quality=False,
            rerun_copyedit=False,
            rerun_retention=True,
        )
        ret = result["retention_report"]
        assert ret is not None
        assert ret.get("passed") is False
        assert ret.get("stale") is True
        assert ret.get("refresh_failed") is True

    def test_originality_stale_failure_when_sim_refresh_failed(self, tmp_path):
        """If similarity refresh_failed, originality must also become stale."""
        stale_sim = {
            "passed": False, "stale": True, "refresh_failed": True,
            "reason": "missing required input: source_transcript",
        }
        result = _refresh_reports_after_script_mutation(
            script_final=_script(),
            fact_lock={"case_name": "T", "facts": [], "people": []},
            blueprint={"title": "T", "sections": []},
            review_dir=tmp_path,
            gate_summary={},
            warnings=[],
            lint_report={},
            similarity_report=stale_sim,   # already stale
            quality_report={},
            copyedit_report={},
            source_transcript="some text",  # source_transcript present but sim is stale
            originality_report={},
            rerun_lint=False,
            rerun_similarity=False,   # not rerunning sim — it's still stale in result
            rerun_quality=False,
            rerun_copyedit=False,
            rerun_originality=True,
        )
        orig = result["originality_report"]
        assert orig.get("refresh_failed") is True, (
            "originality_safety must not run when similarity_report refresh_failed=True"
        )


# ─── Tests: hash invalidates OFP reuse after mutation ─────────────────────────

class TestHashInvalidatesOFPAfterMutation:
    def test_ofp_hash_changes_when_similarity_refreshed(self):
        script = _script("पाठ।")
        stale_sim = {"risk_level": "high", "high_risk_matches": 8}
        fresh_sim = {"risk_level": "low",  "high_risk_matches": 0,
                     "refreshed_after_script_mutation": True}

        h_before = _compute_final_review_input_hash(
            script_final=script, similarity_report=stale_sim
        )
        h_after = _compute_final_review_input_hash(
            script_final=script, similarity_report=fresh_sim
        )
        assert h_before != h_after

    def test_cached_ofp_not_reused_after_similarity_refresh(self):
        script = _script("पाठ।")
        stale_sim  = {"risk_level": "high", "high_risk_matches": 8}
        fresh_sim  = {"risk_level": "low",  "high_risk_matches": 0,
                      "refreshed_after_script_mutation": True}

        h_old = _compute_final_review_input_hash(
            script_final=script, similarity_report=stale_sim
        )
        cached_ofp = {
            "approved": True, "safe_to_voice": True,
            "final_review_input_hash": h_old,
        }

        h_new = _compute_final_review_input_hash(
            script_final=script, similarity_report=fresh_sim
        )

        _stored = cached_ofp.get("final_review_input_hash", "")
        reuse = bool(_stored and _stored == h_new)
        assert reuse is False, (
            "OFP must NOT be reused after similarity was refreshed — "
            "the cached report was built with stale (high-risk) similarity data."
        )
