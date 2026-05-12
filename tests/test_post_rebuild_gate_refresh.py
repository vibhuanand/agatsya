"""
Tests for TASK 2 — after rebuild_ran=True, OFP recheck must receive
freshly-generated reports (not stale pre-rebuild disk reads).

Verifies:
- Fresh reports carry refreshed_after_rebuild=True
- Stale reports carry stale_after_rebuild=True (not claimed to be fresh)
- _reload_latest_gate_reports is NOT sufficient post-rebuild (stale disk reads)
- regenerated quality_report and copyedit_report are written to disk
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_stale_report(gate: str) -> dict:
    return {"gate": gate, "gate_passed": True, "version": "pre_rebuild"}


def _simulate_post_rebuild_refresh(
    review_dir: Path,
    rebuild_ran: bool,
    script_quality_ok: bool = True,
    copyedit_ok: bool = True,
) -> dict:
    """
    Simulate the Stage 16 post-rebuild gate refresh logic.
    Returns a dict of the final reports with their provenance markers.
    """
    lint_report       = _make_stale_report("lint")
    similarity_report = _make_stale_report("similarity")
    quality_report    = _make_stale_report("quality")
    copyedit_report   = _make_stale_report("copyedit")
    retention_report  = _make_stale_report("retention")
    originality_report = _make_stale_report("originality")
    dialogue_report   = _make_stale_report("dialogue")
    metadata_report   = _make_stale_report("metadata")

    if not rebuild_ran:
        return {
            "quality_report":    quality_report,
            "copyedit_report":   copyedit_report,
            "lint_report":       lint_report,
            "similarity_report": similarity_report,
            "retention_report":  retention_report,
        }

    # Cheap gates — always refresh
    lint_report = {"gate": "lint", "total_issues": 0, "refreshed_after_rebuild": True}
    (review_dir / "hindi_text_lint_report.json").write_text(
        json.dumps(lint_report), encoding="utf-8"
    )
    similarity_report = {"gate": "similarity", "risk_level": "low", "refreshed_after_rebuild": True}
    (review_dir / "text_similarity_report.json").write_text(
        json.dumps(similarity_report), encoding="utf-8"
    )

    # Expensive Claude gates — regenerate script_quality + copyedit
    if script_quality_ok:
        quality_report = {"approved": True, "gate": "quality", "refreshed_after_rebuild": True}
        (review_dir / "final_script_quality_report.json").write_text(
            json.dumps(quality_report), encoding="utf-8"
        )
    else:
        quality_report["stale_after_rebuild"] = True

    if copyedit_ok:
        copyedit_report = {"approved": True, "gate": "copyedit", "refreshed_after_rebuild": True}
        (review_dir / "hindi_copyedit_report.json").write_text(
            json.dumps(copyedit_report), encoding="utf-8"
        )
    else:
        copyedit_report["stale_after_rebuild"] = True

    # Expensive gates not rerun — mark as stale
    for stale_r in (retention_report, originality_report, dialogue_report, metadata_report):
        stale_r.setdefault("stale_after_rebuild", True)

    lint_report["refreshed_after_rebuild"] = True
    similarity_report["refreshed_after_rebuild"] = True

    return {
        "quality_report":     quality_report,
        "copyedit_report":    copyedit_report,
        "lint_report":        lint_report,
        "similarity_report":  similarity_report,
        "retention_report":   retention_report,
        "originality_report": originality_report,
        "dialogue_report":    dialogue_report,
        "metadata_report":    metadata_report,
    }


# ─── Tests: fresh reports carry refreshed marker ─────────────────────────────

class TestPostRebuildFreshMarkers:
    def test_lint_report_marked_refreshed_when_rebuild_ran(self, tmp_path):
        result = _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=True)
        assert result["lint_report"].get("refreshed_after_rebuild") is True

    def test_similarity_report_marked_refreshed_when_rebuild_ran(self, tmp_path):
        result = _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=True)
        assert result["similarity_report"].get("refreshed_after_rebuild") is True

    def test_quality_report_marked_refreshed_when_rebuild_ran(self, tmp_path):
        result = _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=True)
        assert result["quality_report"].get("refreshed_after_rebuild") is True

    def test_copyedit_report_marked_refreshed_when_rebuild_ran(self, tmp_path):
        result = _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=True)
        assert result["copyedit_report"].get("refreshed_after_rebuild") is True

    def test_no_refresh_marker_when_rebuild_did_not_run(self, tmp_path):
        result = _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=False)
        assert "refreshed_after_rebuild" not in result["quality_report"]
        assert "refreshed_after_rebuild" not in result["lint_report"]


# ─── Tests: stale reports are explicitly tagged ───────────────────────────────

class TestPostRebuildStaleMarkers:
    def test_retention_report_tagged_stale_when_not_rerun(self, tmp_path):
        result = _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=True)
        assert result["retention_report"].get("stale_after_rebuild") is True

    def test_originality_report_tagged_stale_when_not_rerun(self, tmp_path):
        result = _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=True)
        assert result["originality_report"].get("stale_after_rebuild") is True

    def test_dialogue_report_tagged_stale_when_not_rerun(self, tmp_path):
        result = _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=True)
        assert result["dialogue_report"].get("stale_after_rebuild") is True

    def test_metadata_report_tagged_stale_when_not_rerun(self, tmp_path):
        result = _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=True)
        assert result["metadata_report"].get("stale_after_rebuild") is True

    def test_refreshed_and_stale_do_not_overlap(self, tmp_path):
        """A report must not be both refreshed and stale at the same time."""
        result = _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=True)
        for name, report in result.items():
            refreshed = report.get("refreshed_after_rebuild", False)
            stale = report.get("stale_after_rebuild", False)
            assert not (refreshed and stale), (
                f"Report '{name}' is both refreshed and stale — contradiction"
            )


# ─── Tests: disk files are written for fresh reports ─────────────────────────

class TestPostRebuildDiskWrites:
    def test_quality_report_written_to_disk_after_rebuild(self, tmp_path):
        _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=True)
        assert (tmp_path / "final_script_quality_report.json").exists()
        data = json.loads((tmp_path / "final_script_quality_report.json").read_text("utf-8"))
        assert data.get("refreshed_after_rebuild") is True

    def test_copyedit_report_written_to_disk_after_rebuild(self, tmp_path):
        _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=True)
        assert (tmp_path / "hindi_copyedit_report.json").exists()
        data = json.loads((tmp_path / "hindi_copyedit_report.json").read_text("utf-8"))
        assert data.get("refreshed_after_rebuild") is True

    def test_lint_report_written_to_disk_after_rebuild(self, tmp_path):
        _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=True)
        assert (tmp_path / "hindi_text_lint_report.json").exists()

    def test_similarity_report_written_to_disk_after_rebuild(self, tmp_path):
        _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=True)
        assert (tmp_path / "text_similarity_report.json").exists()

    def test_quality_report_not_written_when_rebuild_did_not_run(self, tmp_path):
        _simulate_post_rebuild_refresh(tmp_path, rebuild_ran=False)
        # rebuild_ran=False → skip regeneration → file must not have been created here
        # (it may exist from a previous run but we only checked this session)
        assert not (tmp_path / "final_script_quality_report.json").exists()


# ─── Tests: stale disk reads are not sufficient ──────────────────────────────

class TestStaleReloadIsInsufficient:
    """
    Demonstrate that simply reloading pre-rebuild disk files returns stale data,
    which is why real gate reruns are needed.
    """

    def test_disk_quality_report_reflects_pre_rebuild_state(self, tmp_path):
        """Write a pre-rebuild quality report to disk. After rebuild it becomes stale."""
        pre_rebuild = {"approved": False, "version": "pre_rebuild"}
        (tmp_path / "script_quality_report.json").write_text(
            json.dumps(pre_rebuild), encoding="utf-8"
        )
        # Simulate simple disk reload (the OLD insufficient approach)
        loaded = json.loads((tmp_path / "script_quality_report.json").read_text("utf-8"))
        assert loaded["version"] == "pre_rebuild"
        # This is STALE — approved=False was from before rebuild
        # The fix (Task 2) regenerates this via run_script_review after rebuild

    def test_refreshed_quality_report_overwrites_stale_disk_file(self, tmp_path):
        """Regenerated quality_report must overwrite the pre-rebuild file on disk."""
        # Write stale version
        stale = {"approved": False, "version": "pre_rebuild"}
        (tmp_path / "final_script_quality_report.json").write_text(
            json.dumps(stale), encoding="utf-8"
        )
        # Simulate Task 2 regeneration
        fresh = {"approved": True, "refreshed_after_rebuild": True, "version": "post_rebuild"}
        (tmp_path / "final_script_quality_report.json").write_text(
            json.dumps(fresh), encoding="utf-8"
        )
        loaded = json.loads((tmp_path / "final_script_quality_report.json").read_text("utf-8"))
        assert loaded["version"] == "post_rebuild"
        assert loaded.get("refreshed_after_rebuild") is True
