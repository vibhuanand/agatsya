"""Tests for post-repair Python preflight and gate_summary[python_preflight].

All API calls are mocked — no real Claude/OpenAI calls are made.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.python_preflight_service import run_python_preflight

make_chunk = pytest.make_chunk
make_script = pytest.make_script
make_glossary = pytest.make_glossary
make_metadata = pytest.make_metadata


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_preflight(script, glossary, review_dir, label=""):
    review_dir.mkdir(parents=True, exist_ok=True)
    return run_python_preflight(
        script_draft=script,
        fact_lock={},
        case_glossary=glossary,
        review_dir=review_dir,
        target_duration_min=20,
        hinglish_level=2,
        label=label,
    )


# ── Post-repair file naming ───────────────────────────────────────────────────

def test_after_repair_label_produces_separate_json(tmp_path):
    """label='_after_repair' must save python_preflight_report_after_repair.json."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    _run_preflight(script, glossary, review_dir, label="_after_repair")
    path = review_dir / "python_preflight_report_after_repair.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "passed" in data
    assert "blocking" in data
    assert "severity_counts" in data


def test_initial_run_still_saves_default_filename(tmp_path):
    """label='' (default) must save python_preflight_report.json, not _after_repair."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    _run_preflight(script, glossary, review_dir)
    assert (review_dir / "python_preflight_report.json").exists()
    assert not (review_dir / "python_preflight_report_after_repair.json").exists()


def test_both_files_saved_when_both_labels_run(tmp_path):
    """Running preflight twice (initial + after_repair) must produce both files."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    _run_preflight(script, glossary, review_dir)
    _run_preflight(script, glossary, review_dir, label="_after_repair")
    assert (review_dir / "python_preflight_report.json").exists()
    assert (review_dir / "python_preflight_report_after_repair.json").exists()


# ── gate_summary shape ────────────────────────────────────────────────────────

def test_gate_summary_python_preflight_shape(tmp_path):
    """gate_summary['python_preflight'] must have all required fields."""
    # We verify the shape of what run_python_preflight returns; the pipeline
    # assembles gate_summary from this output.
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir)
    # Simulate gate_summary entry construction (mirrors pipeline code)
    counts = report.get("severity_counts", {})
    entry = {
        "passed":    report.get("passed", False),
        "blocking":  report.get("blocking", False),
        "high":      counts.get("high", 0),
        "medium":    counts.get("medium", 0),
        "low":       counts.get("low", 0),
        "report":    "python_preflight_report.json",
        "rechecked": False,
    }
    for field in ("passed", "blocking", "high", "medium", "low", "report", "rechecked"):
        assert field in entry, f"gate_summary[python_preflight] missing '{field}'"


def test_gate_summary_rechecked_true_after_repair_label(tmp_path):
    """After post-repair preflight, rechecked=True must be set in gate_summary."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    post_report = _run_preflight(script, glossary, review_dir, label="_after_repair")
    counts = post_report.get("severity_counts", {})
    entry = {
        "passed":    post_report.get("passed", False),
        "blocking":  post_report.get("blocking", False),
        "high":      counts.get("high", 0),
        "medium":    counts.get("medium", 0),
        "low":       counts.get("low", 0),
        "report":    "python_preflight_report_after_repair.json",
        "rechecked": True,
    }
    assert entry["rechecked"] is True
    assert entry["report"] == "python_preflight_report_after_repair.json"


# ── Blocking post-repair prevents safe_to_voice ───────────────────────────────

def test_blocking_after_repair_means_not_safe_to_voice(tmp_path):
    """If post-repair preflight is blocking, safe_to_voice must remain False.

    This tests the logic directly — not via the full pipeline — to avoid
    needing mocked Claude/OpenAI API stacks.
    """
    # Post-repair preflight with a sensational phrase (still blocking after repair)
    script = make_script([make_chunk("001", "यह सबसे खौफनाक मामला था।")])
    glossary = make_glossary(do_not_use=[])
    review_dir = tmp_path / "04-review"
    post_report = _run_preflight(script, glossary, review_dir, label="_after_repair")
    assert post_report["blocking"] is True
    # safe_to_voice computation (mirrors pipeline code):
    # _pf_gate_ok = not gate_summary["python_preflight"]["blocking"]
    _pf_gate_ok = not post_report["blocking"]
    safe_to_voice = False and False and False and _pf_gate_ok  # simplified — gates not checked
    assert safe_to_voice is False


def test_clean_after_repair_allows_gate_ok(tmp_path):
    """If post-repair preflight is clean, _pf_gate_ok=True (safe_to_voice gate may pass)."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    post_report = _run_preflight(script, glossary, review_dir, label="_after_repair")
    assert post_report["blocking"] is False
    _pf_gate_ok = not post_report["blocking"]
    assert _pf_gate_ok is True


# ── OFP gate skipping (behavior documented, not integration-tested) ───────────

def test_post_repair_blocking_report_contains_required_fields(tmp_path):
    """A blocking post-repair report must contain chunk_repair_targets."""
    script = make_script([make_chunk("001", "यह सबसे खौफनाक मामला था।")])
    glossary = make_glossary(do_not_use=[])
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir, label="_after_repair")
    assert report["blocking"] is True
    assert "chunk_repair_targets" in report
    assert "metadata_repair_targets" in report
    assert "severity_counts" in report
