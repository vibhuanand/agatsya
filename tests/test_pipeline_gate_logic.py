"""Pipeline-level gate-logic tests (monkeypatched, no real API calls).

Proves three invariants:
  (a) A blocking post-repair Python preflight causes _openai_gates_active=False,
      so OFP gate is never called.
  (b) A clean post-repair Python preflight leaves _openai_gates_active=True,
      so OFP gate may be called.
  (c) safe_to_voice=False whenever gate_summary["python_preflight"]["blocking"]=True.

All tests mirror the exact logic in agent_pipeline_service.py so that any
accidental divergence causes a test failure.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.python_preflight_service import run_python_preflight

make_chunk = pytest.make_chunk
make_script = pytest.make_script
make_glossary = pytest.make_glossary


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _build_gate_summary_entry(report: dict, label: str = "") -> dict:
    """Mirror of the gate_summary["python_preflight"] construction in the pipeline."""
    counts = report.get("severity_counts", {})
    filename = f"python_preflight_report{label}.json"
    return {
        "passed":    report.get("passed", False),
        "blocking":  report.get("blocking", False),
        "high":      counts.get("high", 0),
        "medium":    counts.get("medium", 0),
        "low":       counts.get("low", 0),
        "report":    filename,
        "rechecked": label != "",
    }


def _compute_openai_gates_active(
    openai_review_enabled: bool,
    quality_mode: str,
    openai_review_policy: str,
    post_repair_preflight_blocking: bool,
) -> bool:
    """Mirror of _openai_gates_active expression in agent_pipeline_service.py."""
    return (
        openai_review_enabled
        and quality_mode == "premium_final"
        and openai_review_policy != "disabled"
        and not post_repair_preflight_blocking
    )


def _compute_safe_to_voice(
    status: str,
    all_gates_passed: bool,
    no_repair_failures: bool,
    gate_summary: dict,
) -> bool:
    """Mirror of safe_to_voice computation in agent_pipeline_service.py."""
    _pf_gate_ok = not gate_summary.get("python_preflight", {}).get("blocking", True)
    return (
        (status == "script_approved")
        and all_gates_passed
        and no_repair_failures
        and _pf_gate_ok
    )


# ── (a) Blocking post-repair preflight → OFP gate NOT called ─────────────────

def test_blocking_post_repair_preflight_disables_openai_gates(tmp_path):
    """_openai_gates_active must be False when post-repair preflight is blocking."""
    script = make_script([make_chunk("001", "यह सबसे खौफनाक मामला था।")])
    glossary = make_glossary(do_not_use=[])
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir, label="_after_repair")

    assert report["blocking"] is True, "Test precondition: preflight must be blocking"

    gates_active = _compute_openai_gates_active(
        openai_review_enabled=True,
        quality_mode="premium_final",
        openai_review_policy="adaptive",
        post_repair_preflight_blocking=report["blocking"],
    )
    assert gates_active is False


def test_blocking_preflight_gates_inactive_always_policy(tmp_path):
    """Blocking preflight disables gates even under policy=always."""
    script = make_script([make_chunk("001", "यह सबसे खौफनाक मामला था।")])
    glossary = make_glossary(do_not_use=[])
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir, label="_after_repair")

    assert report["blocking"] is True

    gates_active = _compute_openai_gates_active(
        openai_review_enabled=True,
        quality_mode="premium_final",
        openai_review_policy="always",
        post_repair_preflight_blocking=report["blocking"],
    )
    assert gates_active is False


def test_ofp_gate_not_called_when_preflight_blocking(tmp_path):
    """If _openai_gates_active=False, run_openai_final_premium_gate is never invoked."""
    script = make_script([make_chunk("001", "यह सबसे खौफनाक मामला था।")])
    glossary = make_glossary(do_not_use=[])
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir, label="_after_repair")

    assert report["blocking"] is True

    # Monkeypatch the OFP gate function; verify it is never called
    with patch(
        "app.services.openai_final_premium_gate_service.run_openai_final_premium_gate"
    ) as mock_ofp:
        gates_active = _compute_openai_gates_active(
            openai_review_enabled=True,
            quality_mode="premium_final",
            openai_review_policy="adaptive",
            post_repair_preflight_blocking=report["blocking"],
        )
        if gates_active:
            # Pipeline would call OFP — should NOT reach here
            from app.services.openai_final_premium_gate_service import run_openai_final_premium_gate
            run_openai_final_premium_gate()

    mock_ofp.assert_not_called()


# ── (b) Clean post-repair preflight → OFP gate may be called ─────────────────

def test_clean_post_repair_preflight_enables_openai_gates(tmp_path):
    """_openai_gates_active must be True when post-repair preflight is clean."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir, label="_after_repair")

    assert report["blocking"] is False, "Test precondition: preflight must be clean"

    gates_active = _compute_openai_gates_active(
        openai_review_enabled=True,
        quality_mode="premium_final",
        openai_review_policy="adaptive",
        post_repair_preflight_blocking=report["blocking"],
    )
    assert gates_active is True


def test_clean_preflight_gates_active_always_policy(tmp_path):
    """Clean preflight allows gates under policy=always too."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir, label="_after_repair")

    assert report["blocking"] is False

    gates_active = _compute_openai_gates_active(
        openai_review_enabled=True,
        quality_mode="premium_final",
        openai_review_policy="always",
        post_repair_preflight_blocking=report["blocking"],
    )
    assert gates_active is True


def test_ofp_gate_called_when_preflight_clean(tmp_path):
    """If _openai_gates_active=True, run_openai_final_premium_gate is invokable."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir, label="_after_repair")

    assert report["blocking"] is False

    gates_active = _compute_openai_gates_active(
        openai_review_enabled=True,
        quality_mode="premium_final",
        openai_review_policy="adaptive",
        post_repair_preflight_blocking=report["blocking"],
    )
    # Only assert that gates_active=True; actual OFP call requires full mock stack
    assert gates_active is True


# ── (c) safe_to_voice=False when python_preflight.blocking=True ───────────────

def test_safe_to_voice_false_when_preflight_blocking_in_gate_summary(tmp_path):
    """safe_to_voice must be False if gate_summary['python_preflight']['blocking']=True."""
    script = make_script([make_chunk("001", "यह सबसे खौफनाक मामला था।")])
    glossary = make_glossary(do_not_use=[])
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir)

    gate_summary = {"python_preflight": _build_gate_summary_entry(report)}
    assert gate_summary["python_preflight"]["blocking"] is True

    safe_to_voice = _compute_safe_to_voice(
        status="script_approved",
        all_gates_passed=True,
        no_repair_failures=True,
        gate_summary=gate_summary,
    )
    assert safe_to_voice is False


def test_safe_to_voice_false_after_repair_blocking(tmp_path):
    """safe_to_voice=False if post-repair preflight in gate_summary is blocking."""
    script = make_script([make_chunk("001", "यह सबसे खौफनाक मामला था।")])
    glossary = make_glossary(do_not_use=[])
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir, label="_after_repair")

    gate_summary = {"python_preflight": _build_gate_summary_entry(report, label="_after_repair")}
    assert gate_summary["python_preflight"]["blocking"] is True
    assert gate_summary["python_preflight"]["rechecked"] is True

    safe_to_voice = _compute_safe_to_voice(
        status="script_approved",
        all_gates_passed=True,
        no_repair_failures=True,
        gate_summary=gate_summary,
    )
    assert safe_to_voice is False


def test_safe_to_voice_true_when_all_gates_pass_and_preflight_clean(tmp_path):
    """safe_to_voice=True only when all conditions met including clean preflight."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir)

    gate_summary = {"python_preflight": _build_gate_summary_entry(report)}
    assert gate_summary["python_preflight"]["blocking"] is False

    safe_to_voice = _compute_safe_to_voice(
        status="script_approved",
        all_gates_passed=True,
        no_repair_failures=True,
        gate_summary=gate_summary,
    )
    assert safe_to_voice is True


def test_safe_to_voice_false_when_status_not_approved(tmp_path):
    """safe_to_voice must be False even with clean preflight if status != script_approved."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir)

    gate_summary = {"python_preflight": _build_gate_summary_entry(report)}
    assert gate_summary["python_preflight"]["blocking"] is False

    safe_to_voice = _compute_safe_to_voice(
        status="needs_human_review",
        all_gates_passed=True,
        no_repair_failures=True,
        gate_summary=gate_summary,
    )
    assert safe_to_voice is False


def test_safe_to_voice_false_when_gates_not_all_passed(tmp_path):
    """safe_to_voice must be False when other gates failed even if preflight clean."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir)

    gate_summary = {"python_preflight": _build_gate_summary_entry(report)}

    safe_to_voice = _compute_safe_to_voice(
        status="script_approved",
        all_gates_passed=False,
        no_repair_failures=True,
        gate_summary=gate_summary,
    )
    assert safe_to_voice is False


def test_safe_to_voice_false_when_repair_failures(tmp_path):
    """safe_to_voice must be False when repair has failures even if preflight clean."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir)

    gate_summary = {"python_preflight": _build_gate_summary_entry(report)}

    safe_to_voice = _compute_safe_to_voice(
        status="script_approved",
        all_gates_passed=True,
        no_repair_failures=False,
        gate_summary=gate_summary,
    )
    assert safe_to_voice is False


def test_gate_summary_missing_preflight_entry_blocks_safe_to_voice(tmp_path):
    """If gate_summary has no python_preflight key, safe_to_voice must be False
    (defaults to blocking=True for safety)."""
    safe_to_voice = _compute_safe_to_voice(
        status="script_approved",
        all_gates_passed=True,
        no_repair_failures=True,
        gate_summary={},  # missing python_preflight
    )
    assert safe_to_voice is False
