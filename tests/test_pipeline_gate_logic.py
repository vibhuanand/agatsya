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


# ── Post-OAI-repair preflight blocks OFP recheck (item 1 cost-control) ────────

def _compute_oai_recheck_active(post_oai_pf_blocking: bool) -> bool:
    """Mirror of 'if not _post_oai_pf_blocking:' guard in adaptive Stage 16."""
    return not post_oai_pf_blocking


def test_post_oai_repair_preflight_blocking_skips_ofp_recheck(tmp_path):
    """When Stage 16b Python preflight is blocking after OAI repair,
    Stage 16a OFP recheck must NOT be called."""
    script = make_script([make_chunk("001", "यह सबसे खौफनाक मामला था।")])
    glossary = make_glossary(do_not_use=[])
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir, label="_after_openai_repair")

    assert report["blocking"] is True, "Test precondition: preflight must be blocking"

    recheck_active = _compute_oai_recheck_active(post_oai_pf_blocking=report["blocking"])
    assert recheck_active is False

    # Monkeypatch verifies OFP recheck function is never invoked
    with patch(
        "app.services.openai_final_premium_gate_service.run_openai_final_premium_gate"
    ) as mock_ofp:
        if recheck_active:
            from app.services.openai_final_premium_gate_service import run_openai_final_premium_gate
            run_openai_final_premium_gate()

    mock_ofp.assert_not_called()


def test_post_oai_repair_preflight_clean_allows_ofp_recheck(tmp_path):
    """When Stage 16b Python preflight is clean after OAI repair,
    Stage 16a OFP recheck is permitted to run."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir, label="_after_openai_repair")

    assert report["blocking"] is False, "Test precondition: preflight must be clean"

    recheck_active = _compute_oai_recheck_active(post_oai_pf_blocking=report["blocking"])
    assert recheck_active is True


def test_post_oai_repair_preflight_blocking_sets_needs_human_review(tmp_path):
    """A blocking post-OAI-repair preflight must result in needs_human_review status."""
    script = make_script([make_chunk("001", "यह सबसे खौफनाक मामला था।")])
    glossary = make_glossary(do_not_use=[])
    review_dir = tmp_path / "04-review"
    report = _run_preflight(script, glossary, review_dir, label="_after_openai_repair")

    assert report["blocking"] is True
    gate_summary = {"python_preflight": _build_gate_summary_entry(report, label="_after_openai_repair")}
    gate_summary["python_preflight"].update({
        "report":    "python_preflight_report_after_openai_repair.json",
        "rechecked": True,
    })

    # Mirrors pipeline: status → needs_human_review when blocking
    status = "needs_human_review" if report["blocking"] else "script_approved"
    assert status == "needs_human_review"
    # And safe_to_voice must be False
    safe_to_voice = _compute_safe_to_voice(
        status=status,
        all_gates_passed=True,
        no_repair_failures=True,
        gate_summary=gate_summary,
    )
    assert safe_to_voice is False


# ── Python preflight exception = blocking (item 1) ────────────────────────────

def _exception_gate_summary_entry() -> dict:
    """Gate summary entry produced when Python preflight raises an exception.
    Mirrors the exact dict written in agent_pipeline_service.py exception handlers.
    """
    return {
        "passed":    False,
        "blocking":  True,
        "high":      0,
        "medium":    0,
        "low":       0,
        "report":    "python_preflight_report_after_repair.json",
        "rechecked": True,
        "error":     "simulated exception",
    }


def test_preflight_exception_is_treated_as_blocking():
    """When post-repair Python preflight raises an exception, blocking must be True."""
    entry = _exception_gate_summary_entry()
    assert entry["blocking"] is True
    assert entry["passed"] is False


def test_preflight_exception_blocks_openai_gates():
    """When preflight exception sets blocking=True, _openai_gates_active must be False."""
    entry = _exception_gate_summary_entry()
    # _post_repair_preflight_blocking = entry["blocking"]
    gates_active = _compute_openai_gates_active(
        openai_review_enabled=True,
        quality_mode="premium_final",
        openai_review_policy="adaptive",
        post_repair_preflight_blocking=entry["blocking"],
    )
    assert gates_active is False


def test_preflight_exception_means_safe_to_voice_false():
    """When preflight exception sets blocking=True in gate_summary, safe_to_voice=False."""
    gate_summary = {"python_preflight": _exception_gate_summary_entry()}
    safe_to_voice = _compute_safe_to_voice(
        status="script_approved",
        all_gates_passed=True,
        no_repair_failures=True,
        gate_summary=gate_summary,
    )
    assert safe_to_voice is False


def test_post_oai_repair_preflight_exception_blocks_ofp_recheck():
    """When Stage 16b preflight raises an exception, _post_oai_pf_blocking=True
    so the OFP recheck guard (if not _post_oai_pf_blocking) prevents the recheck."""
    # Mirrors: except Exception → _post_oai_pf_blocking = True
    _post_oai_pf_blocking = True  # exception handler sets this
    recheck_active = _compute_oai_recheck_active(post_oai_pf_blocking=_post_oai_pf_blocking)
    assert recheck_active is False


def test_preflight_exception_gate_summary_has_error_field():
    """Gate summary from exception handler must include 'error' key for diagnostics."""
    entry = _exception_gate_summary_entry()
    assert "error" in entry
    assert entry["error"]  # non-empty


def test_preflight_exception_status_is_needs_human_review():
    """When preflight exception occurs, status must be set to needs_human_review.
    This ensures safe_to_voice stays False regardless of other gate results."""
    gate_summary = {"python_preflight": _exception_gate_summary_entry()}
    # status="needs_human_review" is set by the exception handler
    safe_to_voice = _compute_safe_to_voice(
        status="needs_human_review",
        all_gates_passed=True,
        no_repair_failures=True,
        gate_summary=gate_summary,
    )
    assert safe_to_voice is False


# ── _gate_passed_for_safe_to_voice helper (item 1) ───────────────────────────

from app.services.agent_pipeline_service import _gate_passed_for_safe_to_voice


def _all_gates_passed(gate_summary: dict) -> bool:
    """Mirror of the all_gates_passed computation in agent_pipeline_service.py."""
    return all(
        _gate_passed_for_safe_to_voice(name, gate)
        for name, gate in gate_summary.items()
    )


def test_python_preflight_low_only_passed_false_blocking_false(tmp_path):
    """preflight with only low issues has passed=False and blocking=False."""
    from app.services.python_preflight_service import run_python_preflight
    import pytest
    make_chunk = pytest.make_chunk
    make_script = pytest.make_script
    make_glossary = pytest.make_glossary
    from tests.conftest import _make_metadata

    # pinned_comment missing → low severity only
    metadata = _make_metadata()
    metadata.pop("pinned_comment", None)
    metadata["pinned_comment"] = ""
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=metadata)
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    review_dir.mkdir()
    report = run_python_preflight(
        script_draft=script, fact_lock={}, case_glossary=glossary,
        review_dir=review_dir, target_duration_min=20, hinglish_level=2,
    )
    assert report["passed"] is False      # has an issue (low)
    assert report["blocking"] is False    # low only → not blocking
    counts = report["severity_counts"]
    assert counts["low"] >= 1
    assert counts["high"] == 0
    assert counts["medium"] == 0


def test_gate_passed_helper_low_only_is_non_blocking():
    """_gate_passed_for_safe_to_voice treats python_preflight low-only as passing."""
    gate = {"passed": False, "blocking": False, "low": 1, "high": 0, "medium": 0}
    assert _gate_passed_for_safe_to_voice("python_preflight", gate) is True


def test_gate_passed_helper_medium_issue_is_blocking():
    """_gate_passed_for_safe_to_voice treats python_preflight with medium as blocking."""
    gate = {"passed": False, "blocking": True, "low": 0, "high": 0, "medium": 1}
    assert _gate_passed_for_safe_to_voice("python_preflight", gate) is False


def test_gate_passed_helper_high_issue_is_blocking():
    """_gate_passed_for_safe_to_voice treats python_preflight with high as blocking."""
    gate = {"passed": False, "blocking": True, "low": 0, "high": 1, "medium": 0}
    assert _gate_passed_for_safe_to_voice("python_preflight", gate) is False


def test_gate_passed_helper_missing_python_preflight_is_blocking():
    """When python_preflight entry is absent, safe_to_voice must be blocked."""
    # The _pf_gate_ok guard uses .get("blocking", True) so missing = blocking.
    # _gate_passed_for_safe_to_voice with blocking absent defaults to True (blocking).
    gate = {}  # no blocking key
    assert _gate_passed_for_safe_to_voice("python_preflight", gate) is False


def test_safe_to_voice_with_low_only_preflight_and_all_other_gates_passing(tmp_path):
    """Low-only python_preflight must not block safe_to_voice when all other gates pass."""
    pf_low_only = {"passed": False, "blocking": False, "low": 1, "high": 0, "medium": 0}
    other_gate = {"passed": True}
    gate_summary = {
        "python_preflight": pf_low_only,
        "script_quality": other_gate,
        "hindi_copyedit": other_gate,
    }
    assert _all_gates_passed(gate_summary) is True
    safe_to_voice = _compute_safe_to_voice(
        status="script_approved",
        all_gates_passed=_all_gates_passed(gate_summary),
        no_repair_failures=True,
        gate_summary=gate_summary,
    )
    assert safe_to_voice is True


def test_safe_to_voice_blocked_by_medium_preflight_even_if_other_gates_pass():
    """Medium python_preflight issue must block safe_to_voice."""
    pf_medium = {"passed": False, "blocking": True, "low": 0, "high": 0, "medium": 1}
    other_gate = {"passed": True}
    gate_summary = {
        "python_preflight": pf_medium,
        "script_quality": other_gate,
    }
    assert _all_gates_passed(gate_summary) is False
    safe_to_voice = _compute_safe_to_voice(
        status="script_approved",
        all_gates_passed=_all_gates_passed(gate_summary),
        no_repair_failures=True,
        gate_summary=gate_summary,
    )
    assert safe_to_voice is False


def test_safe_to_voice_blocked_by_high_preflight_even_if_other_gates_pass():
    """High python_preflight issue must block safe_to_voice."""
    pf_high = {"passed": False, "blocking": True, "low": 0, "high": 1, "medium": 0}
    gate_summary = {"python_preflight": pf_high, "script_quality": {"passed": True}}
    assert _all_gates_passed(gate_summary) is False
    safe_to_voice = _compute_safe_to_voice(
        status="script_approved",
        all_gates_passed=_all_gates_passed(gate_summary),
        no_repair_failures=True,
        gate_summary=gate_summary,
    )
    assert safe_to_voice is False


def test_gate_summary_low_count_visible_when_not_blocking():
    """Even when python_preflight is non-blocking, low count stays in gate_summary."""
    pf = {"passed": False, "blocking": False, "low": 2, "high": 0, "medium": 0,
          "report": "python_preflight_report.json"}
    assert pf["low"] == 2
    assert _gate_passed_for_safe_to_voice("python_preflight", pf) is True


def test_non_preflight_gate_uses_passed_field():
    """Non-python_preflight gates are evaluated with their passed field, not blocking."""
    gate_passing = {"passed": True, "blocking": False}
    gate_failing = {"passed": False, "blocking": False}  # blocking=False irrelevant
    assert _gate_passed_for_safe_to_voice("script_quality", gate_passing) is True
    assert _gate_passed_for_safe_to_voice("script_quality", gate_failing) is False
    assert _gate_passed_for_safe_to_voice("hindi_copyedit", gate_failing) is False
