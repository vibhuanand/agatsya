from __future__ import annotations

from app.services.agent_pipeline_service import _gate_passed_for_safe_to_voice


def test_required_gate_with_stale_after_mutation_cannot_pass():
    assert _gate_passed_for_safe_to_voice(
        "originality_safety",
        {"passed": True, "stale_after_mutation": True},
    ) is False


def test_required_gate_with_refresh_failed_cannot_pass():
    assert _gate_passed_for_safe_to_voice(
        "originality_safety",
        {"passed": True, "refresh_failed": True},
    ) is False
