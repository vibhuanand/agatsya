from __future__ import annotations

from app.services.agent_pipeline_service import _retention_failure_is_localized, _retention_score


def test_broad_retention_failure_after_repair_stops_before_openai_repair():
    report = {
        "approved": False,
        "overall_retention_score": 6,
        "curiosity_gap_score": 5,
        "pacing_score": 6,
        "chunk_repair_targets": [
            {"chunk_id": "001_hook", "problem": "weak hook"},
            {"chunk_id": "004_investigation", "problem": "flat pacing"},
            {"chunk_id": "008_court", "problem": "no payoff"},
            {"chunk_id": "012_close", "problem": "weak ending"},
        ],
    }

    assert _retention_score(report) == 6.0
    assert _retention_failure_is_localized(report) is False


def test_localized_retention_issue_can_route_to_targeted_repair():
    report = {
        "approved": False,
        "overall_retention_score": 8,
        "curiosity_gap_score": 7,
        "pacing_score": 8,
        "chunk_repair_targets": [
            {
                "chunk_id": "001_hook",
                "issue_type": "retention_hook",
                "problem": "opening hook needs sharper central question",
                "repair_instruction": "strengthen the first 30 seconds only",
            }
        ],
    }

    assert _retention_failure_is_localized(report) is True
