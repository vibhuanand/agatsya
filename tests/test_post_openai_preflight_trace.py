from __future__ import annotations

from app.services.agent_pipeline_service import _preflight_blocker_trace


def test_post_openai_blocking_trace_includes_issue_location_attempts_and_next_action():
    report = {
        "issues": [
            {
                "severity": "high",
                "type": "source_shaped_reconstruction",
                "chunk_id": "004_reconstruction",
                "problem": "Copied source-shaped reconstruction remains.",
            }
        ],
        "metadata_issues": [
            {
                "severity": "medium",
                "type": "metadata_unsupported_superlative",
                "field": "youtube_metadata.recommended_title",
                "problem": "Unsafe title phrase remains.",
            }
        ],
    }

    trace = _preflight_blocker_trace(
        report,
        python_fix_attempted=True,
        claude_repair_attempted=True,
        openai_repair_attempted=True,
    )

    assert trace[0]["issue_type"] == "source_shaped_reconstruction"
    assert trace[0]["location"] == "004_reconstruction"
    assert trace[0]["python_fix_attempted"] is True
    assert trace[0]["claude_repair_attempted"] is True
    assert trace[0]["openai_repair_attempted"] is True
    assert trace[0]["recommended_next_action"]
    assert trace[1]["location"] == "youtube_metadata.recommended_title"
