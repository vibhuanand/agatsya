from __future__ import annotations

from app.services.repair_routing_service import run_repair_routing


def _script() -> dict:
    return {
        "hindi_narration_chunks": [
            {"chunk_id": "001_investigation", "section_title": "Investigation", "text": ""},
            {"chunk_id": "002_reconstruction", "section_title": "Court reconstruction", "text": ""},
            {"chunk_id": "003_evidence", "section_title": "Evidence sequence", "text": ""},
        ]
    }


def _preflight() -> dict:
    return {
        "blocking": True,
        "issues": [
            {
                "severity": "high",
                "type": "source_shaped_reconstruction",
                "chunk_id": "002_reconstruction",
                "problem": "same source-shaped reconstruction remains",
            }
        ],
        "chunk_repair_targets": [
            {
                "chunk_id": "002_reconstruction",
                "issue_type": "source_shaped_reconstruction",
                "problem": "same source-shaped reconstruction remains",
                "repair_instruction": "rebuild cluster",
            }
        ],
    }


def test_first_source_cluster_routes_to_claude_grouped_repair():
    plan = run_repair_routing(
        {"python_preflight": _preflight()},
        openai_repair_max_chunks=4,
        script_draft=_script(),
        max_cluster_size=4,
        previous_root_cause_attempts={},
    )

    assert plan["route"] == "claude_grouped_repair"
    assert plan["claude_repair_targets"]
    assert not plan["openai_repair_targets"]


def test_same_root_after_one_claude_attempt_routes_to_openai_cluster():
    first = run_repair_routing(
        {"python_preflight": _preflight()},
        openai_repair_max_chunks=4,
        script_draft=_script(),
        max_cluster_size=4,
        previous_root_cause_attempts={},
    )
    key = first["source_copy_reconstruction_clusters"][0]["root_cause_key"]

    second = run_repair_routing(
        {"python_preflight": _preflight()},
        openai_repair_max_chunks=4,
        script_draft=_script(),
        max_cluster_size=4,
        previous_root_cause_attempts={key: 1},
    )

    assert second["route"] == "openai_targeted"
    assert second["claude_repair_skipped_due_previous_failure"] is True
    assert key in second["repeat_root_cause_detected"]
    assert second["openai_repair_targets"]
    assert not second["claude_repair_targets"]
