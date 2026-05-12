from __future__ import annotations

import json

from app.services.openai_targeted_chunk_repair_service import _build_user_content
from app.services.repair_routing_service import run_repair_routing


def _script() -> dict:
    return {
        "hindi_narration_chunks": [
            {"chunk_id": "001_investigation", "section_title": "Investigation", "text": "a"},
            {"chunk_id": "002_reconstruction", "section_title": "Court reconstruction", "text": "b"},
            {"chunk_id": "003_evidence", "section_title": "Evidence", "text": "c"},
            {"chunk_id": "004_memory", "section_title": "Memory", "text": "d"},
        ]
    }


def test_source_copy_after_claude_routes_to_openai_cluster():
    preflight = {
        "issues": [{"severity": "high", "type": "exact_source_quote_copy", "chunk_id": "002_reconstruction", "problem": "copied quote"}],
        "chunk_repair_targets": [{
            "chunk_id": "002_reconstruction",
            "issue_type": "exact_source_quote_copy",
            "problem": "copied quote",
            "repair_instruction": "remove source quote",
        }],
    }
    first = run_repair_routing({"python_preflight": preflight}, 4, script_draft=_script(), max_cluster_size=4)
    key = first["source_copy_reconstruction_clusters"][0]["root_cause_key"]

    plan = run_repair_routing(
        {"python_preflight": preflight},
        4,
        script_draft=_script(),
        max_cluster_size=4,
        previous_root_cause_attempts={key: 1},
    )

    assert plan["route"] == "openai_targeted"
    assert all(t["repair_type"] == "source_copy_reconstruction_cluster" for t in plan["openai_repair_targets"])


def test_non_source_minor_issue_does_not_route_to_openai_cluster():
    preflight = {
        "issues": [{"severity": "medium", "type": "hinglish_level", "chunk_id": "002_reconstruction", "problem": "miss कर"}],
        "chunk_repair_targets": [{
            "chunk_id": "002_reconstruction",
            "issue_type": "hinglish_level_mismatch",
            "problem": "miss कर",
            "repair_instruction": "fix Hindi",
        }],
    }

    plan = run_repair_routing({"python_preflight": preflight}, 4, script_draft=_script(), max_cluster_size=4)

    assert plan["source_copy_reconstruction_clusters"] == []
    assert plan["openai_repair_targets"] == []


def test_openai_repair_payload_is_chunk_scoped_not_full_script():
    content = _build_user_content(
        target={"chunk_id": "002_reconstruction", "issue_type": "source_shaped_reconstruction", "problem": "copied", "repair_instruction": "repair"},
        current_chunk={"chunk_id": "002_reconstruction", "text": "target only"},
        fact_lock={"verified_people": [{"name": "A"}], "legal_outcome": {"trial_result": "x"}},
        blueprint={"main_hook": "hook", "emotional_anchor": "anchor", "sensitivity_rules": ["safe"]},
        hinglish_level=2,
    )
    payload = json.loads(content)

    assert "chunk_to_repair" in payload
    assert payload["chunk_to_repair"]["text"] == "target only"
    assert "hindi_narration_chunks" not in payload
    assert "full_script" not in payload
