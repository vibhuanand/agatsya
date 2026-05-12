from __future__ import annotations

import json

import pytest

from app.services.agent_pipeline_service import _filter_openai_targets_to_blockers
from app.services.openai_targeted_chunk_repair_service import _build_user_content
from app.services.python_preflight_service import run_python_preflight

make_chunk = pytest.make_chunk
make_metadata = pytest.make_metadata
make_script = pytest.make_script
make_glossary = pytest.make_glossary


def test_openai_repair_receives_only_current_blocking_chunks():
    targets = [
        {"chunk_id": "001_hook", "repair_instruction": "fix hook"},
        {"chunk_id": "004_evidence", "repair_instruction": "fix source copy"},
        {"chunk_id": "009_close", "repair_instruction": "fix ending"},
    ]
    preflight = {
        "chunk_repair_targets": [
            {"chunk_id": "004_evidence", "issue_type": "exact_source_quote_copy"}
        ],
        "issues": [{"chunk_id": "004_evidence", "severity": "high"}],
    }

    filtered = _filter_openai_targets_to_blockers(targets, preflight)

    assert [t["chunk_id"] for t in filtered] == ["004_evidence"]


def test_openai_repair_payload_is_compact_not_full_package():
    payload = json.loads(
        _build_user_content(
            target={"chunk_id": "004_evidence", "issue_type": "source_copy", "problem": "copied", "repair_instruction": "paraphrase"},
            current_chunk={"chunk_id": "004_evidence", "text": "target chunk only"},
            fact_lock={"verified_people": [{"name": "A"}], "legal_outcome": {"trial_result": "x"}},
            blueprint={"main_hook": "hook", "emotional_anchor": "anchor", "sensitivity_rules": ["safe"]},
            hinglish_level=2,
        )
    )

    assert payload["chunk_to_repair"]["text"] == "target chunk only"
    assert "hindi_narration_chunks" not in payload
    assert "full_script" not in payload


def test_deterministic_metadata_issues_are_fixed_before_openai(tmp_path):
    metadata = make_metadata(title="Most brutal case with 27 stab wounds")
    script = make_script([make_chunk("001_hook", "साफ़ पाठ।")], metadata=metadata)

    report = run_python_preflight(
        script_draft=script,
        fact_lock={},
        case_glossary=make_glossary(),
        review_dir=tmp_path / "04-review",
        target_duration_min=20,
        hinglish_level=2,
    )

    assert report["metadata_python_fixes_applied"]
    assert "most brutal" not in script["youtube_metadata"]["recommended_title"].lower()
    assert "27 stab wounds" not in script["youtube_metadata"]["recommended_title"].lower()
