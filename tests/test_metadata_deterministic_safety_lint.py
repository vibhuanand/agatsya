from __future__ import annotations

import pytest

from app.services.python_preflight_service import run_python_preflight

make_chunk = pytest.make_chunk
make_metadata = pytest.make_metadata
make_script = pytest.make_script
make_glossary = pytest.make_glossary


def _run(script: dict, tmp_path):
    return run_python_preflight(
        script_draft=script,
        fact_lock={},
        case_glossary=make_glossary(),
        review_dir=tmp_path / "04-review",
        target_duration_min=20,
        hinglish_level=2,
    )


def test_graphic_injury_detail_in_metadata_is_removed_before_ai(tmp_path):
    metadata = make_metadata(
        title="Most chilling case with 41 injuries",
        tags=["true crime Hindi", "41 injuries", "case evidence"] + [f"tag{i}" for i in range(20)],
        extra={"thumbnail_options": [{"thumbnail_text": "41 injuries", "angle": "unsafe"}]},
    )
    script = make_script([make_chunk("001_hook", "साफ़ पाठ।")], metadata=metadata)

    report = _run(script, tmp_path)
    serialized = str(script["youtube_metadata"]).lower()

    assert "41 injuries" not in serialized
    assert report["metadata_python_fixes_applied"]


def test_unsupported_superlative_is_replaced_deterministically(tmp_path):
    metadata = make_metadata(title="सबसे क्रूर मामला जिसने सच दिखाया")
    script = make_script([make_chunk("001_hook", "साफ़ पाठ।")], metadata=metadata)

    report = _run(script, tmp_path)

    assert "सबसे क्रूर" not in script["youtube_metadata"]["recommended_title"]
    assert report["metadata_python_fixes_applied"]


def test_neutral_factual_metadata_passes_without_safety_target(tmp_path):
    metadata = make_metadata(title="Calgary का मामला और अदालत का फैसला")
    script = make_script([make_chunk("001_hook", "साफ़ पाठ।")], metadata=metadata)

    report = _run(script, tmp_path)
    issue_types = {i["type"] for i in report["metadata_issues"]}

    assert "metadata_graphic_injury_detail" not in issue_types
    assert "metadata_unsupported_superlative" not in issue_types
    assert "metadata_explicit_sexual_violence" not in issue_types
