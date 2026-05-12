from __future__ import annotations

import pytest

from app.services.python_preflight_service import run_python_preflight

make_chunk = pytest.make_chunk
make_script = pytest.make_script
make_glossary = pytest.make_glossary


def _run(script: dict, tmp_path, *, source: str = "") -> dict:
    return run_python_preflight(
        script_draft=script,
        fact_lock={},
        case_glossary=make_glossary(),
        review_dir=tmp_path / "04-review",
        target_duration_min=20,
        hinglish_level=2,
        source_transcript=source,
    )


def test_copied_long_english_quote_blocks_in_reconstruction_chunk(tmp_path):
    source = (
        "The detective said we found the same story repeated again and again "
        "inside the interview room."
    )
    chunk = make_chunk(
        "006_reconstruction",
        "जाँच के इस हिस्से में The detective said we found the same story repeated again and again दिखता है।",
    )
    script = make_script([chunk])

    report = _run(script, tmp_path, source=source)

    assert report["blocking"] is True
    assert report["source_shaped_reconstruction_detected"] is True
    assert any(i["type"] == "source_shaped_reconstruction" for i in report["issues"])


def test_paraphrased_hindi_legal_framing_passes(tmp_path):
    source = "The detective said we found the same story repeated again and again."
    chunk = make_chunk(
        "006_reconstruction",
        "जाँच में अधिकारियों ने बयान की समानता को साक्ष्य की तरह देखा, लेकिन यहाँ असली शब्दों को दोहराना ज़रूरी नहीं है।",
    )
    script = make_script([chunk])

    report = _run(script, tmp_path, source=source)

    assert report["blocking"] is False
    assert report["source_shaped_reconstruction_detected"] is False


def test_explicit_sensitive_wording_blocks_in_reconstruction_chunk(tmp_path):
    chunk = make_chunk(
        "005_final_hours",
        "उस रात की घटना में sexual assault और hands tied जैसी बातें सामने आईं।",
    )
    script = make_script([chunk])

    report = _run(script, tmp_path)

    assert report["blocking"] is True
    assert any(i["type"] == "explicit_sensitive_violence" for i in report["issues"])


def test_names_dates_places_do_not_trigger_source_copy_false_positive(tmp_path):
    source = "Calgary Alberta Canada Supreme Court of Canada 2017."
    chunk = make_chunk(
        "004_court",
        "2017 में Supreme Court of Canada ने मामले के अंतिम कानूनी निष्कर्ष पर मुहर लगाई।",
    )
    script = make_script([chunk])

    report = _run(script, tmp_path, source=source)

    assert report["blocking"] is False
