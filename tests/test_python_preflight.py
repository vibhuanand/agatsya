"""Tests for python_preflight_service.run_python_preflight().

The xfail test documents the known झींगुर hardcoding bug (Phase 3 fix).
All other tests must pass on the current codebase.
"""
from __future__ import annotations

import json

import pytest

from app.services.python_preflight_service import run_python_preflight

make_chunk = pytest.make_chunk
make_metadata = pytest.make_metadata
make_script = pytest.make_script
make_glossary = pytest.make_glossary


# ── Helper ────────────────────────────────────────────────────────────────────

def _run(script, glossary, tmp_path, *, target=20, hinglish=2):
    review_dir = tmp_path / "04-review"
    return run_python_preflight(
        script_draft=script,
        fact_lock={},
        case_glossary=glossary,
        review_dir=review_dir,
        target_duration_min=target,
        hinglish_level=hinglish,
    )


# ── Passing cases ─────────────────────────────────────────────────────────────

def test_clean_script_passes(tmp_path):
    script = make_script([make_chunk("001", "नागपुर अदालत ने फैसला सुनाया।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert result["passed"] is True
    assert result["issues"] == []
    assert result["metadata_issues"] == []


def test_report_saved_to_disk(tmp_path):
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    _run(script, glossary, tmp_path)
    saved = json.loads((tmp_path / "04-review" / "python_preflight_report.json").read_text())
    assert "passed" in saved
    assert "issues" in saved


def test_miss_kar_flagged_at_hinglish_level_2(tmp_path):
    script = make_script([make_chunk("001", "वह उसे miss कर रही थी।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path, hinglish=2)
    assert result["passed"] is False
    types = [i["type"] for i in result["issues"]]
    assert "hinglish_level" in types


def test_miss_kar_not_flagged_at_hinglish_level_4(tmp_path):
    script = make_script([make_chunk("001", "वह उसे miss कर रही थी।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path, hinglish=4)
    types = [i["type"] for i in result["issues"]]
    assert "hinglish_level" not in types


def test_unsupported_first_claim_flagged(tmp_path):
    script = make_script([make_chunk("001", "यह भारत में पहला मामला था।")])
    glossary = make_glossary(allow_first_claim=False)
    result = _run(script, glossary, tmp_path)
    assert result["passed"] is False
    types = [i["type"] for i in result["issues"]]
    assert "unsupported_legal_claim" in types


def test_first_claim_allowed_when_flag_set(tmp_path):
    script = make_script([make_chunk("001", "यह भारत में पहला मामला था।")])
    glossary = make_glossary(allow_first_claim=True)
    result = _run(script, glossary, tmp_path)
    types = [i["type"] for i in result["issues"]]
    assert "unsupported_legal_claim" not in types


def test_generic_do_not_use_term_in_narration_flagged(tmp_path):
    """Forbidden terms from case_glossary.do_not_use are caught in narration chunks.

    The old hardcoded _GRAPHIC_CHILD_TERMS list was case-specific. The generic
    mechanism is: add the term to do_not_use in the glossary → preflight catches it.
    """
    script = make_script([make_chunk("001", "यह सबसे भयानक मामला था।")])
    # "सबसे भयानक" is in the default do_not_use list
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert result["passed"] is False
    assert result["blocking"] is True
    types = [i["type"] for i in result["issues"]]
    assert "case_glossary" in types


def test_title_too_long_in_metadata(tmp_path):
    long_title = "अ" * 101
    script = make_script(
        [make_chunk("001", "साफ़ पाठ।")],
        metadata=make_metadata(title=long_title),
    )
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert result["passed"] is False
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "title_length" in meta_types


def test_tag_count_too_few_flagged(tmp_path):
    script = make_script(
        [make_chunk("001", "साफ़ पाठ।")],
        metadata=make_metadata(tags=["tag1", "tag2"]),
    )
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "tag_count" in meta_types


def test_forbidden_term_in_metadata_flagged(tmp_path):
    metadata = make_metadata()
    metadata["description"] = "सबसे भयानक मामला।"
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=metadata)
    glossary = make_glossary(do_not_use=["सबसे भयानक"])
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "metadata_forbidden_term" in meta_types


def test_estimated_duration_in_report(tmp_path):
    # 110 wpm default; 220 words ≈ 2.0 min
    words = " ".join(["word"] * 220)
    script = make_script([make_chunk("001", words)])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert "estimated_duration_min" in result
    assert result["estimated_duration_min"] > 0


# ── New output shape (Phase 4) ────────────────────────────────────────────────

def test_report_has_blocking_field(tmp_path):
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert "blocking" in result
    assert isinstance(result["blocking"], bool)


def test_report_has_severity_counts(tmp_path):
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    sc = result.get("severity_counts", {})
    assert "high" in sc and "medium" in sc and "low" in sc


def test_clean_script_has_blocking_false(tmp_path):
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert result["blocking"] is False


def test_high_issue_sets_blocking_true(tmp_path):
    """A high-severity issue (unsupported legal claim) must set blocking=True."""
    script = make_script([make_chunk("001", "यह भारत का पहला मामला था।")])
    glossary = make_glossary(allow_first_claim=False)
    result = _run(script, glossary, tmp_path)
    assert result["blocking"] is True
    assert result["severity_counts"]["high"] > 0


def test_medium_issue_sets_blocking_true(tmp_path):
    """A medium-severity issue must also set blocking=True."""
    script = make_script([make_chunk("001", "वह उसे miss कर रही थी।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path, hinglish=2)
    assert result["blocking"] is True
    assert result["severity_counts"]["medium"] > 0


def test_low_only_issue_does_not_block(tmp_path):
    """A low-severity issue alone must NOT set blocking=True."""
    metadata = make_metadata()
    metadata["chapters"] = [{"title": "भाग 1", "timestamp": "0:00"}]
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=metadata)
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    # chapters_before_audio is low — should not block
    assert result["blocking"] is False
    assert result["severity_counts"]["low"] >= 1


def test_metadata_repair_targets_populated_on_title_too_long(tmp_path):
    """Metadata issues must create structured metadata_repair_targets entries."""
    long_title = "अ" * 101
    script = make_script(
        [make_chunk("001", "साफ़ पाठ।")],
        metadata=make_metadata(title=long_title),
    )
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert "metadata_repair_targets" in result
    assert len(result["metadata_repair_targets"]) > 0
    first = result["metadata_repair_targets"][0]
    assert "field" in first
    assert "issue_type" in first
    assert "repair_instruction" in first


def test_metadata_repair_targets_empty_when_clean(tmp_path):
    """No metadata issues → metadata_repair_targets must be an empty list."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert result["metadata_repair_targets"] == []


def test_forbidden_name_variant_in_narration_high_severity(tmp_path):
    """Forbidden name variants in narration must be high severity and blocking."""
    script = make_script([make_chunk("001", "Kyla Jordan ने खेला।")])
    glossary = make_glossary()
    # Inject forbidden_name_variants into the glossary directly
    glossary["forbidden_name_variants"] = ["Kyla Jordan"]
    result = _run(script, glossary, tmp_path)
    assert result["blocking"] is True
    types = [i["type"] for i in result["issues"]]
    assert "forbidden_name_variant" in types


# ── Known bug: झींगुर hardcoding fires for ANY case ──────────────────────────

def test_jhingur_in_non_ladybug_script_is_not_flagged(tmp_path):
    """A non-ladybug case mentioning झींगुर (cricket sounds as atmosphere) must NOT
    be flagged — झींगुर is only forbidden when the case glossary says so.

    Phase 3/4 fix: the hardcoded '`if "झींगुर" in text`' check was removed.
    Forbidden terms now come exclusively from case_glossary.do_not_use.
    """
    script = make_script([
        make_chunk("001", "रात को झींगुर की आवाज़ें सुनाई देती थीं।")
    ])
    # do_not_use does NOT contain झींगुर — this is a generic murder case
    glossary = make_glossary(do_not_use=["सबसे भयानक"])
    result = _run(script, glossary, tmp_path)
    assert result["passed"] is True
    assert result["blocking"] is False


# ── YouTube safety phrase checks ──────────────────────────────────────────────

def test_sensational_hindi_phrase_blocked(tmp_path):
    """Hardcoded sensational phrases must be caught regardless of glossary."""
    script = make_script([make_chunk("001", "यह सबसे खौफनाक मामला था।")])
    glossary = make_glossary(do_not_use=[])  # phrase NOT in do_not_use
    result = _run(script, glossary, tmp_path)
    assert result["blocking"] is True
    types = [i["type"] for i in result["issues"]]
    assert "youtube_safety_phrase" in types


def test_ruh_kamp_phrase_blocked(tmp_path):
    """'रूह कांप जाएगी' must always be blocked."""
    script = make_script([make_chunk("001", "रूह कांप जाएगी यह देखकर।")])
    glossary = make_glossary(do_not_use=[])
    result = _run(script, glossary, tmp_path)
    types = [i["type"] for i in result["issues"]]
    assert "youtube_safety_phrase" in types


def test_sensational_phrase_in_metadata_blocked(tmp_path):
    """Sensational phrases in metadata must be caught at metadata level."""
    metadata = make_metadata()
    metadata["description"] = (
        "हिला देने वाला सच — " + metadata["description"]
    )
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=metadata)
    glossary = make_glossary(do_not_use=[])
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "youtube_safety_phrase" in meta_types


def test_unverified_media_claim_blocked(tmp_path):
    """'caught on camera' is an unverified media claim and must be high severity."""
    script = make_script([
        make_chunk("001", "यह पूरा incident caught on camera था।")
    ])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert result["blocking"] is True
    issues = [i for i in result["issues"] if i["type"] == "unverified_media_claim"]
    assert issues
    assert issues[0]["severity"] == "high"


def test_unverified_claim_allowed_when_glossary_permits(tmp_path):
    """When allow_verified_media_claims=True, unverified claim phrases must not be flagged."""
    script = make_script([
        make_chunk("001", "यह पूरा incident caught on camera था।")
    ])
    glossary = make_glossary()
    glossary["allow_verified_media_claims"] = True
    result = _run(script, glossary, tmp_path)
    types = [i["type"] for i in result["issues"]]
    assert "unverified_media_claim" not in types


def test_leaked_phrase_blocked(tmp_path):
    """'leaked' is an unverified media claim and must be caught."""
    script = make_script([make_chunk("001", "एक leaked video सामने आया।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    types = [i["type"] for i in result["issues"]]
    assert "unverified_media_claim" in types


# ── YouTube metadata protection checks ───────────────────────────────────────

def test_duplicate_tags_flagged(tmp_path):
    """Duplicate tags must be caught at medium severity."""
    tags = ["हत्याकांड", "नागपुर", "हत्याकांड", "true crime",  # हत्याकांड repeated
             "crime", "law", "justice", "DNA", "CCTV", "murder",
             "verdict", "conviction", "court", "India", "2024"]
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=make_metadata(tags=tags))
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "duplicate_tags" in meta_types


def test_unrelated_tag_flagged(tmp_path):
    """'bollywood' tag on a true crime video is off-topic keyword stuffing."""
    tags = [
        "हत्याकांड", "नागपुर", "true crime", "crime", "law", "justice",
        "DNA", "CCTV", "murder", "verdict", "conviction", "court",
        "India", "2024", "bollywood",
    ]
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=make_metadata(tags=tags))
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "unrelated_tags" in meta_types


def test_unrelated_tag_allowed_when_glossary_permits(tmp_path):
    """An unrelated tag explicitly allowed in youtube_metadata_rules must not be flagged."""
    tags = [
        "हत्याकांड", "नागपुर", "true crime", "crime", "law", "justice",
        "DNA", "CCTV", "murder", "verdict", "conviction", "court",
        "India", "2024", "trending",
    ]
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=make_metadata(tags=tags))
    glossary = make_glossary()
    glossary["youtube_metadata_rules"]["allow_unrelated_tags"] = ["trending"]
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "unrelated_tags" not in meta_types


def test_thumbnail_text_one_word_flagged(tmp_path):
    """Single-word thumbnail text is too short (needs 2–5 words)."""
    metadata = make_metadata()
    metadata["thumbnail_options"] = [{"thumbnail_text": "सच", "angle": "theme"}]
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=metadata)
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "thumbnail_text_length" in meta_types


def test_thumbnail_text_six_words_flagged(tmp_path):
    """Six-word thumbnail text is too long (max 5 words)."""
    metadata = make_metadata()
    metadata["thumbnail_options"] = [
        {"thumbnail_text": "देविका राठी का असली सच क्या", "angle": "theme"}
    ]
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=metadata)
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "thumbnail_text_length" in meta_types


def test_thumbnail_text_three_words_passes(tmp_path):
    """Three-word thumbnail text (default fixture) must pass."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])  # default thumbnail has 3 words
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "thumbnail_text_length" not in meta_types


def test_description_too_short_flagged(tmp_path):
    """A description with fewer than 100 words must produce description_too_short issue."""
    metadata = make_metadata()
    metadata["description"] = "बहुत छोटा विवरण।"
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=metadata)
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "description_too_short" in meta_types


def test_description_100_words_passes(tmp_path):
    """The default fixture description (100+ words) must not trigger description_too_short."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "description_too_short" not in meta_types


def test_pinned_comment_missing_flagged(tmp_path):
    """Absent pinned_comment must produce low-severity pinned_comment_missing issue."""
    metadata = make_metadata()
    metadata["pinned_comment"] = ""
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=metadata)
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "pinned_comment_missing" in meta_types
    # low severity — must NOT block
    assert result["blocking"] is False


def test_pinned_comment_present_passes(tmp_path):
    """Default fixture has pinned_comment='नमन।' — must not flag."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "pinned_comment_missing" not in meta_types


# ── label / post-repair file naming ──────────────────────────────────────────

def test_label_saves_separate_file(tmp_path):
    """label='_after_repair' must save python_preflight_report_after_repair.json."""
    import json as _json
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    review_dir = tmp_path / "04-review"
    review_dir.mkdir(parents=True, exist_ok=True)
    run_python_preflight(
        script_draft=script,
        fact_lock={},
        case_glossary=glossary,
        review_dir=review_dir,
        target_duration_min=20,
        hinglish_level=2,
        label="_after_repair",
    )
    after_file = review_dir / "python_preflight_report_after_repair.json"
    assert after_file.exists()
    data = _json.loads(after_file.read_text(encoding="utf-8"))
    assert "passed" in data
    assert "blocking" in data


def test_no_label_saves_default_file(tmp_path):
    """Default label='' must save python_preflight_report.json."""
    script = make_script([make_chunk("001", "साफ़ पाठ।")])
    glossary = make_glossary()
    _run(script, glossary, tmp_path)
    assert (tmp_path / "04-review" / "python_preflight_report.json").exists()


# ── Graphic content checks ────────────────────────────────────────────────────

def test_graphic_content_phrase_blocked_in_narration(tmp_path):
    """Generic graphic content phrase must be high severity and blocking."""
    script = make_script([make_chunk("001", "The body was found dismembered.")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert result["blocking"] is True
    types = [i["type"] for i in result["issues"]]
    assert "graphic_content" in types
    severities = [i["severity"] for i in result["issues"] if i["type"] == "graphic_content"]
    assert all(s == "high" for s in severities)


def test_sexualized_victim_phrase_blocked(tmp_path):
    """Sexualized victim framing must be high severity."""
    script = make_script([
        make_chunk("001", "The script included sexual assault details of the victim.")
    ])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    types = [i["type"] for i in result["issues"]]
    assert "sexualized_victim_framing" in types


def test_child_harm_phrase_blocked(tmp_path):
    """Child harm recreation phrase must be high severity and blocking."""
    script = make_script([make_chunk("001", "This was a child abuse scene reconstruction.")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    assert result["blocking"] is True
    types = [i["type"] for i in result["issues"]]
    assert "child_harm_content" in types


def test_real_audio_blocked(tmp_path):
    """'real audio' must be caught as unverified_media_claim (high severity)."""
    script = make_script([make_chunk("001", "Listen to the real audio from the scene.")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    types = [i["type"] for i in result["issues"]]
    assert "unverified_media_claim" in types


def test_leaked_footage_blocked(tmp_path):
    """'leaked footage' must be caught as unverified_media_claim (high severity)."""
    script = make_script([make_chunk("001", "Here is the leaked footage from the investigation.")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    types = [i["type"] for i in result["issues"]]
    assert "unverified_media_claim" in types


def test_real_scream_blocked(tmp_path):
    """'real scream' must be caught as unverified_media_claim."""
    script = make_script([make_chunk("001", "You can hear the real scream in this clip.")])
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    types = [i["type"] for i in result["issues"]]
    assert "unverified_media_claim" in types


def test_thumbnail_shock_word_flagged(tmp_path):
    """'SHOCKING' in thumbnail text must produce thumbnail_shock_word issue."""
    metadata = make_metadata()
    metadata["thumbnail_options"] = [
        {"thumbnail_text": "SHOCKING truth revealed", "angle": "clickbait"}
    ]
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=metadata)
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "thumbnail_shock_word" in meta_types


def test_graphic_content_in_metadata_flagged(tmp_path):
    """Graphic content phrases in metadata must produce graphic_content_metadata issue."""
    metadata = make_metadata()
    metadata["description"] = metadata["description"] + " The mutilated body was found near the river."
    script = make_script([make_chunk("001", "साफ़ पाठ।")], metadata=metadata)
    glossary = make_glossary()
    result = _run(script, glossary, tmp_path)
    meta_types = [i["type"] for i in result["metadata_issues"]]
    assert "graphic_content_metadata" in meta_types
