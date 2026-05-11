"""
Tests for report_normalization_service.

Guards against the crash:
    TypeError: sequence item N: expected str instance, dict found

which occurs when Claude returns required_fixes / issues as a list of dicts
and the pipeline joins them with "; ".join(...).
"""
from __future__ import annotations

import pytest
from app.services.report_normalization_service import (
    stringify_report_item,
    stringify_report_list,
    safe_join_report_items,
)


# ── stringify_report_item ─────────────────────────────────────────────────────

class TestStringifyReportItem:

    def test_string_passthrough(self):
        assert stringify_report_item("hello") == "hello"

    def test_empty_string(self):
        assert stringify_report_item("") == ""

    def test_int(self):
        assert stringify_report_item(42) == "42"

    def test_float(self):
        assert stringify_report_item(3.14) == "3.14"

    def test_bool_true(self):
        assert stringify_report_item(True) == "True"

    def test_bool_false(self):
        assert stringify_report_item(False) == "False"

    def test_dict_with_problem_key(self):
        item = {"problem": "title is too long", "severity": "high"}
        result = stringify_report_item(item)
        assert "title is too long" in result
        assert isinstance(result, str)

    def test_dict_with_issue_key(self):
        item = {"issue": "missing description"}
        assert stringify_report_item(item) == "missing description"

    def test_dict_with_message_key(self):
        item = {"message": "score below threshold"}
        assert stringify_report_item(item) == "score below threshold"

    def test_dict_with_required_fix_key(self):
        item = {"required_fix": "shorten the title to under 60 chars"}
        assert stringify_report_item(item) == "shorten the title to under 60 chars"

    def test_dict_with_field_and_suggested_fix(self):
        item = {"field": "title", "suggested_fix": "use a shorter title"}
        result = stringify_report_item(item)
        assert "title" in result
        assert "shorter" in result
        assert isinstance(result, str)

    def test_dict_with_chunk_id_and_issue_type(self):
        item = {"chunk_id": "003_events", "issue_type": "pacing", "problem": "too slow"}
        result = stringify_report_item(item)
        assert "too slow" in result  # "problem" is higher priority

    def test_dict_no_useful_keys_falls_back_to_json(self):
        item = {"unknown_key": "some value", "another": 42}
        result = stringify_report_item(item)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be valid compact JSON or something readable
        assert "unknown_key" in result or "some value" in result

    def test_list_of_strings(self):
        result = stringify_report_item(["a", "b", "c"])
        assert result == "a; b; c"

    def test_list_with_dicts(self):
        items = [{"problem": "x"}, "plain string"]
        result = stringify_report_item(items)
        assert "x" in result
        assert "plain string" in result
        assert isinstance(result, str)

    def test_none_becomes_str(self):
        # None is not str/int/float/bool/dict/list — falls to str(item)
        result = stringify_report_item(None)
        assert result == "None"


# ── stringify_report_list ─────────────────────────────────────────────────────

class TestStringifyReportList:

    def test_none_returns_empty(self):
        assert stringify_report_list(None) == []

    def test_empty_list(self):
        assert stringify_report_list([]) == []

    def test_list_of_strings(self):
        assert stringify_report_list(["a", "b"]) == ["a", "b"]

    def test_list_of_dicts(self):
        value = [{"problem": "x"}, {"problem": "y"}]
        result = stringify_report_list(value)
        assert result == ["x", "y"]

    def test_list_mixed_str_and_dict(self):
        value = ["plain fix", {"problem": "dict fix"}]
        result = stringify_report_list(value)
        assert result[0] == "plain fix"
        assert result[1] == "dict fix"

    def test_single_dict_becomes_single_item_list(self):
        value = {"problem": "only one"}
        result = stringify_report_list(value)
        assert result == ["only one"]

    def test_string_becomes_single_item_list(self):
        assert stringify_report_list("single fix") == ["single fix"]

    def test_limit_applied(self):
        value = ["a", "b", "c", "d", "e"]
        result = stringify_report_list(value, limit=3)
        assert result == ["a", "b", "c"]

    def test_limit_with_dicts(self):
        value = [{"problem": str(i)} for i in range(5)]
        result = stringify_report_list(value, limit=2)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)

    def test_returns_all_strings(self):
        value = ["str", {"problem": "dict"}, 42, True, None]
        result = stringify_report_list(value)
        assert all(isinstance(s, str) for s in result)


# ── safe_join_report_items ────────────────────────────────────────────────────

class TestSafeJoinReportItems:

    def test_list_of_strings(self):
        result = safe_join_report_items(["fix a", "fix b", "fix c"])
        assert result == "fix a; fix b; fix c"

    def test_does_not_raise_on_dict_items(self):
        # This is the exact crash that occurred in production
        value = ["a", {"problem": "b"}, {"field": "title", "suggested_fix": "c"}]
        result = safe_join_report_items(value)
        assert isinstance(result, str)
        assert "a" in result
        assert "b" in result

    def test_limit_respected(self):
        value = ["a", "b", "c", "d", "e"]
        result = safe_join_report_items(value, limit=3)
        # Only first 3 items joined
        assert result == "a; b; c"

    def test_empty_list(self):
        assert safe_join_report_items([]) == ""

    def test_none_returns_empty_string(self):
        assert safe_join_report_items(None) == ""

    def test_single_string(self):
        assert safe_join_report_items("only fix") == "only fix"

    def test_single_dict(self):
        result = safe_join_report_items({"problem": "needs fixing"})
        assert result == "needs fixing"

    def test_custom_sep(self):
        result = safe_join_report_items(["a", "b"], sep=" | ")
        assert result == "a | b"

    def test_mixed_types_no_crash(self):
        # All these types should be handled without TypeError
        value = [
            "string fix",
            {"problem": "dict with problem key"},
            {"issue_type": "pacing", "chunk_id": "003"},
            {"unknown_key": "value"},
            42,
            True,
        ]
        result = safe_join_report_items(value, limit=6)
        assert isinstance(result, str)
        # Should contain the string fix
        assert "string fix" in result


# ── Integration: simulate pipeline warning generation ─────────────────────────

class TestPipelineWarningGeneration:
    """
    Simulate the exact patterns used in agent_pipeline_service.py
    to ensure no TypeError on dict required_fixes.
    """

    def test_originality_safety_gate_warning(self):
        required_fixes = [
            "Fix 1 as string",
            {"problem": "Fix 2 as dict", "severity": "high"},
            {"field": "script", "suggested_fix": "Fix 3"},
        ]
        # The exact pattern from agent_pipeline_service.py
        warning = (
            "Originality/safety gate FAILED. Required fixes: "
            + safe_join_report_items(required_fixes, limit=3)
            + (f" (+{len(required_fixes)-3} more)" if len(required_fixes) > 3 else "")
        )
        assert isinstance(warning, str)
        assert "Fix 1 as string" in warning
        assert "Fix 2 as dict" in warning

    def test_metadata_gate_warning_with_dict_fixes(self):
        meta_required_fixes = [
            {"problem": "title too long", "field": "recommended_title"},
            {"problem": "description missing keywords"},
            {"problem": "tags unrelated to case"},
            {"problem": "fourth fix"},
        ]
        warning = (
            "Metadata quality gate FAILED. Required fixes: "
            + safe_join_report_items(meta_required_fixes, limit=3)
            + (f" (+{len(meta_required_fixes)-3} more)" if len(meta_required_fixes) > 3 else "")
        )
        assert isinstance(warning, str)
        assert "title too long" in warning
        assert "(+1 more)" in warning  # 4 items, limit=3

    def test_dialogue_gate_warning_empty_fixes(self):
        required_fixes: list = []
        # Should not crash even when empty
        warning = (
            "Recreated dialogue gate FAILED. Required fixes: "
            + safe_join_report_items(required_fixes, limit=3)
        )
        assert isinstance(warning, str)

    def test_openai_gate_warning_with_mixed_fixes(self):
        fixes = [
            "Reduce source verbatim content",
            {"issue": "opening too similar to source", "severity": "high"},
        ]
        warning = (
            "OpenAI originality/YT risk gate FAILED. Required fixes: "
            + (safe_join_report_items(fixes, limit=3) if fixes else "see report")
            + (f" (+{len(fixes)-3} more)" if len(fixes) > 3 else "")
            + " — See report."
        )
        assert isinstance(warning, str)
        assert "Reduce source verbatim" in warning
        assert "opening too similar" in warning

    def test_stringify_report_list_normalizes_for_gate_summary(self):
        """
        gate_summary["required_fixes"] must only contain strings,
        not raw dicts that would confuse API consumers.
        """
        raw_fixes = [
            "String fix",
            {"problem": "Dict fix A"},
            {"issue_type": "pacing", "chunk_id": "005"},
        ]
        normalized = stringify_report_list(raw_fixes)
        assert all(isinstance(s, str) for s in normalized)
        assert normalized[0] == "String fix"
        assert normalized[1] == "Dict fix A"


# ── Gate service *_issues fields ─────────────────────────────────────────────
#
# Regression tests for the six fields that _python_validate_gate() in each gate
# service passes to safe_join_report_items().  Claude may return any of these as
# a list of dicts rather than a list of strings.

class TestGateIssueFieldsNoCrash:
    """Prove that no gate-service *_issues field crashes when it contains dicts."""

    # ── originality_safety_gate_service ──────────────────────────────────────

    def test_copying_issues_as_dicts(self):
        copying_issues = [{"problem": "copied wording"}, {"problem": "verbatim phrase found"}]
        result = safe_join_report_items(copying_issues, limit=2, sep=", ")
        assert isinstance(result, str)
        assert "copied wording" in result

    def test_copying_issues_empty(self):
        result = safe_join_report_items([], limit=2, sep=", ")
        assert result == ""

    def test_copying_issues_mixed(self):
        copying_issues = ["plain string", {"problem": "dict issue"}]
        result = safe_join_report_items(copying_issues, limit=2, sep=", ")
        assert "plain string" in result
        assert "dict issue" in result

    def test_ad_safety_issues_as_dicts(self):
        ad_safety_issues = [{"problem": "too graphic"}, {"problem": "graphic violence description"}]
        result = safe_join_report_items(ad_safety_issues, limit=2, sep=", ")
        assert isinstance(result, str)
        assert "too graphic" in result

    def test_ad_safety_issues_fallback_or(self):
        # Empty list → empty string → the "or 'see report'" in the f-string kicks in
        result = safe_join_report_items([], limit=2, sep=", ") or "see report"
        assert result == "see report"

    # ── metadata_quality_gate_service ────────────────────────────────────────

    def test_title_issues_as_dicts(self):
        title_issues = [{"problem": "title too sensational"}, {"field": "recommended_title"}]
        result = safe_join_report_items(title_issues, limit=2, sep=", ")
        assert isinstance(result, str)
        assert "title too sensational" in result

    def test_monetization_risks_as_dicts(self):
        monetization_risks = [
            {"problem": "child victim framing risky"},
            {"problem": "graphic crime description"},
        ]
        result = safe_join_report_items(monetization_risks, limit=2, sep=", ")
        assert isinstance(result, str)
        assert "child victim framing risky" in result

    def test_monetization_risks_limit_respected(self):
        monetization_risks = [
            {"problem": "risk A"},
            {"problem": "risk B"},
            {"problem": "risk C"},
        ]
        result = safe_join_report_items(monetization_risks, limit=2, sep=", ")
        assert "risk A" in result
        assert "risk B" in result
        assert "risk C" not in result

    # ── recreated_dialogue_quality_gate_service ───────────────────────────────

    def test_scene_issues_as_dicts(self):
        scene_issues = [
            {"problem": "missing recreated label"},
            {"chunk_id": "003_events", "problem": "label not prominent"},
        ]
        result = safe_join_report_items(scene_issues, limit=2, sep=", ")
        assert isinstance(result, str)
        assert "missing recreated label" in result

    def test_scene_issues_old_list_of_strings_still_works(self):
        # Ensure we didn't break the happy path where Claude returns strings
        scene_issues = ["label missing", "tone too assertive"]
        result = safe_join_report_items(scene_issues, limit=2, sep=", ")
        assert result == "label missing, tone too assertive"

    # ── script_review_service ────────────────────────────────────────────────

    def test_fact_issues_as_dicts(self):
        fact_issues = [{"problem": "wrong date cited"}, {"problem": "victim name misspelled"}]
        result = safe_join_report_items(fact_issues, limit=2, sep=", ")
        assert isinstance(result, str)
        assert "wrong date cited" in result

    def test_language_issues_as_dicts(self):
        language_issues = [{"problem": "unnatural Hindi"}, {"problem": "colloquial mismatch"}]
        result = safe_join_report_items(language_issues, limit=2, sep=", ")
        assert isinstance(result, str)
        assert "unnatural Hindi" in result

    def test_generic_issues_key_fallback(self):
        # When Claude returns a dict with no preferred key, falls back to JSON — no crash
        mystery_issues = [{"unknown_key": "some value"}, {"another": 42}]
        result = safe_join_report_items(mystery_issues, limit=2, sep=", ")
        assert isinstance(result, str)
        assert len(result) > 0

    # ── cross-field: ensure or-fallback pattern works for all fields ──────────

    def test_all_fields_empty_list_or_fallback(self):
        fields = [
            "copying_issues", "ad_safety_issues", "title_issues",
            "monetization_risks", "scene_issues", "fact_issues", "language_issues",
        ]
        fake_report: dict = {f: [] for f in fields}
        for field in fields:
            result = safe_join_report_items(fake_report.get(field, []), limit=2, sep=", ") or "see report"
            assert result == "see report", f"Expected 'see report' fallback for {field}"

    def test_all_fields_dict_items_no_crash(self):
        fake_report = {
            "copying_issues":    [{"problem": "copied wording"}],
            "ad_safety_issues":  [{"problem": "too graphic"}],
            "title_issues":      [{"problem": "title too sensational"}],
            "monetization_risks":[{"problem": "child victim framing risky"}],
            "scene_issues":      [{"problem": "missing recreated label"}],
            "fact_issues":       [{"problem": "wrong date cited"}],
            "language_issues":   [{"problem": "unnatural Hindi"}],
        }
        for field, items in fake_report.items():
            result = safe_join_report_items(items, limit=2, sep=", ")
            assert isinstance(result, str), f"{field} produced non-str: {result!r}"
            assert len(result) > 0, f"{field} produced empty string unexpectedly"
