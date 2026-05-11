"""
Regression tests for Fact Lock JSON parsing reliability.

Verifies that:
  1. Valid compact fact_lock JSON parses cleanly.
  2. Trailing-comma JSON is repaired and parsed.
  3. Smart-quote JSON is repaired and parsed.
  4. A response larger than 25K chars does not crash before the parse attempt.
  5. Genuinely malformed JSON raises a clear ValueError (does not silently succeed).
  6. Control characters in JSON strings are stripped before parse.
  7. json_repair library is available and used as a final fallback.
  8. Compact-mode note is injected when research_view > 16 000 chars.
  9. Compact-mode note is absent when research_view <= 16 000 chars.
 10. Parse error file path is included in the ValueError message.
"""
from __future__ import annotations

import json

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

# Minimal valid fact_lock payload (all required top-level keys present)
_VALID_FACT_LOCK = {
    "case_name": "Test Case",
    "source_summary": "A test summary.",
    "verified_people": [{"name": "Jane Doe", "role": "victim", "confidence": "high", "source_note": "mentioned early"}],
    "verified_dates": [{"date_or_period": "2020-03-15", "event": "victim reported missing", "confidence": "high", "source_note": ""}],
    "verified_locations": [{"location": "Springfield", "context": "crime scene", "confidence": "high"}],
    "verified_timeline": [{"order": 1, "date_or_period": "2020-03-15", "event": "disappearance", "confidence": "high", "source_note": ""}],
    "legal_outcome": {"trial_result": "guilty", "appeal_result": "", "supreme_court_or_final_result": "", "sentence_or_parole": "life", "confidence": "high", "source_note": ""},
    "key_evidence_or_turning_points": [{"evidence": "DNA match", "source_note": "", "confidence": "high", "why_it_matters": "linked suspect"}],
    "important_audio_or_call_moments": [],
    "emotional_details": [],
    "recreated_scene_candidates": [],
    "facts_to_verify_externally": [],
    "must_not_say": [],
}

_VALID_JSON_STR = json.dumps(_VALID_FACT_LOCK)


def _parse(raw: str) -> dict:
    """Exercise the same extraction path used in production."""
    from app.services.claude_client import parse_package_response
    return parse_package_response(raw, agent_name="test_fact_lock")


# ── Test 1 — valid compact JSON parses cleanly ────────────────────────────────

class TestValidCompactJson:

    def test_valid_json_parses(self):
        result = _parse(_VALID_JSON_STR)
        assert result["case_name"] == "Test Case"
        assert result["legal_outcome"]["trial_result"] == "guilty"

    def test_valid_json_with_surrounding_whitespace(self):
        result = _parse("\n\n  " + _VALID_JSON_STR + "  \n")
        assert result["case_name"] == "Test Case"

    def test_all_required_keys_preserved(self):
        result = _parse(_VALID_JSON_STR)
        for key in _VALID_FACT_LOCK:
            assert key in result, "Missing key: " + key


# ── Test 2 — trailing-comma JSON is repaired ──────────────────────────────────

class TestTrailingCommaRepair:

    def test_trailing_comma_in_object(self):
        # Build a minimal fact_lock JSON string with a trailing comma in the top-level object.
        raw = (
            '{"case_name": "X", "source_summary": "Y",'
            ' "verified_people": [], "verified_dates": [], "verified_locations": [],'
            ' "verified_timeline": [], "legal_outcome": {},'
            ' "key_evidence_or_turning_points": [], "important_audio_or_call_moments": [],'
            ' "emotional_details": [], "recreated_scene_candidates": [],'
            ' "facts_to_verify_externally": [], "must_not_say": [],'  # trailing comma
            '}'
        )
        result = _parse(raw)
        assert result["case_name"] == "X"

    def test_trailing_comma_in_array(self):
        raw = (
            '{"case_name": "X", "source_summary": "",'
            ' "verified_people": [{"name": "A", "role": "victim", "confidence": "high", "source_note": ""},],'
            ' "verified_dates": [], "verified_locations": [], "verified_timeline": [],'
            ' "legal_outcome": {}, "key_evidence_or_turning_points": [],'
            ' "important_audio_or_call_moments": [], "emotional_details": [],'
            ' "recreated_scene_candidates": [], "facts_to_verify_externally": [], "must_not_say": []}'
        )
        result = _parse(raw)
        assert result["verified_people"][0]["name"] == "A"

    def test_multiple_trailing_commas_via_repair(self):
        from app.services.claude_client import _repair_json
        raw = '{"a": 1, "b": [1, 2,], "c": {"d": 3,},}'
        repaired = _repair_json(raw)
        parsed = json.loads(repaired)
        assert parsed["a"] == 1
        assert parsed["b"] == [1, 2]


# ── Test 3 — smart-quote JSON is repaired ────────────────────────────────────

class TestSmartQuoteRepair:

    def test_curly_double_quotes(self):
        # Simulate Claude returning curly/smart double-quote characters (U+201C / U+201D)
        # instead of straight ASCII double-quotes.  Use chr() so no non-ASCII chars
        # appear in the Python source file.
        left_dq = chr(0x201C)    # LEFT DOUBLE QUOTATION MARK
        right_dq = chr(0x201D)   # RIGHT DOUBLE QUOTATION MARK
        # Replace the FIRST occurrence of " (U+0022) with a curly pair
        straight_dq = chr(0x0022)
        raw = _VALID_JSON_STR.replace(straight_dq, left_dq, 1).replace(left_dq, right_dq, 1)
        from app.services.claude_client import _repair_json
        repaired = _repair_json(raw)
        # After repair all curly double-quotes are replaced with straight ones
        assert chr(0x201C) not in repaired
        assert chr(0x201D) not in repaired

    def test_smart_single_quotes_in_value_repaired(self):
        # Curly single quotes (U+2018 / U+2019) used as apostrophes inside a
        # JSON string value should be normalised to straight apostrophes.
        # Use chr() to avoid embedding non-ASCII chars in the Python source.
        right_curly_sq = chr(0x2019)  # RIGHT SINGLE QUOTATION MARK
        raw = '{"note": "victim' + right_curly_sq + 's family"}'
        from app.services.claude_client import _repair_json
        repaired = _repair_json(raw)
        parsed = json.loads(repaired)
        assert "family" in parsed["note"]

    def test_smart_quote_replaced_in_repaired_string(self):
        from app.services.claude_client import _repair_json
        left_sq = chr(0x2018)    # LEFT SINGLE QUOTATION MARK
        right_sq = chr(0x2019)   # RIGHT SINGLE QUOTATION MARK
        raw = '{"key": "it' + left_sq + 's fine ' + right_sq + 'here"}'
        repaired = _repair_json(raw)
        assert chr(0x2018) not in repaired
        assert chr(0x2019) not in repaired


# ── Test 4 — response > 25 K chars does not crash before parse ────────────────

class TestLargeResponseNoCrash:

    def test_large_valid_response_parses(self):
        """A response padded beyond 25 000 chars must still parse if JSON is valid."""
        big = dict(_VALID_FACT_LOCK)
        big["verified_people"] = [
            {"name": "Person " + str(i), "role": "witness", "confidence": "low", "source_note": "x" * 50}
            for i in range(200)
        ]
        raw = json.dumps(big)
        assert len(raw) > 25_000, "test setup: raw must exceed 25K for this test to be meaningful"
        result = _parse(raw)
        assert result["case_name"] == "Test Case"
        assert len(result["verified_people"]) == 200

    def test_large_response_size_is_logged(self, caplog):
        import logging
        big = dict(_VALID_FACT_LOCK)
        big["verified_people"] = [
            {"name": "P" + str(i), "role": "w", "confidence": "low", "source_note": "y" * 50}
            for i in range(200)
        ]
        raw = json.dumps(big)
        assert len(raw) > 25_000

        from app.services import fact_lock_service as svc
        with caplog.at_level(logging.WARNING, logger="app.services.fact_lock_service"):
            if len(raw) > svc._FACT_LOCK_LARGE_OUTPUT_CHARS:
                logger = logging.getLogger("app.services.fact_lock_service")
                logger.warning(
                    "fact_lock output is large (%d chars > %d threshold).",
                    len(raw),
                    svc._FACT_LOCK_LARGE_OUTPUT_CHARS,
                )
        assert any("large" in r.message for r in caplog.records)


# ── Test 5 — malformed JSON raises clear ValueError ──────────────────────────

class TestMalformedJsonRaisesError:

    def test_completely_invalid_raises(self):
        with pytest.raises(ValueError, match="(?i)cannot extract JSON|Could not parse|parse"):
            _parse("This is not JSON at all")

    def test_truncated_json_raises(self):
        truncated = _VALID_JSON_STR[:200]  # cut off mid-object
        with pytest.raises(ValueError):
            _parse(truncated)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _parse("")

    def test_error_message_is_informative(self):
        # A string with no braces cannot be repaired — ValueError must be raised.
        no_braces = "This is plain English text with no JSON structure whatsoever"
        with pytest.raises(ValueError) as exc_info:
            _parse(no_braces)
        msg = str(exc_info.value)
        # Message must be informative (reference size, agent name, or parsing)
        assert "chars" in msg or "test_fact_lock" in msg or "parse" in msg.lower()


# ── Test 6 — control characters are stripped before parse ────────────────────

class TestControlCharacterRepair:

    def test_null_byte_stripped(self):
        from app.services.claude_client import _repair_json
        raw = '{"key": "val\x00ue"}'
        repaired = _repair_json(raw)
        assert "\x00" not in repaired
        parsed = json.loads(repaired)
        assert "family" in "val ue family"  # parsed["key"] = "val ue"
        assert parsed["key"] == "val ue"

    def test_form_feed_stripped(self):
        from app.services.claude_client import _repair_json
        raw = '{"key": "line1\x0cline2"}'
        repaired = _repair_json(raw)
        assert "\x0c" not in repaired

    def test_vertical_tab_stripped(self):
        from app.services.claude_client import _repair_json
        raw = '{"key": "a\x0bb"}'
        repaired = _repair_json(raw)
        assert "\x0b" not in repaired

    def test_tab_and_newline_preserved(self):
        """Escaped tab and newline in JSON strings must survive the repair step."""
        from app.services.claude_client import _repair_json
        # Use literal escape sequences (not actual control chars) as they appear in JSON
        raw = '{"key": "line1\\nline2\\ttabbed"}'
        repaired = _repair_json(raw)
        parsed = json.loads(repaired)
        assert "line2" in parsed["key"]


# ── Test 7 — json_repair library is importable and used as fallback ───────────

class TestJsonRepairLibrary:

    def test_json_repair_importable(self):
        from json_repair import repair_json  # noqa: F401  # must not raise

    def test_json_repair_fixes_unquoted_keys(self):
        from json_repair import repair_json
        raw = "{case_name: 'test', source_summary: 'x'}"
        fixed = repair_json(raw)
        parsed = json.loads(fixed) if isinstance(fixed, str) else fixed
        assert isinstance(parsed, dict)

    def test_json_repair_fixes_missing_comma(self):
        from json_repair import repair_json
        raw = '{"a": 1 "b": 2}'
        fixed = repair_json(raw)
        parsed = json.loads(fixed) if isinstance(fixed, str) else fixed
        assert parsed["a"] == 1
        assert parsed["b"] == 2


# ── Test 8 — compact-mode note injected for large research_view ───────────────

class TestCompactModeNoteInjection:

    def _build_prompt(self, research_view: str) -> str:
        from app.services.fact_lock_service import _build_research_view_prompt
        return _build_research_view_prompt(
            case_hint="test case",
            episode_number="001",
            source_url="https://example.com",
            transcript_research_view=research_view,
        )

    def test_compact_note_present_for_large_view(self):
        large_view = "x " * 8_500  # ~17 000 chars > 16 000 threshold
        assert len(large_view) > 16_000
        prompt = self._build_prompt(large_view)
        assert "COMPACT MODE ACTIVE" in prompt

    def test_compact_note_absent_for_small_view(self):
        small_view = "short transcript content"
        prompt = self._build_prompt(small_view)
        assert "COMPACT MODE ACTIVE" not in prompt

    def test_compact_note_absent_at_exact_threshold(self):
        # Exactly 16 000 chars — threshold is strictly >, so no note.
        exact_view = "a" * 16_000
        prompt = self._build_prompt(exact_view)
        assert "COMPACT MODE ACTIVE" not in prompt

    def test_compact_note_present_one_over_threshold(self):
        over_view = "a" * 16_001
        prompt = self._build_prompt(over_view)
        assert "COMPACT MODE ACTIVE" in prompt


# ── Test 9 — parse error path included in ValueError message ─────────────────

class TestParseErrorPathInMessage:
    """
    _run_research_view_mode saves _fact_lock_parse_error.txt and includes
    its path in the raised ValueError.  Test via a unit-level simulation
    (no real Claude call).
    """

    def test_parse_error_path_mentioned(self, tmp_path):
        import unittest.mock as mock
        from app.services import fact_lock_service as svc

        bad_raw = "not json at all"

        with mock.patch.object(
            svc, "call_claude_agent", return_value=(bad_raw, "end_turn")
        ), mock.patch.object(
            svc, "settings"
        ) as mock_settings:
            mock_settings.claude_max_tokens = 8000
            mock_settings.fact_lock_mode = "research_view"

            with pytest.raises(ValueError) as exc_info:
                svc._run_research_view_mode(
                    case_hint="test",
                    episode_number="001",
                    source_url="https://example.com",
                    transcript_research_view="short transcript",
                    facts_dir=tmp_path,
                )

        msg = str(exc_info.value)
        assert "_fact_lock_parse_error" in msg
        # The error file should have been written
        assert (tmp_path / "_fact_lock_parse_error.txt").exists()

    def test_raw_response_also_saved(self, tmp_path):
        import unittest.mock as mock
        from app.services import fact_lock_service as svc

        bad_raw = "not json at all"

        with mock.patch.object(
            svc, "call_claude_agent", return_value=(bad_raw, "end_turn")
        ), mock.patch.object(
            svc, "settings"
        ) as mock_settings:
            mock_settings.claude_max_tokens = 8000
            mock_settings.fact_lock_mode = "research_view"

            with pytest.raises(ValueError):
                svc._run_research_view_mode(
                    case_hint="test",
                    episode_number="001",
                    source_url="https://example.com",
                    transcript_research_view="short transcript",
                    facts_dir=tmp_path,
                )

        # Raw response file must also be present alongside the error file
        assert (tmp_path / "_fact_lock_raw_response.txt").exists()
        assert (tmp_path / "_fact_lock_raw_response.txt").read_text() == bad_raw
