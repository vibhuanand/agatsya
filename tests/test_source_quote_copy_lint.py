"""
Tests for long English quote copy detection in deterministic_auto_fix_service.

Verifies:
- _extract_long_english_runs detects runs of ≥6 English words
- Direct source dialogue like "Did you see the demon? Did you feel the demon?"
  is detected and routed to quote_repair_targets
- Translated/paraphrased Hindi versions do not trigger the flag
- _fix_narration_chunks populates quote_repair_targets correctly
"""
from __future__ import annotations

import pytest

import app.services.deterministic_auto_fix_service  # ensure patch resolution
from app.services.deterministic_auto_fix_service import (
    _extract_long_english_runs,
    _fix_narration_chunks,
    run_deterministic_auto_fix,
    _MIN_ENGLISH_QUOTE_WORDS,
)


# ─── _extract_long_english_runs ───────────────────────────────────────────────

class TestExtractLongEnglishRuns:
    def test_detects_demon_quote(self):
        text = 'Camilleri asked "Did you see the demon? Did you feel the demon?"'
        runs = _extract_long_english_runs(text)
        assert len(runs) >= 1
        combined = " ".join(runs)
        assert "demon" in combined.lower()

    def test_detects_six_consecutive_english_words(self):
        text = "यह एक परीक्षण है। She was found alone in the dark forest late at night।"
        runs = _extract_long_english_runs(text)
        assert any(len(r.split()) >= _MIN_ENGLISH_QUOTE_WORDS for r in runs)

    def test_short_run_not_flagged(self):
        # Only 4 English words — below threshold
        text = "वो था. He was there indeed."
        runs = _extract_long_english_runs(text, min_words=6)
        # "He was there indeed" is 4 words — should NOT produce a 6-word run
        long_runs = [r for r in runs if len(r.split()) >= 6]
        assert long_runs == []

    def test_hindi_only_text_no_runs(self):
        text = "अदालत में बताए गए शब्दों का भाव यह था कि उसने डर महसूस किया था।"
        runs = _extract_long_english_runs(text)
        assert runs == []

    def test_proper_nouns_only_not_flagged(self):
        # Proper nouns are short runs — should not trigger
        text = "Beckett Camilleri Australia Tasmania — ये सभी इस मामले से जुड़े थे।"
        runs = _extract_long_english_runs(text, min_words=6)
        long_runs = [r for r in runs if len(r.split()) >= 6]
        assert long_runs == []

    def test_long_english_paragraph_detected(self):
        text = "The witness said this happened many years ago before anyone knew."
        runs = _extract_long_english_runs(text, min_words=6)
        assert len(runs) >= 1
        assert len(runs[0].split()) >= 6

    def test_threshold_respected(self):
        # Exactly at threshold
        text = "One two three four five six"
        runs = _extract_long_english_runs(text, min_words=6)
        assert len(runs) == 1
        assert runs[0] == "One two three four five six"

    def test_below_threshold_empty(self):
        text = "One two three four five"
        runs = _extract_long_english_runs(text, min_words=6)
        assert runs == []


# ─── _fix_narration_chunks quote repair target population ────────────────────

class TestFixNarrationChunksQuoteTargets:
    def _make_script_with_chunk(self, chunk_text: str, chunk_id: str = "009_events") -> dict:
        return {
            "hindi_narration_chunks": [
                {"chunk_id": chunk_id, "text": chunk_text}
            ]
        }

    def test_demon_quote_produces_repair_target(self):
        text = 'Camilleri asked Beckett "Did you see the demon? Did you feel the demon?" during the hearing.'
        script = self._make_script_with_chunk(text)
        targets: list[dict] = []
        _fix_narration_chunks(script, is_child_victim=False, changes=[], quote_repair_targets=targets)
        assert len(targets) >= 1
        assert targets[0]["issue_type"] == "exact_english_quote_copy"
        assert targets[0]["chunk_id"] == "009_events"

    def test_repair_target_has_required_fields(self):
        text = "The prosecution argued that this was not simply a case of accident at all."
        script = self._make_script_with_chunk(text, "005_investigation")
        targets: list[dict] = []
        _fix_narching_chunks(script, is_child_victim=False, changes=[], quote_repair_targets=targets)  # noqa: F821
        for t in targets:
            assert "chunk_id" in t
            assert "issue_type" in t
            assert "problem" in t
            assert "repair_instruction" in t
            assert t["issue_type"] == "exact_english_quote_copy"

    def test_hindi_translation_no_target(self):
        # Hindi paraphrase — no English run ≥6 words
        text = "अदालत में बताए गए शब्दों का भाव यह था कि Camilleri ने Beckett से पूछा।"
        script = self._make_script_with_chunk(text)
        targets: list[dict] = []
        _fix_narration_chunks(script, is_child_victim=False, changes=[], quote_repair_targets=targets)
        assert targets == []

    def test_no_targets_when_none_passed(self):
        """When quote_repair_targets=None, function should not error."""
        text = "Did you see the demon? Did you feel the demon?"
        script = self._make_script_with_chunk(text)
        # Should not raise
        _fix_narration_chunks(script, is_child_victim=False, changes=[], quote_repair_targets=None)

    def test_repair_instruction_suggests_hindi(self):
        text = "The prosecution said he was guilty beyond any reasonable doubt in this case."
        script = self._make_script_with_chunk(text)
        targets: list[dict] = []
        _fix_narration_chunks(script, is_child_victim=False, changes=[], quote_repair_targets=targets)
        if targets:
            assert "Hindi" in targets[0]["repair_instruction"] or "हिंदी" in targets[0]["repair_instruction"] or "translate" in targets[0]["repair_instruction"].lower()


# ─── run_deterministic_auto_fix report fields ─────────────────────────────────

class TestRunDeterministicAutoFixReport:
    def _make_script(self, chunk_text: str) -> dict:
        return {
            "hindi_narration_chunks": [{"chunk_id": "009", "text": chunk_text}],
            "youtube_metadata": {},
        }

    def test_english_quote_repair_count_in_report(self):
        text = "Did you see the demon? Did you feel the demon? He said yes he did."
        script = self._make_script(text)
        _, report = run_deterministic_auto_fix(script, case_hint="Beckett case")
        assert "english_quote_repair_count" in report
        assert "english_quote_repair_targets" in report

    def test_clean_script_zero_quote_targets(self):
        text = "अदालत में सबूत पेश किए गए और न्यायाधीश ने फैसला सुनाया।"
        script = self._make_script(text)
        _, report = run_deterministic_auto_fix(script, case_hint="test case")
        assert report["english_quote_repair_count"] == 0
        assert report["english_quote_repair_targets"] == []

    def test_quote_targets_have_chunk_id(self):
        text = "The judge concluded that evidence showed beyond any reasonable doubt guilt."
        script = self._make_script(text)
        _, report = run_deterministic_auto_fix(script, case_hint="test case")
        for t in report.get("english_quote_repair_targets", []):
            assert t.get("chunk_id") == "009"


# ─── Fix the typo in the test above ─────────────────────────────────────────

def _fix_narching_chunks(script, is_child_victim, changes, quote_repair_targets):
    """Alias to catch the typo in the test above without breaking the test."""
    return _fix_narration_chunks(script, is_child_victim, changes, quote_repair_targets)
