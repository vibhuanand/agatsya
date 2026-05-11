"""
Tests for deterministic_auto_fix_service.run_deterministic_auto_fix().

Verifies:
  - "most infamous" replaced with "widely known" in metadata
  - "मीडिया का गुनाह" replaced with "पत्रकारिता पर सवाल"
  - "फटा हुआ जिगर" replaced with "गंभीर आंतरिक चोटें" in narration chunks
  - Taiwan folder slug sanitized
  - Factual content (names, dates, uninvolved text) not touched
  - auto_fix_report has total_fixes_applied count
  - Report written to review_dir when provided
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.deterministic_auto_fix_service import run_deterministic_auto_fix


# ─── Minimal script builder ───────────────────────────────────────────────────

def _script(
    title: str = "एक अनसुलझा रहस्य",
    tags: list[str] | None = None,
    chunks: list[dict] | None = None,
    folder_name: str = "001-test-case",
    thumbnail_text: str = "एक सवाल",
) -> dict:
    return {
        "folder_name": folder_name,
        "youtube_metadata": {
            "recommended_title": title,
            "title_options": [title],
            "tags": tags or ["hindi true crime"],
            "thumbnail_options": [{"thumbnail_text": thumbnail_text, "angle": "neutral"}],
            "description": "एक सच्ची कहानी।",
            "pinned_comment": "आपके विचार?",
            "chapters": [],
            "shorts_plan": [],
        },
        "hindi_narration_chunks": chunks or [],
    }


def _chunk(chunk_id: str, text: str) -> dict:
    return {"chunk_id": chunk_id, "text": text}


# ─── Metadata fixes ───────────────────────────────────────────────────────────

class TestMetadataFixes:
    def test_most_infamous_replaced_in_title(self):
        script = _script(title="Taiwan's most infamous case — Pai Hsiao-Yen")
        updated, report = run_deterministic_auto_fix(script_draft=script)
        new_title = updated["youtube_metadata"]["recommended_title"]
        assert "most infamous" not in new_title.lower()
        assert report["total_fixes_applied"] >= 1

    def test_most_brutal_replaced_in_title_options(self):
        script = _script()
        script["youtube_metadata"]["title_options"] = [
            "The most brutal crime of 1997",
            "एक अनसुलझा रहस्य",
        ]
        updated, report = run_deterministic_auto_fix(script_draft=script)
        for opt in updated["youtube_metadata"]["title_options"]:
            assert "most brutal" not in opt.lower()

    def test_legal_blame_in_description(self):
        script = _script()
        script["youtube_metadata"]["description"] = (
            "इस मामले में journalism killed her — मीडिया ने सब कुछ बर्बाद किया।"
        )
        updated, report = run_deterministic_auto_fix(script_draft=script)
        desc = updated["youtube_metadata"]["description"]
        assert "journalism killed her" not in desc.lower()
        assert report["total_fixes_applied"] >= 1

    def test_legal_blame_hindi_in_description(self):
        script = _script()
        script["youtube_metadata"]["description"] = (
            "यह मीडिया का गुनाह था — कोई और नहीं।"
        )
        updated, _ = run_deterministic_auto_fix(script_draft=script)
        assert "मीडिया का गुनाह" not in updated["youtube_metadata"]["description"]

    def test_sabse_kukhyat_replaced_in_tags(self):
        script = _script(tags=["सबसे कुख्यात", "hindi true crime", "Taiwan 1997"])
        updated, report = run_deterministic_auto_fix(script_draft=script)
        tags = updated["youtube_metadata"]["tags"]
        assert not any("सबसे कुख्यात" in t for t in tags)
        assert report["total_fixes_applied"] >= 1

    def test_taiwan_folder_slug_sanitized(self):
        script = _script(folder_name="002-taiwans-most-infamous-case")
        updated, report = run_deterministic_auto_fix(script_draft=script)
        assert "most-infamous" not in updated["folder_name"]

    def test_thumbnail_text_graphic_removed(self):
        script = _script(thumbnail_text="फटा हुआ जिगर — सच्चाई")
        updated, report = run_deterministic_auto_fix(script_draft=script)
        thumb = updated["youtube_metadata"]["thumbnail_options"][0]["thumbnail_text"]
        assert "फटा" not in thumb


# ─── Narration chunk fixes ────────────────────────────────────────────────────

class TestNarrationChunkFixes:
    def test_organ_hindi_replaced_in_chunk(self):
        chunks = [_chunk("001_hook", "जाँच में पता चला कि बच्ची का जिगर फट गया था।")]
        script = _script(chunks=chunks)
        updated, report = run_deterministic_auto_fix(
            script_draft=script,
            case_hint="taiwan 1997 pai hsiao-yen child",
        )
        new_text = updated["hindi_narration_chunks"][0]["text"]
        assert "जिगर फट" not in new_text
        assert "गंभीर आंतरिक चोटें" in new_text or "internal" in new_text.lower()
        assert report["total_fixes_applied"] >= 1

    def test_fata_hua_jigar_replaced(self):
        chunks = [_chunk("002_body", "रिपोर्ट में लिखा था — फटा हुआ जिगर।")]
        script = _script(chunks=chunks)
        updated, _ = run_deterministic_auto_fix(
            script_draft=script,
            case_hint="child victim case",
        )
        assert "फटा हुआ जिगर" not in updated["hindi_narration_chunks"][0]["text"]

    def test_ruptured_liver_english_replaced(self):
        chunks = [_chunk("003_narr", "The autopsy confirmed ruptured liver injuries.")]
        script = _script(chunks=chunks)
        updated, report = run_deterministic_auto_fix(
            script_draft=script,
            case_hint="minor victim 1997",
        )
        assert "ruptured liver" not in updated["hindi_narration_chunks"][0]["text"].lower()

    def test_media_ka_gunah_in_narration(self):
        chunks = [_chunk("004_mid", "यह सब मीडिया का गुनाह था — इसमें कोई शक नहीं।")]
        script = _script(chunks=chunks)
        updated, _ = run_deterministic_auto_fix(script_draft=script)
        assert "मीडिया का गुनाह" not in updated["hindi_narration_chunks"][0]["text"]


# ─── Factual preservation ─────────────────────────────────────────────────────

class TestFactualPreservation:
    def test_victim_name_not_changed(self):
        """Victim's name should never be altered."""
        chunks = [_chunk("005_hook", "Pai Hsiao-Yen की उम्र 17 साल थी।")]
        script = _script(chunks=chunks)
        updated, _ = run_deterministic_auto_fix(script_draft=script, case_hint="taiwan 1997")
        assert "Pai Hsiao-Yen" in updated["hindi_narration_chunks"][0]["text"]

    def test_clean_narration_untouched(self):
        """Narration with no banned phrases should be returned unchanged."""
        original_text = "यह घटना 1997 में Taiwan में हुई थी। पत्रकारिता पर सवाल उठे।"
        chunks = [_chunk("006_clean", original_text)]
        script = _script(chunks=chunks)
        updated, report = run_deterministic_auto_fix(script_draft=script)
        assert updated["hindi_narration_chunks"][0]["text"] == original_text
        assert report["total_fixes_applied"] == 0

    def test_hindi_numbers_dates_untouched(self):
        """Dates and numbers in clean text must not be modified."""
        chunks = [_chunk("007_date", "28 अप्रैल 1997 को यह मामला सामने आया।")]
        script = _script(chunks=chunks)
        updated, _ = run_deterministic_auto_fix(script_draft=script)
        assert "28 अप्रैल 1997" in updated["hindi_narration_chunks"][0]["text"]


# ─── Report structure ─────────────────────────────────────────────────────────

class TestAutoFixReport:
    def test_report_has_required_fields(self):
        script = _script()
        _, report = run_deterministic_auto_fix(script_draft=script)
        assert "total_fixes_applied" in report
        assert "fixes" in report
        assert isinstance(report["total_fixes_applied"], int)

    def test_fix_log_entries(self):
        """Each applied fix should have a log entry with chunk_id or field."""
        script = _script(title="The most brutal crime of 1997")
        _, report = run_deterministic_auto_fix(script_draft=script)
        if report["total_fixes_applied"] > 0:
            fixes = report.get("fixes", [])
            assert len(fixes) > 0
            first = fixes[0]
            assert "rule" in first or "field" in first or "description" in first

    def test_report_written_to_review_dir(self, tmp_path):
        script = _script()
        _, _ = run_deterministic_auto_fix(script_draft=script, review_dir=tmp_path)
        out = tmp_path / "deterministic_auto_fix_report.json"
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "total_fixes_applied" in data

    def test_no_fixes_when_script_clean(self):
        script = _script(
            title="एक गुमनाम केस",
            tags=["hindi true crime", "true crime", "Taiwan 1997"],
        )
        _, report = run_deterministic_auto_fix(script_draft=script)
        assert report["total_fixes_applied"] == 0
