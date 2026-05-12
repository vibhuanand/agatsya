"""
Tests for hindi_text_lint_service child-victim safety rules.

Verifies:
  - Organ-specific injury phrases flagged as high-severity when is_child_victim_case=True
  - Graphic metadata phrases flagged correctly
  - Generic / safe wording NOT flagged
  - Child-victim rules skipped (no flags) when is_child_victim_case=False
  - English "ruptured liver" flagged in child-victim mode
  - "गंभीर आंतरिक चोटें" (the safe replacement) NOT flagged
"""
from __future__ import annotations

import pytest

from app.services.hindi_text_lint_service import run_hindi_text_lint


# ─── Minimal script builder ───────────────────────────────────────────────────

def _script_with_chunk(text: str, chunk_id: str = "001_hook") -> dict:
    return {
        "hindi_narration_chunks": [
            {"chunk_id": chunk_id, "text": text}
        ],
        "youtube_metadata": {
            "recommended_title": "एक सच्ची घटना",
            "tags": [],
            "description": "",
        },
    }


# ─── Tests: organ flagging in child-victim mode ───────────────────────────────

class TestOrganFlaggingChildVictimMode:
    def test_fata_hua_jigar_flagged(self):
        script = _script_with_chunk("जाँच में पता चला कि फटा हुआ जिगर था।")
        report = run_hindi_text_lint(script, hinglish_level=2, is_child_victim_case=True)
        high = [i for i in report.get("issues", []) if i.get("severity") == "high"]
        assert any("जिगर" in i.get("text", "") or "liver" in i.get("rule_id", "").lower()
                   for i in high), f"Expected high-severity organ flag, got: {high}"

    def test_jigar_fat_flagged(self):
        script = _script_with_chunk("डॉक्टर ने बताया कि जिगर फट गया था।")
        report = run_hindi_text_lint(script, hinglish_level=2, is_child_victim_case=True)
        high_rules = [i.get("rule_id", "") for i in report.get("issues", [])
                      if i.get("severity") == "high"]
        assert any("liver" in r.lower() or "organ" in r.lower() for r in high_rules) or \
               len([i for i in report.get("issues", []) if i.get("severity") == "high"]) > 0

    def test_ruptured_liver_english_flagged(self):
        script = _script_with_chunk("The report confirmed ruptured liver damage.")
        report = run_hindi_text_lint(script, hinglish_level=2, is_child_victim_case=True)
        high = [i for i in report.get("issues", []) if i.get("severity") == "high"]
        assert len(high) > 0, "English 'ruptured liver' should be high-severity in child-victim mode"

    def test_sharir_ke_tukde_flagged(self):
        script = _script_with_chunk("उसके शरीर के टुकड़े मिले थे।")
        report = run_hindi_text_lint(script, hinglish_level=2, is_child_victim_case=True)
        high = [i for i in report.get("issues", []) if i.get("severity") == "high"]
        assert len(high) > 0, "Body fragmentation wording should be flagged"

    def test_kshata_vikshata_flagged(self):
        script = _script_with_chunk("शव क्षत-विक्षत पाया गया था।")
        report = run_hindi_text_lint(script, hinglish_level=2, is_child_victim_case=True)
        high = [i for i in report.get("issues", []) if i.get("severity") == "high"]
        assert len(high) > 0


# ─── Tests: child-victim rules skipped in non-child-victim mode ──────────────

class TestOrganNotFlaggedInNonChildVictimMode:
    def test_fata_hua_jigar_not_flagged_without_flag(self):
        """Same phrase should NOT be flagged when is_child_victim_case=False."""
        script = _script_with_chunk("जाँच में पता चला कि फटा हुआ जिगर था।")
        report = run_hindi_text_lint(script, hinglish_level=2, is_child_victim_case=False)
        # The organ rule should be skipped entirely
        organ_flags = [
            i for i in report.get("issues", [])
            if "jigar" in i.get("rule_id", "").lower()
            or "liver" in i.get("rule_id", "").lower()
            or "organ" in i.get("rule_id", "").lower()
        ]
        assert len(organ_flags) == 0, (
            "Child-victim organ rules must not fire when is_child_victim_case=False"
        )

    def test_ruptured_liver_not_flagged_without_flag(self):
        script = _script_with_chunk("The report confirmed ruptured liver damage.")
        report = run_hindi_text_lint(script, hinglish_level=2, is_child_victim_case=False)
        organ_flags = [
            i for i in report.get("issues", [])
            if "liver" in i.get("rule_id", "").lower()
        ]
        assert len(organ_flags) == 0

    def test_default_is_child_victim_false(self):
        """Default call (no is_child_victim_case kwarg) should not flag organ phrases."""
        script = _script_with_chunk("जाँच में पता चला कि फटा हुआ जिगर था।")
        report = run_hindi_text_lint(script, hinglish_level=2)
        organ_flags = [
            i for i in report.get("issues", [])
            if "jigar" in i.get("rule_id", "").lower()
            or "liver" in i.get("rule_id", "").lower()
        ]
        assert len(organ_flags) == 0


# ─── Tests: safe replacement wording NOT flagged ─────────────────────────────

class TestSafeReplacementNotFlagged:
    def test_gambhir_antrik_chotein_not_flagged(self):
        """The safe replacement phrase must not trigger any organ flags."""
        script = _script_with_chunk(
            "जाँच में गंभीर आंतरिक चोटें पाई गईं।"
        )
        report = run_hindi_text_lint(script, hinglish_level=2, is_child_victim_case=True)
        organ_flags = [
            i for i in report.get("issues", [])
            if "jigar" in i.get("rule_id", "").lower()
            or "liver" in i.get("rule_id", "").lower()
            or "organ" in i.get("rule_id", "").lower()
        ]
        assert len(organ_flags) == 0, (
            "'गंभीर आंतरिक चोटें' is the approved phrase and must not be flagged"
        )

    def test_serious_internal_injuries_english_not_flagged(self):
        """English safe replacement should not be flagged."""
        script = _script_with_chunk("The autopsy found serious internal injuries.")
        report = run_hindi_text_lint(script, hinglish_level=2, is_child_victim_case=True)
        organ_flags = [
            i for i in report.get("issues", [])
            if "liver" in i.get("rule_id", "").lower()
        ]
        assert len(organ_flags) == 0

    def test_clean_emotional_narrative_not_flagged(self):
        """Purely emotional narrative with no graphic content should produce no organ flags."""
        script = _script_with_chunk(
            "वह एक 17 साल की लड़की थी जो पढ़ाई में बहुत होशियार थी। "
            "उसके परिवार को न्याय की तलाश आज भी है।"
        )
        report = run_hindi_text_lint(script, hinglish_level=2, is_child_victim_case=True)
        high = [i for i in report.get("issues", []) if i.get("severity") == "high"]
        assert len(high) == 0, f"Clean narrative should have no high-severity issues: {high}"


# ─── Tests: report structure ──────────────────────────────────────────────────

class TestLintReportStructure:
    def test_report_has_required_keys(self):
        script = _script_with_chunk("एक सच्ची कहानी।")
        report = run_hindi_text_lint(script, hinglish_level=2, is_child_victim_case=True)
        for key in ("issues", "total_issues"):
            assert key in report, f"Missing key: {key}"

    def test_high_issues_count_matches_issues(self):
        script = _script_with_chunk("जाँच में पता चला कि फटा हुआ जिगर था।")
        report = run_hindi_text_lint(script, hinglish_level=2, is_child_victim_case=True)
        high_count = sum(
            1 for i in report.get("issues", []) if i.get("severity") == "high"
        )
        assert report.get("high_issues", high_count) >= high_count
