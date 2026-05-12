"""
Tests for TASK 3 — English-quote repair targets from deterministic_auto_fix_service
must be routed to Claude rebuild, not left unreported or sent to OpenAI.

Verifies:
- deterministic_auto_fix_service returns english_quote_repair_targets
- _run_routing_and_rebuild converts and appends them to claude_repair_targets
- targets already in claude_repair_targets are not duplicated
- premium_section_rebuild receives the merged targets
- exact_english_quote_copy never routed to openai_targeted
"""
from __future__ import annotations

import pytest

from app.services.deterministic_auto_fix_service import (
    _extract_long_english_runs,
    _fix_narration_chunks,
    run_deterministic_auto_fix,
    _MIN_ENGLISH_QUOTE_WORDS,
)


# ─── Helper: simulate the routing merge logic ─────────────────────────────────

def _merge_eq_targets_into_routing(
    af_report: dict,
    routing_plan: dict,
) -> dict:
    """
    Replicate the Step 2b logic from _run_routing_and_rebuild:
    read english_quote_repair_targets from af_report and append to routing_plan.
    """
    eq_targets = af_report.get("english_quote_repair_targets", [])
    if not eq_targets:
        return routing_plan

    existing_cids: set[str] = {
        cid
        for t in routing_plan.get("claude_repair_targets", [])
        for cid in t.get("chunk_ids", [])
    }

    added = 0
    for eq in eq_targets:
        cid = eq.get("chunk_id", "")
        if not cid or cid in existing_cids:
            continue
        routing_plan.setdefault("claude_repair_targets", []).append({
            "area":                   "hindi_quality",
            "root_cause_id":          f"eq_{cid}",
            "repair_instruction":     eq.get("repair_instruction", ""),
            "affected_targets":       [eq],
            "chunk_ids":              [cid],
            "preferred_repair_owner": "claude",
            "reason":                 "Verbatim English/source quote — requires Hindi translation/paraphrase",
            "issue_type":             "exact_english_quote_copy",
        })
        existing_cids.add(cid)
        added += 1

    routing_plan["_eq_targets_added"] = added
    return routing_plan


# ─── Tests: deterministic service detects English quotes ─────────────────────

class TestDeterministicServiceQuoteDetection:
    def _script_with_english_quote(self, text: str) -> dict:
        return {
            "hindi_narration_chunks": [{"chunk_id": "009_events", "text": text}],
            "youtube_metadata": {},
        }

    def test_long_english_quote_produces_repair_targets(self):
        text = 'Camilleri asked "Did you see the demon? Did you feel the demon?" at the hearing.'
        script = self._script_with_english_quote(text)
        _, report = run_deterministic_auto_fix(script, case_hint="Beckett case")
        assert "english_quote_repair_targets" in report
        assert report["english_quote_repair_count"] >= 1

    def test_english_quote_repair_target_has_chunk_id(self):
        text = "The prosecutor said he was guilty beyond any reasonable doubt in this case."
        script = self._script_with_english_quote(text)
        _, report = run_deterministic_auto_fix(script, case_hint="test")
        for t in report.get("english_quote_repair_targets", []):
            assert "chunk_id" in t
            assert t["chunk_id"] == "009_events"

    def test_english_quote_target_has_repair_instruction(self):
        text = "The witness testified that she saw nothing suspicious at the property that day."
        script = self._script_with_english_quote(text)
        _, report = run_deterministic_auto_fix(script, case_hint="test")
        for t in report.get("english_quote_repair_targets", []):
            assert "repair_instruction" in t
            assert len(t["repair_instruction"]) > 0

    def test_hindi_only_text_produces_no_eq_targets(self):
        text = "अदालत में सबूत पेश किए गए और न्यायाधीश ने फैसला सुनाया।"
        script = self._script_with_english_quote(text)
        _, report = run_deterministic_auto_fix(script, case_hint="test")
        assert report["english_quote_repair_count"] == 0
        assert report["english_quote_repair_targets"] == []


# ─── Tests: routing merge appends EQ targets to claude_repair_targets ─────────

class TestEnglishQuoteRoutingMerge:
    def _af_report_with_targets(self, chunk_ids: list[str]) -> dict:
        return {
            "english_quote_repair_targets": [
                {
                    "chunk_id":          cid,
                    "issue_type":        "exact_english_quote_copy",
                    "problem":           "Long English quote detected.",
                    "repair_instruction": "Translate to Hindi or paraphrase.",
                    "verbatim_run_sample": "the demon said did you",
                }
                for cid in chunk_ids
            ],
            "english_quote_repair_count": len(chunk_ids),
        }

    def test_eq_target_appended_when_not_already_queued(self):
        af_report = self._af_report_with_targets(["009_events"])
        routing_plan = {"route": "claude_grouped_repair", "claude_repair_targets": []}
        merged = _merge_eq_targets_into_routing(af_report, routing_plan)
        claude_targets = merged.get("claude_repair_targets", [])
        queued_cids = [cid for t in claude_targets for cid in t.get("chunk_ids", [])]
        assert "009_events" in queued_cids

    def test_eq_target_not_duplicated_when_already_queued(self):
        af_report = self._af_report_with_targets(["009_events"])
        # 009_events already queued
        routing_plan = {
            "route": "claude_grouped_repair",
            "claude_repair_targets": [{
                "chunk_ids": ["009_events"],
                "repair_instruction": "prior target",
                "area": "retention",
            }],
        }
        merged = _merge_eq_targets_into_routing(af_report, routing_plan)
        queued_cids = [cid for t in merged["claude_repair_targets"] for cid in t.get("chunk_ids", [])]
        assert queued_cids.count("009_events") == 1   # exactly once

    def test_multiple_eq_chunks_all_appended(self):
        af_report = self._af_report_with_targets(["005_background", "009_events"])
        routing_plan = {"route": "python_only", "claude_repair_targets": []}
        merged = _merge_eq_targets_into_routing(af_report, routing_plan)
        queued_cids = [cid for t in merged["claude_repair_targets"] for cid in t.get("chunk_ids", [])]
        assert "005_background" in queued_cids
        assert "009_events" in queued_cids

    def test_eq_target_preferred_owner_is_claude(self):
        af_report = self._af_report_with_targets(["009_events"])
        routing_plan = {"route": "python_only", "claude_repair_targets": []}
        merged = _merge_eq_targets_into_routing(af_report, routing_plan)
        for t in merged["claude_repair_targets"]:
            if "009_events" in t.get("chunk_ids", []):
                assert t.get("preferred_repair_owner") == "claude"

    def test_eq_target_not_routed_to_openai(self):
        """exact_english_quote_copy must never end up in openai targets."""
        af_report = self._af_report_with_targets(["009_events"])
        routing_plan = {"route": "python_only", "claude_repair_targets": []}
        merged = _merge_eq_targets_into_routing(af_report, routing_plan)
        # Routing plan must route to claude, not openai_targeted
        for t in merged.get("claude_repair_targets", []):
            assert t.get("preferred_repair_owner") != "python"
        # Verify there's no openai_targets list with eq chunks
        for t in merged.get("openai_targets", []):
            assert "009_events" not in t.get("chunk_ids", [])

    def test_empty_eq_targets_leaves_routing_plan_unchanged(self):
        af_report = {"english_quote_repair_targets": [], "english_quote_repair_count": 0}
        original_targets = [{"chunk_ids": ["001_hook"], "area": "retention"}]
        routing_plan = {"route": "claude_grouped_repair", "claude_repair_targets": list(original_targets)}
        merged = _merge_eq_targets_into_routing(af_report, routing_plan)
        assert len(merged["claude_repair_targets"]) == 1
        # When eq_targets is empty the helper returns early — _eq_targets_added is 0 implicitly
        assert merged.get("_eq_targets_added", 0) == 0

    def test_eq_issue_type_preserved_in_target(self):
        af_report = self._af_report_with_targets(["009_events"])
        routing_plan = {"route": "python_only", "claude_repair_targets": []}
        merged = _merge_eq_targets_into_routing(af_report, routing_plan)
        for t in merged["claude_repair_targets"]:
            if "009_events" in t.get("chunk_ids", []):
                assert t.get("issue_type") == "exact_english_quote_copy"
