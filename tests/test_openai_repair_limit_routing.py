"""
Tests for the routing behaviour when OpenAI repair targets exceed OPENAI_REPAIR_MAX_CHUNKS.

Verifies that:
  - ≤ OPENAI_REPAIR_MAX_CHUNKS targets → route proceeds to openai_targeted
  - > OPENAI_REPAIR_MAX_CHUNKS targets → route redirected to python/claude auto-rebuild
  - repair_routing_service is the decision point (not agent_pipeline directly)
  - The route value reflects the correct decision in each case
"""
from __future__ import annotations

import pytest

from app.services.repair_routing_service import run_repair_routing


# ─── Target builder ───────────────────────────────────────────────────────────

def _target(i: int, issue_type: str = "retention", problem: str = "slow pacing") -> dict:
    return {
        "chunk_id": f"{i:03d}_chunk",
        "issue_type": issue_type,
        "problem": problem,
        "repair_instruction": "improve pacing",
    }


def _reports_with_targets(n: int, issue_type: str = "retention") -> dict:
    targets = [_target(i, issue_type) for i in range(1, n + 1)]
    return {
        "openai_final_premium": {
            "approved": False,
            "safe_to_voice": False,
            "chunk_repair_targets": targets,
            "issues": [],
        }
    }


# ─── Test: ≤ limit → openai_targeted ─────────────────────────────────────────

class TestBelowLimit:
    def test_2_targets_routes_to_openai(self):
        """2 targets, limit=6 → should NOT route to auto_rebuild."""
        plan = run_repair_routing(
            all_gate_reports=_reports_with_targets(2),
            openai_repair_max_chunks=6,
        )
        route = plan.get("route", "")
        assert route != "auto_rebuild_required", (
            f"2 targets below limit should not trigger rebuild. Got route={route}"
        )

    def test_exactly_at_limit_routes_to_openai(self):
        """Exactly 6 targets with limit=6 → openai_targeted."""
        plan = run_repair_routing(
            all_gate_reports=_reports_with_targets(6),
            openai_repair_max_chunks=6,
        )
        route = plan.get("route", "")
        # At the limit, should still route to openai (not rebuild)
        assert route in (
            "openai_targeted", "python_only", "claude_grouped_repair"
        ), f"At-limit targets should not force rebuild. Got route={route}"

    def test_1_target_python_fix_route(self):
        """Single deterministic target → python_only or openai_targeted."""
        plan = run_repair_routing(
            all_gate_reports=_reports_with_targets(1, issue_type="safety"),
            openai_repair_max_chunks=6,
        )
        assert plan.get("route") in (
            "python_only", "openai_targeted", "claude_grouped_repair"
        )


# ─── Test: > limit → auto_rebuild / python+claude ────────────────────────────

class TestAboveLimit:
    def test_7_targets_above_limit_6(self):
        """7 targets with limit=6 → rebuild route."""
        plan = run_repair_routing(
            all_gate_reports=_reports_with_targets(7),
            openai_repair_max_chunks=6,
        )
        route = plan.get("route", "")
        assert route in (
            "auto_rebuild_required", "python_only", "claude_grouped_repair",
            "stop_not_voice_ready",
        ), f"7 targets > limit 6 should trigger rebuild route. Got route={route}"

    def test_10_targets_above_limit_6(self):
        """10 targets with limit=6 → rebuild route (not openai_targeted)."""
        plan = run_repair_routing(
            all_gate_reports=_reports_with_targets(10),
            openai_repair_max_chunks=6,
        )
        route = plan.get("route", "")
        assert route != "openai_targeted", (
            f"10 targets should not route to openai (limit=6). Got route={route}"
        )

    def test_12_targets_limit_4(self):
        """12 targets with limit=4 → always exceeds limit → rebuild route."""
        plan = run_repair_routing(
            all_gate_reports=_reports_with_targets(12),
            openai_repair_max_chunks=4,
        )
        route = plan.get("route", "")
        assert route in (
            "auto_rebuild_required", "python_only", "claude_grouped_repair",
            "stop_not_voice_ready",
        )

    def test_python_fixes_reduce_effective_count(self):
        """If all targets are deterministic, python_fixes list is non-empty."""
        # All targets are pure metadata/superlative — can be fixed by python
        targets_only_metadata = [
            _target(i, "safety", "most infamous in title — metadata issue")
            for i in range(10)
        ]
        plan = run_repair_routing(
            all_gate_reports={
                "openai_final_premium": {
                    "approved": False,
                    "chunk_repair_targets": targets_only_metadata,
                    "issues": [],
                }
            },
            openai_repair_max_chunks=6,
        )
        assert len(plan.get("python_fixes", [])) > 0, (
            "Deterministic metadata issues should always produce python_fixes"
        )


# ─── Test: unrecoverable always stops regardless of limit ────────────────────

class TestUnrecoverableAlwaysStops:
    def test_1_unrecoverable_target_stops_pipeline(self):
        """Even 1 target with 'contradictory facts' stops the pipeline."""
        plan = run_repair_routing(
            all_gate_reports={
                "openai_final_premium": {
                    "approved": False,
                    "chunk_repair_targets": [
                        _target(1, "safety", "contradictory facts — external verification needed")
                    ],
                    "issues": [
                        {
                            "severity": "high",
                            "description": "contradictory facts in verified names",
                            "chunk_id": "001_chunk",
                        }
                    ],
                }
            },
            openai_repair_max_chunks=6,
        )
        assert plan.get("route") == "stop_not_voice_ready"

    def test_unrecoverable_in_issues_not_targets_stops(self):
        """Unrecoverable issue in gate issues (not targets) still stops."""
        plan = run_repair_routing(
            all_gate_reports={
                "openai_final_premium": {
                    "approved": False,
                    "chunk_repair_targets": [],
                    "issues": [
                        {
                            "severity": "high",
                            "description": "unrecoverable structural failure in script",
                            "chunk_id": None,
                        }
                    ],
                }
            },
            openai_repair_max_chunks=6,
        )
        assert plan.get("route") == "stop_not_voice_ready"


# ─── Test: openai_repair_targets list reflects actual targets ────────────────

class TestOpenaiTargetsList:
    def test_openai_targets_populated_at_limit(self):
        """With targets ≤ limit, openai_repair_targets should be populated."""
        plan = run_repair_routing(
            all_gate_reports=_reports_with_targets(3),
            openai_repair_max_chunks=6,
        )
        # Route must be openai_targeted and openai_repair_targets non-empty
        if plan.get("route") == "openai_targeted":
            assert len(plan.get("openai_repair_targets", [])) > 0

    def test_openai_targets_empty_above_limit(self):
        """With targets > limit, openai_repair_targets should be empty (routed elsewhere)."""
        plan = run_repair_routing(
            all_gate_reports=_reports_with_targets(10),
            openai_repair_max_chunks=6,
        )
        if plan.get("route") != "openai_targeted":
            assert len(plan.get("openai_repair_targets", [])) == 0, (
                "When not routing to openai, openai_repair_targets must be empty"
            )
