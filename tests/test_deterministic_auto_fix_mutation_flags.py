"""
Tests for TASK 3 — Stage 12 deterministic auto-fix sets artifact_state mutation
flags accurately by inspecting the changes list, not just the total count.

Verifies:
- Changes with context "chunk:<id>" → script_mutated
- Changes with context "recreated_dialogue:<id>" → dialogue_mutated
- Changes with context metadata fields (title_options, recommended_title,
  description, tag, thumbnail_text, pinned_comment) → metadata_mutated
- Changes with context "folder_name" → no OFP-relevant mutation
- Mixed changes → correct set of mutation flags
- final_review_inputs_mutated is True whenever any relevant change exists
- _ran_any_repair flag correctly reflects content mutations
"""
from __future__ import annotations

import pytest

from app.services.agent_pipeline_service import ArtifactState
from app.services.deterministic_auto_fix_service import run_deterministic_auto_fix


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _simulate_stage12_mutation_flags(changes: list[dict]) -> tuple[ArtifactState, bool]:
    """
    Replicate Stage 12 mutation-flag logic from agent_pipeline_service.py.

    Returns (artifact_state, _ran_any_repair).
    """
    _METADATA_CONTEXTS = frozenset({
        "title_options", "recommended_title", "description",
        "tag", "thumbnail_text", "pinned_comment", "tags",
    })
    state = ArtifactState()
    ran_any_repair = False

    for chg in changes:
        ctx = chg.get("context", "")
        if ctx.startswith("chunk:"):
            ran_any_repair = True
            state.mark_script_mutated("stage12_deterministic_auto_fix_narration")
        elif ctx.startswith("recreated_dialogue:"):
            ran_any_repair = True
            state.mark_dialogue_mutated("stage12_deterministic_auto_fix_dialogue")
        elif ctx in _METADATA_CONTEXTS:
            ran_any_repair = True
            state.mark_metadata_mutated("stage12_deterministic_auto_fix_metadata")
        # "folder_name" → slug only, no OFP-relevant mutation

    return state, ran_any_repair


# ─── Tests: narration change marks script_mutated ────────────────────────────

class TestNarrationChangeMarksScriptMutated:
    def test_chunk_context_sets_script_mutated(self):
        changes = [
            {"context": "chunk:001", "description": "Child-victim safety fix",
             "occurrences": 1},
        ]
        state, ran = _simulate_stage12_mutation_flags(changes)
        assert state.script_mutated is True

    def test_chunk_context_sets_final_review_inputs_mutated(self):
        changes = [{"context": "chunk:003", "occurrences": 1, "description": "Fix"}]
        state, ran = _simulate_stage12_mutation_flags(changes)
        assert state.final_review_inputs_mutated is True

    def test_chunk_context_sets_ran_any_repair(self):
        changes = [{"context": "chunk:007", "occurrences": 1, "description": "Phrasing fix"}]
        _, ran = _simulate_stage12_mutation_flags(changes)
        assert ran is True

    def test_chunk_context_does_not_set_metadata_or_dialogue(self):
        changes = [{"context": "chunk:002", "occurrences": 1, "description": "Fix"}]
        state, _ = _simulate_stage12_mutation_flags(changes)
        assert state.metadata_mutated is False
        assert state.dialogue_mutated is False

    def test_multiple_chunk_changes_source_deduplicated(self):
        changes = [
            {"context": "chunk:001", "occurrences": 1, "description": "Fix A"},
            {"context": "chunk:003", "occurrences": 1, "description": "Fix B"},
        ]
        state, _ = _simulate_stage12_mutation_flags(changes)
        assert state.script_mutated is True
        # source deduplication — same source should appear only once
        assert state.mutation_sources.count(
            "stage12_deterministic_auto_fix_narration"
        ) == 1


# ─── Tests: metadata change marks metadata_mutated ───────────────────────────

class TestMetadataChangeMarksMetadataMutated:
    @pytest.mark.parametrize("ctx", [
        "title_options", "recommended_title", "description",
        "tag", "thumbnail_text", "pinned_comment", "tags",
    ])
    def test_metadata_context_sets_metadata_mutated(self, ctx):
        changes = [{"context": ctx, "occurrences": 1, "description": "Metadata fix"}]
        state, ran = _simulate_stage12_mutation_flags(changes)
        assert state.metadata_mutated is True, (
            f"context={ctx!r} must set metadata_mutated"
        )
        assert ran is True
        assert state.final_review_inputs_mutated is True

    def test_metadata_context_does_not_set_script_or_dialogue(self):
        changes = [{"context": "description", "occurrences": 1, "description": "Fix"}]
        state, _ = _simulate_stage12_mutation_flags(changes)
        assert state.script_mutated is False
        assert state.dialogue_mutated is False


# ─── Tests: dialogue change marks dialogue_mutated ───────────────────────────

class TestDialogueChangeMarksDialogueMutated:
    def test_recreated_dialogue_context_sets_dialogue_mutated(self):
        changes = [
            {"context": "recreated_dialogue:scene_001",
             "description": "Added missing label", "occurrences": 1},
        ]
        state, ran = _simulate_stage12_mutation_flags(changes)
        assert state.dialogue_mutated is True

    def test_recreated_dialogue_context_sets_final_review_mutated(self):
        changes = [{"context": "recreated_dialogue:scene_002", "occurrences": 1,
                    "description": "Disclaimer added"}]
        state, _ = _simulate_stage12_mutation_flags(changes)
        assert state.final_review_inputs_mutated is True

    def test_recreated_dialogue_does_not_set_script_or_metadata(self):
        changes = [{"context": "recreated_dialogue:scene_001", "occurrences": 1,
                    "description": "Fix"}]
        state, _ = _simulate_stage12_mutation_flags(changes)
        assert state.script_mutated is False
        assert state.metadata_mutated is False


# ─── Tests: folder_name change is NOT an OFP mutation ────────────────────────

class TestFolderSlugChangeIsNotOFPMutation:
    def test_folder_name_context_does_not_set_any_mutation_flag(self):
        changes = [
            {"context": "folder_name",
             "description": "Sanitized folder slug to remove unsupported superlative",
             "occurrences": 1},
        ]
        state, ran = _simulate_stage12_mutation_flags(changes)
        assert state.script_mutated is False
        assert state.metadata_mutated is False
        assert state.dialogue_mutated is False
        assert state.final_review_inputs_mutated is False
        assert ran is False

    def test_folder_name_only_does_not_set_ran_any_repair(self):
        changes = [{"context": "folder_name", "occurrences": 1, "description": "Slug fix"}]
        _, ran = _simulate_stage12_mutation_flags(changes)
        assert ran is False


# ─── Tests: mixed changes → correct combined flags ───────────────────────────

class TestMixedChangesProduceCorrectFlags:
    def test_chunk_plus_metadata_sets_both(self):
        changes = [
            {"context": "chunk:001", "occurrences": 1, "description": "Safety fix"},
            {"context": "description", "occurrences": 1, "description": "Metadata fix"},
        ]
        state, ran = _simulate_stage12_mutation_flags(changes)
        assert state.script_mutated is True
        assert state.metadata_mutated is True
        assert state.dialogue_mutated is False
        assert ran is True

    def test_all_three_plus_folder_sets_three_flags(self):
        changes = [
            {"context": "chunk:001",                   "occurrences": 1, "description": "Narration"},
            {"context": "description",                  "occurrences": 1, "description": "Meta"},
            {"context": "recreated_dialogue:scene_1",   "occurrences": 1, "description": "Dialogue"},
            {"context": "folder_name",                  "occurrences": 1, "description": "Slug"},
        ]
        state, ran = _simulate_stage12_mutation_flags(changes)
        assert state.script_mutated is True
        assert state.metadata_mutated is True
        assert state.dialogue_mutated is True
        assert state.final_review_inputs_mutated is True
        assert ran is True

    def test_no_changes_no_mutations(self):
        state, ran = _simulate_stage12_mutation_flags([])
        assert state.script_mutated is False
        assert state.metadata_mutated is False
        assert state.dialogue_mutated is False
        assert state.final_review_inputs_mutated is False
        assert ran is False


# ─── Tests: live run_deterministic_auto_fix produces inspectable changes ──────

class TestLiveAutoFixChangesAreInspectable:
    """Verify the live service returns changes with inspectable context fields."""

    def test_metadata_fix_produces_metadata_context(self):
        """A script with a banned tag produces a 'tags' context change."""
        script = {
            "hindi_narration_chunks": [{"chunk_id": "001", "text": "Normal text."}],
            "youtube_metadata": {
                "title": "Test",
                "tags": ["most infamous"],   # actual banned tag
                "description": "",
            },
            "recreated_dialogues": {"items": []},
        }
        _, report = run_deterministic_auto_fix(
            script_draft=script,
            case_hint="",
        )
        contexts = [c["context"] for c in report.get("changes", [])]
        metadata_ctxs = [
            c for c in contexts
            if c in {"tags", "title_options", "recommended_title", "description",
                     "tag", "thumbnail_text", "pinned_comment"}
        ]
        assert len(metadata_ctxs) > 0, (
            "Banned tag fix must produce a change with metadata context"
        )
        # Verify the mutation-flag logic sees it as metadata
        state, ran = _simulate_stage12_mutation_flags(report["changes"])
        assert state.metadata_mutated is True

    def test_dialogue_fix_produces_recreated_dialogue_context(self):
        """A scene without a label produces a 'recreated_dialogue:' context change."""
        script = {
            "hindi_narration_chunks": [{"chunk_id": "001", "text": "पाठ।"}],
            "youtube_metadata": {"title": "T"},
            "recreated_dialogues": {
                "items": [
                    {
                        "scene_id": "scene_01",
                        "label_on_screen": "",    # missing label — triggers fix
                        "dialogue": [{"speaker": "A", "text": "Test"}],
                    }
                ]
            },
        }
        _, report = run_deterministic_auto_fix(
            script_draft=script,
            case_hint="",
        )
        contexts = [c["context"] for c in report.get("changes", [])]
        dialogue_ctxs = [c for c in contexts if c.startswith("recreated_dialogue:")]
        assert len(dialogue_ctxs) > 0, (
            "Missing label fix must produce a change with recreated_dialogue context"
        )
        state, ran = _simulate_stage12_mutation_flags(report["changes"])
        assert state.dialogue_mutated is True

    def test_narration_fix_produces_chunk_context(self):
        """A child-victim organ reference in a chunk produces a 'chunk:' context change."""
        script = {
            "hindi_narration_chunks": [
                {"chunk_id": "001", "text": "उसका फटा हुआ जिगर मिला।"}
            ],
            "youtube_metadata": {"title": "T"},
            "recreated_dialogues": {"items": []},
        }
        _, report = run_deterministic_auto_fix(
            script_draft=script,
            case_hint="child victim",   # triggers organ-rule replacement
        )
        contexts = [c["context"] for c in report.get("changes", [])]
        chunk_ctxs = [c for c in contexts if c.startswith("chunk:")]
        assert len(chunk_ctxs) > 0, (
            "Child-victim organ fix must produce a change with chunk: context"
        )
        state, ran = _simulate_stage12_mutation_flags(report["changes"])
        assert state.script_mutated is True
