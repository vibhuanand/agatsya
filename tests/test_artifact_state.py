"""
Tests for ArtifactState dataclass — session 4, TASK 1.

Verifies:
- Initial state is all-False
- mark_script_mutated sets script_mutated and final_review_inputs_mutated
- mark_metadata_mutated sets metadata_mutated and final_review_inputs_mutated
- mark_dialogue_mutated sets dialogue_mutated and final_review_inputs_mutated
- mutation_sources accumulates deduped sources
- to_dict returns a serializable snapshot
- Multiple mark calls on different fields all set final_review_inputs_mutated
"""
from __future__ import annotations

import pytest

from app.services.agent_pipeline_service import ArtifactState


class TestArtifactStateInitialState:
    def test_all_false_initially(self):
        state = ArtifactState()
        assert state.script_mutated is False
        assert state.metadata_mutated is False
        assert state.dialogue_mutated is False
        assert state.final_review_inputs_mutated is False

    def test_mutation_sources_empty_initially(self):
        state = ArtifactState()
        assert state.mutation_sources == []


class TestMarkScriptMutated:
    def test_sets_script_mutated(self):
        state = ArtifactState()
        state.mark_script_mutated("stage6_script_repair")
        assert state.script_mutated is True

    def test_sets_final_review_inputs_mutated(self):
        state = ArtifactState()
        state.mark_script_mutated("stage6_script_repair")
        assert state.final_review_inputs_mutated is True

    def test_appends_source(self):
        state = ArtifactState()
        state.mark_script_mutated("stage6_script_repair")
        assert "stage6_script_repair" in state.mutation_sources

    def test_does_not_set_metadata_or_dialogue(self):
        state = ArtifactState()
        state.mark_script_mutated("some_source")
        assert state.metadata_mutated is False
        assert state.dialogue_mutated is False

    def test_deduplicates_source(self):
        state = ArtifactState()
        state.mark_script_mutated("stage6")
        state.mark_script_mutated("stage6")
        assert state.mutation_sources.count("stage6") == 1


class TestMarkMetadataMutated:
    def test_sets_metadata_mutated(self):
        state = ArtifactState()
        state.mark_metadata_mutated("stage13a_metadata_repair")
        assert state.metadata_mutated is True

    def test_sets_final_review_inputs_mutated(self):
        state = ArtifactState()
        state.mark_metadata_mutated("stage13a_metadata_repair")
        assert state.final_review_inputs_mutated is True

    def test_does_not_set_script_or_dialogue(self):
        state = ArtifactState()
        state.mark_metadata_mutated("some_source")
        assert state.script_mutated is False
        assert state.dialogue_mutated is False


class TestMarkDialogueMutated:
    def test_sets_dialogue_mutated(self):
        state = ArtifactState()
        state.mark_dialogue_mutated("stage12_deterministic_auto_fix")
        assert state.dialogue_mutated is True

    def test_sets_final_review_inputs_mutated(self):
        state = ArtifactState()
        state.mark_dialogue_mutated("stage12_deterministic_auto_fix")
        assert state.final_review_inputs_mutated is True

    def test_does_not_set_script_or_metadata(self):
        state = ArtifactState()
        state.mark_dialogue_mutated("some_source")
        assert state.script_mutated is False
        assert state.metadata_mutated is False


class TestMutationSourcesAccumulation:
    def test_multiple_marks_accumulate_sources(self):
        state = ArtifactState()
        state.mark_script_mutated("stage6")
        state.mark_metadata_mutated("stage13a")
        state.mark_dialogue_mutated("stage12")
        assert set(state.mutation_sources) == {"stage6", "stage13a", "stage12"}

    def test_deduplication_across_different_marks(self):
        state = ArtifactState()
        state.mark_script_mutated("repair")
        state.mark_dialogue_mutated("repair")  # same source
        assert state.mutation_sources.count("repair") == 1

    def test_all_flags_set_after_all_marks(self):
        state = ArtifactState()
        state.mark_script_mutated("s")
        state.mark_metadata_mutated("m")
        state.mark_dialogue_mutated("d")
        assert state.script_mutated is True
        assert state.metadata_mutated is True
        assert state.dialogue_mutated is True
        assert state.final_review_inputs_mutated is True


class TestToDict:
    def test_returns_dict(self):
        state = ArtifactState()
        d = state.to_dict()
        assert isinstance(d, dict)

    def test_initial_to_dict_all_false(self):
        state = ArtifactState()
        d = state.to_dict()
        assert d["script_mutated"] is False
        assert d["metadata_mutated"] is False
        assert d["dialogue_mutated"] is False
        assert d["final_review_inputs_mutated"] is False
        assert d["mutation_sources"] == []

    def test_to_dict_reflects_mutations(self):
        state = ArtifactState()
        state.mark_script_mutated("x")
        d = state.to_dict()
        assert d["script_mutated"] is True
        assert d["final_review_inputs_mutated"] is True
        assert "x" in d["mutation_sources"]

    def test_to_dict_is_serializable(self):
        import json
        state = ArtifactState()
        state.mark_script_mutated("stage6")
        state.mark_metadata_mutated("stage13a")
        s = json.dumps(state.to_dict())
        assert "stage6" in s
        assert "stage13a" in s
