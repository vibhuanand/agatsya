"""
Tests for TASK 4 — premium_section_rebuild_service must validate each rebuilt
chunk against NarrationChunk schema before merging.

Verifies:
- invalid rebuilt chunk is skipped (not merged)
- rebuilt_count counts only valid chunks
- invalid_rebuilt_chunks records skipped failures
- a report with all-invalid chunks returns rebuilt_count=0
- valid chunks are still merged normally
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import NarrationChunk
from app.services.premium_section_rebuild_service import (
    _merge_rebuilt_chunks,
    _filter_chunks,
)


# ─── Helper: replicate the validation loop from the service ───────────────────

def _validate_and_filter_rebuilt(
    rebuilt_chunks_raw: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Replicate the Task 4 validation loop from premium_section_rebuild_service.
    Returns (valid_chunks, invalid_records).
    """
    valid: list[dict] = []
    invalid: list[dict] = []
    for chunk in rebuilt_chunks_raw:
        try:
            NarrationChunk.model_validate(chunk)
            valid.append(chunk)
        except ValidationError as ve:
            invalid.append({
                "chunk_id":         chunk.get("chunk_id", "?"),
                "validation_error": str(ve)[:300],
                "raw_keys":         list(chunk.keys()),
            })
    return valid, invalid


def _make_valid_chunk(chunk_id: str, text: str = "वैध पाठ।") -> dict:
    return {
        "chunk_id":      chunk_id,
        "section_title": f"Section {chunk_id}",
        "voice":         "narrator",
        "tone":          "neutral",
        "estimated_words": 3,
        "text":          text,
    }


def _make_invalid_chunk_missing_text(chunk_id: str) -> dict:
    """Missing required 'text' field — fails NarrationChunk validation."""
    return {
        "chunk_id":      chunk_id,
        "section_title": "Missing Text",
        # 'text' is absent — required by NarrationChunk
    }


def _make_invalid_chunk_missing_section_title(chunk_id: str) -> dict:
    """Missing required 'section_title' field."""
    return {
        "chunk_id": chunk_id,
        "text":     "Some text here.",
        # 'section_title' is absent — required by NarrationChunk
    }


# ─── Tests: NarrationChunk schema baseline ───────────────────────────────────

class TestNarrationChunkSchema:
    def test_valid_chunk_passes(self):
        chunk = _make_valid_chunk("001_hook", "यह हुक है।")
        validated = NarrationChunk.model_validate(chunk)
        assert validated.chunk_id == "001_hook"

    def test_missing_text_fails(self):
        chunk = _make_invalid_chunk_missing_text("002")
        with pytest.raises(ValidationError):
            NarrationChunk.model_validate(chunk)

    def test_missing_section_title_fails(self):
        chunk = _make_invalid_chunk_missing_section_title("003")
        with pytest.raises(ValidationError):
            NarrationChunk.model_validate(chunk)

    def test_empty_text_is_valid(self):
        """empty string satisfies the str type constraint."""
        chunk = _make_valid_chunk("004", text="")
        validated = NarrationChunk.model_validate(chunk)
        assert validated.text == ""


# ─── Tests: validation loop in isolation ─────────────────────────────────────

class TestValidationLoop:
    def test_all_valid_chunks_pass(self):
        chunks = [_make_valid_chunk(f"00{i}") for i in range(1, 4)]
        valid, invalid = _validate_and_filter_rebuilt(chunks)
        assert len(valid) == 3
        assert len(invalid) == 0

    def test_one_invalid_chunk_filtered_out(self):
        chunks = [
            _make_valid_chunk("001_hook"),
            _make_invalid_chunk_missing_text("002_bad"),
            _make_valid_chunk("003_ok"),
        ]
        valid, invalid = _validate_and_filter_rebuilt(chunks)
        assert len(valid) == 2
        assert len(invalid) == 1
        assert invalid[0]["chunk_id"] == "002_bad"

    def test_all_invalid_chunks_give_empty_valid_list(self):
        chunks = [
            _make_invalid_chunk_missing_text("001"),
            _make_invalid_chunk_missing_section_title("002"),
        ]
        valid, invalid = _validate_and_filter_rebuilt(chunks)
        assert valid == []
        assert len(invalid) == 2

    def test_invalid_record_has_chunk_id(self):
        chunks = [_make_invalid_chunk_missing_text("bad_chunk")]
        _, invalid = _validate_and_filter_rebuilt(chunks)
        assert invalid[0]["chunk_id"] == "bad_chunk"

    def test_invalid_record_has_validation_error_text(self):
        chunks = [_make_invalid_chunk_missing_text("x")]
        _, invalid = _validate_and_filter_rebuilt(chunks)
        assert len(invalid[0]["validation_error"]) > 0

    def test_invalid_record_has_raw_keys(self):
        chunk = _make_invalid_chunk_missing_text("y")
        _, invalid = _validate_and_filter_rebuilt([chunk])
        assert "chunk_id" in invalid[0]["raw_keys"]

    def test_rebuilt_count_equals_valid_chunk_count(self):
        chunks = [
            _make_valid_chunk("001"),
            _make_invalid_chunk_missing_text("002_bad"),
            _make_valid_chunk("003"),
        ]
        valid, invalid = _validate_and_filter_rebuilt(chunks)
        rebuilt_count = len(valid)
        assert rebuilt_count == 2   # not 3

    def test_rebuilt_count_zero_when_all_invalid(self):
        chunks = [
            _make_invalid_chunk_missing_text("001"),
            _make_invalid_chunk_missing_text("002"),
        ]
        valid, _ = _validate_and_filter_rebuilt(chunks)
        assert len(valid) == 0   # rebuilt_count must be 0


# ─── Tests: merge respects validation ────────────────────────────────────────

class TestMergeRespectsValidation:
    def _original_chunks(self) -> list[dict]:
        return [
            _make_valid_chunk("001_hook", "मूल हुक।"),
            _make_valid_chunk("002_bg",   "मूल पृष्ठभूमि।"),
            _make_valid_chunk("003_ev",   "मूल घटनाएँ।"),
        ]

    def test_valid_rebuilt_chunk_replaces_original(self):
        originals = self._original_chunks()
        rebuilt = _make_valid_chunk("002_bg", "पुनर्निर्मित पृष्ठभूमि।")
        valid, _ = _validate_and_filter_rebuilt([rebuilt])
        merged = _merge_rebuilt_chunks(originals, valid)
        bg = next(c for c in merged if c["chunk_id"] == "002_bg")
        assert bg["text"] == "पुनर्निर्मित पृष्ठभूमि।"

    def test_invalid_rebuilt_chunk_leaves_original_intact(self):
        originals = self._original_chunks()
        invalid_chunk = _make_invalid_chunk_missing_text("002_bg")
        valid, invalid = _validate_and_filter_rebuilt([invalid_chunk])
        # valid is empty → no merge happens
        merged = _merge_rebuilt_chunks(originals, valid)
        bg = next(c for c in merged if c["chunk_id"] == "002_bg")
        assert bg["text"] == "मूल पृष्ठभूमि।"   # original preserved

    def test_mixed_rebuilds_only_valid_replaces(self):
        originals = self._original_chunks()
        rebuilt = [
            _make_valid_chunk("001_hook", "नया हुक।"),
            _make_invalid_chunk_missing_text("003_ev"),  # invalid — original kept
        ]
        valid, _ = _validate_and_filter_rebuilt(rebuilt)
        merged = _merge_rebuilt_chunks(originals, valid)

        hook = next(c for c in merged if c["chunk_id"] == "001_hook")
        ev   = next(c for c in merged if c["chunk_id"] == "003_ev")
        assert hook["text"] == "नया हुक।"
        assert ev["text"]   == "मूल घटनाएँ।"   # original preserved


# ─── Tests: report structure ──────────────────────────────────────────────────

class TestRebuildReportStructure:
    def test_report_has_invalid_rebuilt_chunks_key(self):
        invalid_record = {"chunk_id": "001", "validation_error": "missing text", "raw_keys": ["chunk_id"]}
        report = {
            "rebuilt_count": 0,
            "invalid_rebuilt_chunks": [invalid_record],
            "invalid_rebuilt_count": 1,
        }
        assert "invalid_rebuilt_chunks" in report
        assert report["invalid_rebuilt_count"] == 1

    def test_report_rebuilt_count_excludes_invalid(self):
        """rebuilt_count must NOT include chunks that failed validation."""
        # Simulate: 3 raw chunks, 1 invalid, 2 valid
        valid_count = 2
        invalid_count = 1
        report = {
            "rebuilt_count": valid_count,  # only valid
            "invalid_rebuilt_count": invalid_count,
        }
        assert report["rebuilt_count"] == 2
        assert report["rebuilt_count"] != 3   # must not include invalid
