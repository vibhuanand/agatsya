"""
Pydantic schemas for Claude agent outputs.

These are validated after each agent call.
A ValidationError stops the pipeline with status=failed.

Keep schemas practical — use Optional / defaults liberally so minor
Claude schema variations don't cause unnecessary pipeline failures.
Only hard-fail on genuinely missing required fields.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Fact Lock sub-models ─────────────────────────────────────────────────────

class VerifiedPerson(BaseModel):
    name: str
    role: str
    confidence: str = "medium"
    source_phrase: str = ""


class VerifiedDate(BaseModel):
    date_or_period: str
    event: str
    confidence: str = "medium"
    source_phrase: str = ""


class VerifiedLocation(BaseModel):
    location: str
    context: str = ""
    confidence: str = "medium"


class VerifiedTimelineEvent(BaseModel):
    order: int
    date_or_period: str
    event: str
    confidence: str = "medium"
    source_phrase: str = ""


class LegalOutcome(BaseModel):
    trial_result: str = ""
    appeal_result: str = ""
    supreme_court_or_final_result: str = ""
    sentence_or_parole: str = ""
    confidence: str = "medium"
    source_phrase: str = ""


class RecreatedSceneCandidate(BaseModel):
    scene_type: str
    why_useful: str = ""
    safety_note: str = ""


# ─── Structured list-item models (richer than plain strings) ─────────────────

class KeyEvidenceItem(BaseModel):
    """
    A piece of evidence or case-turning-point extracted from the transcript.
    Use .evidence for narration content, .source_phrase for grounding,
    .confidence to prioritise, .why_it_matters for story context.
    """
    evidence: str
    source_phrase: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "medium"
    why_it_matters: Optional[str] = None


class AudioMomentItem(BaseModel):
    """
    A notable audio / call moment (911 call, police interview, wiretap, confession).
    .call_type names the recording type; .description is the story-relevant content;
    .safety_note signals recreated-scene restrictions.
    """
    call_type: str
    description: str
    source_phrase: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "medium"
    safety_note: Optional[str] = None


class EmotionalDetailItem(BaseModel):
    """
    A human moment that makes the victim real — personality, family, memorial.
    Use .detail for narration, .story_use to guide where it fits the blueprint.
    """
    detail: str
    source_phrase: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "medium"
    story_use: Optional[str] = None


class FactToVerifyItem(BaseModel):
    """
    A claim that needs external verification before it can be stated as fact.
    .reason explains why it needs checking; .confidence reflects transcript certainty.
    """
    fact: str
    reason: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "low"


# ─── Fact Lock schema ─────────────────────────────────────────────────────────

class FactLock(BaseModel):
    case_name: str
    source_summary: str
    verified_people: List[VerifiedPerson]
    verified_dates: List[VerifiedDate] = []
    verified_locations: List[VerifiedLocation] = []
    verified_timeline: List[VerifiedTimelineEvent] = []
    legal_outcome: LegalOutcome
    key_evidence_or_turning_points: List[KeyEvidenceItem] = []
    important_audio_or_call_moments: List[AudioMomentItem] = []
    emotional_details: List[EmotionalDetailItem] = []
    recreated_scene_candidates: List[RecreatedSceneCandidate] = []
    facts_to_verify_externally: List[FactToVerifyItem] = []
    must_not_say: List[str] = []

    @field_validator("verified_people")
    @classmethod
    def at_least_one_person(cls, v: list) -> list:
        if not v:
            raise ValueError("verified_people must contain at least one entry")
        return v


# ─── Normalization: tolerant pre-validation pass ──────────────────────────────

def normalize_fact_lock_payload(payload: dict) -> dict:
    """
    Normalize Claude's fact_lock output to match FactLock schema before validation.

    Claude sometimes returns plain strings for the four structured-list fields.
    This converts them to the expected object shape so Pydantic validation succeeds
    on well-formed content, while still failing hard on genuinely bad output.

    Conversions applied:

      key_evidence_or_turning_points:
        "string"  →  {"evidence": "string", "confidence": "medium"}
        dict without "evidence"  →  tries common alt keys, falls back to str()

      important_audio_or_call_moments:
        "string"  →  {"call_type": "unknown", "description": "string", "confidence": "medium"}

      emotional_details:
        "string"  →  {"detail": "string", "confidence": "medium"}

      facts_to_verify_externally:
        "string"  →  {"fact": "string", "confidence": "low"}

    Structured dicts are passed through untouched (Pydantic will validate them).
    """

    def _to_key_evidence(item: Any) -> dict:
        if isinstance(item, str):
            return {"evidence": item, "confidence": "medium"}
        if isinstance(item, dict):
            if "evidence" not in item:
                for alt in ("text", "description", "point", "finding", "detail"):
                    if alt in item:
                        item = dict(item)
                        item["evidence"] = item.pop(alt)
                        return item
                # No recognisable key — stringify the whole object
                return {"evidence": str(item), "confidence": "medium"}
            return item
        return {"evidence": str(item), "confidence": "medium"}

    def _to_audio_moment(item: Any) -> dict:
        if isinstance(item, str):
            return {"call_type": "unknown", "description": item, "confidence": "medium"}
        if isinstance(item, dict):
            d = dict(item)
            if "description" not in d:
                for alt in ("text", "moment", "content", "detail", "summary"):
                    if alt in d:
                        d["description"] = d.pop(alt)
                        break
                else:
                    d["description"] = str(item)
            d.setdefault("call_type", "unknown")
            return d
        return {"call_type": "unknown", "description": str(item), "confidence": "medium"}

    def _to_emotional_detail(item: Any) -> dict:
        if isinstance(item, str):
            return {"detail": item, "confidence": "medium"}
        if isinstance(item, dict):
            d = dict(item)
            if "detail" not in d:
                for alt in ("text", "description", "emotion", "moment", "memory"):
                    if alt in d:
                        d["detail"] = d.pop(alt)
                        break
                else:
                    d["detail"] = str(item)
            return d
        return {"detail": str(item), "confidence": "medium"}

    def _to_fact_to_verify(item: Any) -> dict:
        if isinstance(item, str):
            return {"fact": item, "confidence": "low"}
        if isinstance(item, dict):
            d = dict(item)
            if "fact" not in d:
                for alt in ("text", "description", "claim", "statement", "detail"):
                    if alt in d:
                        d["fact"] = d.pop(alt)
                        break
                else:
                    d["fact"] = str(item)
            return d
        return {"fact": str(item), "confidence": "low"}

    out = dict(payload)
    out["key_evidence_or_turning_points"] = [
        _to_key_evidence(x) for x in payload.get("key_evidence_or_turning_points", [])
    ]
    out["important_audio_or_call_moments"] = [
        _to_audio_moment(x) for x in payload.get("important_audio_or_call_moments", [])
    ]
    out["emotional_details"] = [
        _to_emotional_detail(x) for x in payload.get("emotional_details", [])
    ]
    out["facts_to_verify_externally"] = [
        _to_fact_to_verify(x) for x in payload.get("facts_to_verify_externally", [])
    ]
    return out


# ─── Script Outline schemas ───────────────────────────────────────────────────

class ScriptOutlineChunk(BaseModel):
    chunk_id: str
    section_title: str
    purpose: str = ""
    tone: str = ""
    target_words: int = 150
    must_include_points: List[str] = []
    source_fact_refs: List[str] = []
    safety_notes: List[str] = []


class RecreatedScenePlanItem(BaseModel):
    scene_id: str
    scene_type: str
    purpose: str = ""
    source_fact_refs: List[str] = []
    safety_boundary: str = ""


class ScriptOutline(BaseModel):
    episode_id: str
    target_duration_min: int
    target_word_count_min: int
    target_word_count_ideal: int
    target_word_count_max: int
    chunks: List[ScriptOutlineChunk]
    recreated_scene_plan: List[RecreatedScenePlanItem] = []

    @field_validator("chunks")
    @classmethod
    def validate_chunk_count(cls, v: list) -> list:
        if len(v) < 1:
            raise ValueError("Script outline must have at least 1 chunk")
        return v


class NarrationChunk(BaseModel):
    chunk_id: str
    section_title: str
    voice: str = "narrator"
    tone: str = ""
    estimated_words: int = 0
    text: str


class RecreatedDialogues(BaseModel):
    items: List[Any] = []


class YoutubeMetadataMinimal(BaseModel):
    title_options: List[str] = []
    recommended_title: str
    description: str = ""
    tags: List[str] = []
    thumbnail_options: List[Any] = []
    chapters: List[Any] = []
    shorts_plan: List[Any] = []
    pinned_comment: str = ""


# ─── Story Blueprint schema ───────────────────────────────────────────────────

class NarrativeSection(BaseModel):
    section_order: int
    section_name: str
    purpose: str = ""
    must_include_facts: List[str] = []


class RecreatedScenePlan(BaseModel):
    scene_type: str
    reason: str = ""
    safety_boundary: str = ""


class StoryBlueprint(BaseModel):
    primary_story_type: str
    secondary_patterns: List[str] = []
    main_hook: str
    emotional_anchor: str
    central_question: str
    narrative_sections: List[NarrativeSection]
    must_include_story_points: List[str] = []
    recreated_scenes_to_use: List[RecreatedScenePlan] = []
    sensitivity_rules: List[str] = []
    title_angle: str = ""
    closing_style: str = ""
    do_not_overuse: List[str] = []

    @field_validator("narrative_sections")
    @classmethod
    def at_least_one_section(cls, v: list) -> list:
        if not v:
            raise ValueError("narrative_sections must contain at least one entry")
        return v


# ─── Script Draft schema (lightweight — full text validation via critic) ──────

class ScriptDraftCaseSummary(BaseModel):
    case_title: str
    location: str = ""
    year: str = ""
    people: List[Any] = []
    timeline: List[Any] = []
    core_story: str = ""
    legal_outcome: str = ""
    sensitive_topics: List[str] = []
    facts_to_verify: List[str] = []
    avoid_in_final_video: List[str] = []


class ScriptDraft(BaseModel):
    episode_id: str
    folder_name: str
    case_summary: ScriptDraftCaseSummary
    hindi_narration_chunks: List[Any]
    recreated_dialogues: Any = {}
    youtube_metadata: Any = {}
    quality_checklist: List[str] = []

    @field_validator("hindi_narration_chunks")
    @classmethod
    def at_least_ten_chunks(cls, v: list) -> list:
        if len(v) < 5:
            raise ValueError(
                f"hindi_narration_chunks has only {len(v)} entries — expected at least 5. "
                "Script is likely truncated."
            )
        return v


# ─── Targeted chunk repair schemas ───────────────────────────────────────────

# Map non-standard issue_type values produced by linters/deterministic services
# to the nearest accepted schema type.  Normalization runs before Pydantic
# validates the Literal, so callers get a clean type without losing intent.
_ISSUE_TYPE_ALIASES: dict[str, str] = {
    # exact_english_quote_copy comes from deterministic_auto_fix_service.
    # It describes a chunk where a long English quote was copied verbatim
    # instead of being translated — nearest semantic type is hindi_naturalness.
    "exact_english_quote_copy": "hindi_naturalness",
}


class ChunkRepairTarget(BaseModel):
    chunk_id: str
    issue_type: Literal[
        "hindi_naturalness",
        "hinglish_level_mismatch",
        "missing_fact",
        "pacing",
        "safety",
        "structure",
        "duration",
        "case_glossary",
    ]
    problem: str
    repair_instruction: str

    @field_validator("issue_type", mode="before")
    @classmethod
    def _normalize_issue_type(cls, v: object) -> object:
        """Normalize known alias types to an accepted Literal value.

        Only maps explicitly listed aliases.  Any other unknown string is
        passed through unchanged so that the Literal check still rejects it
        with a clear validation error.
        """
        if isinstance(v, str):
            return _ISSUE_TYPE_ALIASES.get(v, v)
        return v


class HinglishLevelAssessment(BaseModel):
    requested_level: int = Field(..., ge=1, le=5)
    detected_level: float = Field(..., ge=1.0, le=5.0)   # float: script can sit between levels
    matches_requested_level: bool
    notes: str = ""


# ─── Script Quality Report schema ─────────────────────────────────────────────

class QualityScores(BaseModel):
    factual_accuracy: int = 0
    story_structure: int = 0
    hindi_naturalness: int = 0
    emotional_depth: int = 0
    retention_hook: int = 0
    safety: int = 0
    monetization_safety: int = 0
    recreated_scene_quality: int = 10   # default 10 if no recreated scenes

    @field_validator(
        "factual_accuracy", "story_structure", "hindi_naturalness",
        "emotional_depth", "retention_hook", "safety",
        "monetization_safety", "recreated_scene_quality",
        mode="before",
    )
    @classmethod
    def clamp_score(cls, v: Any) -> int:
        try:
            return max(0, min(10, int(v)))
        except (TypeError, ValueError):
            return 0


# ─── Gate report schemas ──────────────────────────────────────────────────────

class MetadataQualityReport(BaseModel):
    gate_passed: bool
    scores: dict = {}
    required_fixes: List[str] = []
    # tolerate any extra fields from Claude

    class Config:
        extra = "allow"


class RetentionQualityReport(BaseModel):
    approved: bool
    overall_retention_score: int = 0
    opening_hook_score: int = 0
    curiosity_gap_score: int = 0
    pacing_score: int = 0
    emotional_arc_score: int = 0
    ending_payoff_score: int = 0
    issues: List[Any] = []
    chunk_repair_targets: List[Any] = []

    class Config:
        extra = "allow"


class OpenAIFinalPremiumReport(BaseModel):
    approved: bool
    safe_to_voice: bool
    # Scores are float so OpenAI can return decimal values (e.g. 8.5, 9.5)
    # without coercion loss. All must be >= 9.0 for approval.
    overall_score:            float = Field(0.0, ge=0.0, le=10.0)
    hindi_quality_score:      float = Field(0.0, ge=0.0, le=10.0)
    retention_score:          float = Field(0.0, ge=0.0, le=10.0)
    originality_score:        float = Field(0.0, ge=0.0, le=10.0)
    youtube_safety_score:     float = Field(0.0, ge=0.0, le=10.0)
    metadata_score:           float = Field(0.0, ge=0.0, le=10.0)
    recreated_dialogue_score: float = Field(10.0, ge=0.0, le=10.0)
    issues: List[Any] = []
    chunk_repair_targets: List[Any] = []
    metadata_repair_required: bool = False
    recreated_dialogue_repair_required: bool = False
    recommendation: str = "approved"

    class Config:
        extra = "allow"


class ScriptQualityReport(BaseModel):
    approved: bool
    scores: QualityScores
    estimated_word_count: int = 0
    estimated_duration_min: float = 0.0
    fact_issues: List[str] = []
    missing_required_points: List[str] = []
    story_structure_issues: List[str] = []
    language_issues: List[str] = []
    safety_issues: List[str] = []
    recreated_scene_issues: List[str] = []
    youtube_metadata_issues: List[str] = []
    monetization_risks: List[str] = []
    cost_mode_issues: List[str] = []
    repair_required: bool = False
    repair_instructions: List[str] = []
    chunk_repair_targets: List[ChunkRepairTarget] = []
    hinglish_level_assessment: Optional[HinglishLevelAssessment] = None
