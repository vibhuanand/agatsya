from __future__ import annotations
from typing import Any, List, Optional
from pydantic import BaseModel, HttpUrl, field_validator

COST_MODES = ("bootstrap", "standard", "premium")
PACKAGE_LEVELS = ("script_first", "full_package", "video_plan_only")


# ─── Input models ────────────────────────────────────────────────────────────

class EpisodeInput(BaseModel):
    youtube_url: str
    episode_number: str
    case_hint: str
    target_duration_min: int = 22
    raw_transcript: str
    style: str = "Agatsya Anand pure Hindi respectful dark true crime"
    cost_mode: str = "premium"
    package_level: str = "script_first"
    enable_gpt_review: bool = False
    hinglish_level: int = 2

    @field_validator("hinglish_level")
    @classmethod
    def validate_hinglish_level(cls, v: int) -> int:
        if v not in (1, 2, 3, 4, 5):
            raise ValueError(f"hinglish_level must be 1–5, got {v}")
        return v

    @field_validator("episode_number")
    @classmethod
    def pad_episode(cls, v: str) -> str:
        return v.zfill(3)

    @field_validator("cost_mode")
    @classmethod
    def validate_cost_mode(cls, v: str) -> str:
        if v not in COST_MODES:
            raise ValueError(
                f"cost_mode must be one of {COST_MODES}, got '{v}'"
            )
        return v

    @field_validator("package_level")
    @classmethod
    def validate_package_level(cls, v: str) -> str:
        if v not in PACKAGE_LEVELS:
            raise ValueError(
                f"package_level must be one of {PACKAGE_LEVELS}, got '{v}'"
            )
        return v


class FullPipelineInput(EpisodeInput):
    enable_voice: bool = False
    enable_assets: bool = False
    enable_render: bool = False


# ─── Claude package sub-models ────────────────────────────────────────────────

class Person(BaseModel):
    name: str
    role: str

class TimelineEvent(BaseModel):
    date: str
    event: str

class CaseSummary(BaseModel):
    case_title: str
    location: str
    year: str
    people: List[Person]
    timeline: List[TimelineEvent]
    core_story: str
    legal_outcome: str
    sensitive_topics: List[str]
    facts_to_verify: List[str]
    avoid_in_final_video: List[str]


class NarrationChunk(BaseModel):
    chunk_id: str
    section_title: str
    voice: str = "narrator"
    tone: str
    text: str


class DialogueLine(BaseModel):
    speaker: str
    text: str
    emotion: Optional[str] = None

class RecreatedScene(BaseModel):
    scene_id: str
    scene_title: str
    label_on_screen: str = "फिर से रचा गया संवाद"
    not_original_audio: bool = True
    safety_note: str
    voices: List[str]
    dialogue: List[DialogueLine]
    sfx: List[str]
    estimated_duration_sec: int

class RecreatedDialogues(BaseModel):
    items: List[RecreatedScene]


class VideoScene(BaseModel):
    scene_id: str
    audio_chunk_id: str
    duration_sec: int
    visual_type: str
    background_type: str
    template_name: str
    ai_prompt: str
    real_asset_keywords: List[str]
    on_screen_text: str
    motion: str
    subtitle: bool = True
    sfx: List[str]
    music_mood: str
    safety_notes: str

class VideoScenePlan(BaseModel):
    format: str = "longform_16x9"
    visual_theme: str = "dark_respectful_true_crime"
    scenes: List[VideoScene]


class YouTubeChapter(BaseModel):
    timestamp: str
    title: str

class YouTubeMetadata(BaseModel):
    title_options: List[str]
    recommended_title: str
    description: str
    tags: List[str]
    chapters: List[YouTubeChapter]
    thumbnail_options: List[str]
    pinned_comment: str


class ShortsItem(BaseModel):
    short_id: str
    title: str
    source_chunk_id: str
    hook_text: str
    duration_sec: int
    visual_note: str

class ShortsPlan(BaseModel):
    items: List[ShortsItem]


# ─── Top-level production package ─────────────────────────────────────────────

class ProductionPackage(BaseModel):
    episode_id: str
    folder_name: str
    case_summary: CaseSummary
    hindi_narration_chunks: List[NarrationChunk]
    recreated_dialogues: RecreatedDialogues
    video_scene_plan: VideoScenePlan
    youtube_metadata: YouTubeMetadata
    shorts_plan: ShortsPlan
    quality_checklist: List[str]


# ─── API responses ─────────────────────────────────────────────────────────────

class QualitySummary(BaseModel):
    approved: bool = True
    scores: Optional[dict] = None
    estimated_word_count: int = 0
    estimated_duration_min: float = 0.0
    repair_required: bool = False


class PackageResponse(BaseModel):
    episode_id: str
    folder_name: str
    episode_dir: str
    status: str = "script_approved"   # script_approved | auto_repair_required | auto_rebuild_required | not_voice_ready_auto_retry_exhausted | needs_human_review | failed
    files: dict[str, str]
    quality_summary: Optional[QualitySummary] = None
    gate_summary: Optional[dict] = None   # premium gate results per gate
    safe_to_voice: bool = False           # True only when ALL premium gates pass
    warnings: List[str] = []
    telemetry: Optional[dict] = None      # model call counts, timing, stage reuse

class FullPipelineResponse(PackageResponse):
    voice_files: List[str] = []
    asset_files: List[str] = []
    render_files: List[str] = []
    pipeline_warnings: List[str] = []


# ─── Video plan (second-stage) models ─────────────────────────────────────────

class VideoPlanRequest(BaseModel):
    episode_id: str   # e.g. "001-meika-jordan"
    cost_mode: str = "premium"

    @field_validator("cost_mode")
    @classmethod
    def validate_cost_mode(cls, v: str) -> str:
        if v not in COST_MODES:
            raise ValueError(
                f"cost_mode must be one of {COST_MODES}, got '{v}'"
            )
        return v


class VideoPlanResponse(BaseModel):
    episode_id: str
    episode_dir: str
    files: dict[str, str]
    warnings: List[str] = []
