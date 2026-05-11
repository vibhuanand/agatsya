import os
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Claude
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"
    claude_max_tokens: int = 12000
    transcript_research_chars: int = 18000

    # OpenAI (premium final review — recommended for production)
    openai_api_key: str = ""
    openai_review_model: str = "gpt-5.5"
    openai_review_enabled: bool = True
    openai_repair_enabled: bool = True
    openai_repair_max_chunks: int = 6

    # Premium retention / revenue optimization
    premium_segmented_fact_lock_threshold: int = 30000

    # ElevenLabs
    elevenlabs_api_key: str = ""
    elevenlabs_narrator_voice_id: str = ""
    elevenlabs_model_id: str = "eleven_multilingual_v2"

    # Asset APIs
    pexels_api_key: str = ""
    pixabay_api_key: str = ""

    # Storage
    storage_base: Path = Path("app/storage")
    episodes_dir: Path = Path("app/storage/episodes")

    # Pipeline control
    max_repair_passes: int = 1
    min_acceptable_duration_ratio: float = 0.8
    # Estimated words-per-minute for serious Hindi true-crime narration (slow, deliberate delivery)
    hindi_narration_wpm: int = 110

    # Fact Lock mode: "research_view" (default, cheap) | "segmented" (thorough)
    fact_lock_mode: str = "research_view"
    fact_lock_segment_chars: int = 7000   # chars per segment in segmented mode

    # Idempotency: skip stages whose output files already exist
    reuse_existing_stage_outputs: bool = False

    # Skip final quality gates (for debugging script generation only — not voice-ready)
    skip_final_gates: bool = False

    # Quality mode
    # premium_build  — Claude + Python only; useful for debugging; not voice-ready
    # premium_final  — Claude + Python + OpenAI; required before ElevenLabs
    # premium_batch  — no OpenAI, no voice/video; for bulk candidate evaluation
    quality_mode: str = "premium_final"

    # OpenAI review policy (applies when quality_mode=premium_final)
    # adaptive  — run one combined OpenAI gate (Hindi Editor only); saves 1 API call
    # always    — run both Hindi Editor + Originality gates (maximum coverage)
    # disabled  — skip OpenAI gates for debugging only; blocks safe_to_voice and is never voice-ready
    openai_review_policy: str = "adaptive"

    # Cost / budget controls
    # Set high for Phase 1 testing — tighten after baseline run data is available
    max_total_model_calls: int = 999   # hard stop across all Claude + OpenAI calls
    max_repair_calls: int = 999        # stop if total repair calls (Claude) exceed this
    max_openai_repair_calls: int = 999 # stop if OpenAI targeted repair calls exceed this

    # Claude prompt caching
    # When true, stable prompt sections (channel rules, output schemas) are marked for caching
    # via the Anthropic prompt-caching beta (cache_control: ephemeral).
    # Reduces cost and latency for repeated runs on the same episode type.
    claude_prompt_cache_enabled: bool = False
    # TTL hint for prompt cache entries (informational — Anthropic manages actual TTL)
    claude_prompt_cache_ttl: str = "5m"

    # Script chunk writer control
    script_chunk_retry_limit: int = 1   # retries per chunk beyond first attempt
    max_script_chunks: int = 16
    min_script_chunks: int = 10

    # Full pipeline gate — set to true only after confirming safe_to_voice=true
    # from /api/episodes/package. POST /api/episodes/full is not production-ready
    # until this flag is explicitly enabled.
    enable_full_pipeline: bool = False

    # Originality Transformation Layer
    # When true (default), runs the Originality Transformation Planner after fact_lock
    # and before script_outline. The plan tells the writer how to create an original
    # Hindi documentary episode rather than translating/paraphrasing the source transcript.
    # Failure is non-fatal (logged as warning) but sets safe_to_voice=False.
    # Set to false only for debugging script generation (not recommended for production).
    originality_transformation_enabled: bool = True

    # Maximum number of high-risk source-copy matches allowed before blocking safe_to_voice.
    # "High-risk" = verbatim English phrase matches of 8+ words from the source transcript.
    # 0 = any high-risk match blocks approval (strict mode, default).
    # Increase only for cases where long proper-noun phrases are unavoidable (e.g. court names).
    source_similarity_max_high_risk_matches: int = 0

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    @field_validator("quality_mode")
    @classmethod
    def validate_quality_mode(cls, value: str) -> str:
        allowed = {"premium_build", "premium_final", "premium_batch"}
        if value not in allowed:
            raise ValueError(f"QUALITY_MODE must be one of {sorted(allowed)}, got {value!r}")
        return value

    @field_validator("openai_review_policy")
    @classmethod
    def validate_openai_review_policy(cls, value: str) -> str:
        allowed = {"adaptive", "always", "disabled"}
        if value not in allowed:
            raise ValueError(
                f"OPENAI_REVIEW_POLICY must be one of {sorted(allowed)}, got {value!r}"
            )
        return value

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
