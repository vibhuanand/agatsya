"""Smoke tests: verify key services import cleanly and settings validators work.

No API calls, no disk I/O, no fixtures needed.
"""
from __future__ import annotations


# ── Import smoke tests ────────────────────────────────────────────────────────

def test_agent_pipeline_imports():
    from app.services import agent_pipeline_service  # noqa: F401
    assert hasattr(agent_pipeline_service, "run_agent_pipeline")


def test_case_glossary_imports():
    from app.services.case_glossary_service import build_case_glossary  # noqa: F401
    assert callable(build_case_glossary)


def test_python_preflight_imports():
    from app.services.python_preflight_service import run_python_preflight  # noqa: F401
    assert callable(run_python_preflight)


def test_stage_manifest_imports():
    from app.services.stage_manifest_service import (  # noqa: F401
        inputs_changed, prompt_changed, save_manifest, load_manifest,
    )


def test_openai_final_premium_gate_imports():
    from app.services.openai_final_premium_gate_service import (  # noqa: F401
        run_openai_final_premium_gate, _THRESHOLDS,
    )
    assert len(_THRESHOLDS) == 7


def test_transcript_cleaner_imports():
    from app.services.transcript_cleaner_service import clean_transcript  # noqa: F401
    assert callable(clean_transcript)


def test_schemas_import():
    from app.schemas import (  # noqa: F401
        FactLock, StoryBlueprint, ScriptDraft,
        ScriptQualityReport, MetadataQualityReport,
        RetentionQualityReport, OpenAIFinalPremiumReport,
        normalize_fact_lock_payload,
    )


def test_models_import():
    from app.models import EpisodeInput, PackageResponse, QualitySummary  # noqa: F401


def test_call_tracker_imports():
    from app.services.call_tracker import (  # noqa: F401
        reset, inc_claude, inc_openai, BudgetExceededError,
    )


# ── Settings / quality mode validators ───────────────────────────────────────

def test_quality_mode_values_accepted():
    from pydantic import ValidationError
    from app.config import Settings

    for mode in ("premium_build", "premium_final", "premium_batch"):
        s = Settings(
            anthropic_api_key="test",
            quality_mode=mode,
            openai_review_policy="adaptive",
        )
        assert s.quality_mode == mode


def test_invalid_quality_mode_rejected():
    from pydantic import ValidationError
    from app.config import Settings
    import pytest

    with pytest.raises((ValidationError, ValueError)):
        Settings(
            anthropic_api_key="test",
            quality_mode="basic",
        )


def test_openai_review_policy_values_accepted():
    from app.config import Settings

    for policy in ("adaptive", "always", "disabled"):
        s = Settings(
            anthropic_api_key="test",
            openai_review_policy=policy,
        )
        assert s.openai_review_policy == policy


def test_invalid_openai_review_policy_rejected():
    from pydantic import ValidationError
    from app.config import Settings
    import pytest

    with pytest.raises((ValidationError, ValueError)):
        Settings(
            anthropic_api_key="test",
            openai_review_policy="never",
        )


def test_default_quality_mode_is_premium_final():
    from app.config import settings
    assert settings.quality_mode == "premium_final"


def test_default_openai_review_policy_is_adaptive():
    from app.config import settings
    assert settings.openai_review_policy == "adaptive"


def test_safe_to_voice_false_when_full_pipeline_disabled():
    """ENABLE_FULL_PIPELINE=False (default) means ElevenLabs never runs."""
    from app.config import settings
    # Default must be False — ElevenLabs only runs after explicit opt-in
    assert settings.enable_full_pipeline is False


def test_skip_final_gates_default_is_false():
    """SKIP_FINAL_GATES must default to False — gates are always active by default."""
    from app.config import settings
    assert settings.skip_final_gates is False
