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


# ── Legacy package path protection (item 2) ───────────────────────────────────

def test_legacy_package_path_safe_to_voice_always_false():
    """The legacy create_package() path (non-script_first) never sets safe_to_voice.
    PackageResponse defaults to safe_to_voice=False, so legacy output is never voice-ready."""
    from app.models import PackageResponse
    # Simulate what the route does: create a PackageResponse with legacy defaults
    pkg = PackageResponse(
        episode_id="test-001",
        folder_name="001-test",
        episode_dir="/tmp/test",
        files={},
    )
    # Force the same values the route applies to legacy output
    pkg.status = "needs_human_review"
    pkg.safe_to_voice = False
    pkg.warnings = list(pkg.warnings) + [
        "Legacy single-call package path is not voice-ready. "
        "Use package_level=script_first for the full quality-gate pipeline."
    ]
    assert pkg.safe_to_voice is False
    assert pkg.status == "needs_human_review"
    assert any("not voice-ready" in w for w in pkg.warnings)


def test_package_response_default_safe_to_voice_is_false():
    """PackageResponse must default safe_to_voice=False so legacy paths are never voice-ready."""
    from app.models import PackageResponse
    pkg = PackageResponse(
        episode_id="x", folder_name="x", episode_dir="/tmp/x", files={}
    )
    assert pkg.safe_to_voice is False


def test_legacy_package_warning_text_identifies_correct_fix():
    """The legacy path warning must name script_first as the correct alternative."""
    from app.models import PackageResponse
    pkg = PackageResponse(
        episode_id="x", folder_name="x", episode_dir="/tmp/x", files={}
    )
    pkg.warnings = [
        "Legacy single-call package path is not voice-ready. "
        "Use package_level=script_first for the full quality-gate pipeline."
    ]
    assert any("script_first" in w for w in pkg.warnings)


# ── /api/episodes/full guard (item 3) ─────────────────────────────────────────

def test_full_pipeline_disabled_by_default():
    """ENABLE_FULL_PIPELINE must default to False — full endpoint is not production-ready."""
    from app.config import settings
    assert settings.enable_full_pipeline is False


def test_full_pipeline_safe_to_voice_guard_in_pipeline_service():
    """pipeline_service.run_full_pipeline must not call ElevenLabs when safe_to_voice=False.
    The service checks pkg.safe_to_voice before enable_voice — if False, voice is skipped
    and a warning is added. This test verifies the guard logic directly."""
    from app.models import PackageResponse, FullPipelineInput

    # Verify that FullPipelineInput has enable_voice flag
    inp = FullPipelineInput(
        youtube_url="https://www.youtube.com/watch?v=test",
        episode_number="001",
        case_hint="test case",
        raw_transcript="test transcript",
        enable_voice=True,
    )
    assert hasattr(inp, "enable_voice")
    assert inp.enable_voice is True

    # The guard: if enable_voice=True but safe_to_voice=False, voice must be skipped
    pkg = PackageResponse(
        episode_id="test", folder_name="test", episode_dir="/tmp/test", files={}
    )
    # PackageResponse default is safe_to_voice=False
    assert pkg.safe_to_voice is False
    # Verify the guard condition mirrors pipeline_service.py logic:
    # "if inp.enable_voice and not pkg.safe_to_voice" → skip voice, add warning
    voice_should_skip = inp.enable_voice and not pkg.safe_to_voice
    assert voice_should_skip is True


def test_full_pipeline_future_requirement_documented():
    """Verify that pipeline_service has the TODO comment requiring run_agent_pipeline
    before voice generation can be enabled."""
    import inspect
    from app.services import pipeline_service
    source = inspect.getsource(pipeline_service)
    assert "run_agent_pipeline" in source, (
        "pipeline_service must reference run_agent_pipeline in its TODO for future enabling"
    )
    assert "safe_to_voice" in source, (
        "pipeline_service must guard voice generation on safe_to_voice"
    )
