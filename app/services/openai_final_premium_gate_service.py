"""
OpenAI Final Premium Gate — combined single-call quality review.

Used when OPENAI_REVIEW_POLICY=adaptive (default).
Replaces running Hindi Editor + Originality gates separately.

Checks (one call, one verdict):
  Hindi grammar/matra/nasalization, Hinglish level, retention quality,
  originality/reuse risk, YouTube monetization safety, metadata completeness,
  recreated dialogue safety, safe_to_voice.

All score thresholds are >= 9 (uniform, strict).

Evidence received from every upstream Claude gate:
  - script_draft            (full final script)
  - fact_lock_summary       (case_name, people, legal_outcome)
  - claude_gate_evidence    (condensed summaries of all Claude gates already run)
  - youtube_metadata        (extracted from script_draft)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.services.openai_client import call_openai_json

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/openai_final_premium_gate.txt")

# All thresholds are uniform at 9.0 — any score below 9.0 blocks approval.
# Stored as float to tolerate decimal scores (e.g. 8.5, 9.5) without coercion loss.
_THRESHOLDS = {
    "overall_score":             9.0,
    "hindi_quality_score":       9.0,
    "retention_score":           9.0,
    "originality_score":         9.0,
    "youtube_safety_score":      9.0,
    "metadata_score":            9.0,
    "recreated_dialogue_score":  9.0,
}


def _build_claude_gate_evidence(
    lint_report: dict,
    copyedit_report: dict,
    quality_report: dict,
    retention_report: dict,
    similarity_report: dict,
    originality_report: dict,
    dialogue_report: dict,
    metadata_report: dict,
) -> dict:
    """
    Build a compact evidence dict from all Claude gate results.
    Keeps only the fields the OpenAI model needs — avoids sending huge raw files.
    """
    evidence: dict = {}

    # Hindi Text Lint (Python-only)
    if lint_report:
        evidence["hindi_text_lint"] = {
            "total_issues": lint_report.get("total_issues", 0),
            "high_issues":  lint_report.get("high_issues", 0),
            "risk_level":   lint_report.get("risk_level", "unknown"),
        }

    # Hindi Copyedit Gate (Claude)
    if copyedit_report:
        evidence["hindi_copyedit"] = {
            "approved":       copyedit_report.get("approved", False),
            "score":          copyedit_report.get("score", 0),
            "grammar_score":  copyedit_report.get("grammar_score", 0),
            "matra_score":    copyedit_report.get("matra_nasalization_score", 0),
            "flow_score":     copyedit_report.get("sentence_flow_score", 0),
            "high_issues":    sum(
                1 for i in copyedit_report.get("issues", [])
                if isinstance(i, dict) and i.get("severity") == "high"
            ),
        }

    # Script Quality Critic (Claude)
    if quality_report:
        evidence["script_quality"] = {
            "approved":               quality_report.get("approved", False),
            "scores":                 quality_report.get("scores", {}),
            "estimated_duration_min": quality_report.get("estimated_duration_min", 0),
            "repair_required":        quality_report.get("repair_required", False),
            "fact_issues":            quality_report.get("fact_issues", [])[:3],
        }

    # Text Similarity Check (Python)
    if similarity_report:
        evidence["text_similarity"] = {
            "risk_level":        similarity_report.get("risk_level", "unknown"),
            "total_match_count": similarity_report.get("total_match_count", 0),
            "high_risk_matches": similarity_report.get("high_risk_matches", 0),
        }

    # Originality Safety Gate (Claude)
    if originality_report:
        evidence["originality_safety"] = {
            "gate_passed":     originality_report.get("gate_passed", False),
            "scores":          originality_report.get("scores", {}),
            "required_fixes":  originality_report.get("required_fixes", [])[:3],
        }

    # Recreated Dialogue Gate (Claude)
    if dialogue_report:
        evidence["recreated_dialogue"] = {
            "gate_passed":       dialogue_report.get("gate_passed", False),
            "no_recreated_scenes": dialogue_report.get("no_recreated_scenes", False),
            "scores":            dialogue_report.get("scores", {}),
            "required_fixes":    dialogue_report.get("required_fixes", [])[:3],
        }

    # Metadata Quality Gate (Claude)
    if metadata_report:
        evidence["metadata_quality"] = {
            "gate_passed":    metadata_report.get("gate_passed", False),
            "scores":         metadata_report.get("scores", {}),
            "required_fixes": metadata_report.get("required_fixes", [])[:3],
        }

    # Retention Quality Gate (Claude — may be empty if no retention blueprint)
    if retention_report:
        evidence["retention_quality"] = {
            "approved":              retention_report.get("approved", False),
            "overall_score":         retention_report.get("overall_retention_score", 0),
            "opening_hook_score":    retention_report.get("opening_hook_score", 0),
            "curiosity_gap_score":   retention_report.get("curiosity_gap_score", 0),
            "pacing_score":          retention_report.get("pacing_score", 0),
            "emotional_arc_score":   retention_report.get("emotional_arc_score", 0),
            "ending_payoff_score":   retention_report.get("ending_payoff_score", 0),
        }

    return evidence


def run_openai_final_premium_gate(
    script_draft: dict,
    fact_lock: dict,
    blueprint: dict,
    hinglish_level: int,
    lint_report: dict,
    copyedit_report: dict,
    quality_report: dict,
    retention_report: dict,
    similarity_report: dict,
    originality_report: dict,
    dialogue_report: dict,
    metadata_report: dict,
    review_dir: Path,
    label: str = "",
) -> dict:
    """
    Run the combined OpenAI Final Premium Gate.

    Single OpenAI call covering all quality dimensions. Receives compact summaries
    from every upstream Claude gate so it can make an informed independent verdict.

    All score thresholds are >= 9. Any score below 9 blocks approval.

    Args:
        script_draft:       Final script (hindi_narration_chunks + youtube_metadata).
        fact_lock:          Verified facts (case_name, people, legal_outcome).
        blueprint:          Story blueprint (story type, sensitivity rules).
        hinglish_level:     Requested language level (1–5).
        lint_report:        Hindi text lint results.
        copyedit_report:    Claude copyedit gate results.
        quality_report:     Claude script quality report.
        retention_report:   Claude retention quality gate results (may be empty).
        similarity_report:  Text similarity check results.
        originality_report: Originality safety gate results.
        dialogue_report:    Recreated dialogue gate results.
        metadata_report:    Metadata quality gate results.
        review_dir:         Path to write report and raw response files.
        label:              File suffix for output files. "" for first pass,
                            "_after_repair" for post-repair recheck. This keeps
                            the initial report intact as a reference.

    Returns:
        Gate report dict with approved, safe_to_voice, scores, issues, chunk_repair_targets.
    """
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    # Build compact fact_lock summary (top 5 people only)
    fact_lock_summary = {
        "case_name":       fact_lock.get("case_name", ""),
        "verified_people": [
            {"name": p.get("name", ""), "role": p.get("role", "")}
            for p in fact_lock.get("verified_people", [])[:5]
        ],
        "legal_outcome":   fact_lock.get("legal_outcome", {}),
    }

    # Build rich Claude gate evidence
    claude_gate_evidence = _build_claude_gate_evidence(
        lint_report=lint_report,
        copyedit_report=copyedit_report,
        quality_report=quality_report,
        retention_report=retention_report,
        similarity_report=similarity_report,
        originality_report=originality_report,
        dialogue_report=dialogue_report,
        metadata_report=metadata_report,
    )

    # Extract youtube_metadata from script_draft (also sent separately for convenience)
    youtube_metadata = script_draft.get("youtube_metadata", {})

    user_content = json.dumps(
        {
            "hinglish_level":       hinglish_level,
            "fact_lock_summary":    fact_lock_summary,
            "script_draft":         script_draft,
            "claude_gate_evidence": claude_gate_evidence,
            "youtube_metadata":     youtube_metadata,
        },
        ensure_ascii=False,
    )

    report = call_openai_json(
        system_prompt=system_prompt,
        user_content=user_content,
        raw_save_path=review_dir / f"_openai_final_premium{label}_raw_response.txt",
        agent_name="openai_final_premium_gate",
    )

    # ── Python threshold validation (all scores must be >= 9) ────────────────
    threshold_failures: list[str] = []

    for score_key, min_val in _THRESHOLDS.items():
        score = report.get(score_key, 0)
        # Coerce to float safely — preserves decimal scores like 8.5, 9.5
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        if score < min_val:
            threshold_failures.append(
                f"[FINAL_GATE] {score_key}={score} below required {min_val}"
            )

    # Any HIGH severity issue also blocks approval
    high_issues = [
        i for i in report.get("issues", [])
        if isinstance(i, dict) and i.get("severity") == "high"
    ]
    if high_issues:
        threshold_failures.append(
            f"[FINAL_GATE] {len(high_issues)} high-severity issue(s) — "
            "must be resolved before audio generation."
        )

    if threshold_failures:
        report["approved"] = False
        report["safe_to_voice"] = False
        existing_issues = list(report.get("issues", []))
        for fail_msg in threshold_failures:
            existing_issues.append({
                "severity": "high",
                "type":     "python_threshold_failure",
                "description": fail_msg,
                "chunk_id": None,
            })
        report["issues"] = existing_issues
        report["_python_failures"] = threshold_failures
        logger.warning(
            "[openai_final_premium_gate] Python threshold failures: %s",
            threshold_failures,
        )
    else:
        # Python thresholds pass — honour Claude's own approved / safe_to_voice verdict
        if not report.get("approved", False):
            report["safe_to_voice"] = False
        else:
            report["safe_to_voice"] = True

    scores_log = {k: report.get(k, "?") for k in _THRESHOLDS}
    logger.info(
        "[openai_final_premium_gate] approved=%s safe_to_voice=%s | scores=%s | "
        "issues=%d (%d high) | chunk_repair_targets=%d",
        report.get("approved", False),
        report.get("safe_to_voice", False),
        scores_log,
        len(report.get("issues", [])),
        len(high_issues),
        len(report.get("chunk_repair_targets", [])),
    )

    # Save report — label controls the filename so first-pass and post-repair
    # reports coexist on disk and can both be inspected after a repair cycle.
    out_path = review_dir / f"openai_final_premium_report{label}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("[openai_final_premium_gate] Report saved → %s", out_path)

    return report
