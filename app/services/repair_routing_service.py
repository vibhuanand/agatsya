"""
Repair Routing Service — pure Python, zero AI cost.

Groups gate failures and OpenAI repair targets into a small number of root causes,
then decides the cheapest repair owner for each:

  python   — deterministic regex/replacement; no model call
  claude   — grouped creative/content repair; one Claude call per root cause
  openai   — small final-polish; only when total targets <= OPENAI_REPAIR_MAX_CHUNKS

Routing rules (in priority order):
  1. Deterministic metadata/title/tag superlative issues → python
  2. Deterministic child-victim organ/graphic wording → python
  3. Missing recreated-dialogue disclaimer → python
  4. Retention pacing / curiosity-gap → claude (grouped by section)
  5. Narration rewrite / structure / originality → claude (grouped by root cause)
  6. Small final polish ≤ OPENAI_REPAIR_MAX_CHUNKS → openai
  7. Too many targets for OpenAI → python + claude grouped (NOT openai bulk)
  8. Contradictory facts / unrecoverable / legal uncertainty → stop (needs_human_review)

Produces:  04-review/repair_routing_plan.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── Category matchers ────────────────────────────────────────────────────────

_METADATA_LEGAL_BLAME_PATTERNS = [
    "most infamous", "most shocking", "most brutal",
    "journalism killed", "media killed", "media's fault",
    "सबसे भयानक", "सबसे डरावना", "सबसे क्रूर",
    "मीडिया का गुनाह", "मीडिया ने मारा", "पत्रकारिता ने मारा",
    "सबसे कुख्यात",
]

_CHILD_VICTIM_ORGAN_PATTERNS = [
    "फटा हुआ जिगर", "जिगर फट", "ruptured liver",
    "शरीर के टुकड़े", "क्षत-विक्षत", "mutilat",
    "corpse photo", "autopsy detail", "severed finger",
    "कटी हुई उँगली",
]

_RETENTION_PATTERNS = [
    "retention", "pacing", "curiosity", "hook", "engagement", "re-engagement",
    "slow", "dead zone", "no payoff", "flat", "boring",
    "curiosity_gap_score", "pacing_score", "overall_retention_score",
]

_ORIGINALITY_PATTERNS = [
    "originality", "verbatim", "copy", "paraphrase", "source",
    "copying_risk", "reused_content_risk", "transformative_value",
]

_HINDI_QUALITY_PATTERNS = [
    "hindi", "grammar", "matra", "gender", "postposition",
    "copyedit", "naturalness", "nasalization",
]

_RECREATED_DIALOGUE_PATTERNS = [
    "recreated", "disclaimer", "label", "dialogue", "simulated",
    "labelling_compliance", "factual_consistency",
]

_UNRECOVERABLE_PATTERNS = [
    "contradictory facts", "missing source facts", "legal uncertainty",
    "malformed", "unrecoverable", "external verification",
]

_CASE_GLOSSARY_PATTERNS = [
    "case_glossary", "glossary", "wrong name", "incorrect name", "inconsistent name",
    "forbidden term", "preferred hindi", "preferred hinglish", "pronunciation",
    "name form", "place name", "victim name", "suspect name", "spelling",
]


def _text_matches_any(text: str, patterns: list[str]) -> bool:
    text_lower = text.lower()
    return any(p.lower() in text_lower for p in patterns)


def _issue_to_area(text: str) -> str:
    """Classify a free-text issue into one of the known repair areas."""
    if _text_matches_any(text, _UNRECOVERABLE_PATTERNS):
        return "unrecoverable"
    if _text_matches_any(text, _CHILD_VICTIM_ORGAN_PATTERNS):
        return "child_victim_safety"
    if _text_matches_any(text, _METADATA_LEGAL_BLAME_PATTERNS):
        return "metadata"
    if _text_matches_any(text, _RECREATED_DIALOGUE_PATTERNS):
        return "recreated_dialogue"
    if _text_matches_any(text, _CASE_GLOSSARY_PATTERNS):
        return "case_glossary"
    if _text_matches_any(text, _RETENTION_PATTERNS):
        return "retention"
    if _text_matches_any(text, _ORIGINALITY_PATTERNS):
        return "originality"
    if _text_matches_any(text, _HINDI_QUALITY_PATTERNS):
        return "hindi_quality"
    return "general"


def _preferred_owner(area: str, is_deterministic: bool) -> str:
    """Choose the cheapest repair owner for an area."""
    if is_deterministic or area in ("metadata", "child_victim_safety", "recreated_dialogue"):
        return "python"
    if area in ("retention", "originality", "hindi_quality", "general", "case_glossary"):
        return "claude"
    return "claude"


# ─── Main routing function ────────────────────────────────────────────────────

def run_repair_routing(
    all_gate_reports: dict[str, dict],
    openai_repair_max_chunks: int,
    review_dir: Path | None = None,
) -> dict:
    """
    Analyse all gate reports and OpenAI repair targets, group them into root causes,
    and decide the cheapest repair route.

    Parameters
    ----------
    all_gate_reports : dict mapping gate name → gate report dict.
        Expected keys (all optional): script_quality, final_quality, copyedit,
        originality_safety, retention, metadata, recreated_dialogue,
        text_similarity, openai_final_premium, python_preflight.
    openai_repair_max_chunks : int   from settings.openai_repair_max_chunks
    review_dir : Path | None         if provided, saves repair_routing_plan.json

    Returns
    -------
    repair_routing_plan dict
    """
    root_causes: list[dict] = []
    python_fixes: list[str] = []
    claude_targets: list[dict] = []
    openai_targets: list[dict] = []
    unrecoverable: list[str] = []
    notes: list[str] = []

    # ── Extract all OAI repair targets ───────────────────────────────────────
    ofp_report = all_gate_reports.get("openai_final_premium", {})
    raw_oai_targets: list[dict] = ofp_report.get("chunk_repair_targets", [])

    # ── Build a combined set of issues from gate reports ─────────────────────
    all_issues: list[dict] = []

    # Script quality issues
    for src_key in ("script_quality", "final_quality"):
        qr = all_gate_reports.get(src_key, {})
        for issue in qr.get("issues", []):
            txt = (
                issue.get("problem", "")
                or issue.get("description", "")
                or str(issue)
            )
            all_issues.append({"text": txt, "severity": issue.get("severity", "medium"), "source": src_key})

    # Copyedit issues
    for issue in all_gate_reports.get("copyedit", {}).get("issues", []):
        txt = issue.get("problem", "") or str(issue)
        all_issues.append({"text": txt, "severity": issue.get("severity", "medium"), "source": "copyedit"})

    # Originality safety
    orig = all_gate_reports.get("originality_safety", {})
    if not orig.get("gate_passed", True):
        for fix in orig.get("required_fixes", []):
            txt = fix if isinstance(fix, str) else str(fix)
            all_issues.append({"text": txt, "severity": "high", "source": "originality_safety"})

    # Retention gate
    ret = all_gate_reports.get("retention", {})
    if not ret.get("gate_passed", True):
        for target in ret.get("chunk_repair_targets", []):
            txt = target.get("problem", "") or str(target)
            all_issues.append({"text": txt, "severity": "high", "source": "retention"})
        # Also check raw scores
        scores = ret.get("scores", {})
        for score_key in ("overall_retention_score", "pacing_score", "curiosity_gap_score"):
            val = scores.get(score_key)
            if val is not None and isinstance(val, (int, float)) and val < 9:
                all_issues.append({
                    "text": f"{score_key}={val} below threshold (9)",
                    "severity": "high" if val < 7 else "medium",
                    "source": "retention",
                })

    # Metadata gate
    meta = all_gate_reports.get("metadata", {})
    if not meta.get("gate_passed", True):
        for fix in meta.get("required_fixes", []):
            txt = fix if isinstance(fix, str) else str(fix)
            all_issues.append({"text": txt, "severity": "high", "source": "metadata"})

    # Recreated dialogue gate
    dial = all_gate_reports.get("recreated_dialogue", {})
    if not dial.get("gate_passed", True):
        for fix in dial.get("required_fixes", []):
            txt = fix if isinstance(fix, str) else str(fix)
            all_issues.append({"text": txt, "severity": "medium", "source": "recreated_dialogue"})

    # Python preflight
    pf = all_gate_reports.get("python_preflight", {})
    for issue in pf.get("issues", []):
        txt = issue.get("description", "") or str(issue)
        sev = issue.get("severity", "low")
        all_issues.append({"text": txt, "severity": sev, "source": "python_preflight"})

    # OAI final premium gate issues (includes unrecoverable high-severity issues)
    for issue in ofp_report.get("issues", []):
        txt = (
            issue.get("description", "")
            or issue.get("problem", "")
            or str(issue)
        )
        sev = issue.get("severity", "medium")
        all_issues.append({
            "text": txt,
            "severity": sev,
            "source": "openai_final_premium",
            "chunk_id": issue.get("chunk_id"),
        })

    # OAI repair targets (treat each as an issue too)
    for t in raw_oai_targets:
        txt = (
            t.get("problem", "")
            or t.get("repair_instruction", "")
            or str(t)
        )
        all_issues.append({
            "text": txt,
            "severity": "high",
            "source": "openai_final_premium",
            "chunk_id": t.get("chunk_id", ""),
            "raw_target": t,
        })

    # ── Group issues into root causes ─────────────────────────────────────────
    area_buckets: dict[str, list[dict]] = {}
    for issue in all_issues:
        area = _issue_to_area(issue["text"])
        area_buckets.setdefault(area, []).append(issue)

    rc_id = 0
    for area, bucket in area_buckets.items():
        rc_id += 1
        severities = [i.get("severity", "low") for i in bucket]
        max_sev = (
            "critical" if "critical" in severities
            else "high" if "high" in severities
            else "medium" if "medium" in severities
            else "low"
        )
        # Determine if all issues in this area are deterministic
        is_det = area in ("metadata", "child_victim_safety", "recreated_dialogue")
        owner = _preferred_owner(area, is_det)

        affected = list({
            i.get("chunk_id", "") or i.get("source", "")
            for i in bucket
            if i.get("chunk_id") or i.get("source")
        })

        # Build a grouped repair instruction
        unique_texts = list({i["text"] for i in bucket})[:4]
        repair_instr = " | ".join(unique_texts)

        rc = {
            "root_cause_id": f"RC{rc_id:02d}",
            "area": area,
            "severity": max_sev,
            "issue_count": len(bucket),
            "affected_targets": affected[:8],
            "preferred_repair_owner": owner,
            "reason": f"{len(bucket)} issue(s) in {area} area grouped under one root cause",
            "repair_instruction": repair_instr[:400],
        }
        root_causes.append(rc)

        if area == "unrecoverable":
            unrecoverable.extend(unique_texts)
        elif owner == "python":
            python_fixes.extend([i["text"] for i in bucket if i["text"]])
        elif owner == "claude":
            # Build a grouped claude target
            chunk_ids = [
                i["chunk_id"] for i in bucket if i.get("chunk_id")
            ]
            claude_targets.append({
                "root_cause_id": rc["root_cause_id"],
                "area": area,
                "severity": max_sev,
                "chunk_ids": list(dict.fromkeys(chunk_ids))[:6],
                "repair_instruction": repair_instr[:400],
            })

    # ── Determine OAI targets ─────────────────────────────────────────────────
    # Only route to OpenAI if total unique chunk targets <= limit
    unique_oai_chunks = list({t.get("chunk_id", "") for t in raw_oai_targets if t.get("chunk_id")})
    if len(unique_oai_chunks) <= openai_repair_max_chunks and not claude_targets:
        # Small enough for direct OpenAI targeted repair
        openai_targets = raw_oai_targets
    else:
        # Too many for OpenAI bulk — keep OAI targets empty; Claude handles via grouped repair
        if raw_oai_targets and not openai_targets:
            notes.append(
                f"OAI targets ({len(unique_oai_chunks)}) exceed OPENAI_REPAIR_MAX_CHUNKS "
                f"({openai_repair_max_chunks}) — routed to Python + Claude grouped repair."
            )

    # ── Choose top-level route ─────────────────────────────────────────────────
    estimated_calls_saved = 0
    if unrecoverable:
        route = "stop_not_voice_ready"
        notes.append(f"Unrecoverable issues found: {'; '.join(unrecoverable[:3])}")
    elif not all_issues:
        route = "openai_targeted"
        notes.append("No issues detected — gate reports all passed.")
    elif python_fixes and not claude_targets and not openai_targets:
        route = "python_only"
        estimated_calls_saved = len(python_fixes)
    elif openai_targets and not claude_targets:
        route = "openai_targeted"
    elif claude_targets:
        if python_fixes:
            route = "claude_grouped_repair"
            notes.append(f"Python will fix {len(python_fixes)} deterministic issues first.")
        else:
            route = "claude_grouped_repair"
        # Claude grouped saves individual-chunk calls
        individual_repairs = sum(len(rc["affected_targets"]) for rc in root_causes if rc["preferred_repair_owner"] == "claude")
        grouped_repairs = len(claude_targets)
        estimated_calls_saved = max(0, individual_repairs - grouped_repairs)
    else:
        route = "claude_grouped_repair"

    plan = {
        "route": route,
        "root_causes": root_causes,
        "python_fixes": list(set(python_fixes))[:20],
        "claude_repair_targets": claude_targets,
        "openai_repair_targets": openai_targets,
        "estimated_model_calls_saved": estimated_calls_saved,
        "unrecoverable_issues": unrecoverable,
        "notes": notes,
        "stats": {
            "total_raw_issues": len(all_issues),
            "root_cause_count": len(root_causes),
            "python_fix_count": len(python_fixes),
            "claude_target_count": len(claude_targets),
            "openai_target_count": len(openai_targets),
            "unique_oai_chunks": len(unique_oai_chunks),
        },
    }

    if review_dir is not None:
        out_path = Path(review_dir) / "repair_routing_plan.json"
        out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "Repair routing plan saved → %s  route=%s  root_causes=%d  "
            "python_fixes=%d  claude_targets=%d  saved_calls=%d",
            out_path, route, len(root_causes), len(python_fixes),
            len(claude_targets), estimated_calls_saved,
        )

    return plan
