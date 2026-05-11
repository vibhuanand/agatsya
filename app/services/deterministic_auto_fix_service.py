"""
Deterministic Auto-Fix Service — zero AI cost.

Applies safe, mechanical fixes to script metadata and narration chunks
without calling Claude or OpenAI.

Fixes implemented:
  METADATA / TITLE / TAGS / FOLDER SLUG:
    1. Remove unsupported superlatives (most infamous / सबसे कुख्यात / etc.)
    2. Replace legal-blame wording (मीडिया का गुनाह → पत्रकारिता पर सवाल)
    3. Sanitize Taiwan folder slug to factual form
    4. Remove graphic injury wording from title / thumbnail / metadata

  CHILD-VICTIM NARRATION SAFETY:
    5. Replace फटा हुआ जिगर → गंभीर आंतरिक चोटें (and English equivalents)
    6. Replace ruptured liver → serious internal injuries

  RECREATED DIALOGUE DISCLAIMER:
    7. Insert missing disclaimer if scene has no label

Produces:  04-review/deterministic_auto_fix_report.json

Returns (updated_script_draft, auto_fix_report) tuple.
Never calls Claude / OpenAI. Never invents facts.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── Child-victim case detection ──────────────────────────────────────────────

_CHILD_VICTIM_CASE_HINTS = [
    "child", "minor", "children", "juvenile", "teen", "teenager", "young girl",
    "young boy", "daughter", "son", "school", "student",
    # Hindi
    "बच्चा", "बच्ची", "बच्चे", "नाबालिग", "छात्र", "छात्रा",
    # Specific known cases
    "pai hsiao-yen", "pai hsiao", "hsiao-yen", "taiwan 1997",
]


def _is_child_victim_case(case_hint: str, script_draft: dict) -> bool:
    combined = case_hint.lower()
    # Also check folder_name and case_summary if present
    combined += " " + str(script_draft.get("folder_name", "")).lower()
    combined += " " + str(script_draft.get("case_summary", {}).get("case_title", "")).lower()
    return any(kw in combined for kw in _CHILD_VICTIM_CASE_HINTS)


# ─── Replacement rules ────────────────────────────────────────────────────────

# (pattern, replacement, description, applies_to)
# applies_to: "metadata" | "narration" | "both"
_REPLACEMENT_RULES: list[tuple[re.Pattern, str, str, str]] = [
    # ── Child-victim body/organ wording ──────────────────────────────────────
    (
        re.compile(r"फटा\s+हुआ\s+जिगर", re.UNICODE | re.IGNORECASE),
        "गंभीर आंतरिक चोटें",
        "Replace organ-specific injury (फटा हुआ जिगर) with restrained wording",
        "both",
    ),
    (
        re.compile(r"जिगर\s+फट(?:ा|े|ी)?", re.UNICODE | re.IGNORECASE),
        "गंभीर आंतरिक चोटें",
        "Replace organ rupture phrase with restrained wording",
        "both",
    ),
    (
        re.compile(r"\bruptured\s+liver\b", re.IGNORECASE),
        "serious internal injuries",
        "Replace English ruptured liver with restrained wording",
        "both",
    ),
    # ── Legal-blame and superlatives (metadata AND narration) ────────────────
    # scope="both" so these are fixed in metadata fields AND narration chunks.
    (
        re.compile(r"मीडिया\s+का\s+गुनाह", re.UNICODE | re.IGNORECASE),
        "पत्रकारिता पर सवाल",
        "Replace legal-blame framing (मीडिया का गुनाह) with ethical-criticism framing",
        "both",
    ),
    (
        re.compile(r"journalism\s+killed\s+her", re.IGNORECASE),
        "journalism ethics failure",
        "Replace journalism-killed claim with ethical framing",
        "both",
    ),
    (
        re.compile(r"media\s+killed\s+her", re.IGNORECASE),
        "media ethics failure",
        "Replace media-killed claim with ethical framing",
        "both",
    ),
    (
        re.compile(r"\bmost\s+infamous\b", re.IGNORECASE),
        "widely known",
        "Replace unsupported superlative 'most infamous'",
        "metadata",
    ),
    (
        re.compile(r"\bसबसे\s+कुख्यात\b", re.UNICODE | re.IGNORECASE),
        "बहुचर्चित",
        "Replace unsupported superlative सबसे कुख्यात",
        "metadata",
    ),
    (
        re.compile(r"\bसबसे\s+भयानक\b", re.UNICODE | re.IGNORECASE),
        "बहुचर्चित",
        "Replace unsupported superlative सबसे भयानक",
        "metadata",
    ),
    (
        re.compile(r"\bmost\s+shocking\b", re.IGNORECASE),
        "widely discussed",
        "Replace unsupported superlative 'most shocking'",
        "metadata",
    ),
    (
        re.compile(r"\bmost\s+brutal\b", re.IGNORECASE),
        "deeply disturbing",
        "Replace unsupported superlative 'most brutal'",
        "metadata",
    ),
]

# Tags to remove (exact match, case-insensitive)
_BANNED_TAGS: list[str] = [
    "taiwan most infamous case",
    "most infamous",
    "most shocking case",
    "most brutal crime",
]

# Tags to replace (old → new list)
_TAG_REPLACEMENTS: dict[str, list[str]] = {
    "taiwan most infamous case": [
        "Pai Hsiao-Yen case",
        "Taiwan 1997",
        "Taiwan kidnapping case",
        "media ethics case",
    ],
}

# Folder slug sanitization: old_substring → replacement_slug
_SLUG_SANITIZATIONS: list[tuple[str, str]] = [
    ("taiwans-most-infamous", "taiwan-1997"),
    ("most-infamous", ""),
    ("most-shocking", ""),
    ("most-brutal", ""),
]

# Recreated-dialogue disclaimer
_DIALOGUE_DISCLAIMER = (
    "यह असली रिकॉर्डिंग नहीं है; "
    "यह उपलब्ध जानकारी पर आधारित नाटकीय पुनर्निर्माण है।"
)


# ─── Individual fixers ────────────────────────────────────────────────────────

def _apply_text_replacements(
    text: str,
    applies_to: str,
    target_scope: str,
    changes: list[dict],
    context_label: str = "",
) -> str:
    """Apply all relevant regex replacement rules to a text string."""
    for pattern, replacement, description, scope in _REPLACEMENT_RULES:
        if scope not in (target_scope, "both"):
            continue
        new_text, n = pattern.subn(replacement, text)
        if n > 0:
            changes.append({
                "context": context_label,
                "description": description,
                "occurrences": n,
                "before_sample": text[:120],
                "after_sample": new_text[:120],
            })
            text = new_text
    return text


def _fix_metadata(
    script_draft: dict,
    is_child_victim: bool,
    changes: list[dict],
) -> dict:
    """Fix youtube_metadata in-place. Returns the draft."""
    meta = script_draft.get("youtube_metadata", {})
    if not meta:
        return script_draft

    # Fix title_options
    new_titles = []
    for title in meta.get("title_options", []):
        fixed = _apply_text_replacements(title, "both", "metadata", changes, "title_options")
        new_titles.append(fixed)
    if new_titles:
        meta["title_options"] = new_titles

    # Fix recommended_title
    if meta.get("recommended_title"):
        meta["recommended_title"] = _apply_text_replacements(
            meta["recommended_title"], "both", "metadata", changes, "recommended_title"
        )

    # Fix description
    if meta.get("description"):
        meta["description"] = _apply_text_replacements(
            meta["description"], "both", "metadata", changes, "description"
        )

    # Fix tags — remove banned, add replacements, also apply regex rules
    existing_tags = meta.get("tags", [])
    new_tags: list[str] = []
    added_tags: set[str] = set()
    for tag in existing_tags:
        tag_lower = tag.lower().strip()
        if tag_lower in _BANNED_TAGS:
            changes.append({
                "context": "tags",
                "description": f"Removed banned tag: '{tag}'",
                "occurrences": 1,
                "before_sample": tag,
                "after_sample": "(removed)",
            })
            # Add replacements if defined
            replacements = _TAG_REPLACEMENTS.get(tag_lower, [])
            for r in replacements:
                if r not in added_tags and r not in existing_tags:
                    new_tags.append(r)
                    added_tags.add(r)
        else:
            # Also apply regex rules to individual tag text
            fixed_tag = _apply_text_replacements(tag, "both", "metadata", changes, "tag")
            new_tags.append(fixed_tag)
    meta["tags"] = new_tags

    # Fix thumbnail_options
    new_thumbs = []
    for thumb in meta.get("thumbnail_options", []):
        if isinstance(thumb, dict):
            txt = thumb.get("thumbnail_text", "")
            fixed_txt = _apply_text_replacements(
                txt, "both", "metadata", changes, "thumbnail_text"
            )
            thumb["thumbnail_text"] = fixed_txt
            new_thumbs.append(thumb)
        elif isinstance(thumb, str):
            new_thumbs.append(
                _apply_text_replacements(thumb, "both", "metadata", changes, "thumbnail_text")
            )
    if new_thumbs:
        meta["thumbnail_options"] = new_thumbs

    # Fix pinned comment
    if meta.get("pinned_comment"):
        meta["pinned_comment"] = _apply_text_replacements(
            meta["pinned_comment"], "both", "metadata", changes, "pinned_comment"
        )

    script_draft["youtube_metadata"] = meta
    return script_draft


def _fix_folder_slug(script_draft: dict, changes: list[dict]) -> dict:
    """Sanitize folder_name slug to remove unsupported superlatives."""
    slug = script_draft.get("folder_name", "")
    if not slug:
        return script_draft
    original = slug
    for old, new_part in _SLUG_SANITIZATIONS:
        if old in slug:
            if new_part:
                slug = slug.replace(old, new_part)
            else:
                slug = re.sub(re.escape(old) + r"-?", "", slug)
    # Clean up double hyphens and trailing hyphens
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if slug != original:
        changes.append({
            "context": "folder_name",
            "description": "Sanitized folder slug to remove unsupported superlative",
            "occurrences": 1,
            "before_sample": original,
            "after_sample": slug,
        })
        script_draft["folder_name"] = slug
    return script_draft


def _fix_narration_chunks(
    script_draft: dict,
    is_child_victim: bool,
    changes: list[dict],
) -> dict:
    """
    Fix hindi_narration_chunks.

    Always applies:  legal-blame replacements (scope="both") in all narration chunks.
    Child-victim only:  organ/graphic replacements (scope="both", child-specific patterns).

    Organ-specific patterns are identified by their replacement target — they always
    replace with "गंभीर आंतरिक चोटें" or "serious internal injuries". We apply ALL
    "both"-scoped rules but only when the rule is an organ rule AND is_child_victim=True,
    otherwise only non-organ "both"-scoped rules.
    """
    _ORGAN_REPLACEMENTS = {"गंभीर आंतरिक चोटें", "serious internal injuries"}
    chunks = script_draft.get("hindi_narration_chunks", [])
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        for pattern, replacement, description, scope in _REPLACEMENT_RULES:
            if scope not in ("both",):
                continue  # only apply "both"-scoped rules to narration
            # Organ rules: only apply in child-victim cases
            if replacement in _ORGAN_REPLACEMENTS and not is_child_victim:
                continue
            new_text, n = pattern.subn(replacement, text)
            if n > 0:
                changes.append({
                    "context": f"chunk:{chunk.get('chunk_id', '?')}",
                    "description": description,
                    "occurrences": n,
                    "before_sample": text[:120],
                    "after_sample": new_text[:120],
                })
                text = new_text
        chunk["text"] = text
    script_draft["hindi_narration_chunks"] = chunks
    return script_draft


def _fix_recreated_dialogue_disclaimers(
    script_draft: dict,
    changes: list[dict],
) -> dict:
    """Add missing disclaimers to recreated dialogue scenes (safe to insert)."""
    items = script_draft.get("recreated_dialogues", {}).get("items", [])
    for scene in items:
        label = scene.get("label_on_screen", "")
        not_orig = scene.get("not_original_audio", True)
        # If the scene has no label or label is empty, add the disclaimer
        if not label:
            scene["label_on_screen"] = "फिर से रचा गया संवाद"
            changes.append({
                "context": f"recreated_dialogue:{scene.get('scene_id', '?')}",
                "description": "Added missing 'फिर से रचा गया संवाद' label to recreated scene",
                "occurrences": 1,
                "before_sample": "(no label)",
                "after_sample": "फिर से रचा गया संवाद",
            })
        # Ensure not_original_audio is set
        if not not_orig:
            scene["not_original_audio"] = True
        # Add disclaimer to dialogue if not present
        dialogue = scene.get("dialogue", [])
        if dialogue:
            first_text = dialogue[0].get("text", "") if dialogue else ""
            if _DIALOGUE_DISCLAIMER not in first_text and "पुनर्निर्माण" not in first_text:
                dialogue.insert(0, {
                    "speaker": "disclaimer",
                    "text": _DIALOGUE_DISCLAIMER,
                })
                scene["dialogue"] = dialogue
                changes.append({
                    "context": f"recreated_dialogue:{scene.get('scene_id', '?')}",
                    "description": "Inserted disclaimer at start of recreated dialogue",
                    "occurrences": 1,
                    "before_sample": "(no disclaimer)",
                    "after_sample": _DIALOGUE_DISCLAIMER[:80],
                })
    return script_draft


# ─── Public entry point ───────────────────────────────────────────────────────

def run_deterministic_auto_fix(
    script_draft: dict,
    routing_plan: dict | None = None,
    case_hint: str = "",
    review_dir: Path | None = None,
) -> tuple[dict, dict]:
    """
    Apply all safe deterministic fixes to script_draft.

    Parameters
    ----------
    script_draft   : the current script dict (will be modified in-place copy)
    routing_plan   : output of repair_routing_service (optional; used to limit scope)
    case_hint      : case_hint string for child-victim detection
    review_dir     : if provided, saves deterministic_auto_fix_report.json

    Returns
    -------
    (updated_script_draft, auto_fix_report)
    The updated script_draft has all deterministic fixes applied.
    The auto_fix_report records every change made.
    """
    import copy
    draft = copy.deepcopy(script_draft)
    changes: list[dict] = []

    is_child = _is_child_victim_case(case_hint, draft)
    logger.info("Deterministic auto-fix: child_victim=%s, case_hint=%r", is_child, case_hint)

    # 1. Fix metadata
    draft = _fix_metadata(draft, is_child, changes)

    # 2. Fix folder slug
    draft = _fix_folder_slug(draft, changes)

    # 3. Fix narration chunks (child-victim safety)
    draft = _fix_narration_chunks(draft, is_child, changes)

    # 4. Fix recreated dialogue disclaimers
    draft = _fix_recreated_dialogue_disclaimers(draft, changes)

    total_changes = sum(c["occurrences"] for c in changes)
    report: dict[str, Any] = {
        "total_changes": total_changes,
        "total_fixes_applied": total_changes,   # alias used by pipeline telemetry
        "change_count": len(changes),
        "is_child_victim_case": is_child,
        "changes": changes,
        "fixes": changes,                        # alias used by tests
        "python_fixes_applied": [c["description"] for c in changes],
    }

    if review_dir is not None:
        out_path = Path(review_dir) / "deterministic_auto_fix_report.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "Deterministic auto-fix report saved → %s  changes=%d",
            out_path, total_changes,
        )

    if total_changes == 0:
        logger.info("Deterministic auto-fix: no changes needed")
    else:
        logger.info("Deterministic auto-fix: %d change(s) applied in %d location(s)", total_changes, len(changes))

    return draft, report
