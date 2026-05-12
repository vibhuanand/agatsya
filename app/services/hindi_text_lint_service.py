"""
Hindi Text Lint Service — deterministic, zero-cost rule-based checks.

Scans the final Hindi narration for known grammar, orthography, phrasing,
and consistency issues that are mechanical enough to catch with regex.
Does NOT rewrite anything — reports issues with suggested fixes.

Lint report is fed into the Hindi Copyedit Gate Agent as ground truth so
Claude doesn't have to re-discover issues that Python already caught.

Rules implemented:
  1. Orthography: सांस → साँस (chandrabindu not anusvara)
  2. Grammar: relative clause with जो when ने is required
  3. Grammar: [X] की जाल (जाल is masculine → का)
  4. Punctuation: sentence fragment  था/है। जिसे → था/है, जिसे
  5. Phrasing: तीन-साढ़े-तीन साल बाद (awkward) → लगभग साढ़े तीन साल बाद
  6. Phrasing: बाल-हत्या → बच्चे की हत्या
  7. Consistency: Canada के (level 1-2) → कनाडा के
  8. Consistency: Canadian → कनाडाई
  9. Repetition: यह सिर्फ़ used more than 2 times

Produces: 04-review/hindi_text_lint_report.json
"""
from __future__ import annotations

import re
import logging
from typing import Iterator

logger = logging.getLogger(__name__)


# ─── Rule definitions ─────────────────────────────────────────────────────────

def _find_all_with_context(pattern: re.Pattern, text: str, window: int = 40) -> list[dict]:
    """
    Return all matches of *pattern* in *text* with surrounding context.
    Each match dict has: matched, context, position.
    """
    hits: list[dict] = []
    for m in pattern.finditer(text):
        start = max(0, m.start() - window)
        end   = min(len(text), m.end() + window)
        hits.append({
            "matched":  m.group(0),
            "context":  text[start:end].strip(),
            "position": m.start(),
        })
    return hits


def _find_repetition(pattern: re.Pattern, text: str, max_ok: int = 2) -> list[dict]:
    """Flag if a pattern occurs more than max_ok times across the full text."""
    matches = pattern.findall(text)
    if len(matches) > max_ok:
        return [{
            "matched":  matches[0],
            "context":  f"Found {len(matches)} times (max {max_ok} recommended)",
            "position": 0,
        }]
    return []


# ─── Per-rule checks ──────────────────────────────────────────────────────────

_RULE_SANS   = re.compile(r"सांस")          # anusvara: should be chandrabindu साँस
_RULE_JO_NE  = re.compile(
    r"जो\s+\S+\s+(?:को|ने|से)\s+\S+\s+(?:घर|काम|जगह|वहाँ|यहाँ)?\s*(?:पर|में|को)?\s*"
    r"(?:देखा|सुना|बताया|कहा|पाया|मिला|गई|गया|आई|आया)\s+था",
    re.UNICODE,
)
_RULE_JAAL   = re.compile(r"\S+\s+की\s+जाल(?:\s|।|,|$)", re.UNICODE)
_RULE_FRAG   = re.compile(r"(?:था|है|थी|हैं|थे)\s*।\s+जिसे", re.UNICODE)
_RULE_TADHA  = re.compile(r"तीन[–-]साढ़े[–-]तीन\s+साल\s+बाद", re.UNICODE)
_RULE_BAAL   = re.compile(r"बाल[–-]हत्या", re.UNICODE)
_RULE_CANADA = re.compile(r"Canada\s+के", re.UNICODE)    # for level 1–2
_RULE_CADIAN = re.compile(r"Canadian\s+(?!Court|government|law|police|charter)",  # not official names
                           re.UNICODE | re.IGNORECASE)
_RULE_SIRF   = re.compile(r"यह\s+सिर्फ़?", re.UNICODE)

# ── Child-victim sensitive-term rules (blocking in metadata; high in narration) ──
_RULE_ORGAN_LIVER  = re.compile(r"फटा\s+हुआ\s+जिगर|जिगर\s+फट", re.UNICODE | re.IGNORECASE)
_RULE_ENG_LIVER    = re.compile(r"\bruptured\s+liver\b", re.IGNORECASE)
_RULE_MUTILATION   = re.compile(r"शरीर\s+के\s+टुकड़े|क्षत[–-]विक्षत|mutilat", re.UNICODE | re.IGNORECASE)
_RULE_SEVERED_FIN  = re.compile(r"कटी\s+हुई\s+उँगली|severed\s+finger", re.UNICODE | re.IGNORECASE)


_RULES: list[dict] = [
    {
        "id":          "sans_chandrabindu",
        "pattern":     _RULE_SANS,
        "severity":    "medium",
        "type":        "matra",
        "description": "Use chandrabindu (ँ) for nasal sound in साँस, not anusvara (ं)",
        "suggested":   "साँस",
        "repetition":  False,
    },
    {
        "id":          "jo_ne_clause",
        "pattern":     _RULE_JO_NE,
        "severity":    "high",
        "type":        "grammar",
        "description": (
            "Relative clause starting with 'जो' when the verb requires 'ने' "
            "— use 'जिसने' / 'जिस ... ने'"
        ),
        "suggested":   "जिसने [name] को ... देखा था",
        "repetition":  False,
    },
    {
        "id":          "jaal_masculine",
        "pattern":     _RULE_JAAL,
        "severity":    "high",
        "type":        "grammar",
        "description": "जाल is masculine — use 'का' not 'की' (e.g. Mr. Big का जाल)",
        "suggested":   "[X] का जाल",
        "repetition":  False,
    },
    {
        "id":          "fragment_jise",
        "pattern":     _RULE_FRAG,
        "severity":    "high",
        "type":        "punctuation",
        "description": (
            "Sentence fragment: 'था/है। जिसे' — 'जिसे' starts a dependent clause "
            "and cannot follow a full stop. Use a comma instead."
        ),
        "suggested":   "था, जिसे / है, जिसे",
        "repetition":  False,
    },
    {
        "id":          "teen_sadhe_teen",
        "pattern":     _RULE_TADHA,
        "severity":    "medium",
        "type":        "awkward_phrase",
        "description": "'तीन-साढ़े-तीन साल बाद' sounds unnatural — prefer 'लगभग साढ़े तीन साल बाद'",
        "suggested":   "लगभग साढ़े तीन साल बाद",
        "repetition":  False,
    },
    {
        "id":          "baal_hatya",
        "pattern":     _RULE_BAAL,
        "severity":    "medium",
        "type":        "awkward_phrase",
        "description": "'बाल-हत्या' is formal/stiff for narration — prefer 'बच्चे की हत्या'",
        "suggested":   "बच्चे की हत्या",
        "repetition":  False,
    },
    {
        "id":          "canada_ke",
        "pattern":     _RULE_CANADA,
        "severity":    "medium",
        "type":        "hinglish_consistency",
        "description": "At hinglish_level 1–2 prefer 'कनाडा के' over 'Canada के' (unless in official name)",
        "suggested":   "कनाडा के",
        "repetition":  False,
    },
    {
        "id":          "canadian_adj",
        "pattern":     _RULE_CADIAN,
        "severity":    "medium",
        "type":        "hinglish_consistency",
        "description": "At hinglish_level 1–2 prefer 'कनाडाई' over 'Canadian' as adjective",
        "suggested":   "कनाडाई",
        "repetition":  False,
    },
    {
        "id":          "yeh_sirf_repetition",
        "pattern":     _RULE_SIRF,
        "severity":    "low",
        "type":        "repetition",
        "description": "'यह सिर्फ़' used more than twice — vary the sentence opener",
        "suggested":   "Vary: 'केवल...', 'बस...', 'महज़...', or restructure the sentence",
        "repetition":  True,
        "max_ok":      2,
    },
    # ── Child-victim sensitive-term rules ────────────────────────────────────
    {
        "id":          "organ_liver_hindi",
        "pattern":     _RULE_ORGAN_LIVER,
        "severity":    "high",
        "type":        "child_victim_safety",
        "description": (
            "Organ-specific injury language (फटा हुआ जिगर / जिगर फट) in narration — "
            "replace with 'गंभीर आंतरिक चोटें' for child-victim cases"
        ),
        "suggested":   "गंभीर आंतरिक चोटें",
        "repetition":  False,
        "child_victim_only": True,
    },
    {
        "id":          "organ_liver_english",
        "pattern":     _RULE_ENG_LIVER,
        "severity":    "high",
        "type":        "child_victim_safety",
        "description": (
            "English organ-specific injury (ruptured liver) — "
            "replace with 'serious internal injuries'"
        ),
        "suggested":   "serious internal injuries",
        "repetition":  False,
        "child_victim_only": True,
    },
    {
        "id":          "mutilation_terms",
        "pattern":     _RULE_MUTILATION,
        "severity":    "high",
        "type":        "child_victim_safety",
        "description": (
            "Mutilation/dismemberment wording (शरीर के टुकड़े / क्षत-विक्षत / mutilat...) "
            "— must not appear in child-victim narration or metadata"
        ),
        "suggested":   "गंभीर चोटें (keep clinical and minimal)",
        "repetition":  False,
        "child_victim_only": True,
    },
    {
        "id":          "severed_finger_repeated",
        "pattern":     _RULE_SEVERED_FIN,
        "severity":    "medium",
        "type":        "child_victim_safety",
        "description": (
            "Severed-finger detail (कटी हुई उँगली / severed finger) — "
            "may appear once for case mechanics but must not be repeated or used as a hook"
        ),
        "suggested":   "Mention once only; remove from hooks, metadata, and thumbnails",
        "repetition":  True,
        "max_ok":      1,
        "child_victim_only": True,
    },
]


# ─── Full narration extractor ─────────────────────────────────────────────────

def _extract_narration_with_ids(script_draft: dict) -> list[tuple[str, str]]:
    """
    Returns list of (chunk_id, text) pairs from hindi_narration_chunks.
    """
    pairs: list[tuple[str, str]] = []
    for chunk in script_draft.get("hindi_narration_chunks", []):
        cid  = chunk.get("chunk_id", "unknown")
        text = chunk.get("text", "")
        if text:
            pairs.append((cid, text))
    return pairs


def _full_narration_text(script_draft: dict) -> str:
    return " ".join(t for _, t in _extract_narration_with_ids(script_draft))


# ─── Public API ───────────────────────────────────────────────────────────────

def run_hindi_text_lint(
    script_draft: dict,
    hinglish_level: int = 2,
    is_child_victim_case: bool = False,
) -> dict:
    """
    Run all deterministic lint rules over the final Hindi script.

    Returns a lint_report dict:
      total_issues  — int count of all flagged occurrences
      risk_level    — "none" | "low" | "medium" | "high"
      issues        — list of per-match issue dicts
      chunk_targets — list of chunk_repair_targets compatible with copyedit gate schema
      rules_skipped — rules that were skipped based on hinglish_level
    """
    chunk_pairs = _extract_narration_with_ids(script_draft)
    full_text   = " ".join(t for _, t in chunk_pairs)

    issues:       list[dict] = []
    chunk_targets: list[dict] = []
    rules_skipped: list[str] = []

    for rule in _RULES:
        # Skip Canada/Canadian consistency rules for hinglish_level >= 3
        if rule["id"] in ("canada_ke", "canadian_adj") and hinglish_level >= 3:
            rules_skipped.append(rule["id"])
            continue
        # Skip child-victim-only rules when the case is not a child-victim case
        if rule.get("child_victim_only", False) and not is_child_victim_case:
            rules_skipped.append(rule["id"])
            continue

        pattern: re.Pattern = rule["pattern"]

        if rule.get("repetition", False):
            max_ok = rule.get("max_ok", 2)
            hits = _find_repetition(pattern, full_text, max_ok)
        else:
            hits = _find_all_with_context(pattern, full_text)

        for hit in hits:
            # Try to identify which chunk contains this match
            matched_chunk_ids: list[str] = []
            if not rule.get("repetition", False):
                for cid, ctext in chunk_pairs:
                    if pattern.search(ctext):
                        matched_chunk_ids.append(cid)
            else:
                # For repetition rule, flag all chunks that contain the phrase
                for cid, ctext in chunk_pairs:
                    if pattern.search(ctext):
                        matched_chunk_ids.append(cid)

            issue = {
                "rule_id":      rule["id"],
                "issue_type":   rule["type"],
                "severity":     rule["severity"],
                "matched":      hit["matched"],
                "context":      hit["context"],
                "description":  rule["description"],
                "suggested":    rule["suggested"],
                "chunk_ids":    matched_chunk_ids,
            }
            issues.append(issue)

            # Emit a chunk_repair_target for every affected chunk
            for cid in matched_chunk_ids:
                chunk_targets.append({
                    "chunk_id":          cid,
                    "issue_type":        "hindi_copyedit",
                    "problem":           f"[{rule['id']}] {rule['description']} — found: '{hit['matched']}'",
                    "repair_instruction": (
                        f"Replace '{hit['matched']}' with the correct form. "
                        f"Suggested: {rule['suggested']}"
                    ),
                    "severity":          rule["severity"],
                    "lint_rule":         rule["id"],
                })

    # Deduplicate chunk_targets by (chunk_id, lint_rule)
    seen: set[tuple[str, str]] = set()
    deduped_targets: list[dict] = []
    for t in chunk_targets:
        key = (t["chunk_id"], t["lint_rule"])
        if key not in seen:
            seen.add(key)
            deduped_targets.append(t)

    high_count   = sum(1 for i in issues if i["severity"] == "high")
    medium_count = sum(1 for i in issues if i["severity"] == "medium")
    total        = len(issues)

    if total == 0:
        risk_level = "none"
    elif high_count > 0:
        risk_level = "high"
    elif medium_count > 2:
        risk_level = "medium"
    else:
        risk_level = "low"

    report = {
        "total_issues":    total,
        "high_issues":     high_count,
        "medium_issues":   medium_count,
        "low_issues":      sum(1 for i in issues if i["severity"] == "low"),
        "risk_level":      risk_level,
        "hinglish_level":  hinglish_level,
        "issues":          issues,
        "chunk_targets":   deduped_targets,
        "rules_skipped":   rules_skipped,
    }

    logger.info(
        "Hindi lint: %d issues (high=%d medium=%d low=%d) risk=%s",
        total, high_count, medium_count, report["low_issues"], risk_level,
    )
    return report
