#!/usr/bin/env python3
"""
Test the transcript cleaner locally — no API calls, no cost.

Reads:    input/test_transcript.txt   (also tries input/test_transscript.txt)
Outputs:  tmp/clean_transcript_preview.txt
          tmp/transcript_cleanup_report.json

Usage:
    python3 scripts/test_transcript_cleanup.py
    python3 scripts/test_transcript_cleanup.py path/to/your_transcript.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from project root without installing the package
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from app.services.transcript_cleaner_service import (
    clean_transcript,
    _LEFTOVER_JUNK_TERMS,
)

# ── Candidate input paths (in priority order) ─────────────────────────────────
_DEFAULT_INPUTS = [
    _ROOT / "input" / "test_transcript.txt",
    _ROOT / "input" / "test_transscript.txt",   # typo variant
]

# Output directory
_TMP_DIR = _ROOT / "tmp"


def _find_input(argv: list[str]) -> Path:
    if len(argv) > 1:
        p = Path(argv[1])
        if not p.is_absolute():
            p = _ROOT / p
        if p.exists():
            return p
        print(f"ERROR: file not found: {p}")
        sys.exit(1)

    for candidate in _DEFAULT_INPUTS:
        if candidate.exists():
            return candidate

    print("ERROR: no test transcript found. Tried:")
    for c in _DEFAULT_INPUTS:
        print(f"  {c}")
    print("Run from the project root or pass a path as the first argument.")
    sys.exit(1)


def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


def main() -> None:
    input_path = _find_input(sys.argv)
    _TMP_DIR.mkdir(parents=True, exist_ok=True)

    out_clean = _TMP_DIR / "clean_transcript_preview.txt"
    out_report = _TMP_DIR / "transcript_cleanup_report.json"

    print(f"Input  : {input_path}")
    print(f"Output : {out_clean}")
    print(f"Report : {out_report}")

    raw = input_path.read_text(encoding="utf-8")

    # Run the cleaner
    clean = clean_transcript(raw, report_path=out_report)
    out_clean.write_text(clean, encoding="utf-8")

    # Load report written by the cleaner
    report = json.loads(out_report.read_text(encoding="utf-8"))

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_section("CLEANUP SUMMARY")
    raw_chars   = report["raw_chars"]
    clean_chars = report["clean_chars"]
    removed_chars = report["removed_chars"]
    removed_pct   = report["removed_pct"]
    print(f"  Raw chars     : {raw_chars:,}")
    print(f"  Clean chars   : {clean_chars:,}")
    print(f"  Removed chars : {removed_chars:,}  ({removed_pct:.1f}%)")
    print(f"  Sponsor blocks removed : {report['sponsor_blocks_removed']}")
    print(f"  Outro blocks removed   : {report['outro_blocks_removed']}")

    if report["warnings"]:
        _print_section("WARNINGS")
        for w in report["warnings"]:
            print(f"  ⚠  {w}")
    else:
        print("\n  ✓  No warnings")

    if report["removed_markers"]:
        _print_section("REMOVED BLOCKS (first 5)")
        for m in report["removed_markers"][:5]:
            print(f"  • {m}")

    # ── Leftover junk check ───────────────────────────────────────────────────
    _print_section("LEFTOVER JUNK CHECK")
    lower_clean = clean.lower()
    all_clear = True
    for term in _LEFTOVER_JUNK_TERMS:
        present = term in lower_clean
        status = "✗  STILL PRESENT" if present else "✓  removed"
        print(f"  {status:<20}  {term!r}")
        if present:
            all_clear = False

    # ── Patreon / supporter check (not in leftover list but task-requested) ──
    for extra in ("patreon", "supporters"):
        present = extra in lower_clean
        status = "✗  STILL PRESENT" if present else "✓  removed"
        print(f"  {status:<20}  {extra!r}")
        if present:
            all_clear = False

    # ── First 300 chars of clean output ──────────────────────────────────────
    _print_section("FIRST 300 CHARS OF CLEANED TRANSCRIPT")
    print(clean[:300])

    # ── Last 300 chars of clean output ───────────────────────────────────────
    _print_section("LAST 300 CHARS OF CLEANED TRANSCRIPT")
    print(clean[-300:])

    # ── Final verdict ─────────────────────────────────────────────────────────
    _print_section("RESULT")
    if all_clear and not report["warnings"]:
        print("  ✓  Clean — no junk terms remain, no warnings.")
    elif all_clear:
        print("  ✓  Junk removed — but check warnings above.")
    else:
        print("  ⚠  Some junk terms remain. Review the report and refine patterns.")
    print(f"\nFull preview → {out_clean}")
    print(f"Full report  → {out_report}\n")


if __name__ == "__main__":
    main()
