#!/usr/bin/env bash
# export_clean_zip.sh — macOS/Linux equivalent of export_clean_zip.ps1
#
# Creates a clean ZIP of the Agatsya Automation repo for sharing with reviewers.
#
# EXCLUDES:
#   .env            — API keys and secrets (NEVER share)
#   .venv/          — local Python virtual environment
#   .git/           — git history (large; reviewers don't need it)
#   app/storage/    — generated episode outputs (can be large)
#   input/          — local input transcripts
#   .pytest_cache/  — pytest artefacts
#   __pycache__/    — compiled bytecode
#   __MACOSX/       — macOS metadata noise
#   .DS_Store       — macOS folder metadata
#
# USAGE (from the repo root):
#   bash scripts/export_clean_zip.sh
#   bash scripts/export_clean_zip.sh agatsya-review.zip
#
# SECURITY REMINDER:
#   - Verify the ZIP does NOT contain .env before sending.
#   - If .env was ever accidentally zipped or shared, rotate all API keys immediately.
#   - Never commit .env to git.

set -euo pipefail

OUTPUT="${1:-agatsya-clean.zip}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Agatsya Automation — Clean Export"
echo "Source : $REPO_ROOT"
echo "Output : $OUTPUT"
echo ""

# Safety check: abort immediately if .env exists and would be included
if find "$REPO_ROOT" -maxdepth 1 -name ".env" | grep -q .; then
    echo "ABORT: .env found in repo root. Remove or exclude it before zipping." >&2
    exit 1
fi

# Remove existing output
rm -f "$OUTPUT"

# Build ZIP using zip(1), excluding sensitive and generated paths
zip -r "$OUTPUT" "$REPO_ROOT" \
    --exclude "*.env" \
    --exclude "*/.env" \
    --exclude "*/.env.*" \
    --exclude "*.pyc" \
    --exclude "*.pyo" \
    --exclude "*/.venv/*" \
    --exclude "*/venv/*" \
    --exclude "*/.git/*" \
    --exclude "*/app/storage/*" \
    --exclude "*/input/*" \
    --exclude "*/.pytest_cache/*" \
    --exclude "*/__pycache__/*" \
    --exclude "*/__MACOSX/*" \
    --exclude "*/.DS_Store" \
    --exclude "*/node_modules/*"

# Final safety check: ensure .env is NOT in the archive
if unzip -l "$OUTPUT" 2>/dev/null | grep -qE "(^|\s)\.env($|\s)"; then
    echo "" >&2
    echo "ABORT: .env was included in $OUTPUT. Deleting ZIP now." >&2
    rm -f "$OUTPUT"
    exit 1
fi

SIZE_MB=$(du -sh "$OUTPUT" | cut -f1)
echo ""
echo "Done: $OUTPUT ($SIZE_MB)"
echo ""
echo "SECURITY CHECKLIST before sending:"
echo "  [ ] Verify .env is NOT in the ZIP:      unzip -l $OUTPUT | grep -E '\\.env'"
echo "  [ ] Verify app/storage/ is NOT present: unzip -l $OUTPUT | grep 'storage/'"
echo "  [ ] Verify .venv/ is NOT present:       unzip -l $OUTPUT | grep '.venv/'"
echo "  [ ] Verify .git/ is NOT present:        unzip -l $OUTPUT | grep '.git/'"
echo "  [ ] If .env was ever shared accidentally, rotate all API keys now"
