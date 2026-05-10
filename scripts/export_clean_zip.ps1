# export_clean_zip.ps1
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
# USAGE (from the repo root, in PowerShell):
#   .\scripts\export_clean_zip.ps1
#   .\scripts\export_clean_zip.ps1 -OutputPath "C:\Shared\agatsya-review.zip"
#
# SECURITY REMINDER:
#   - Verify the ZIP does NOT contain .env before sending.
#   - If .env was ever accidentally zipped or shared, rotate all API keys immediately.
#   - Never commit .env to git.

param(
    [string]$OutputPath = "agatsya-clean.zip"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Agatsya Automation — Clean Export"
Write-Host "Source : $repoRoot"
Write-Host "Output : $OutputPath"
Write-Host ""

# Collect files, excluding sensitive and generated content
$excludePatterns = @(
    "*.env",
    ".env",
    ".env.*",
    "*.pyc",
    "*.pyo"
)

$excludeDirs = @(
    ".venv",
    "venv",
    ".git",
    "app\storage",
    "app/storage",
    "input",
    ".pytest_cache",
    "__pycache__",
    "__MACOSX",
    "node_modules"
)

# Build the exclusion filter for Get-ChildItem
$files = Get-ChildItem -Path $repoRoot -Recurse -File | Where-Object {
    $relativePath = $_.FullName.Substring($repoRoot.Length + 1)
    $skip = $false

    # Skip excluded directories
    foreach ($dir in $excludeDirs) {
        $normalized = $dir.Replace("/", "\")
        if ($relativePath.StartsWith($normalized + "\") -or $relativePath -eq $normalized) {
            $skip = $true
            break
        }
    }

    # Skip .DS_Store files
    if ($_.Name -eq ".DS_Store") { $skip = $true }

    # Skip .env files (any name starting with .env)
    if ($_.Name -match "^\.env") { $skip = $true }

    -not $skip
}

# Safety check: abort if .env would be included
$envFiles = $files | Where-Object { $_.Name -match "^\.env" }
if ($envFiles) {
    Write-Error "ABORT: .env file(s) would be included in ZIP. Check exclusion logic."
    exit 1
}

Write-Host "Files to include: $($files.Count)"

# Remove existing output if present
if (Test-Path $OutputPath) {
    Remove-Item $OutputPath -Force
}

# Create the ZIP
Add-Type -Assembly System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($OutputPath, 'Create')

foreach ($file in $files) {
    $relativePath = $file.FullName.Substring($repoRoot.Length + 1)
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
        $zip, $file.FullName, $relativePath
    ) | Out-Null
}

$zip.Dispose()

Write-Host ""
Write-Host "Done: $OutputPath ($([Math]::Round((Get-Item $OutputPath).Length / 1MB, 2)) MB)"
Write-Host ""
Write-Host "SECURITY CHECKLIST before sending:"
Write-Host "  [ ] Verify .env is NOT in the ZIP"
Write-Host "  [ ] Verify app/storage/ is NOT in the ZIP"
Write-Host "  [ ] If .env was ever shared accidentally, rotate all API keys now"
