# Agatsya Automation

AI-powered Hindi true-crime YouTube production engine.  
One YouTube source → complete episode production package in pure Hindi.

---

## Architecture

```
SOURCE INPUT
  → POST /api/episodes/package

     [1] Transcript Cleaner     Python — strip timestamp noise, preserve spoken content
     [2] Fact Lock Agent        Claude — extract verified facts, dates, legal outcome
     [3] Story Blueprint Agent  Claude — classify story type, plan narrative structure
     [4] Script Writer Pipeline (5 internal stages):
         [4a] Script Outline Agent      Claude — plan 12–16 chunk specs from fact_lock + blueprint
         [4b] Narration Chunk Writer    Claude — write one chunk at a time (retryable)
         [4c] Recreated Dialogue Agent  Claude — short labelled dialogue scenes
         [4d] Metadata Agent            Claude — minimal YouTube titles + description
         [4e] Script Assembler          Python — combine all pieces into script_draft.json
     [5] Quality Critic Agent   Claude — review against fact_lock; approve or list per-chunk issues
     [6] Targeted Chunk Repair  Claude — repair only flagged chunks, one chunk at a time (one pass max)
     [7] Final Quality Review   Claude — post-repair check (only if repair ran)

  → SCRIPT PACKAGE (saved to disk: 02-facts/, 03-script/, 02-package/)

  → POST /api/episodes/video-plan  (after human script review)
     [8] Video Plan Agent       Claude — scene plan, asset keywords, Shorts, full metadata

  → POST /api/episodes/full (optional stages)
     → VOICE GENERATION (ElevenLabs)
     → VISUAL ASSET FETCH (Pexels/Pixabay)
     → ASSET GUARDRAILS
     → VIDEO RENDERER (FFmpeg draft)
```

### Why controlled multi-agent instead of one giant prompt?

| Problem with single-call | How multi-agent solves it |
|---|---|
| Claude hallucinates facts when writing and planning simultaneously | Fact Lock runs first — script writer gets verified facts only |
| Every case forced into same structure | Story Blueprint classifies story type and adapts narrative plan |
| No way to catch mistakes before spending ElevenLabs credits | Quality Critic blocks bad scripts before audio generation |
| Truncated JSON on long episodes | Each agent produces a focused, smaller JSON payload |
| Hard to debug what went wrong | Each agent saves its raw response for inspection |

---

## Cost Modes

Every episode request includes a `cost_mode` field that controls how Claude plans the video and which production tools are used.

| Mode | Default | Use when |
|---|---|---|
| `bootstrap` | **Yes** | Starting out, testing, low-budget production |
| `standard` | No | Channel is growing, a few hero AI video scenes OK |
| `premium` | No | Full production quality, AI video throughout |

### Bootstrap (recommended for early production)

Bootstrap is low-cost because it skips ElevenLabs, AI video generation, asset fetching, and rendering by default — not because it uses fewer Claude calls. It still runs the full 7-agent pipeline for script quality.

- Runs the full 7-agent pipeline (fact lock → blueprint → script → critic → repair) using Claude Sonnet
- No AI video clips — uses template cards, stock images, AI stills, captions
- No ElevenLabs audio, no asset API calls, no video rendering (unless explicitly enabled)
- Max 5 AI still images, max 20 real asset keywords
- Prefers: `template_intro`, `timeline_card`, `recreated_call`, `document_card`, `court_card`, `location_context`
- Ideal episode length: 15–22 minutes
- Works with only `ANTHROPIC_API_KEY` set

### Standard

- Allows 1–3 AI hero video clips for pivotal moments
- Max 10 AI still images, max 30 real assets
- Higher visual variety — still cost-conscious

### Premium

This is the primary production mode for Agatsya Automation. All new episodes use `cost_mode: "premium"` by default.

- Up to 10 AI video clips, up to 20 AI still images
- All safety and dignity rules apply
- **Two-layer review pipeline** — Claude produces and self-checks; OpenAI acts as independent senior editor and safety reviewer
- `safe_to_voice` is only `true` when ALL gates pass — never generate audio otherwise

**The `estimated_cost_policy.json` file is written to `06-review/` for every episode.**

### Premium Gate Pipeline

Premium mode runs two review layers after script production:

**Claude layer** — production and self-review:
1. Fact Lock Agent (auto-switches to `segmented` mode for long premium transcripts ≥ 30 000 chars)
2. Story Blueprint Agent
3. **Retention Blueprint Agent** (premium only — designs viewer experience arc, curiosity gaps, re-engagement moments, shorts candidates)
4. Hindi Script Writer (uses retention blueprint to place retention goals per chunk)
5. Script Quality Critic (Claude + Python score enforcement)
6. Targeted Chunk Repair (if needed, one pass)
7. Final Script Quality Review (after repair)
8. Hindi Text Lint (Python — deterministic rules, free)
9. Hindi Copyedit Gate (Claude + Python)
10. Copyedit Targeted Repair (if needed, one pass + re-check)
11. **Retention Quality Gate** (Claude + Python — 8 scoring dimensions; premium only when retention blueprint present)
12. Retention Targeted Repair (if needed, one pass + re-check)
13. Text Similarity Check (Python — phrase matching, free)
14. Originality Safety Gate (Claude + Python)
15. Recreated Dialogue Gate (Claude + Python)
16. Metadata Quality Gate (Claude + Python — 9 scoring dimensions including CTR, thumbnail, curiosity, originality)
17. Metadata Repair (Claude — if gate 16 fails, one targeted pass repairing only youtube_metadata)
18. Recheck Metadata Gate (Claude — after repair)

**OpenAI/ChatGPT layer** — independent senior editor, safety reviewer, and targeted fixer:

19. **Premium Hindi Editor Gate** (GPT) — independent second opinion on grammar, matra/nasalization, naturalness, Hinglish consistency, legal clarity, flow, and a secondary retention flow check
20. **Originality & YouTube Risk Gate** (GPT) — independent copyright risk, reuse risk, ad-safety, and victim dignity review
21. **OpenAI Targeted Chunk Repair** (GPT) — if either gate 19 or 20 fails with specific chunk targets, repair only those chunks (max `OPENAI_REPAIR_MAX_CHUNKS=6`); then recheck the failed gate(s) once
    - If too many chunks need repair (> max), no repair runs → `needs_human_review`
    - If repair fails for any chunk → original kept, `openai_repair_failed=true`, `safe_to_voice=false`
    - For broad source-copying risk with no specific chunk targets → no repair, `needs_human_review`

Claude produces and self-checks. OpenAI independently evaluates the final output, repairs targeted issues, then rechecks — acting as a fresh pair of eyes at every step.

**`safe_to_voice` is the authoritative signal.** It is `true` only when ALL of these hold:

| # | Gate | Model | Key thresholds |
|---|---|---|---|
| 1 | Script Quality | Claude + Python | All premium scores met |
| 2 | Hindi Text Lint | Python (free) | chandrabindu, gender, fragments, phrasing |
| 3 | Hindi Copyedit | Claude + Python | `overall ≥ 9`, `grammar ≥ 9`, `matra ≥ 9`, `flow ≥ 9`, no HIGH issues |
| 4 | Retention Quality | Claude + Python | `overall ≥ 9`, `hook ≥ 9`, `arc ≥ 9`, `curiosity ≥ 8`, `pacing ≥ 8`, `ending ≥ 8`, no HIGH issues |
| 5 | Originality Safety | Claude + Python | `copying_risk ≤ 2`, `transformative ≥ 9`, `ad_safety ≥ 9` |
| 6 | Recreated Dialogue | Claude + Python | `overall ≥ 9`, `labelling ≥ 9`, `dignity ≥ 9` |
| 7 | Metadata Quality | Claude + Python | `monetization = 10`, `clickability ≥ 8`, `copyright_risk ≤ 2`, `ctr ≥ 7`, `thumbnail ≥ 7`, `curiosity ≥ 7`, `originality ≥ 7` |
| 8 | OpenAI Hindi Editor | GPT + Python | `overall ≥ 9`, `grammar ≥ 9`, `matra ≥ 9`, `flow ≥ 9`, no HIGH issues |
| 9 | OpenAI Originality/YT Risk | GPT + Python | `copying ≤ 2`, `transformative ≥ 9`, `yt_safety ≥ 9`, `metadata ≥ 9` |
| — | Zero repair failures | Python | `claude_script_repair_failed=false`, `copyedit_repair_failed=false`, `metadata_repair_failed=false`, `retention_repair_failed=false`, `openai_repair_failed=false` |

Gate 4 (Retention Quality) is only active when a Retention Blueprint was successfully generated. If the Retention Blueprint agent fails, the gate is marked `skipped` and does not block approval.

If `OPENAI_REVIEW_ENABLED=false`, gates 8–9 are marked `skipped` and do NOT block approval (not recommended for production).

**ElevenLabs must never be run when `safe_to_voice=false`.** ElevenLabs credits are non-refundable. The `safe_to_voice` field is the authoritative signal — it is `true` only when every gate passes and zero repair failures occurred.

**POST `/api/episodes/full` is not production-ready** — set `ENABLE_FULL_PIPELINE=true` in `.env` only after confirming `safe_to_voice=true` from `/api/episodes/package`.

**Automatic repair behaviour:**
- Hindi Copyedit gate fails → Claude targeted chunk repair → recheck once
- Metadata Quality gate fails → Claude metadata-only repair (fixes youtube_metadata only, does not touch narration) → recheck once
- Retention Quality gate fails with `chunk_repair_targets` → Claude retention repair → recheck once
- Retention Quality gate fails without chunk targets → no automated repair → `needs_human_review`
- OpenAI Hindi Editor or Originality gate fails with `chunk_repair_targets` → OpenAI targeted chunk repair (max 6 chunks) → recheck the failed gate(s) once
- Broad source-copying risk (no specific chunk targets) → no automated repair → `needs_human_review`
- All other gate failures → read `required_fixes` in the gate report → fix manually

Gate reports are saved to `04-review/`:
- `hindi_text_lint_report.json`
- `hindi_copyedit_report.json` + `hindi_copyedit_repair_report.json` (if repair ran)
- `retention_quality_report.json` (if retention blueprint present; overwritten with recheck result if repair ran)
- `retention_repair_targets.json` + `retention_repair_report.json` (if retention repair ran)
- `text_similarity_report.json`
- `originality_safety_gate_report.json`
- `recreated_dialogue_gate_report.json`
- `metadata_quality_gate_report.json` (overwritten with recheck result if repair ran)
- `_metadata_repair_raw_response.txt` + `metadata_repair_report.json` (if metadata repair ran)
- `openai_premium_hindi_editor_report.json` (overwritten with recheck result if repair ran)
- `openai_originality_youtube_risk_report.json` (overwritten with recheck result if repair ran)
- `openai_repair_targets.json` (if OpenAI repair ran)
- `openai_repair_report.json` (if OpenAI repair ran)

**Shorts strategy** — when the Retention Blueprint is generated, a `shorts_strategy.json` file is written to `02-package/` containing the shorts candidates with hooks, source sections, and CTAs to the full episode. Shorts are curiosity-based or investigatively interesting — never exploitative.

### OpenAI configuration

```bash
OPENAI_API_KEY=sk-...
OPENAI_REVIEW_MODEL=gpt-5.5
OPENAI_REVIEW_ENABLED=true
OPENAI_REPAIR_ENABLED=true
OPENAI_REPAIR_MAX_CHUNKS=6
```

| Var | Default | Effect |
|---|---|---|
| `OPENAI_REVIEW_ENABLED` | `true` | Run gates 14–15. Set `false` to skip (marks as skipped, not blocking). |
| `OPENAI_REPAIR_ENABLED` | `true` | Run OpenAI targeted repair if gates 14–15 fail with chunk targets. |
| `OPENAI_REPAIR_MAX_CHUNKS` | `6` | If more chunks need repair than this limit, skip repair and set `needs_human_review`. |

If `OPENAI_API_KEY` is missing when `OPENAI_REVIEW_ENABLED=true`, the pipeline sets `status=needs_human_review` and `safe_to_voice=false` without crashing. Warning: `"OpenAI review enabled but OPENAI_API_KEY is missing. Do not run ElevenLabs."`

### `gate_summary` format

The API response includes a `gate_summary` object with one entry per gate plus a `repair_failures` audit block and a top-level `safe_to_voice` flag:

```json
{
  "script_quality":                  { "passed": true,  "scores": {...} },
  "hindi_copyedit":                  { "passed": true,  "score": 9, "grammar_score": 9, ... },
  "retention_quality":               { "passed": true,  "overall_retention_score": 9, "opening_hook_score": 9, "recheck": false },
  "originality_safety":              { "passed": true,  "scores": {...} },
  "recreated_dialogue":              { "passed": true,  "scores": {...} },
  "metadata_quality":                { "passed": true,  "scores": {...} },
  "openai_premium_hindi_editor":     { "passed": true,  "overall_score": 9, "recheck": true },
  "openai_originality_youtube_risk": { "passed": true,  "copying_risk": 1,  "recheck": false },
  "repair_failures": {
    "claude_script_repair_failed": false,
    "copyedit_repair_failed":      false,
    "metadata_repair_failed":      false,
    "retention_repair_failed":     false,
    "openai_repair_failed":        false,
    "passed":                      true
  },
  "safe_to_voice": true
}
```

`recheck: true` means the gate was re-run after targeted repair. The report file contains the recheck result.

When no Retention Blueprint was generated (standard mode or blueprint failure), `retention_quality` is `{ "passed": true, "skipped": true, "reason": "..." }` and does not block approval.

> **No automated quality guarantee.** This pipeline significantly reduces risk but cannot guarantee YouTube monetization approval, copyright safety, or zero strikes. Final human review is always recommended before publishing. Claude and GPT reduce the probability of obvious issues — they do not provide legal or contractual guarantees.

---

## Quickstart

### 1. Clone and configure

```bash
git clone <repo>
cd agatsya-automation

cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY at minimum
```

### 2. Run with Docker (recommended)

```bash
docker compose up --build
```

Server starts at `http://localhost:8000`.

### 3. Verify the server is up

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"ok","claude_model":"claude-sonnet-4-6","voice_enabled":false,"pexels_enabled":false,"pixabay_enabled":false}
```

### 4. Generate a production package

The sample payload uses `cost_mode: premium` — the default and required production mode.

```bash
curl -X POST http://localhost:8000/api/episodes/package \
  -H "Content-Type: application/json" \
  -d @samples/create_episode_payload.json
```

On success the response contains `episode_dir` and `files` — a list of every file written to disk.

To pretty-print (requires `jq`):

```bash
curl -X POST http://localhost:8000/api/episodes/package \
  -H "Content-Type: application/json" \
  -d @samples/create_episode_payload.json | jq .
```

To test a different cost mode inline:

```bash
curl -X POST http://localhost:8000/api/episodes/package \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=5bttM6SYuLE",
    "episode_number": "001",
    "case_hint": "Meika Jordan",
    "target_duration_min": 22,
    "cost_mode": "standard",
    "raw_transcript": "Meika Jordan was a six-year-old girl from Calgary..."
  }'
```

### 3. Run locally (no Docker)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # fill in your keys

uvicorn app.main:app --reload --port 8000
```

---

## Transcript Handling

Agatsya Automation handles long transcripts without sending the full text to Claude on every call.

### How it works

| File | What it contains |
|---|---|
| `01-input/source_transcript.txt` | Full original transcript — saved verbatim, never modified |
| `01-input/transcript_research_view.txt` | Compressed research view actually sent to Claude |

The research view splits the transcript budget into three sections:

```
[BEGINNING EXCERPT]   40% of budget  ← setup, victim introduction, early facts
[MIDDLE EXCERPT]      20% of budget  ← investigation, key events
[ENDING EXCERPT]      40% of budget  ← verdict, appeal, legal outcome, final twist
```

If the transcript fits within the budget, it is sent in full with no compression.

### Why ending gets the same weight as beginning

True-crime cases turn on their ending — verdict, appeal result, Supreme Court ruling, final revelation. Simple beginning-only truncation misses these. The beginning/middle/end split preserves all three critical zones within the same token budget.

### Configuration

Default budget is 18,000 characters (~4,500 words, roughly sufficient for most podcast transcripts).

To increase (for very long transcripts):

```env
# .env
TRANSCRIPT_RESEARCH_CHARS=24000
```

To decrease (for cost reduction in testing):

```env
TRANSCRIPT_RESEARCH_CHARS=12000
```

### If the output is missing case details

1. Check `01-input/transcript_research_view.txt` — confirm the key facts appear in the beginning, middle, or ending section
2. If a critical fact falls in a compressed gap, it will appear in `facts_to_verify` in `02-package/case_summary.json` — Claude is instructed to flag missing facts rather than invent them
3. Increase `TRANSCRIPT_RESEARCH_CHARS` and re-run

---

## Running a premium test

### Running a clean premium test (OpenAI Final Premium Gate)

Recommended `.env` for a **fresh, fully-gated** premium run — no stage reuse, all Claude and OpenAI gates active, combined Final Premium Gate enabled:

```env
QUALITY_MODE=premium_final
OPENAI_REVIEW_POLICY=adaptive
OPENAI_REVIEW_ENABLED=true
SKIP_FINAL_GATES=false
REUSE_EXISTING_STAGE_OUTPUTS=false
MAX_TOTAL_MODEL_CALLS=80
CLAUDE_PROMPT_CACHE_ENABLED=true
# Required — the Final Premium Gate is an OpenAI call:
OPENAI_API_KEY=sk-...
```

**What the Final Premium Gate does**: a single OpenAI call that scores the fully-assembled script across 8 dimensions (Hindi grammar, Hinglish level, retention, originality, YouTube safety, metadata completeness, recreated dialogue safety, and safe-to-voice sign-off). Every upstream Claude gate's result is sent as compact evidence so OpenAI can make an informed independent verdict.

**All 7 scores must be ≥ 9** for `approved=true` and `safe_to_voice=true`. Any score below 9 or any HIGH-severity issue blocks approval.

> **Always run with `REUSE_EXISTING_STAGE_OUTPUTS=false`** for a final pre-production test. Stale cached outputs from a previous run can pass the gate on outdated content.

> **Do not run ElevenLabs unless the response shows:**
> ```json
> "status": "script_approved",
> "safe_to_voice": true
> ```
> Both fields must be true. `status=script_approved` alone is not sufficient — the OpenAI Final Premium Gate may still have blocked `safe_to_voice`.

#### What happens if the OpenAI Final Premium Gate fails and repair runs

If the gate fails but returns `chunk_repair_targets`, the pipeline automatically:
1. Runs OpenAI targeted chunk repair (Stage 16)
2. Refreshes Hindi lint + text similarity on the repaired script
3. Reloads all upstream gate reports from disk
4. Rechecks the Final Premium Gate once (Stage 16a) — saved as `openai_final_premium_report_after_repair.json`

If the recheck also fails: `status=needs_human_review`, `safe_to_voice=false`. No further automatic repair runs. The original first-pass report is preserved at `openai_final_premium_report.json` for comparison.

### Running a cheaper debug rerun

For re-running when only a later stage changed and earlier outputs are still valid:

```env
REUSE_EXISTING_STAGE_OUTPUTS=true
QUALITY_MODE=premium_build    # skip OpenAI entirely
SKIP_FINAL_GATES=false        # still run Claude gates
```

> If the transcript or `hinglish_level` changed since the last run, the stage manifest will warn you that cached outputs may be stale. Set `REUSE_EXISTING_STAGE_OUTPUTS=false` for a full re-run.

---

## Troubleshooting

### Package generation fails / JSON parse error

Every agent saves its raw Claude response **before** parsing. If a stage fails, check the corresponding file:

```
app/storage/episodes/<folder>/02-facts/_fact_lock_raw_response.txt
app/storage/episodes/<folder>/02-facts/_story_blueprint_raw_response.txt
app/storage/episodes/<folder>/03-script/_script_writer_raw_response.txt
app/storage/episodes/<folder>/04-review/_script_quality_raw_response.txt
```

The error message names the agent that failed, e.g.:
```
Script Writer Agent JSON parse failed: ...
Raw response saved at: app/storage/episodes/001-meika-jordan/03-script/_script_writer_raw_response.txt
```

Common causes:
- `CLAUDE_MAX_TOKENS` too low → Claude truncated mid-JSON. Raise to `12000` or higher in `.env`.
- Model returned markdown instead of pure JSON → check the corresponding prompt file in `app/prompts/`.

### Script is `needs_human_review`

Read the quality report to see what failed:

```bash
cat app/storage/episodes/<folder>/04-review/script_quality_report.json
```

The `chunk_repair_targets` array in `script_quality_report.json` shows exactly which chunks need fixing and why.
The `script_repair_report.json` in `04-review/` shows what the targeted repair did to each chunk.

To investigate a specific chunk failure:
1. Check `04-review/chunk_repair_targets.json` for the repair instruction
2. Check `03-script/chunks/_repair_raw_<chunk_id>.txt` for the raw Claude response
3. Edit `03-script/hindi_narration_chunks.json` directly for a manual fix, or
4. Re-run the endpoint — the pipeline will re-run repair on a fresh draft

### Container healthcheck failing

Make sure `curl` is installed in the image (it is, as of this build). Rebuild if upgrading from an older image:

```bash
docker compose up --build --force-recreate
```

### `safe_to_voice: false` with `OPENAI_API_KEY` missing

If OpenAI gates are enabled but `OPENAI_API_KEY` is not set, the pipeline will:
- Skip stages 14–15 (OpenAI Hindi Editor and Originality gates)
- Set the Hindi Editor gate to `passed: false, reason: "OPENAI_API_KEY missing"`
- Force `status=needs_human_review` and `safe_to_voice=false`

This is intentional — without the second layer of review, the pipeline cannot confirm audio readiness.

**To fix:** Add your OpenAI key to `.env`:
```bash
OPENAI_API_KEY=sk-...
```

**If you want to run without OpenAI review** — use `QUALITY_MODE` or `OPENAI_REVIEW_POLICY`:
```bash
# Reduce to one OpenAI call (Hindi Editor only — default, recommended)
OPENAI_REVIEW_POLICY=adaptive

# Skip all OpenAI gates — does not block approval but lowers confidence
OPENAI_REVIEW_POLICY=disabled

# Or skip all OpenAI gates via quality mode (no OpenAI, no voice/video)
QUALITY_MODE=premium_build   # Claude + Python only, debug mode — NOT voice-ready
QUALITY_MODE=premium_batch   # bulk candidate evaluation — NOT voice-ready
```

With `QUALITY_MODE=premium_build` or `premium_batch`, or `OPENAI_REVIEW_POLICY=disabled`, all OpenAI gates are marked `skipped` and do not block `safe_to_voice`. The script must still pass all Claude gates.

### ElevenLabs / Pexels / Pixabay not configured

`POST /api/episodes/package` works with **only** `ANTHROPIC_API_KEY` set. The other keys are needed only when calling `POST /api/episodes/full` with `enable_voice`, `enable_assets`, or `enable_render` set to `true`.

---

## Bootstrap Production Flow (Recommended)

### Stage 1 — Generate the script package

```bash
# Build the payload (if you haven't already)
python3 scripts/build_payload.py

# Run the controlled multi-agent pipeline
curl -X POST http://localhost:8000/api/episodes/package \
  -H "Content-Type: application/json" \
  -d @samples/create_episode_full_payload.json | jq .
```

The pipeline runs 18–25 focused Claude calls internally (1 fact lock + 1 blueprint + 1 outline + 12–16 chunks + 1 dialogue + 1 metadata + 1–2 review/repair + 3 premium gates) and returns:

```json
{
  "episode_id": "001-meika-jordan",
  "status": "script_approved",
  "safe_to_voice": true,
  "quality_summary": {
    "approved": true,
    "scores": {
      "factual_accuracy": 10,
      "safety": 10,
      "monetization_safety": 10,
      "hindi_naturalness": 9,
      "story_structure": 9,
      "retention_hook": 9,
      "emotional_depth": 9,
      "recreated_scene_quality": 10
    },
    "estimated_word_count": 2940,
    "estimated_duration_min": 24.5,
    "repair_required": false
  },
  "gate_summary": {
    "originality_safety":  { "passed": true,  "scores": { ... } },
    "recreated_dialogue":  { "passed": true,  "no_scenes": false, "scores": { ... } },
    "metadata_quality":    { "passed": true,  "scores": { ... } }
  },
  "files": { ... },
  "warnings": []
}
```

Status values:
- `script_approved` — Quality Critic approved, all chunk repairs succeeded
- `needs_human_review` — Script generated but quality issues remain; review before audio
- `failed` — Pipeline error; check logs

**`safe_to_voice`** — `true` only in premium mode when ALL of the following hold:
- `status == "script_approved"`
- All 5 quality gates passed (script quality, Hindi copyedit, originality, recreated dialogue, metadata)
- Zero script repair failures (`script_repair_report.chunks_failed == 0`)
- Zero copyedit repair failures (`hindi_copyedit_repair_report.chunks_failed == 0`)

Non-premium runs always return `safe_to_voice: false`. **Do not run ElevenLabs unless `safe_to_voice: true`.**

### Stage 2 — Review the script

```bash
# Read the narration text
cat app/storage/episodes/001-meika-jordan/03-script/hindi_narration_full.txt

# Review the quality report
cat app/storage/episodes/001-meika-jordan/04-review/script_quality_report.json

# Review what facts were extracted
cat app/storage/episodes/001-meika-jordan/02-facts/fact_lock.json

# Review the narrative plan
cat app/storage/episodes/001-meika-jordan/02-facts/story_blueprint.json
```

If `status=needs_human_review`, read the quality report and edit `03-script/hindi_narration_chunks.json` directly before proceeding.

### Stage 3 — Generate video plan

Once the script is approved:

```bash
curl -X POST http://localhost:8000/api/episodes/video-plan \
  -H "Content-Type: application/json" \
  -d '{"episode_id": "001-meika-jordan", "cost_mode": "bootstrap"}' | jq .
```

Generates:
- `episode_video_plan.json` — scene-by-scene visual plan
- `asset_keywords.txt` — searchable keywords for Pexels/Pixabay
- `shorts_plan.json` — YouTube Shorts plan
- `youtube_metadata.json` — full title, description, tags, chapters, pinned comment

### Stage 4 — Audio generation

> ⚠️ **Only run this when `safe_to_voice: true`.** ElevenLabs credits are non-refundable.
>
> `safe_to_voice: true` requires ALL of:
> 1. `status == "script_approved"`
> 2. `hindi_copyedit_report.approved == true`
> 3. `retention_quality_report.approved == true` (or gate skipped — no retention blueprint)
> 4. `originality_safety_gate_report.gate_passed == true`
> 5. `recreated_dialogue_gate_report.gate_passed == true`
> 6. `metadata_quality_gate_report.gate_passed == true`
> 7. Zero script chunk repair failures (`gate_summary.repair_failures.claude_script_repair_failed == false`)
> 8. Zero copyedit repair failures (`gate_summary.repair_failures.copyedit_repair_failed == false`)
> 9. Zero retention repair failures (`gate_summary.repair_failures.retention_repair_failed == false`)
>
> ElevenLabs credits are non-refundable. Never generate audio when `safe_to_voice: false`
> or `status` is `needs_human_review`.
>
> To check all gates quickly:
> ```bash
> cat app/storage/episodes/001-meika-jordan/04-review/hindi_copyedit_report.json | jq .approved
> cat app/storage/episodes/001-meika-jordan/04-review/retention_quality_report.json | jq .approved
> cat app/storage/episodes/001-meika-jordan/04-review/originality_safety_gate_report.json | jq .gate_passed
> cat app/storage/episodes/001-meika-jordan/04-review/recreated_dialogue_gate_report.json | jq .gate_passed
> cat app/storage/episodes/001-meika-jordan/04-review/metadata_quality_gate_report.json | jq .gate_passed
> ```

```bash
curl -X POST http://localhost:8000/api/episodes/full \
  -H "Content-Type: application/json" \
  -d @samples/create_episode_full_payload.json | jq .
```

### Why multi-agent instead of a single prompt?

| Problem | Solution |
|---|---|
| Claude hallucinates facts when writing and planning at once | Fact Lock runs first — script writer gets verified facts only |
| Same template forced on every case | Story Blueprint classifies the case and adapts the narrative structure |
| No quality gate before ElevenLabs costs money | Quality Critic blocks bad scripts before audio generation |
| Long episode JSON truncated mid-output | Each agent produces a small focused JSON payload |
| Can't tell what went wrong | Every agent saves its raw Claude response for inspection |

| `package_level` | What runs |
|---|---|
| `script_first` (default) | Full 7-agent pipeline — fact lock → blueprint → script → review → repair |
| `full_package` | Legacy single-call Claude (for short episodes or testing only) |

> ⚠️ **Do not use `full_package` for 22-minute episodes.** The combined JSON exceeds Claude's output token limit and produces truncated results.

---

## API Endpoints

### GET /health

```bash
curl http://localhost:8000/health
```

Returns which integrations are enabled based on keys present.

---

### POST /api/episodes/package

Generate the full production package using the controlled multi-agent pipeline.  
Does **not** generate audio, fetch assets, or render video.

```bash
curl -X POST http://localhost:8000/api/episodes/package \
  -H "Content-Type: application/json" \
  -d @samples/create_episode_payload.json | jq .
```

**Input fields:**

| Field | Required | Description |
|---|---|---|
| `youtube_url` | Yes | Source video URL (for reference/credit tracking) |
| `episode_number` | Yes | Episode number string, e.g. `"001"` |
| `case_hint` | Yes | Case name, e.g. `"Meika Jordan"` |
| `target_duration_min` | Yes | Target episode length in minutes (15–30) |
| `raw_transcript` | Yes | Source transcript text (used as research, not copied) |
| `style` | No | Style descriptor (default: Agatsya Anand pure Hindi...) |
| `cost_mode` | No | `bootstrap` / `standard` / `premium` (default: `premium`) |
| `hinglish_level` | No | Language style 1–5 (default: `2`). See [Hinglish Level](#hinglish-level) below |
| `enable_gpt_review` | No | `true` to run GPT quality review after Claude (default: `false`) |

**Response:** Episode folder path + list of all generated file paths.

---

### POST /api/episodes/video-plan

Second-stage: generate video scene plan, asset keywords, Shorts plan, and full YouTube metadata from an approved script.

```bash
curl -X POST http://localhost:8000/api/episodes/video-plan \
  -H "Content-Type: application/json" \
  -d @samples/video_plan_payload.json | jq .
```

Input:

| Field | Required | Description |
|---|---|---|
| `episode_id` | Yes | Folder name, e.g. `"001-meika-jordan"` |
| `cost_mode` | No | `bootstrap` / `standard` / `premium` (default: `premium`) |

Reads `hindi_narration_chunks.json`, `recreated_dialogues.json`, and `case_summary.json` from the episode folder. Calls Claude with the video-plan prompt. Writes `episode_video_plan.json`, `asset_keywords.txt`, `shorts_plan.json`, and full `youtube_metadata.json`. Merges everything back into `production_package.json`.

Returns 404 if the episode folder does not exist. Returns 422 if the script files are still deferred placeholders.

---

### POST /api/episodes/full

> ⚠️ **Not production-ready. Use `/api/episodes/package` first.**
>
> `/api/episodes/package` is the recommended first automation endpoint. It runs the full
> script pipeline and returns a reviewable script without spending ElevenLabs credits.
>
> Only call `/api/episodes/full` (with `enable_voice=true`) when ALL of the following are true:
> - `status` is `script_approved`
> - `warnings` array is empty (no chunk repair failures)
> - You have reviewed the script in `03-script/hindi_narration_full.txt`
>
> **Never trigger ElevenLabs audio generation when `status=needs_human_review`.**
> This status means either the Quality Critic rejected the script, or one or more chunk repairs
> failed and the original unrepaired chunks were kept. ElevenLabs credits are non-refundable.

Run the package pipeline, then optionally trigger audio, assets, and rendering.

```bash
curl -X POST http://localhost:8000/api/episodes/full \
  -H "Content-Type: application/json" \
  -d @samples/create_episode_full_payload.json | jq .
```

Additional input fields over `/package`:

| Field | Default | Description |
|---|---|---|
| `enable_voice` | `false` | Generate MP3 audio via ElevenLabs — **only use when `status=script_approved`** |
| `enable_assets` | `false` | Fetch visual assets from Pexels/Pixabay |
| `enable_render` | `false` | Render draft MP4 slideshow via FFmpeg |

---

## Output Folder Structure

```
app/storage/episodes/001-meika-jordan/

  01-input/
    source_transcript.txt          ← raw transcript (unchanged)
    clean_transcript.txt           ← timestamp-stripped spoken content
    transcript_research_view.txt   ← 40/20/40 beginning/middle/end excerpt
    input_payload.json             ← full input parameters

  02-facts/                        ← agent outputs: facts and narrative plan
    fact_lock.json                 ← verified facts, dates, people, legal outcome
    _fact_lock_raw_response.txt    ← raw Claude output (debugging)
    story_blueprint.json           ← story type, hook, sections, sensitivity rules
    _story_blueprint_raw_response.txt

  03-script/                       ← agent outputs: Hindi narration
    script_draft.json              ← script as written by Script Writer agent
    hindi_narration_full_draft.txt ← readable narration text (draft)
    hindi_narration_chunks_draft.json
    recreated_dialogues_draft.json
    elevenlabs_chunks_draft.json
    youtube_metadata_draft.json
    _script_writer_raw_response.txt
    script_final.json              ← final script (repaired or = draft)
    hindi_narration_full.txt       ← final narration (approve this before audio)
    hindi_narration_chunks.json
    recreated_dialogues.json
    elevenlabs_chunks.json
    youtube_metadata.json
    _script_repair_raw_response.txt  ← (only if repair ran)

  04-review/                       ← quality reports
    script_quality_report.json     ← draft review: approved + chunk_repair_targets
    _script_quality_raw_response.txt
    chunk_repair_targets.json      ← (if repair ran) list of chunks to repair
    script_repair_report.json      ← (if repair ran) per-chunk repair results
    hinglish_level_assessment.json ← (if repair ran) requested vs detected level
    final_script_quality_report.json ← (if repair ran) post-repair review

  02-package/                      ← backward-compat copy for video-plan endpoint
    production_package.json        ← = script_final.json
    case_summary.json
    hindi_narration_full.txt
    hindi_narration_chunks.json
    recreated_dialogues.json
    elevenlabs_chunks.json
    youtube_metadata.json
    episode_video_plan.json        ← placeholder until /api/episodes/video-plan
    shorts_plan.json               ← placeholder
    asset_keywords.txt             ← placeholder
    _claude_raw_response.txt       ← = script_writer raw response

  03-audio/
    000_disclaimer.mp3             ← (if enable_voice=true)
    001_hook.mp3
    ...

  04-assets/
    real-candidates/               ← asset metadata JSONs from Pexels/Pixabay
    approved/                      ← manually reviewed & approved assets
    generated/                     ← AI-generated images (future)

  05-renders/
    draft_001-meika-jordan.mp4     ← (if enable_render=true)

  06-review/
    asset_guardrail_policy.json    ← guardrail rules reference
    estimated_cost_policy.json     ← cost mode policy for this episode
    quality_checklist.json         ← QA checklist
    guardrail_results.json         ← (if enable_assets=true)
```

---

## Retention Blueprint and Revenue Optimization

In premium mode, after the Story Blueprint, the pipeline generates a **Retention Blueprint** — a viewer experience arc designed to maximize audience retention, CTR, and subscriber conversion.

**What the Retention Blueprint produces:**
- `opening_hook_strategy` — exact first 5 seconds, first 30 seconds, central question, viewer promise
- `retention_curve` — per 2-minute range: tension level, curiosity gap planted, payoff or transition
- `re_engagement_moments` — specific moments (by minute) that re-hook viewers who are drifting
- `pattern_interrupts` — timeline shifts, audio recreations, court turns, emotional memories
- `subscriber_conversion_moment` — the natural premium moment where a subscribe invitation fits
- `ending_strategy` — how the episode resolves the emotional and factual promise of the opening
- `shorts_candidates` — 2–4 specific moments that work as standalone 40–60 second Shorts clips
- `title_thumbnail_angles` — 3+ title/thumbnail combinations with CTR reasoning and risk notes

**How it feeds the script:**
- The Script Outline Agent uses the retention blueprint to assign per-chunk `retention_goal`, `curiosity_gap`, `viewer_payoff`, and `pattern_interrupt` fields
- The Narration Chunk Writer receives these fields as explicit viewer-experience goals for each chunk
- The Retention Quality Gate scores the final narration across 8 dimensions and triggers targeted repair

**Retention Quality Gate thresholds (premium):**

| Dimension | Threshold |
|---|---|
| `overall_retention_score` | ≥ 9 |
| `opening_hook_score` | ≥ 9 |
| `first_30_seconds_score` | ≥ 9 |
| `emotional_arc_score` | ≥ 9 |
| `curiosity_gap_score` | ≥ 8 |
| `pacing_score` | ≥ 8 |
| `midpoint_retention_score` | ≥ 8 |
| `ending_payoff_score` | ≥ 8 |
| HIGH severity issues | 0 |

**Content rules — non-negotiable:**
- No sensationalism, no outrage-based hooks. Every hook is curiosity-based or empathy-based.
- Thumbnail text: 2–5 words maximum, no graphic or exploitative language
- Titles must not use "भारत की पहली", "सबसे भयानक", "आप यकीन नहीं करेंगे", or unverified superlatives
- Shorts must drive viewers to the full episode — not be self-contained replacements
- Subscriber conversion moment must feel genuine, not transactional

**Output files:**
```
02-facts/
  retention_blueprint.json                 ← retention and revenue optimization plan
  _retention_blueprint_raw_response.txt

02-package/
  shorts_strategy.json                     ← shorts candidates + title/thumbnail angles

04-review/
  retention_quality_report.json            ← gate scores across 8 dimensions
  _retention_quality_raw_response.txt
  retention_repair_targets.json            ← (if repair ran)
  retention_repair_report.json             ← (if repair ran)
```

The Retention Blueprint agent is **non-fatal** — if it fails, the pipeline continues with standard narrative structure and `retention_quality` is marked `skipped` in `gate_summary`.

---

## OpenAI Premium Review Gates

In premium mode, GPT runs as an **independent senior editor and safety reviewer** after all Claude gates have passed. It does not replace Claude — it acts as a second pair of eyes.

The gate behaviour depends on `OPENAI_REVIEW_POLICY`:

**`adaptive` (default)** — Single combined **Final Premium Gate** call covering all dimensions in one pass:
- Hindi grammar / matra / nasalization errors
- Hinglish level consistency (actual vs requested level)
- Retention quality (opening hook, curiosity gaps, re-engagement, ending payoff)
- Originality and content reuse risk
- YouTube monetization safety (title, thumbnail, tags)
- Metadata completeness (≥ 2 thumbnail_options, 15–25 tags, chapters present)
- Recreated dialogue safety (labels, no voice imitation, no exploitation)
- Final safe-to-voice sign-off

Saves 1 OpenAI API call per episode vs `always` mode. The legacy Hindi Editor and Originality gates are marked `skipped` and do not run.

**`always`** — Combined Final Premium Gate **plus** the legacy Hindi Editor gate and Originality/YouTube Risk gate. All three must pass for `safe_to_voice=true`. Use for maximum coverage before important releases.

**`disabled`** — All OpenAI gates skipped. Does not block `safe_to_voice`, but lowers confidence. Not recommended for production.

### Configuration

Add to `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_REVIEW_MODEL=gpt-5.5
OPENAI_REVIEW_ENABLED=true
OPENAI_REVIEW_POLICY=adaptive
```

If `OPENAI_API_KEY` is not set and `OPENAI_REVIEW_ENABLED=true`, the pipeline returns `status=needs_human_review` and `safe_to_voice=false` without crashing. If `OPENAI_REVIEW_ENABLED=false`, all OpenAI gates are marked `skipped` and do not block script approval (not recommended for production).

### Output files

```
04-review/
  openai_final_premium_report.json                ← combined Final Premium Gate result (adaptive / always)
  _openai_final_premium_raw_response.txt          ← raw GPT response
  openai_premium_hindi_editor_report.json         ← legacy Hindi Editor gate (always mode only)
  _openai_premium_hindi_editor_raw_response.txt
  openai_originality_youtube_risk_report.json     ← legacy Originality/YT Risk gate (always mode only)
  _openai_originality_youtube_risk_raw_response.txt
```

### Cost note

`adaptive` (default) — 1 OpenAI API call per episode.  
`always` — 3 OpenAI API calls per episode.  
`disabled` — 0 OpenAI API calls.  
All OpenAI calls run only after all Claude gates pass.

> **No automated quality guarantee.** Claude + GPT reduce the probability of obvious issues but cannot guarantee YouTube monetization approval or zero copyright claims. Final human review is recommended before publishing.

---

## Adding ElevenLabs

1. Get your API key from [elevenlabs.io](https://elevenlabs.io)
2. Get your narrator voice ID from the ElevenLabs voice library
3. Add to `.env`:

```env
ELEVENLABS_API_KEY=your_key_here
ELEVENLABS_NARRATOR_VOICE_ID=your_voice_id_here
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
```

4. Call the full endpoint with `"enable_voice": true`

The service will generate one MP3 per narration chunk into `03-audio/`.  
Chunk IDs are stable, so re-running only regenerates missing files.

**Recommended Hindi voice:** Search ElevenLabs for multilingual voices that support Hindi. The `eleven_multilingual_v2` model handles Devanagari well.

---

## Adding Asset APIs

### Pexels (free commercial license)
1. Sign up at [pexels.com/api](https://www.pexels.com/api/)
2. Add `PEXELS_API_KEY=your_key` to `.env`

### Pixabay (free commercial license)
1. Sign up at [pixabay.com/api/docs](https://pixabay.com/api/docs/)
2. Add `PIXABAY_API_KEY=your_key` to `.env`

Then call with `"enable_assets": true`.

---

## Asset Guardrail Policy

| Status | Rule |
|---|---|
| **Auto-safe** | City/location visuals, court buildings, maps, generic hospital/street/house, symbolic non-person visuals |
| **Manual review** | Victim photos, family photos, children, case-specific real people, news/editorial images |
| **Blocked** | Graphic crime scene, autopsy, watermarked image, podcast screenshots, private social media, unclear-license image |

Guardrail results are written to `06-review/guardrail_results.json`.  
Review `manual_review` items before using in video.

---

## Hinglish Level

Every episode has a `hinglish_level` (1–5) that controls language style across all writing and review agents.

| Level | Description | Use when |
|---|---|---|
| `1` | Almost pure Hindi. Only unavoidable proper nouns in English/Roman script. | Maximum formality, archive/documentary style |
| `2` | Mostly Hindi. English allowed only for proper nouns and widely recognised terms. **Default.** | Child-victim cases, legal-appeal stories, serious true-crime |
| `3` | Natural spoken Hinglish OK for pacing — avoid at emotional/legal moments. | General true-crime, older adult cases |
| `4` | Hinglish-heavy YouTube style. Still respectful. | Lighter cases, younger audience |
| `5` | Very casual Hinglish throughout. | Not recommended for sensitive or child-victim stories |

**For the Meika Jordan episode:** `hinglish_level: 2` — serious child-victim case, maximum Hindi.

The Quality Critic checks the actual Hinglish level of every chunk against the requested level and flags mismatches. If any chunks use too much English for the requested level, they are added to `chunk_repair_targets` with `issue_type: "hinglish_level_mismatch"` and repaired automatically.

The `hinglish_level_assessment` field in the quality report and the `04-review/hinglish_level_assessment.json` file show the requested vs detected level for the episode.

---

## Targeted Chunk Repair

The repair stage works on individual chunks rather than regenerating the entire script. This solves the truncation problem that occurs when Claude tries to output a full 40KB repaired script in one response.

**How it works:**

1. The Quality Critic identifies specific chunks that need repair and adds them to `chunk_repair_targets` in the quality report — each entry names the `chunk_id`, `issue_type`, `problem`, and `repair_instruction`.
2. The Targeted Chunk Repair service loads each flagged chunk from `03-script/chunks/<chunk_id>.json`.
3. One Claude call per chunk: sends only the chunk text + fact_lock + blueprint summary + hinglish_level + repair instruction. Returns one small `NarrationChunk` JSON.
4. Retries once on parse/validation failure.
5. If a chunk repair fails after both attempts: keeps the original chunk, adds a warning, marks the episode `needs_human_review` — but does not fail the entire pipeline.
6. Merges all repaired chunks back into the script and re-saves the assembly files.

**Issue types the critic can flag:**

| `issue_type` | What it catches |
|---|---|
| `hindi_naturalness` | Unnatural Hindi phrasing, awkward constructions |
| `hinglish_level_mismatch` | English/Hindi ratio does not match requested hinglish_level |
| `missing_fact` | A required fact from fact_lock is absent from the chunk |
| `pacing` | Chunk is too fast, too slow, or jarring |
| `safety` | Safety rule violation (missing label, graphic content) |
| `structure` | Chunk is out of order or missing its purpose |
| `duration` | Chunk is significantly over/under target word count |

**Audit files written to `04-review/`:**

| File | Contents |
|---|---|
| `chunk_repair_targets.json` | The full list of repair targets from the quality report |
| `script_repair_report.json` | Per-chunk repair results: status, words before/after, errors |
| `hinglish_level_assessment.json` | Requested vs detected Hinglish level with notes |

**Repaired chunk files written to `03-script/chunks/`:**

| File | Contents |
|---|---|
| `repaired_<chunk_id>.json` | Repaired chunk (validated NarrationChunk) |
| `_repair_raw_<chunk_id>.txt` | Raw Claude response for debugging |

The old `script_repair_service.py` is kept as a legacy fallback but is no longer called by the pipeline.

---

## Content Rules (enforced via prompt)

- Narration language is controlled by `hinglish_level` (default: 2 = mostly Hindi)
- Source transcript is research input only — not copied
- Recreated scenes are labelled "फिर से रचा गया संवाद"
- No real voice imitation — generic voice descriptors only
- No graphic or exploitative language
- No child suffering recreation
- No automatic use of real victim/family/minor photos
- Victim dignity is respected throughout
- Mandatory disclaimer chunk at start

---

## What Is Still Needed for Production Video

The MVP generates the complete content brain, audio, and draft layout. For a fully rendered production video you will additionally need:

1. **Motion graphics renderer** — a tool like Remotion, After Effects, or a custom Puppeteer/Canvas pipeline to render scene cards with your visual brand
2. **AI image generation** — integrate Stable Diffusion, Midjourney API, or FLUX to generate cinematic scene backgrounds from the `ai_prompt` fields in `episode_video_plan.json`
3. **AI video generation** — integrate Kling, Runway, or Pika for short motion clips from ai_prompts
4. **Asset downloader** — download and cache approved image/video URLs from `04-assets/real-candidates/` into `04-assets/approved/`
5. **Audio mixing** — FFmpeg pipeline to merge narration MP3s with background music and SFX
6. **Timeline assembly** — sync narration audio durations with scene durations; currently estimated from word count
7. **Subtitle renderer** — burn Devanagari subtitles using FFmpeg `subtitles` filter or an ASS/SRT pipeline
8. **Thumbnail generator** — Pillow or Canva API integration using `youtube_metadata.thumbnail_options`
9. **YouTube upload** — YouTube Data API v3 upload with metadata from `youtube_metadata.json`
10. **Quality check automation** — automated scan for flagged content before upload

---

## Development Notes

### Agent architecture

Each agent is a separate Python module with one clear responsibility:

| Agent | Prompt file | Input | Output |
|---|---|---|---|
| Transcript Cleaner | Python only | raw_transcript | clean_transcript |
| Fact Lock | `fact_lock_agent.txt` | research_view | fact_lock.json |
| Story Blueprint | `story_blueprint_agent.txt` | fact_lock | story_blueprint.json |
| Script Outline | `script_outline_agent.txt` | fact_lock + blueprint | script_outline.json |
| Chunk Writer ×N | `narration_chunk_writer_agent.txt` | outline chunk spec + fact_lock | chunks/{chunk_id}.json |
| Dialogue Writer | `recreated_dialogue_agent.txt` | scene plan + audio moments | recreated_dialogues_draft.json |
| Metadata Writer | `metadata_agent.txt` | fact_lock + blueprint + chunk summaries | youtube_metadata_draft.json |
| Script Assembler | Python only | all chunk files | script_draft.json |
| Quality Critic | `script_quality_critic_agent.txt` | draft + fact_lock + blueprint | quality_report.json (with chunk_repair_targets) |
| Targeted Chunk Repair ×M | `targeted_chunk_repair_agent.txt` | one chunk + fact_lock + blueprint + repair_instruction | repaired_{chunk_id}.json |

### Token efficiency

- Fact Lock and Blueprint receive only the research_view (~18K chars), not the full transcript
- Script Writer receives only fact_lock + blueprint (compact JSON, ~5-8K chars)
- Quality Critic receives the script draft + fact_lock + blueprint — largest prompt, but all within 200K context
- Repair Agent receives the same inputs as the critic plus repair_instructions

### Why chunked script generation?

The old single-call script writer produced ~40KB of JSON in one response. Claude's output
tends to truncate mid-JSON when a response is this large, causing parse failures.

The new pipeline breaks this into focused calls:

| Call | Output size | Can fail alone? |
|---|---|---|
| Script Outline | ~3KB | Yes — stops pipeline |
| Narration Chunk ×N | ~1–2KB each | Yes — retried once, then stops |
| Recreated Dialogue | ~2KB | No — falls back to empty |
| Metadata | ~0.5KB | No — falls back to case name |
| Script Assembler | Python, no Claude | N/A |

If one chunk fails both attempts, only that chunk needs investigation — not the entire script.
Existing chunks are saved to `03-script/chunks/` and reused on re-run (idempotent).

### Fact-lock extraction modes

`FACT_LOCK_MODE` in `.env` controls how the Fact Lock agent processes the transcript:

| Mode | Description | Use when |
|---|---|---|
| `research_view` (default) | One Claude call using the 40/20/40 beginning/middle/end excerpt | Most transcripts — cheap and fast |
| `segmented` | Splits the clean transcript into ~7000-char segments, runs one Claude call per segment, then merges with deduplication | Very long or complex cases where important details fall in the middle sections |

**Auto-segmented for long premium transcripts:** When `cost_mode=premium` and the transcript is ≥ `PREMIUM_SEGMENTED_FACT_LOCK_THRESHOLD` characters (default 30 000), the pipeline automatically switches to `segmented` mode — even if `FACT_LOCK_MODE=research_view`. This ensures that long episodes (25+ minutes) do not miss facts from the middle of the transcript. The threshold is configurable:

```bash
PREMIUM_SEGMENTED_FACT_LOCK_THRESHOLD=30000
```

In `segmented` mode, individual segment results are saved for debugging:
```
02-facts/segments/
  fact_segment_001.json    ← parsed facts from segment 1
  fact_segment_002.json    ← parsed facts from segment 2
  _segment_001_raw.txt     ← raw Claude response for segment 1
  ...
```

Adjust segment size with `FACT_LOCK_SEGMENT_CHARS=7000` (default). Smaller values → more segments → more Claude calls.

### Cost and quality control

The pipeline exposes four knobs to balance cost, speed, and quality coverage.

#### `QUALITY_MODE`

| Mode | OpenAI gates | Voice-ready | Use when |
|---|---|---|---|
| `premium_final` | ✅ per policy | ✅ if all gates pass | Production episodes |
| `premium_build` | ❌ skipped | ❌ never | Debugging script generation |
| `premium_batch` | ❌ skipped | ❌ never | Bulk candidate evaluation |

#### `OPENAI_REVIEW_POLICY` (only applies when `QUALITY_MODE=premium_final`)

| Policy | Gates run | OpenAI calls | Recommended for |
|---|---|---|---|
| `adaptive` | Combined Final Premium Gate (all dimensions in one call) | 1 per episode | Most production runs (default) |
| `always` | Final Premium Gate + legacy Hindi Editor + legacy Originality/YT Risk | 3 per episode | Maximum coverage, important releases |
| `disabled` | None | 0 per episode | Budget debugging — lowers confidence, not recommended for production |

In `adaptive` mode, the combined gate checks: Hindi grammar, Hinglish level, retention, originality, YouTube safety, metadata completeness, recreated dialogue, and safe-to-voice — all in a single call.  
In `always` mode, the combined gate runs first, then two additional independent gates cross-check the same content.  
The `gate_summary.openai_final_premium` key is present in both modes. Legacy keys (`openai_premium_hindi_editor`, `openai_originality_youtube_risk`) are present in `always` mode and marked `skipped` in `adaptive` mode.

#### `SKIP_FINAL_GATES`

```bash
SKIP_FINAL_GATES=true
```

Bypasses **all** premium quality gates (stages 8–16). Output is never voice-ready. For debugging script generation logic without running any review AI. Remove before audio production.

#### Budget guards

The pipeline aborts with `BudgetExceededError` if any limit is exceeded:

| Variable | Default | Guards against |
|---|---|---|
| `MAX_TOTAL_MODEL_CALLS` | `80` | Runaway cost from looping repair stages |
| `MAX_REPAIR_CALLS` | `12` | Claude chunk repair calls per episode |
| `MAX_OPENAI_REPAIR_CALLS` | `6` | OpenAI targeted repair calls per episode |

If a budget guard fires, the error message names the agent and limit that was exceeded.

#### Telemetry

Every pipeline response now includes a `telemetry` field:

```json
"telemetry": {
  "model_calls": {
    "claude_total": 18,
    "openai_total": 2,
    "repair_claude": 3,
    "repair_openai": 0,
    "total": 20
  },
  "stage_timing_sec": {
    "fact_lock": 12.4,
    "story_blueprint": 8.1,
    "script_writer": 47.2,
    "quality_review": 9.8,
    "openai_hindi_editor": 14.3
  },
  "stage_reuse": ["fact_lock", "story_blueprint"]
}
```

Use `stage_timing_sec` to identify bottlenecks and `stage_reuse` to confirm which stages were skipped on re-runs.

#### Prompt caching

When `CLAUDE_PROMPT_CACHE_ENABLED=true` (default), agent calls made via `call_claude_agent_cached()` mark the stable system prompt with `cache_control: ephemeral` using the Anthropic prompt-caching beta. This can reduce cost and latency when the same episode type is run repeatedly.

**Important:** prompt caching is only active for services wired to `call_claude_agent_cached()`. Not every service uses the cached variant — most still use the standard `call_claude_agent()`. The primary cost-control mechanisms in this pipeline are:

- **Stage reuse** (`REUSE_EXISTING_STAGE_OUTPUTS=true`) — skip stages whose output files already exist
- **Compact prompts** — each agent receives only the fields it needs
- **Deterministic Python checks** — text lint, word count, text similarity never call an LLM
- **OpenAI final-gate consolidation** — `OPENAI_REVIEW_POLICY=adaptive` runs a single combined OpenAI call instead of three separate gate calls

Do not rely on `CLAUDE_PROMPT_CACHE_ENABLED` alone for cost savings until all services are confirmed to use `call_claude_agent_cached()`.

### Idempotent stage execution

When `REUSE_EXISTING_STAGE_OUTPUTS=true`, a stage is skipped if its output file already exists on disk. This is useful when debugging a later stage — you can re-run the pipeline without paying for the earlier stages again.

```env
# .env
REUSE_EXISTING_STAGE_OUTPUTS=true
```

Stages that check for existing output:

| Stage | File checked |
|---|---|
| Transcript Cleaner | `01-input/clean_transcript.txt` |
| Fact Lock | `02-facts/fact_lock.json` |
| Story Blueprint | `02-facts/story_blueprint.json` |
| Script Writer | `03-script/script_draft.json` |
| Quality Critic | `04-review/script_quality_report.json` |

Default is `false` — always re-run every stage.

### Python-side quality validation

After Claude returns a quality report, the pipeline runs its own score check in Python, independent of Claude's `approved` field:

| Score dimension | Minimum required |
|---|---|
| `factual_accuracy` | 9 / 10 |
| `safety` | 10 / 10 |
| `monetization_safety` | 9 / 10 |
| `hindi_naturalness` | 8 / 10 |
| `story_structure` | 8 / 10 |
| `retention_hook` | 8 / 10 |

If any score is below its minimum, the pipeline overrides `approved=True` to `approved=False` and prepends the specific failure reasons to `repair_instructions`. This prevents a hallucinated approval from letting a weak script through to audio generation.

Duration is also checked using a Python word count (not Claude's estimate): the script must be at least 80% of the target duration (configurable via `MIN_ACCEPTABLE_DURATION_RATIO`).

### Debugging

Every agent saves its raw Claude response before parsing:
- `02-facts/_fact_lock_raw_response.txt`
- `02-facts/_story_blueprint_raw_response.txt`
- `03-script/chunks/_raw_{chunk_id}.txt` — chunk writer raw responses
- `03-script/chunks/_repair_raw_{chunk_id}.txt` — chunk repair raw responses (if repair ran)
- `04-review/_script_quality_raw_response.txt`

If any stage fails with a JSON parse error, check the corresponding `_raw_response.txt` or `_raw_{chunk_id}.txt`.

### Cost per episode (approximate, bootstrap mode)

| Stage | Model | Tokens (approx) |
|---|---|---|
| Fact Lock | Claude Sonnet | ~5K in / ~1K out |
| Story Blueprint | Claude Sonnet | ~3K in / ~0.5K out |
| Script Writer | Claude Sonnet | ~8K in / ~6K out |
| Quality Critic | Claude Sonnet | ~12K in / ~0.5K out |
| Chunk Repair ×M (if needed) | Claude Sonnet | ~6K in / ~0.5K out per chunk |

Total per episode: ~28–42K tokens (without repair). Repair adds ~6K tokens per flagged chunk.
At Sonnet pricing this is a few cents per episode.
- JSON extraction is robust against markdown fences and trailing commas
- All services are stateless and independently testable
- Add `--reload` to uvicorn for hot reload during development

---

## Docker Commands Reference

```bash
# Build and start
docker compose up --build

# Run in background
docker compose up -d --build

# View logs
docker compose logs -f agatsya

# Stop
docker compose down

# Rebuild after code changes
docker compose up --build --force-recreate

# Open shell in container
docker compose exec agatsya bash
```

---

## Production Principles

### Who does what

| Layer | Tool | Role |
|---|---|---|
| **Python** | Built-in | Deterministic checks — Hindi text lint, word-count scoring, text similarity phrase matching, schema validation. Zero API cost. Always runs. |
| **Claude** | Anthropic API | Production engine — fact extraction, story architecture, full Hindi narration (12–16 chunks), targeted chunk repair, and all Claude-layer quality gates (copyedit, retention, originality, dialogue, metadata). |
| **OpenAI** | OpenAI API | Final premium judge and editor — one combined gate at the end (OPENAI_REVIEW_POLICY=adaptive) covering Hindi grammar, Hinglish level, retention, originality, YouTube safety, metadata, and recreated dialogue. Independent from Claude. |

### Core production rules

**Claude writes and self-checks. OpenAI independently judges and optionally repairs.**

- Claude produces the full script and runs all intermediate quality checks.
- OpenAI runs once at the end as an independent senior editor with no access to Claude's intermediate reasoning — only the final script and compact summaries of Claude's gate results.
- OpenAI targeted repair runs only if the final gate fails and returns specific `chunk_repair_targets`. It repairs only the named chunks — no full-script rewrites.
- If too many chunks need repair (> `OPENAI_REPAIR_MAX_CHUNKS`), no repair runs and the status is set to `needs_human_review`.

**No ElevenLabs unless `status=script_approved` AND `safe_to_voice=true`.**

Both fields must be true. `status=script_approved` alone is not sufficient. ElevenLabs credits are non-refundable — the `safe_to_voice` field is the authoritative go/no-go signal.

### Key environment settings for production

```env
# Production quality mode — Claude + OpenAI; required before ElevenLabs
QUALITY_MODE=premium_final

# Combined single OpenAI gate at end (saves 1 API call vs "always")
OPENAI_REVIEW_POLICY=adaptive

# For debug reruns of a failed late stage (skips already-completed early stages)
REUSE_EXISTING_STAGE_OUTPUTS=true

# Full pipeline endpoint disabled by default — enable only after safe_to_voice=true confirmed
ENABLE_FULL_PIPELINE=false
```

### Clean test command

Run a full clean premium test (no stage reuse, all gates active):

```bash
# Ensure .env has:
# QUALITY_MODE=premium_final
# OPENAI_REVIEW_POLICY=adaptive
# REUSE_EXISTING_STAGE_OUTPUTS=false
# SKIP_FINAL_GATES=false
# OPENAI_API_KEY=sk-...

curl -X POST http://localhost:8000/api/episodes/package \
  -H "Content-Type: application/json" \
  -d @samples/create_episode_payload.json | jq '{status, safe_to_voice, gate_summary}'
```

Expected output when all gates pass:
```json
{
  "status": "script_approved",
  "safe_to_voice": true,
  "gate_summary": { ... }
}
```

### Resuming from a failed late stage

If a late stage fails (e.g. the OpenAI gate), set `REUSE_EXISTING_STAGE_OUTPUTS=true` in `.env` before rerunning. Early stages (fact lock, blueprint, script writer) will be loaded from disk. Only the failed stage and everything after it will re-execute.

```bash
# In .env:
REUSE_EXISTING_STAGE_OUTPUTS=true

# Then rerun the same endpoint — the pipeline will skip stages whose output files exist
curl -X POST http://localhost:8000/api/episodes/package \
  -H "Content-Type: application/json" \
  -d @samples/create_episode_payload.json | jq '{status, safe_to_voice}'
```

**Important:** If you changed the transcript, `hinglish_level`, `cost_mode`, or any prompt file since the last run, set `REUSE_EXISTING_STAGE_OUTPUTS=false` to force a full rerun. The stage manifest will warn you if inputs changed while reuse is enabled.

### `/api/episodes/full` is disabled by default

`POST /api/episodes/full` is guarded by `ENABLE_FULL_PIPELINE=false`. Do not enable it until you have confirmed `safe_to_voice=true` from `/api/episodes/package`. ElevenLabs voice generation is irreversible — there is no way to undo a bad audio run.
```
