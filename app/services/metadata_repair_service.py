"""
Metadata Repair Service — fixes youtube_metadata when the Metadata Quality Gate fails.

Does NOT rewrite the narration script. Repairs only the metadata object, then saves:
  03-script/script_draft.json          (youtube_metadata key updated)
  03-script/script_final.json          (youtube_metadata key updated)
  02-package/youtube_metadata.json     (overwritten)
  04-review/metadata_repair_report.json

Produces per run:
  04-review/_metadata_repair_raw_response.txt
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.services.claude_client import call_claude_agent, parse_package_response
from app.services.prompt_utils import get_channel_rules

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("app/prompts/metadata_repair_agent.txt")


def run_metadata_repair(
    script_draft: dict,
    fact_lock: dict,
    gate_report: dict,
    script_dir: Path,
    review_dir: Path,
) -> tuple[dict, dict]:
    """
    Repair the youtube_metadata section of script_draft using Claude.

    Returns (updated_script_draft, repair_report).
    The caller is responsible for re-running the metadata gate after this.
    """
    current_metadata = script_draft.get("youtube_metadata", {})

    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = (
        template
        .replace("{channel_rules}", get_channel_rules())
        .replace("{fact_lock_json}", json.dumps(fact_lock, ensure_ascii=False))
        .replace("{current_metadata_json}", json.dumps(current_metadata, ensure_ascii=False))
        .replace("{gate_report_json}", json.dumps(gate_report, ensure_ascii=False))
    )

    raw_response, stop_reason = call_claude_agent(prompt, agent_name="metadata_repair")

    raw_path = review_dir / "_metadata_repair_raw_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")

    if stop_reason == "max_tokens":
        logger.warning("metadata_repair hit max_tokens — repaired metadata may be incomplete")

    try:
        repaired_metadata = parse_package_response(raw_response, agent_name="metadata_repair")
    except ValueError as exc:
        raise ValueError(
            f"Metadata repair JSON parse failed: {exc}\n"
            f"Raw response saved at: {raw_path}"
        ) from exc

    if not isinstance(repaired_metadata, dict):
        raise ValueError(
            f"Metadata repair returned unexpected type {type(repaired_metadata).__name__}. "
            f"Expected dict. Raw response saved at: {raw_path}"
        )

    # Merge repaired metadata back into script_draft
    script_updated = dict(script_draft)
    script_updated["youtube_metadata"] = repaired_metadata

    # Save updated assembly files
    pkg_dir = script_dir.parent / "02-package"

    def _save(path: Path, data: object) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Update script_final.json and script_draft.json in 03-script/
    for fname in ("script_final.json", "script_draft.json"):
        fpath = script_dir / fname
        if fpath.exists():
            try:
                existing = json.loads(fpath.read_text(encoding="utf-8"))
                existing["youtube_metadata"] = repaired_metadata
                _save(fpath, existing)
            except Exception as exc:
                logger.warning("Could not update %s: %s", fpath, exc)

    # Overwrite 03-script/youtube_metadata.json (alongside script files)
    _save(script_dir / "youtube_metadata.json", repaired_metadata)

    # Overwrite 02-package/youtube_metadata.json
    if pkg_dir.exists():
        _save(pkg_dir / "youtube_metadata.json", repaired_metadata)

    repair_report = {
        "status": "metadata_repaired",
        "fields_present": list(repaired_metadata.keys()),
        "thumbnail_options_count": len(repaired_metadata.get("thumbnail_options", [])),
        "tags_count": len(repaired_metadata.get("tags", [])),
        "title_options_count": len(repaired_metadata.get("title_options", [])),
    }
    _save(review_dir / "metadata_repair_report.json", repair_report)

    logger.info(
        "Metadata repair complete — %d tags, %d thumbnail options, %d title options",
        repair_report["tags_count"],
        repair_report["thumbnail_options_count"],
        repair_report["title_options_count"],
    )

    return script_updated, repair_report
