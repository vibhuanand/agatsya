from __future__ import annotations

from app.services.repair_routing_service import run_repair_routing


def _chunk(cid: str, title: str) -> dict:
    return {
        "chunk_id": cid,
        "section_title": title,
        "text": "x",
    }


def _script() -> dict:
    return {
        "hindi_narration_chunks": [
            _chunk("001_intro", "Opening"),
            _chunk("002_investigation", "Investigation begins"),
            _chunk("003_reconstruction", "Court reconstruction"),
            _chunk("004_evidence", "Evidence sequence"),
            _chunk("005_aftermath", "Aftermath transition"),
            _chunk("006_memory", "Memory and tribute"),
        ]
    }


def _preflight(chunk_id: str = "003_reconstruction") -> dict:
    return {
        "blocking": True,
        "issues": [
            {
                "severity": "high",
                "type": "source_shaped_reconstruction",
                "chunk_id": chunk_id,
                "problem": "source-shaped reconstruction",
            }
        ],
        "chunk_repair_targets": [
            {
                "chunk_id": chunk_id,
                "issue_type": "source_shaped_reconstruction",
                "problem": "source-shaped reconstruction",
                "repair_instruction": "rebuild",
            }
        ],
    }


def test_source_copy_issue_creates_cluster_target():
    plan = run_repair_routing(
        {"python_preflight": _preflight()},
        openai_repair_max_chunks=6,
        script_draft=_script(),
        max_cluster_size=4,
    )

    clusters = plan["source_copy_reconstruction_clusters"]
    assert len(clusters) == 1
    assert clusters[0]["repair_type"] == "source_copy_reconstruction_cluster"
    assert "003_reconstruction" in clusters[0]["target_chunk_ids"]
    assert plan["source_shaped_reconstruction_detected"] is True


def test_adjacent_related_chunks_included_but_unrelated_excluded():
    plan = run_repair_routing(
        {"python_preflight": _preflight()},
        openai_repair_max_chunks=6,
        script_draft=_script(),
        max_cluster_size=4,
    )

    ids = plan["source_copy_reconstruction_clusters"][0]["target_chunk_ids"]
    assert "002_investigation" in ids
    assert "004_evidence" in ids
    assert "006_memory" not in ids


def test_cluster_respects_max_size():
    plan = run_repair_routing(
        {"python_preflight": _preflight()},
        openai_repair_max_chunks=6,
        script_draft=_script(),
        max_cluster_size=2,
    )

    ids = plan["source_copy_reconstruction_clusters"][0]["target_chunk_ids"]
    assert len(ids) == 2
