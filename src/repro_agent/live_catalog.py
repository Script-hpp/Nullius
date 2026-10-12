"""Server-owned live Discovery claims, separate from the curated benchmark catalog.

Stored in the existing logs volume. Browsers submit only an ID, never a claim or
task override. Atomic writes preserve candidates across reloads and redeploys.
"""

import hashlib
import json
import threading

from . import catalog as cat
from .dossier import REPO_ROOT
from .site_export import write_json_atomic

PATH = REPO_ROOT / "logs" / "live_candidates.json"
CACHE_PATH = cat.DISCOVERY_CACHE
REPLAY_PATH = REPO_ROOT / "logs" / "discovery_replay.json"
_lock = threading.RLock()


def load_entries() -> list[dict]:
    with _lock:
        return json.loads(PATH.read_text()) if PATH.exists() else []


def find_entry(candidate_id: str) -> dict | None:
    return next((entry for entry in load_entries() if entry["id"] == candidate_id), None)


def eligibility(entry: dict) -> dict:
    try:
        paper, _ = cat.build_paper(entry)
        return {"runnable": True, "reason": None, "warnings": paper["warnings"]}
    except cat.NotRunnable as exc:
        reason = str(exc)
        if "Cora ohne matched_task" in reason:
            reason = "The Cora split is not established. We cannot choose between the public split and random splits without evidence."
        return {"runnable": False, "reason": reason, "warnings": []}


def save_candidate(event: dict) -> dict:
    """Called only for validated candidate events from the server Discovery agent."""
    entry = cat.entry_from_candidate(event, "live discovery")
    # A paper can yield different methods, datasets, targets or protocols.
    identity = {key: event.get(key) for key in ("arxiv_id", "dataset", "claim", "protocol", "github_url")}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    entry["id"] += "_" + digest
    eligible = eligibility(entry)
    published = {**event, "catalog_id": entry["id"], "runnable": eligible["runnable"],
                 "run_reason": eligible["reason"], "run_warnings": eligible["warnings"]}
    entry["discovery_event"] = published
    with _lock:
        entries = load_entries()
        existing = next((e for e in entries if e["id"] == entry["id"]), None)
        if existing is None:
            entries.append(entry)
        else:
            # A cached search must not erase a completed experiment.
            existing["discovery_event"] = published
        write_json_atomic(PATH, entries)
    return published


def save_outcome(entry: dict, outcome: cat.CandidateOutcome) -> None:
    with _lock:
        entries = load_entries()
        stored = cat.find_entry(entries, entry["id"])
        cat.apply_outcome(stored, outcome)
        write_json_atomic(PATH, entries)


def candidate_events() -> list[dict]:
    return sorted((e["discovery_event"] for e in load_entries()), key=lambda e: e.get("ts", 0))[-30:]


def import_existing_cache() -> None:
    """One-time import of server-validated cache v2/v3, without search or model calls.

    This lets already deployed Discovery results remain usable after the upgrade.
    Never modifies the old cache, benchmark files, or an existing live catalog.
    """
    if PATH.exists() or not CACHE_PATH.exists():
        return
    stamp = CACHE_PATH.stat().st_mtime
    for candidate, _ in cat.read_cache_candidates(CACHE_PATH):
        save_candidate({**candidate, "type": "candidate", "ts": stamp, "cached": True,
                        "community_hits": len(candidate.get("community") or [])})


def save_replay(events: list[dict]) -> None:
    write_json_atomic(REPLAY_PATH, events)


def latest_replay() -> list[dict]:
    return json.loads(REPLAY_PATH.read_text()) if REPLAY_PATH.exists() else []
