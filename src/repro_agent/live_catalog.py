"""Server-owned live Discovery claims, separate from the curated benchmark catalog.

Stored in the existing logs volume. Browsers submit only an ID, never a claim or
task override. Atomic writes preserve candidates across reloads and redeploys.
"""

import hashlib
import json
import re
import threading
import time
from pathlib import Path

from . import catalog as cat
from .dossier import REPO_ROOT
from .site_export import write_json_atomic

PATH = REPO_ROOT / "logs" / "live_candidates.json"
CACHE_PATH = cat.DISCOVERY_CACHE
REPLAY_PATH = REPO_ROOT / "logs" / "discovery_replay.json"
PREFLIGHT_DIR = REPO_ROOT / "logs" / "preflight"
NOT_PREFLIGHTED = ("The readiness check has not run for this claim yet, so it is not offered for a one-click check. "
                   "New searches check every claim they find.")
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


def offer(entry: dict) -> dict:
    """A found claim gets a start button only after its readiness check passed (loop.Orchestrator.preflight):
    the target is bound to its table, the authors' code is read and the spec has no conflict."""
    eligible = eligibility(entry)
    if not eligible["runnable"]:
        return eligible
    readiness = entry.get("readiness")
    if not readiness:
        return {"runnable": False, "reason": NOT_PREFLIGHTED, "warnings": []}
    if not readiness.get("ready"):
        return {"runnable": False, "reason": readiness.get("reason") or NOT_PREFLIGHTED, "warnings": [],
                "stage": readiness.get("stage")}
    return {**eligible, "readiness": readiness}


CURATED_NOT_OFFERED = ("Live checks are offered for the featured entries, which were re-checked with the current pipeline. "
                       "This entry shows its recorded result.")


def curated_offer(entry: dict) -> dict:
    """A curated catalog entry gets a live check only when it is featured (catalog.jury_ready): re-checked with the current
    pipeline and passed. A withdrawn or never re-checked entry keeps its recorded result but no start button."""
    eligible = eligibility(entry)
    if not eligible["runnable"]:
        return eligible
    ready, _ = cat.jury_ready(entry)
    if not ready:
        override = (entry.get("jury_override") or {}).get("reason")
        return {"runnable": False, "reason": override or CURATED_NOT_OFFERED, "warnings": []}
    return eligible


def preflight_path(candidate_id: str) -> Path:
    return PREFLIGHT_DIR / (re.sub(r"[^A-Za-z0-9_.-]", "_", candidate_id) + ".pkl")


def save_readiness(candidate_id: str, result: dict) -> dict:
    readiness = {"ready": bool(result.get("ready")), "stage": result.get("stage"), "reason": result.get("reason"),
                 "mode": result.get("mode"), "repo": result.get("repo"), "cost_gross_usd": result.get("cost_gross_usd"),
                 "spec_enforced": result.get("spec_enforced"), "spec_missing": result.get("spec_missing"),
                 "checked_at": time.time()}
    with _lock:
        entries = load_entries()
        entry = next((e for e in entries if e["id"] == candidate_id), None)
        if entry is not None:
            entry["readiness"] = readiness
            entry.get("discovery_event", {})["readiness"] = readiness
            write_json_atomic(PATH, entries)
    return readiness


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


# ---- one offer per claim -------------------------------------------------------------------------------------------
# A paper can be found several times (different quote, repository link or protocol note give different ids). Saved finds
# with the same paper (arXiv id without version), dataset/task, metric, target and horizon are ONE claim for offering:
# one card, one start button. The others stay on disk with their readiness state untouched; they are only hidden.
# Different targets of one paper (0.366 vs 0.371) are different claims.
_spec_cache: dict[tuple[str, float], tuple[int, int] | None] = {}


def claim_key(entry: dict) -> tuple | None:
    cand = entry.get("candidate") or {}
    claim = cand.get("claim") or {}
    metric = str(claim.get("metric") or "").lower()
    metric = "accuracy" if metric == "acc" else metric
    try:
        target = float(claim.get("target"))
    except (TypeError, ValueError):
        return None   # no exact target: never grouped
    if metric == "accuracy" and target > 1:
        target /= 100
    arxiv = re.sub(r"v\d+$", "", str(entry.get("arxiv_id") or ""))
    task = claim.get("matched_task") or cand.get("dataset")
    if not arxiv or not task or not metric:
        return None
    return (arxiv, str(cand.get("dataset")).lower(), str(task).lower(), metric, round(target, 9), claim.get("horizon"))


def _spec_counts(entry: dict) -> tuple[int, int] | None:
    """(enforced spec values, missing values) of the readiness check; older checks are read from their saved state."""
    readiness = entry.get("readiness") or {}
    if readiness.get("spec_enforced") is not None:
        return int(readiness["spec_enforced"]), int(readiness.get("spec_missing") or 0)
    path = preflight_path(entry["id"])
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    key = (str(path), stamp)
    if key not in _spec_cache:
        try:
            import pickle   # own file on the logs volume, the same trust as the run that resumes from it
            spec = pickle.loads(path.read_bytes())["state"].get("spec")
            _spec_cache[key] = (len(spec.enforced()), len(spec.missing)) if spec is not None else None
        except Exception:
            _spec_cache[key] = None
    return _spec_cache[key]


def _rank(entry: dict) -> tuple:
    """Representative of one claim, deterministic: a ready find beats a not-ready one; then more enforced spec values,
    then fewer missing values, then the newest find, then the id. A find with a recorded outcome ranks first and is
    never hidden: its evidence stays in the catalog export."""
    enforced, missing = _spec_counts(entry) or (-1, 0)
    ts = (entry.get("discovery_event") or {}).get("ts") or 0
    return (entry.get("status") != cat.UNCHECKED, bool((entry.get("readiness") or {}).get("ready")),
            enforced, -missing, ts, entry["id"])


def duplicates(entries: list[dict] | None = None) -> dict[str, str]:
    """{hidden id: representative id} over the saved finds."""
    entries = load_entries() if entries is None else entries
    groups: dict[tuple, list[dict]] = {}
    for entry in entries:
        key = claim_key(entry)
        if key is not None:
            groups.setdefault(key, []).append(entry)
    hidden: dict[str, str] = {}
    for group in groups.values():
        if len(group) > 1:
            best = max(group, key=_rank)
            hidden.update({e["id"]: best["id"] for e in group
                           if e is not best and e.get("status") == cat.UNCHECKED})
    return hidden


def offered_entries(entries: list[dict] | None = None) -> list[dict]:
    entries = load_entries() if entries is None else entries
    hidden = duplicates(entries)
    return [e for e in entries if e["id"] not in hidden]


def duplicate_reason(representative_id: str) -> str:
    return (f"This claim was found more than once. The check is offered on the earlier find {representative_id}; "
            "start it there.")


def candidate_events() -> list[dict]:
    entries = offered_entries()
    return sorted((e["discovery_event"] for e in entries), key=lambda e: e.get("ts", 0))[-30:]


def import_existing_cache() -> None:
    """One-time import of accepted cache versions, without search or model calls.

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
    """Recorded events; cards and readiness of a hidden duplicate are left out, so the replay shows no double offer."""
    events = json.loads(REPLAY_PATH.read_text()) if REPLAY_PATH.exists() else []
    hidden = duplicates()
    seen = {e.get("catalog_id") for e in events if isinstance(e, dict) and e.get("type") == "candidate"}
    out = []
    for e in events:
        rep = hidden.get(e.get("catalog_id")) if isinstance(e, dict) and e.get("type") in ("candidate", "readiness") else None
        if rep is None:
            out.append(e)
        elif e["type"] == "candidate" and rep not in seen:   # the card of the same claim is the representative's
            stored = find_entry(rep)
            if stored is not None:
                seen.add(rep)
                out.append({**stored["discovery_event"], "ts": e.get("ts", stored["discovery_event"].get("ts"))})
    return out
