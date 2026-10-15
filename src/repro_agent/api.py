import asyncio
import json
import gzip
import logging
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import catalog as cat
from . import live_gate, site_export, web_guard, spend_guard, live_catalog
from .discovery import DiscoveryAgent
from .dossier import REPO_ROOT


@asynccontextmanager
async def lifespan(_app):
    restore_counters()   # Tageslimits ueberleben Neustart/Redeploy (logs/live_jobs.jsonl)
    try:
        live_catalog.import_existing_cache()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.error("Could not restore Discovery candidates: %s", sanitize_text(str(exc), 300))
    yield


# Keine automatische Doku (/docs, /redoc, /openapi.json) auf dem oeffentlichen Server. Kein CORS: die Web-App ist same-origin.
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.add_middleware(web_guard.GuardMiddleware, state=app.state)   # Rate-Limit je IP, Body-Limit, Security-Header

# Global limits
job_semaphore = asyncio.Semaphore(1)
daily_limit = {"date": None, "count": 0}
MAX_DAILY_JOBS = int(os.environ.get("MAX_DAILY_JOBS", 5))
JOB_TIMEOUT_S = 2700            # Stream reports timeout; slot stays held until the worker exits
DISCOVERY_TIMEOUT_S = 1800            # search (~2 min) plus a readiness check per found claim (~3 min each)
STREAM_MAX_S = 3600             # laengste Dauer einer SSE-Verbindung (Reconnect mit from=<n> liefert den Rest)
_held: set[str] = set()         # Job-IDs, die den Semaphor halten (genau einmal freigeben, auch bei Watchdog + Worker-Ende)
_budgets: dict[str, spend_guard.JobBudget] = {}


def _run_budgeted(job_id, worker):
    token = spend_guard.current_budget.set(_budgets[job_id])
    try:
        worker()
    finally:
        spend_guard.current_budget.reset(token)


def _new_budget(kind):
    try:
        limit = spend_guard.number("MAX_COST_PER_DISCOVERY_USD" if kind == "discovery" else "MAX_COST_PER_RUN_USD", .25 if kind == "discovery" else .50)
        if limit <= 0:
            raise ValueError("A positive job budget is required.")
        return spend_guard.JobBudget(limit, live=True)
    except (ValueError, spend_guard.BudgetExceeded) as exc:
        raise HTTPException(503, detail="Live spending limits are misconfigured.") from exc


def _record_start(kind, ip):
    try:
        live_gate.record_start(kind, ip)
    except OSError as exc:
        raise HTTPException(503, detail="Live accounting storage is unavailable.") from exc


def restore_counters() -> None:
    """Tageszaehler aus dem persistenten Start-Log neu aufbauen (nach Neustart). Setzt nie etwas zurueck."""
    today = datetime.date.today()
    for e in live_gate.todays_starts():
        if e["kind"] == "reproduction":
            if repro_daily["date"] != today:
                repro_daily["date"], repro_daily["count"] = today, 0
            repro_daily["count"] += 1
        else:
            if daily_limit["date"] != today:
                daily_limit["date"], daily_limit["count"] = today, 0
            daily_limit["count"] += 1
        ip_limits.restore(e["kind"], e.get("ip", ""))


def _release_lock(job_id: str) -> None:
    """Semaphor einmal je Job freigeben (Watchdog und Worker koennen beide enden)."""
    if job_id in _held:
        _held.discard(job_id)
        _budgets.pop(job_id, None)
        job_semaphore.release()


class StreamSlots:
    """Gleichzeitig offene SSE-Verbindungen: global und je IP begrenzt (jede haelt eine Coroutine und pollt).
    Jeder Platz verfaellt nach STREAM_MAX_S + 60 s von selbst, falls ein abgebrochener Client ihn nie freigibt."""

    def __init__(self) -> None:
        self.slots: dict[int, tuple[str, float]] = {}
        self._next = 0

    def _purge(self) -> None:
        now = time.monotonic()
        for k in [k for k, (_, exp) in self.slots.items() if exp < now]:
            del self.slots[k]

    def acquire(self, ip: str) -> int | None:
        self._purge()
        if len(self.slots) >= int(live_gate._f("MAX_SSE_STREAMS", 30)):
            return None
        if sum(1 for i, _ in self.slots.values() if i == ip) >= int(live_gate._f("MAX_SSE_PER_IP", 4)):
            return None
        self._next += 1
        self.slots[self._next] = (ip, time.monotonic() + STREAM_MAX_S + 60)
        return self._next

    def release(self, token: int) -> None:
        self.slots.pop(token, None)


stream_slots = StreamSlots()

# In-memory registry for jobs and their queues
@dataclass
class DiscoveryJob:
    """All events since start. A dropped connection (proxy, network) reconnects with from=<n>; the search keeps running."""
    events: list = field(default_factory=list)
    done: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, evt: dict) -> None:
        with self.lock:
            self.events.append(evt)
            if evt.get("type") == "done":
                self.done = True


jobs: Dict[str, DiscoveryJob] = {}
background_tasks = set()
logger = logging.getLogger(__name__)

def _known_arxiv_ids() -> set[str]:
    """Papers in the catalog or saved live finds; a fresh search skips them instead of reading them again."""
    try:
        entries = cat.load_catalog(CATALOG_PATH) + live_catalog.load_entries()
    except (OSError, ValueError):
        return set()
    return {re.sub(r"v\d+$", "", str(e["arxiv_id"])) for e in entries if e.get("arxiv_id")}


class DiscoverRequest(BaseModel):
    datasets: list[str] = Field(["Cora", "ETTh1"], max_length=8)


@app.post("/api/discover")
async def start_discovery(req: DiscoverRequest, request: Request):
    from .tasks import TASKS
    datasets = list(dict.fromkeys(req.datasets))   # Duplikate raus: ["Cora"] * 1000 waere 1000 Tavily-Suchen
    if not datasets:
        raise HTTPException(status_code=400, detail="No dataset given.")
    for ds in datasets:
        if ds not in TASKS:
            raise HTTPException(status_code=400, detail="Dataset is not supported.")

    ip = live_gate.client_ip(request)
    check_live("discovery", ip)
    budget = _new_budget("discovery")
    today = datetime.date.today()
    if daily_limit["date"] != today:
        daily_limit["date"], daily_limit["count"] = today, 0
    _record_start("discovery", ip)
    daily_limit["count"] += 1
    ip_limits.add("discovery", ip)

    job_id = str(uuid.uuid4())
    job = DiscoveryJob()
    jobs[job_id] = job
    start_event = {"type": "discovery_start", "ts": time.time(), "datasets": datasets}
    job.add(start_event)
    recording = [start_event]
    recording_lock = threading.Lock()
    recording_closed = False

    def emit(evt, worker_finished=False):
        nonlocal recording_closed
        event = sanitize(evt)
        stamp = event.get("ts")
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            event["ts"] = time.time()
        with recording_lock:
            if not recording_closed:
                recording.append(event)
                if event["type"] == "done":
                    recording_closed = True
                    try:
                        live_catalog.save_replay(recording)
                    except OSError:
                        logger.error("Discovery job %s replay could not be saved", job_id)
                        job.add({"type": "warning", "ts": time.time(),
                                 "message": "The search ended, but its replay could not be saved."})
            if not job.done:   # after the watchdog's end, late worker events only free the slot
                job.add(event)
        if worker_finished:
            loop.call_soon_threadsafe(_release_lock, job_id)

    # Run the discovery loop in a background thread or task.
    # We use asyncio.to_thread because the DiscoveryAgent uses synchronous httpx/Tavily calls.
    def discovery_worker():
        def on_event(evt):
            budget.check()
            # A closed browser tab does not stop the search: its result is saved and the stream can be reopened.
            if evt.get("type") == "candidate":
                evt = live_catalog.save_candidate(evt)
                representative = live_catalog.duplicates().get(evt.get("catalog_id"))
                if representative:   # same claim as an earlier saved find: no second card, no paid readiness check
                    evt = {**evt, "duplicate_of": representative, "runnable": False,
                           "run_reason": live_catalog.duplicate_reason(representative)}
                else:
                    found.append(evt)
            emit(evt)

        found: list[dict] = []
        try:
            agent = DiscoveryAgent()   # im try: ein Fehler im Konstruktor darf die Sperre nie dauerhaft halten
            agent.discover(datasets, on_event=on_event, known_ids=_known_arxiv_ids())
            _preflight_found(found, emit, budget)
        except Exception as e:
            message = sanitize_text(f"{type(e).__name__}: {e}", 300)
            logger.error("Discovery job %s failed: %s", job_id, message)
            emit({"type": "error", "message": message})
        finally:
            if current_job.get("job_id") == job_id:
                current_job.clear()
            emit({"type": "done"}, worker_finished=True)
            loop.call_soon_threadsafe(lambda: loop.call_later(JOB_KEEP_SECONDS, jobs.pop, job_id, None))   # nie gelesene Warteschlangen wegraeumen

    loop = asyncio.get_running_loop()
    await job_semaphore.acquire()
    _held.add(job_id)
    _budgets[job_id] = budget
    current_job.update({"kind": "discovery", "job_id": job_id, "catalog_id": None, "started_at": time.time()})
    def on_timeout():
        emit({"type": "error", "ts": time.time(), "message": "Search took too long and was stopped."})
        emit({"type": "done", "ts": time.time()})
    _arm_watchdog(job_id, DISCOVERY_TIMEOUT_S, on_timeout)
    # Start the worker thread
    task = asyncio.create_task(asyncio.to_thread(_run_budgeted, job_id, discovery_worker))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

    return {"job_id": job_id}


PREFLIGHT_UNFINISHED = "The readiness check could not finish, so this claim is not offered for a one-click check."


def make_preflight_orchestrator():
    """Same configuration as a quick live check, so the check's saved state can start that run. Tests replace this."""
    return make_orchestrator(True)


def preflight_claim(cid: str) -> dict:
    """One readiness check of a saved discovery claim under its own spending limit; stores and returns the result."""
    result = {"ready": False, "stage": "infrastructure", "reason": PREFLIGHT_UNFINISHED}
    orch = None
    token = None
    try:
        limit = spend_guard.number("MAX_COST_PER_PREFLIGHT_USD", .35)
        token = spend_guard.current_budget.set(spend_guard.JobBudget(limit, live=True))
        orch = make_preflight_orchestrator()
        result = orch.preflight(cid, "live_catalog", save_path=live_catalog.preflight_path(cid))
    except Exception as exc:   # budget, provider, sandbox: not a finding about the paper
        logger.error("Readiness check for %s failed: %s", cid, sanitize_text(f"{type(exc).__name__}: {exc}", 300))
    finally:
        if token is not None:
            spend_guard.current_budget.reset(token)
        close = getattr(orch, "close", None)
        if close:
            close()
    return live_catalog.save_readiness(cid, result)


def _preflight_found(found: list[dict], emit, search_budget) -> None:
    """Run every step a check takes before its first paid code generation for each new runnable claim. Only claims
    that pass get a start button; the others show why. Each check has its own spending limit."""
    for event in found:
        cid = event.get("catalog_id")
        if not cid or not event.get("runnable"):
            continue
        if search_budget.cancelled.is_set():
            emit({"type": "readiness", "catalog_id": cid, "state": "blocked", "stage": "time", "reason": PREFLIGHT_UNFINISHED})
            continue
        emit({"type": "readiness", "catalog_id": cid, "state": "checking"})
        readiness = preflight_claim(cid)
        emit({"type": "readiness", "catalog_id": cid, "state": "ready" if readiness["ready"] else "blocked",
              "stage": readiness["stage"], "reason": sanitize_text(readiness["reason"] or "", 300) or None,
              "cost_gross_usd": readiness["cost_gross_usd"]})


def _arm_watchdog(job_id: str, seconds: float, on_timeout) -> None:
    """Cancel future paid requests; keep the slot until the actual worker exits."""
    async def watch():
        await asyncio.sleep(seconds)
        if job_id in _held:
            _budgets[job_id].cancelled.set()
            on_timeout()

    task = asyncio.create_task(watch())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)


@app.get("/api/discover/{job_id}/stream")
async def stream_discovery(job_id: str, request: Request, start: int = Query(0, alias="from")):
    """SSE from index `start` (or Last-Event-ID + 1); a reconnect after a dropped connection gets the remaining events."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    last = request.headers.get("last-event-id")
    first = int(last) + 1 if last and last.isascii() and last.isdigit() and len(last) < 9 else max(0, start)
    slot = stream_slots.acquire(live_gate.client_ip(request))
    if slot is None:
        raise HTTPException(status_code=429, detail="Too many open streams.")

    async def event_generator():
        try:
            async for chunk in _buffered_events(job, request, first, time.monotonic()):
                yield chunk
        finally:
            stream_slots.release(slot)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


# ---- Live-Reproduktion eines Katalogeintrags ------------------------------------------------------
# Discovery und Reproduktion teilen sich job_semaphore (ein Job gleichzeitig). Die API schreibt catalog.yaml NIE um
# (ein Schnelllauf mit 3 Seeds wuerde das kuratierte Urteil ueberschreiben); Live-Laeufe werden nur exportiert.
MAX_DAILY_REPRO_JOBS = int(os.environ.get("MAX_DAILY_REPRO_JOBS", 3))
QUICK_SEEDS = 3
JOB_KEEP_SECONDS = 3600
CATALOG_PATH = cat.CATALOG_YAML          # Tests biegen diese Pfade um
repro_daily = {"date": None, "count": 0}
current_job: Dict[str, Any] = {}         # {} = nichts laeuft; sonst kind, job_id, catalog_id, started_at


@dataclass
class ReproJob:
    id: str
    catalog_id: str
    quick: bool
    created: float = field(default_factory=time.time)
    events: list = field(default_factory=list)   # alle Events ab Start, Reconnect liefert sie erneut aus
    done: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, evt: dict) -> None:
        with self.lock:
            self.events.append(evt)


repro_jobs: Dict[str, ReproJob] = {}
REPRO_ARCHIVE_DIR = REPO_ROOT / 'logs' / 'reproduction_jobs'


def _archive_job(job: ReproJob) -> None:
    """Keep sanitized terminal evidence available after memory pruning/restarts."""
    try:
        with job.lock:
            payload = {'id': job.id, 'catalog_id': job.catalog_id, 'quick': job.quick,
                       'created': job.created, 'done': True, 'events': sanitize(job.events)}
        REPRO_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        path = REPRO_ARCHIVE_DIR / f'{job.id}.json.gz'
        temp = path.with_suffix('.tmp-' + uuid.uuid4().hex)
        try:
            temp.write_bytes(gzip.compress(json.dumps(payload, allow_nan=False).encode()))
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    except (OSError, ValueError, TypeError):
        logger.error('Could not archive completed reproduction job %s', job.id)


def _load_archived_job(job_id: str) -> ReproJob | None:
    try:
        if str(uuid.UUID(job_id)) != job_id:
            return None
        path = REPRO_ARCHIVE_DIR / f'{job_id}.json.gz'
        payload = json.loads(gzip.decompress(path.read_bytes()))
        if payload.get('id') != job_id or payload.get('done') is not True or not isinstance(payload.get('events'), list):
            return None
        return ReproJob(job_id, payload['catalog_id'], payload['quick'], created=payload['created'],
                        events=payload['events'], done=True)
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        return None


def make_orchestrator(quick: bool, author_code: bool = True):
    """Orchestrator fuer einen Live-Lauf. Tests ersetzen diese Funktion. Standard: Nachbau aus Paper und Autorencode
    (`loop.product_kwargs`); `author_code=False` schaltet auf den reinen Paper-Nachbau."""
    from .loop import Orchestrator, product_kwargs

    return Orchestrator(**product_kwargs(author_code), seeds=QUICK_SEEDS if quick else None, llm_cache=False,
                        objection_experiments=True, objection_quick=quick)   # Live: kein Cache, das Modell laeuft wirklich; Einwand-Experimente (Schnellmodus begrenzt)


def sanitize_text(text: str, limit: int | None = None) -> str:
    """Keine Geheimnisse und keine absoluten Pfade mit Nutzernamen nach aussen (web_guard.redact_text)."""
    out = web_guard.redact_text(text, str(REPO_ROOT))
    return out[:limit] if limit else out


def sanitize(obj):
    if isinstance(obj, str):
        return sanitize_text(obj)
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, Path):
        return sanitize_text(str(obj))
    return obj


class Refused(Exception):
    """Live-Start abgelehnt: maschinenlesbarer Grund, keine Kontostaende."""

    def __init__(self, status: int, reason: str, reset_at: str | None = None) -> None:
        self.status, self.reason, self.reset_at = status, reason, reset_at


@app.exception_handler(Refused)
async def _refused(_request: Request, exc: Refused):
    msg = live_gate.MESSAGES[exc.reason]
    return JSONResponse(status_code=exc.status, content={"reason": exc.reason, "message": msg, "detail": msg, "reset_at": exc.reset_at})


ip_limits = live_gate.IpLimits()


def refusal(kind: str, ip: str) -> Refused | None:
    """Warum ein Live-Start jetzt nicht moeglich ist (Reihenfolge: Budget, Tavily, Tageslimit, Limit je IP, Sperre). None = frei.
    Reine Lesefunktion: zaehlt nichts hoch."""
    reason = live_gate.floor_reason()
    if reason:
        return Refused(503, reason)
    today = datetime.date.today()
    daily, cap = (repro_daily, MAX_DAILY_REPRO_JOBS) if kind == "reproduction" else (daily_limit, MAX_DAILY_JOBS)
    used = daily["count"] if daily["date"] == today else 0
    if used >= cap or ip_limits.used_up(kind, ip):
        return Refused(429, "daily_limit", live_gate.next_reset_iso())
    if job_semaphore.locked():
        return Refused(429, "busy")
    return None


def check_live(kind: str, ip: str) -> None:
    r = refusal(kind, ip)
    if r:
        raise r


@app.get("/api/status")
async def live_status(request: Request):
    """Ist Live jetzt moeglich? Ohne Geheimnisse und ohne Kontostand. `live` gilt fuer mindestens eine Art; `kinds` je Art."""
    ip = live_gate.client_ip(request)
    kinds = {}
    for kind in ("reproduction", "discovery"):
        r = refusal(kind, ip)
        kinds[kind] = {"live": r is None, "reason": r.reason if r else None, "reset_at": r.reset_at if r else None}
    open_kinds = [k for k in kinds.values() if k["live"]]
    if open_kinds:
        return {"live": True, "reason": None, "reset_at": None, "kinds": kinds}
    # Alle Arten gesperrt: der gemeinsame, schwerste Grund (Budget vor Tavily vor Tageslimit vor Sperre)
    order = ["budget_floor", "tavily_floor", "daily_limit", "busy"]
    reason = min((k["reason"] for k in kinds.values()), key=order.index)
    reset = next((k["reset_at"] for k in kinds.values() if k["reason"] == reason and k["reset_at"]), None)
    return {"live": False, "reason": reason, "reset_at": reset, "kinds": kinds}


def full_runs_allowed() -> bool:
    return os.environ.get("ALLOW_FULL_RUNS") == "1"


class ReproduceRequest(BaseModel):
    catalog_id: str = Field(max_length=120)
    quick: bool = True             # der Server erzwingt die Schnellpruefung, solange ALLOW_FULL_RUNS nicht 1 ist
    author_code: bool = True   # False: reiner Paper-Nachbau (Schalter, kein Standard)


@app.get("/api/discover/candidates")
async def saved_discovery_candidates():
    try:
        return sanitize(live_catalog.candidate_events())
    except (OSError, ValueError):
        raise HTTPException(503, detail="Saved discovery claims are unavailable.")


@app.get("/api/discover/replay")
async def latest_discovery_replay():
    try:
        return sanitize(live_catalog.latest_replay())
    except (OSError, ValueError):
        raise HTTPException(503, detail="Saved discovery replay is unavailable.")


@app.get("/api/reproduce/eligibility")
async def reproduction_eligibility():
    try:
        result = {entry["id"]: live_catalog.curated_offer(entry) for entry in cat.load_catalog(CATALOG_PATH)}
        # A saved live find with a curated entry of the same id is judged by the curated entry (its lock stays).
        entries = live_catalog.load_entries()
        hidden = live_catalog.duplicates(entries)
        result.update({entry["id"]: ({"runnable": False, "reason": live_catalog.duplicate_reason(hidden[entry["id"]]),
                                      "warnings": [], "duplicate_of": hidden[entry["id"]]}
                                     if entry["id"] in hidden else live_catalog.offer(entry))
                       for entry in entries if entry["id"] not in result})
        return sanitize(result)
    except (OSError, ValueError):
        raise HTTPException(503, detail="Check eligibility is unavailable.")


def _prune_jobs() -> None:
    now = time.time()
    for jid in [j for j, job in repro_jobs.items() if job.done and now - job.created > JOB_KEEP_SECONDS]:
        del repro_jobs[jid]


@app.post("/api/reproduce")
async def start_reproduction(req: ReproduceRequest, request: Request):
    ip = live_gate.client_ip(request)
    check_live("reproduction", ip)
    budget = _new_budget("reproduction")
    if not full_runs_allowed():
        req.quick = True
    entry = next((e for e in cat.load_catalog(CATALOG_PATH) if e["id"] == req.catalog_id), None)
    dynamic = entry is None
    if dynamic:
        try:
            entry = live_catalog.find_entry(req.catalog_id)
        except (OSError, ValueError):
            raise HTTPException(503, detail="Saved discovery claims are unavailable.")
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown catalog entry")
    if dynamic:
        try:
            representative = live_catalog.duplicates().get(entry["id"])
        except (OSError, ValueError):
            raise HTTPException(503, detail="Saved discovery claims are unavailable.")
        if representative:
            raise HTTPException(status_code=409, detail=live_catalog.duplicate_reason(representative))
    eligible = live_catalog.offer(entry) if dynamic else live_catalog.curated_offer(entry)
    if not eligible["runnable"]:
        raise HTTPException(status_code=400, detail=f"Not runnable: {sanitize_text(eligible['reason'], 300)}")

    today = datetime.date.today()
    if repro_daily["date"] != today:
        repro_daily["date"], repro_daily["count"] = today, 0
    _record_start("reproduction", ip)
    repro_daily["count"] += 1
    ip_limits.add("reproduction", ip)

    _prune_jobs()
    job = ReproJob(id=str(uuid.uuid4()), catalog_id=entry["id"], quick=req.quick)
    repro_jobs[job.id] = job
    job.add({"ts": time.time(), "type": "queued", "catalog_id": entry["id"], "quick": req.quick,
                 "entry": sanitize(site_export.catalog_json([entry])[0]),
                 "mode_requested": "nachbau_autorencode" if req.author_code else "nachbau"})
    loop = asyncio.get_running_loop()
    await job_semaphore.acquire()
    _held.add(job.id)
    _budgets[job.id] = budget
    current_job.update({"kind": "reproduction", "job_id": job.id, "catalog_id": entry["id"], "started_at": time.time()})

    def on_timeout():
        job.add({"ts": time.time(), "type": "error", "message": "The check took too long and was stopped.", "status": "timeout"})
        job.done = True
        _archive_job(job)
    _arm_watchdog(job.id, JOB_TIMEOUT_S, on_timeout)

    def worker():
        orch = None
        try:
            job.add({"ts": time.time(), "type": "started", "catalog_id": entry["id"]})
            orch = make_orchestrator(req.quick, req.author_code)
            orch.on_event = lambda evt: job.add(sanitize(evt))
            prepared = live_catalog.preflight_path(entry["id"]) if dynamic else None
            outcome = (cat.run_candidate(orch, entry, source="live_catalog",
                                         prepared_path=prepared if prepared and prepared.exists() else None)
                       if dynamic else cat.run_candidate(orch, entry))
            budget.check()
            if dynamic:
                live_catalog.save_outcome(entry, outcome)
                site_export.write_catalog(CATALOG_PATH, site_export.OUT, live_catalog.offered_entries())
            if outcome.status == cat.CHECKED and outcome.dossier is not None and outcome.dossier.verdicts:
                seeds = getattr(orch, "seeds", None) or cat.build_paper(entry)[0]["seeds"]
                note = f"live quick check, {seeds} seeds" if req.quick else f"live run, {seeds} seeds"
                ref = site_export.export_live_run(outcome.run_dir, CATALOG_PATH, site_export.OUT, note, seeds,
                                                  extra_entries=live_catalog.offered_entries())
                job.add({"ts": time.time(), "type": "finished", "run_id": ref["id"],
                         "verdict": sanitize(cat.verdict_dict(entry, outcome.dossier, outcome.run_dir)),
                         "cost_gross_usd": round(outcome.cost_gross_usd, 4)})
            else:
                job.add({"ts": time.time(), "type": "error", "message": sanitize_text(outcome.reason or "Run failed", 300),
                         "status": outcome.status})
        except Exception as exc:   # kein Absturz der API durch einen Lauf
            if not job.done:
                job.add({"ts": time.time(), "type": "error", "message": sanitize_text(f"{type(exc).__name__}: {exc}", 300)})
        finally:
            job.done = True
            _archive_job(job)
            if orch is not None and hasattr(orch, 'close'):
                try:
                    orch.close()
                except Exception:
                    logger.error('Could not close resources for reproduction job %s', job.id)
            if current_job.get("job_id") == job.id:
                current_job.clear()
            loop.call_soon_threadsafe(_release_lock, job.id)

    task = asyncio.create_task(asyncio.to_thread(_run_budgeted, job.id, worker))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return {"job_id": job.id}


@app.get("/api/reproduce/{job_id}/stream")
async def stream_reproduction(job_id: str, request: Request, start: int = Query(0, alias="from")):
    """SSE ab Index `start` (oder Last-Event-ID + 1). Alle Events bleiben im Speicher, ein Reconnect liefert sie erneut."""
    job = repro_jobs.get(job_id) or _load_archived_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    last = request.headers.get("last-event-id")
    first = int(last) + 1 if last and last.isascii() and last.isdigit() and len(last) < 9 else max(0, start)
    slot = stream_slots.acquire(live_gate.client_ip(request))
    if slot is None:
        raise HTTPException(status_code=429, detail="Too many open streams.")

    async def event_generator():
        try:
            async for chunk in _buffered_events(job, request, first, time.monotonic()):
                yield chunk
        finally:
            stream_slots.release(slot)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


async def _buffered_events(job: "ReproJob | DiscoveryJob", request: Request, i: int, started: float):
    idle = 0.0
    while True:
        if await request.is_disconnected() or time.monotonic() - started > STREAM_MAX_S:
            break
        with job.lock:
            batch = job.events[i:]
        for evt in batch:
            yield f"id: {i}\ndata: {json.dumps(evt, default=str)}\n\n"
            i += 1
            idle = 0.0
        if job.done and not batch:
            with job.lock:
                if i >= len(job.events):
                    break
        await asyncio.sleep(0.2)
        idle += 0.2
        if idle >= 15:
            yield ": keepalive\n\n"
            idle = 0.0


@app.get("/api/health")
async def health():
    """Healthcheck fuer Container/Proxy: kein Job-Zustand, keine Dateizugriffe."""
    return {"ok": True}


@app.get("/api/jobs/current")
async def current_job_status():
    if not current_job:
        return {"busy": False}
    return {"busy": True, **current_job}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
