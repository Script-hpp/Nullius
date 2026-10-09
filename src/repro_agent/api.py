import asyncio
import json
import os
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
from . import live_gate, site_export, web_guard
from .discovery import DiscoveryAgent
from .dossier import REPO_ROOT


@asynccontextmanager
async def lifespan(_app):
    restore_counters()   # Tageslimits ueberleben Neustart/Redeploy (logs/live_jobs.jsonl)
    yield


# Keine automatische Doku (/docs, /redoc, /openapi.json) auf dem oeffentlichen Server. Kein CORS: die Web-App ist same-origin.
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.add_middleware(web_guard.GuardMiddleware, state=app.state)   # Rate-Limit je IP, Body-Limit, Security-Header

# Global limits
job_semaphore = asyncio.Semaphore(1)
daily_limit = {"date": None, "count": 0}
MAX_DAILY_JOBS = int(os.environ.get("MAX_DAILY_JOBS", 5))
JOB_TIMEOUT_S = 2700            # Wanduhr je Reproduktion: danach meldet der Stream einen Fehler und die Sperre wird frei
DISCOVERY_TIMEOUT_S = 900
STREAM_MAX_S = 3600             # laengste Dauer einer SSE-Verbindung (Reconnect mit from=<n> liefert den Rest)
_held: set[str] = set()         # Job-IDs, die den Semaphor halten (genau einmal freigeben, auch bei Watchdog + Worker-Ende)


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
jobs: Dict[str, asyncio.Queue] = {}
background_tasks = set()

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
    today = datetime.date.today()
    if daily_limit["date"] != today:
        daily_limit["date"], daily_limit["count"] = today, 0
    daily_limit["count"] += 1
    ip_limits.add("discovery", ip)
    live_gate.record_start("discovery", ip)

    job_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    jobs[job_id] = queue

    # Run the discovery loop in a background thread or task.
    # We use asyncio.to_thread because the DiscoveryAgent uses synchronous httpx/Tavily calls.
    def discovery_worker():
        def on_event(evt):
            # Put the event into the queue in a thread-safe manner
            if job_id not in jobs:
                raise RuntimeError("Client disconnected, aborting discovery loop.")
            try:
                loop.call_soon_threadsafe(queue.put_nowait, sanitize(evt))
            except Exception:
                pass

        try:
            agent = DiscoveryAgent()   # im try: ein Fehler im Konstruktor darf die Sperre nie dauerhaft halten
            agent.discover(datasets, on_event=on_event)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": sanitize_text(f"{type(e).__name__}: {e}", 300)})
        finally:
            if current_job.get("job_id") == job_id:
                current_job.clear()
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})
            loop.call_soon_threadsafe(_release_lock, job_id)
            loop.call_soon_threadsafe(lambda: loop.call_later(JOB_KEEP_SECONDS, jobs.pop, job_id, None))   # nie gelesene Warteschlangen wegraeumen

    loop = asyncio.get_running_loop()
    await job_semaphore.acquire()
    _held.add(job_id)
    current_job.update({"kind": "discovery", "job_id": job_id, "catalog_id": None, "started_at": time.time()})
    _arm_watchdog(job_id, DISCOVERY_TIMEOUT_S, lambda: queue.put_nowait({"type": "error", "message": "Search took too long and was stopped."}))
    # Start the worker thread
    task = asyncio.create_task(asyncio.to_thread(discovery_worker))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

    return {"job_id": job_id}


def _arm_watchdog(job_id: str, seconds: float, on_timeout) -> None:
    """Haengt ein Job (Sandbox, Netz), meldet der Stream nach `seconds` einen Fehler und die Sperre wird frei.
    Der Worker-Thread laeuft weiter, ist aber durch MAX_COST_PER_RUN_USD und das globale Budget nach oben begrenzt."""
    async def watch():
        await asyncio.sleep(seconds)
        if job_id in _held:
            try:
                on_timeout()
            finally:
                if current_job.get("job_id") == job_id:
                    current_job.clear()
                _release_lock(job_id)

    task = asyncio.create_task(watch())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)


@app.get("/api/discover/{job_id}/stream")
async def stream_discovery(job_id: str, request: Request):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    slot = stream_slots.acquire(live_gate.client_ip(request))
    if slot is None:
        raise HTTPException(status_code=429, detail="Too many open streams.")
    queue = jobs[job_id]
    started = time.monotonic()

    async def event_generator():
        try:
            while True:
                # Disconnect if client leaves
                if await request.is_disconnected() or time.monotonic() - started > STREAM_MAX_S:
                    break

                try:
                    # Wait for an event with a 15-second timeout for keepalive
                    evt = await asyncio.wait_for(queue.get(), timeout=15.0)
                    
                    if evt.get("type") == "done":
                        break
                        
                    yield f"data: {json.dumps(evt)}\n\n"
                    
                except asyncio.TimeoutError:
                    # Emit keepalive to prevent proxy disconnects
                    yield ": keepalive\n\n"
        finally:
            # Cleanup when the stream is closed
            stream_slots.release(slot)
            if job_id in jobs:
                del jobs[job_id]
                
    return StreamingResponse(event_generator(), media_type="text/event-stream")


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


def _prune_jobs() -> None:
    now = time.time()
    for jid in [j for j, job in repro_jobs.items() if job.done and now - job.created > JOB_KEEP_SECONDS]:
        del repro_jobs[jid]


@app.post("/api/reproduce")
async def start_reproduction(req: ReproduceRequest, request: Request):
    ip = live_gate.client_ip(request)
    check_live("reproduction", ip)
    if not full_runs_allowed():
        req.quick = True
    entry = next((e for e in cat.load_catalog(CATALOG_PATH) if e["id"] == req.catalog_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown catalog entry")
    try:
        cat.build_paper(entry)
    except cat.NotRunnable as exc:
        raise HTTPException(status_code=400, detail=f"Not runnable: {sanitize_text(str(exc), 300)}")

    today = datetime.date.today()
    if repro_daily["date"] != today:
        repro_daily["date"], repro_daily["count"] = today, 0
    repro_daily["count"] += 1
    ip_limits.add("reproduction", ip)
    live_gate.record_start("reproduction", ip)

    _prune_jobs()
    job = ReproJob(id=str(uuid.uuid4()), catalog_id=entry["id"], quick=req.quick)
    repro_jobs[job.id] = job
    job.add({"ts": time.time(), "type": "queued", "catalog_id": entry["id"], "quick": req.quick,
                 "mode_requested": "nachbau_autorencode" if req.author_code else "nachbau"})
    loop = asyncio.get_running_loop()
    await job_semaphore.acquire()
    _held.add(job.id)
    current_job.update({"kind": "reproduction", "job_id": job.id, "catalog_id": entry["id"], "started_at": time.time()})

    def on_timeout():
        job.add({"ts": time.time(), "type": "error", "message": "The check took too long and was stopped.", "status": "timeout"})
        job.done = True
    _arm_watchdog(job.id, JOB_TIMEOUT_S, on_timeout)

    def worker():
        try:
            job.add({"ts": time.time(), "type": "started", "catalog_id": entry["id"]})
            orch = make_orchestrator(req.quick, req.author_code)
            orch.on_event = lambda evt: job.add(sanitize(evt))
            outcome = cat.run_candidate(orch, entry)
            if outcome.status == cat.CHECKED and outcome.dossier is not None and outcome.dossier.verdicts:
                seeds = getattr(orch, "seeds", None) or cat.build_paper(entry)[0]["seeds"]
                note = f"live quick check, {seeds} seeds" if req.quick else f"live run, {seeds} seeds"
                ref = site_export.export_live_run(outcome.run_dir, CATALOG_PATH, site_export.OUT, note, seeds)
                job.add({"ts": time.time(), "type": "finished", "run_id": ref["id"],
                         "verdict": sanitize(cat.verdict_dict(entry, outcome.dossier, outcome.run_dir)),
                         "cost_gross_usd": round(outcome.cost_gross_usd, 4)})
            else:
                job.add({"ts": time.time(), "type": "error", "message": sanitize_text(outcome.reason or "Run failed", 300),
                         "status": outcome.status})
        except Exception as exc:   # kein Absturz der API durch einen Lauf
            job.add({"ts": time.time(), "type": "error", "message": sanitize_text(f"{type(exc).__name__}: {exc}", 300)})
        finally:
            job.done = True
            if current_job.get("job_id") == job.id:
                current_job.clear()
            loop.call_soon_threadsafe(_release_lock, job.id)

    task = asyncio.create_task(asyncio.to_thread(worker))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return {"job_id": job.id}


@app.get("/api/reproduce/{job_id}/stream")
async def stream_reproduction(job_id: str, request: Request, start: int = Query(0, alias="from")):
    """SSE ab Index `start` (oder Last-Event-ID + 1). Alle Events bleiben im Speicher, ein Reconnect liefert sie erneut."""
    job = repro_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    last = request.headers.get("last-event-id")
    first = int(last) + 1 if last and last.isascii() and last.isdigit() and len(last) < 9 else max(0, start)
    slot = stream_slots.acquire(live_gate.client_ip(request))
    if slot is None:
        raise HTTPException(status_code=429, detail="Too many open streams.")

    async def event_generator():
        try:
            async for chunk in _repro_events(job, request, first, time.monotonic()):
                yield chunk
        finally:
            stream_slots.release(slot)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


async def _repro_events(job: ReproJob, request: Request, i: int, started: float):
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
