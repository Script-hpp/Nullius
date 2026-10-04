import asyncio
import getpass
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import catalog as cat
from . import live_gate, site_export
from .discovery import DiscoveryAgent
from .dossier import REPO_ROOT

app = FastAPI()

# Global limits
import datetime
job_semaphore = asyncio.Semaphore(1)
daily_limit = {"date": None, "count": 0}
import os
MAX_DAILY_JOBS = int(os.environ.get("MAX_DAILY_JOBS", 5))

# In-memory registry for jobs and their queues
jobs: Dict[str, asyncio.Queue] = {}
background_tasks = set()

class DiscoverRequest(BaseModel):
    datasets: list[str] = ["Cora", "ETTh1"]

@app.post("/api/discover")
async def start_discovery(req: DiscoverRequest, request: Request):
    from .tasks import TASKS
    for ds in req.datasets:
        if ds not in TASKS:
            raise HTTPException(status_code=400, detail=f"Dataset {ds} is not supported.")
            
    ip = live_gate.client_ip(request)
    check_live("discovery", ip)
    today = datetime.date.today()
    if daily_limit["date"] != today:
        daily_limit["date"], daily_limit["count"] = today, 0
    daily_limit["count"] += 1
    ip_limits.add("discovery", ip)
    
    job_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    jobs[job_id] = queue
    
    # Run the discovery loop in a background thread or task.
    # We use asyncio.to_thread because the DiscoveryAgent uses synchronous httpx/Tavily calls.
    def discovery_worker():
        agent = DiscoveryAgent()
        def on_event(evt):
            # Put the event into the queue in a thread-safe manner
            if job_id not in jobs:
                raise RuntimeError("Client disconnected, aborting discovery loop.")
            try:
                loop.call_soon_threadsafe(queue.put_nowait, evt)
            except Exception:
                pass
                
        try:
            agent.discover(req.datasets, on_event=on_event)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
        finally:
            current_job.clear()
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})
            loop.call_soon_threadsafe(job_semaphore.release)
            
    loop = asyncio.get_running_loop()
    await job_semaphore.acquire()
    current_job.update({"kind": "discovery", "job_id": job_id, "catalog_id": None, "started_at": time.time()})
    # Start the worker thread
    task = asyncio.create_task(asyncio.to_thread(discovery_worker))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    
    return {"job_id": job_id}


@app.get("/api/discover/{job_id}/stream")
async def stream_discovery(job_id: str, request: Request):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    queue = jobs[job_id]
    
    async def event_generator():
        try:
            while True:
                # Disconnect if client leaves
                if await request.is_disconnected():
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


def make_orchestrator(quick: bool):
    """Orchestrator fuer einen Live-Lauf. Tests ersetzen diese Funktion."""
    from .loop import Orchestrator

    return Orchestrator(graph_repair=True, setup_v2=True, seeds=QUICK_SEEDS if quick else None, llm_cache=False)   # Live: kein Cache, das Modell laeuft wirklich


def sanitize_text(text: str, limit: int | None = None) -> str:
    """Keine absoluten Pfade mit Nutzernamen nach aussen: Repo-Wurzel -> ".", Home -> "~"."""
    out = text.replace(str(REPO_ROOT), ".")
    home = str(Path.home())
    out = out.replace(home, "~")
    try:
        user = getpass.getuser()
        if len(user) > 2:
            out = out.replace(user, "user")
    except Exception:
        pass
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


class ReproduceRequest(BaseModel):
    catalog_id: str
    quick: bool = True


def _prune_jobs() -> None:
    now = time.time()
    for jid in [j for j, job in repro_jobs.items() if job.done and now - job.created > JOB_KEEP_SECONDS]:
        del repro_jobs[jid]


@app.post("/api/reproduce")
async def start_reproduction(req: ReproduceRequest, request: Request):
    ip = live_gate.client_ip(request)
    check_live("reproduction", ip)
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

    _prune_jobs()
    job = ReproJob(id=str(uuid.uuid4()), catalog_id=entry["id"], quick=req.quick)
    repro_jobs[job.id] = job
    job.add({"ts": time.time(), "type": "queued", "catalog_id": entry["id"], "quick": req.quick})
    loop = asyncio.get_running_loop()
    await job_semaphore.acquire()
    current_job.update({"kind": "reproduction", "job_id": job.id, "catalog_id": entry["id"], "started_at": time.time()})

    def worker():
        try:
            job.add({"ts": time.time(), "type": "started", "catalog_id": entry["id"]})
            orch = make_orchestrator(req.quick)
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
            current_job.clear()
            loop.call_soon_threadsafe(job_semaphore.release)

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
    first = int(last) + 1 if last and last.isdigit() else max(0, start)

    async def event_generator():
        i, idle = first, 0.0
        while True:
            if await request.is_disconnected():
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

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/jobs/current")
async def current_job_status():
    if not current_job:
        return {"busy": False}
    return {"busy": True, **current_job}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
