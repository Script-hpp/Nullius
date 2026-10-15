"""Live-Reproduktion ueber die API mit Fake-Orchestrator: Sperre, Limits, Zugang, Stream, Reconnect, Sanitizing. Kein Netz, kein Geld."""

import asyncio
import dataclasses
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from repro_agent import api, live_gate, site_export
from repro_agent import catalog as cat
from repro_agent.dossier import REPO_ROOT, Attempt
from repro_agent.llm import BillingBlocked
from test_catalog import cand, dossier_with, entry


class FakeOrch:
    """Schreibt events.jsonl und dossier.json wie der Loop und ruft den Live-Hook."""
    instances: list = []
    mode = "ok"
    gate: threading.Event | None = None

    def __init__(self, quick, author_code=True):
        self.author_code = author_code
        self.seeds = 3 if quick else 10
        self.on_event = None
        self.last_run_dir = None
        self.spent_gross_usd = 0.0123
        FakeOrch.instances.append(self)

    def preflight(self, paper_id, source="live_catalog", save_path=None):
        return {"ready": True, "stage": "ready", "reason": None, "cost_gross_usd": 0.0}

    def run(self, paper_id, source="benchmark", prepared_path=None):
        assert source in ("catalog", "live_catalog")
        run_dir = self.root / "runs" / paper_id / "20261001-120000"
        run_dir.mkdir(parents=True)
        self.last_run_dir = run_dir
        evts = [("run_start", {"paper_id": paper_id, "seeds": self.seeds}), ("plan", {"text": f"see {REPO_ROOT}/secret and {Path.home()}/x"}),
                ("attempt", {"seed": 0, "measured": 0.8})]
        with (run_dir / "events.jsonl").open("w") as f:
            for t, d in evts:
                rec = {"ts": time.time(), "type": t, **d}
                f.write(json.dumps(rec) + "\n")
                self.on_event(rec)
        self.on_event({"ts": time.time(), "type": "llm_call", "role": "code", "cost_net_usd": 0.001, "total_net_usd": 0.001})
        if FakeOrch.gate is not None:
            FakeOrch.gate.wait(5)
        if FakeOrch.mode == "billing":
            raise BillingBlocked("gesperrt")
        if FakeOrch.mode == "crash":
            raise RuntimeError(f"boom at {REPO_ROOT}/src/x.py")
        d = dossier_with("nicht reproduziert", (0.78, 0.79))
        d.paper_id = paper_id
        d.attempts = [Attempt(claim_id=d.claims[0].id, branch_id=f"b{i}", parent_snapshot="s", code_hash="h", seed=i,
                              status="abweichung", measured=m, error_summary=None) for i, m in enumerate((0.78, 0.79))]
        (run_dir / "dossier.json").write_text(json.dumps(dataclasses.asdict(d)))
        (run_dir / "zeugnis.md").write_text("z")
        return d


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr("repro_agent.llm.COST_LOG_PATH", tmp_path / "costs.jsonl")
    monkeypatch.setattr("repro_agent.tavily_usage.TAVILY_LOG_PATH", tmp_path / "tavily_usage.jsonl")
    monkeypatch.setenv("BUDGET_TOTAL_USD", "30")
    monkeypatch.setenv("TAVILY_CREDITS_TOTAL", "1000")
    monkeypatch.setenv("MAX_DAILY_REPRO_PER_IP", "100")       # TestClient ist immer dieselbe IP
    monkeypatch.setattr(api, "ip_limits", live_gate.IpLimits())
    e = entry(cand())
    etth = entry(cand("2601.21866", "ETTh1", claim={"method": "M", "metric": "mse", "target": 0.35, "horizon": 192, "protocol": "unklar"}, protocol="unklar"))
    f = tmp_path / "catalog.yaml"
    cat.save_catalog([e, etth], f)
    monkeypatch.setattr(api, "CATALOG_PATH", f)
    monkeypatch.setattr(site_export, "OUT", tmp_path / "data")
    monkeypatch.setattr(api, "job_semaphore", asyncio.Semaphore(1))
    monkeypatch.setattr(api, "repro_daily", {"date": None, "count": 0})
    monkeypatch.setattr(api, "MAX_DAILY_REPRO_JOBS", 3)
    api.repro_jobs.clear()
    api.current_job.clear()
    FakeOrch.instances, FakeOrch.mode, FakeOrch.gate = [], "ok", None

    def make(quick, author_code=True):
        o = FakeOrch(quick, author_code)
        o.root = tmp_path
        return o
    monkeypatch.setattr(api, "make_orchestrator", make)
    with TestClient(api.app) as client:
        yield SimpleNamespace(client=client, tmp=tmp_path, cat=f, id=e["id"], etth=etth["id"])


def start(env, **over):
    body = {"catalog_id": env.id, "quick": True, **over}
    return env.client.post("/api/reproduce", json=body)


def read_stream(env, job_id, **params):
    events = []
    with env.client.stream("GET", f"/api/reproduce/{job_id}/stream", params=params) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def test_author_code_is_the_default_and_the_switch_reaches_the_orchestrator(env):
    job = start(env).json()["job_id"]
    evts = read_stream(env, job)
    assert FakeOrch.instances[0].author_code is True and evts[0]["mode_requested"] == "nachbau_autorencode"
    job = start(env, author_code=False).json()["job_id"]
    evts = read_stream(env, job)
    assert FakeOrch.instances[1].author_code is False and evts[0]["mode_requested"] == "nachbau"


def test_no_access_code_needed_and_unknown_id(env):
    r = start(env, catalog_id="disc_nope")
    assert r.status_code == 404
    assert FakeOrch.instances == []


def test_not_runnable_entry_is_400_with_reason(env):
    r = start(env, catalog_id=env.etth)
    assert r.status_code == 400 and "Horizont 192" in r.json()["detail"]


def test_full_run_streams_events_and_exports_without_touching_catalog(env):
    before = env.cat.read_text()
    job = start(env).json()["job_id"]
    evts = read_stream(env, job)
    types = [e["type"] for e in evts]
    assert types[:2] == ["queued", "started"] and types[-1] == "finished"
    assert {"run_start", "plan", "attempt", "llm_call"} <= set(types)
    fin = evts[-1]
    assert fin["run_id"] == "disc_2511_08094--20261001-120000" and fin["verdict"]["urteil"] == "nicht reproduziert"
    assert FakeOrch.instances[0].seeds == 3                                   # quick = 3 Seeds
    data = env.tmp / "data"
    index = json.loads((data / "index.json").read_text())
    (ref,) = index
    assert ref["id"] == fin["run_id"] and ref["live"] is True and ref["note"] == "live quick check, 3 seeds" and ref["seeds"] == 3
    assert ref["catalog_id"] == "disc_2511_08094" and (data / ref["paper_id"] / ref["timestamp"] / "dossier.json").exists()
    assert env.cat.read_text() == before                                      # API schreibt catalog.yaml nie
    assert not list(data.glob("*.tmp"))


def test_full_run_note_and_seeds(env, monkeypatch):
    monkeypatch.setenv("ALLOW_FULL_RUNS", "1")
    job = start(env, quick=False).json()["job_id"]
    read_stream(env, job)
    (ref,) = json.loads((env.tmp / "data" / "index.json").read_text())
    assert FakeOrch.instances[0].seeds == 10 and ref["note"] == "live run, 10 seeds"


def test_paths_are_sanitized_in_stream(env):
    job = start(env).json()["job_id"]
    plan = next(e for e in read_stream(env, job) if e["type"] == "plan")
    assert str(REPO_ROOT) not in plan["text"] and str(Path.home()) not in plan["text"] and "./secret" in plan["text"]


def test_second_job_is_refused_while_running_and_current_reports_busy(env):
    FakeOrch.gate = threading.Event()
    job = start(env).json()["job_id"]
    for _ in range(50):
        if FakeOrch.instances and FakeOrch.instances[0].on_event:
            break
        time.sleep(0.05)
    cur = env.client.get("/api/jobs/current").json()
    assert cur["busy"] is True and cur["kind"] == "reproduction" and cur["catalog_id"] == env.id and cur["job_id"] == job
    assert start(env).status_code == 429                                      # Sperre
    FakeOrch.gate.set()
    read_stream(env, job)
    assert env.client.get("/api/jobs/current").json() == {"busy": False}
    assert start(env).status_code == 200                                      # Sperre wieder frei


def test_discovery_and_reproduction_share_the_lock(env, monkeypatch):
    class Locked:
        def locked(self):
            return True
    monkeypatch.setattr(api, "job_semaphore", Locked())
    assert start(env).status_code == 429


def test_daily_limit(env, monkeypatch):
    monkeypatch.setattr(api, "MAX_DAILY_REPRO_JOBS", 2)
    for _ in range(2):
        read_stream(env, start(env).json()["job_id"])
    r = start(env)
    assert r.status_code == 429 and r.json()["reason"] == "daily_limit"


def test_crash_becomes_error_event_with_short_sanitized_message(env):
    FakeOrch.mode = "crash"
    evts = read_stream(env, start(env).json()["job_id"])
    err = evts[-1]
    assert err["type"] == "error" and err["status"] == "lauf_fehlgeschlagen" and "boom" in err["message"]
    assert str(REPO_ROOT) not in err["message"] and len(err["message"]) <= 300
    assert not (env.tmp / "data" / "index.json").exists()                     # nichts exportiert
    assert api.current_job == {}                                              # Sperre frei


def test_billing_blocked_is_error_and_frees_the_lock(env):
    FakeOrch.mode = "billing"
    err = read_stream(env, start(env).json()["job_id"])[-1]
    assert err["type"] == "error" and "BillingBlocked" in err["message"] and err["status"] == "ungeprueft"
    assert start(env).status_code == 200


def test_reconnect_from_index_and_last_event_id(env):
    job = start(env).json()["job_id"]
    all_ = read_stream(env, job)
    part = read_stream(env, job, **{"from": 3})
    assert part == all_[3:]
    with env.client.stream("GET", f"/api/reproduce/{job}/stream", headers={"Last-Event-ID": "1"}) as r:
        got = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert got == all_[2:]


def test_stream_unknown_job_404(env):
    assert env.client.get("/api/reproduce/nope/stream").status_code == 404


def test_current_when_idle(env):
    assert env.client.get("/api/jobs/current").json() == {"busy": False}


def test_budget_floor_blocks_reproduction_before_any_orchestrator(env, monkeypatch):
    monkeypatch.setenv("LIVE_MIN_BUDGET_USD", "31")
    r = start(env)
    assert r.status_code == 503 and r.json()["reason"] == "budget_floor"
    assert FakeOrch.instances == [] and not api.repro_jobs and api.repro_daily["count"] == 0


def test_ip_limit_one_reproduction_per_day(env, monkeypatch):
    monkeypatch.setenv("MAX_DAILY_REPRO_PER_IP", "1")
    read_stream(env, start(env).json()["job_id"])
    r = start(env)
    assert r.status_code == 429 and r.json()["reason"] == "daily_limit"


def test_quick_check_is_enforced_unless_full_runs_are_allowed(env):
    read_stream(env, start(env, quick=False).json()["job_id"])
    assert FakeOrch.instances[0].seeds == 3                                   # quick=false wird auf die Schnellpruefung gesetzt


def test_catalog_id_length_is_limited(env):
    assert start(env, catalog_id="x" * 500).status_code == 422


def test_watchdog_holds_the_lock_until_worker_exits(env, monkeypatch):
    monkeypatch.setattr(api, "JOB_TIMEOUT_S", 0.3)
    FakeOrch.gate = threading.Event()
    job = start(env).json()["job_id"]
    evts = read_stream(env, job)                                               # endet, obwohl der Worker haengt
    assert evts[-1]["type"] == "error" and evts[-1]["status"] == "timeout"
    assert api.current_job["job_id"] == job and api.job_semaphore.locked()
    assert api._budgets[job].cancelled.is_set()
    assert start(env).status_code == 429
    FakeOrch.gate.set()                                                        # Worker endet spaeter: kein doppeltes Freigeben
    time.sleep(0.5)
    assert api.job_semaphore._value == 1
    assert api.current_job == {}
    assert not any(e["type"] == "finished" for e in api.repro_jobs[job].events)


def test_stream_slot_limit_returns_429(env, monkeypatch):
    job = start(env).json()["job_id"]
    read_stream(env, job)
    monkeypatch.setenv("MAX_SSE_STREAMS", "0")
    assert env.client.get(f"/api/reproduce/{job}/stream").status_code == 429


def test_last_event_id_garbage_does_not_crash(env):
    job = start(env).json()["job_id"]
    read_stream(env, job)
    for bad in ("9" * 30, "-1", "x"):
        with env.client.stream("GET", f"/api/reproduce/{job}/stream", headers={"Last-Event-ID": bad}) as r:
            assert r.status_code == 200


def test_failed_admission_log_blocks_paid_job(env, monkeypatch):
    def fail(*args):
        raise OSError("read-only storage")
    monkeypatch.setattr(live_gate, "record_start", fail)
    assert start(env).status_code == 503
    assert FakeOrch.instances == []
    assert not api.job_semaphore.locked()


@pytest.mark.parametrize("value", ["nan", "bad", "0", "-1"])
def test_invalid_job_budget_blocks_without_holding_slot(env, monkeypatch, value):
    monkeypatch.setenv("MAX_COST_PER_RUN_USD", value)
    assert start(env).status_code == 503
    assert FakeOrch.instances == []
    assert not api.job_semaphore.locked()


@pytest.mark.parametrize('blocked', ['target', 'method'])
def test_unverified_target_or_missing_method_is_not_exported_as_a_paper_result(env, monkeypatch, blocked):
    from repro_agent.claims_source import TargetVerificationBlocked
    from repro_agent.method_guard import MethodImplementationBlocked

    error = TargetVerificationBlocked if blocked == 'target' else MethodImplementationBlocked
    def run(self, paper_id, source='catalog'):
        raise error('Claim verification stopped' if blocked == 'target' else 'Incomplete method: paper is not assessable')

    monkeypatch.setattr(FakeOrch, 'run', run)
    events = read_stream(env, start(env).json()['job_id'])
    assert events[-1]['type'] == 'error' and events[-1]['status'] == cat.UNCHECKED
    assert not any(event['type'] == 'finished' for event in events)
    assert not (env.tmp / 'data' / 'index.json').exists()
    assert not api.job_semaphore.locked()
