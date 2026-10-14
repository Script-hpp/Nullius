"""Schutz des oeffentlichen Pfads: Redaktion, Security-Header, Body-/Rate-Limit, Tageskosten, Fail-closed, persistente Zaehler. Kein Netz, kein Geld."""

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from repro_agent import api, live_gate, site_export, web_guard


@pytest.fixture(autouse=True)
def gate(tmp_path, monkeypatch):
    monkeypatch.setattr("repro_agent.llm.COST_LOG_PATH", tmp_path / "costs.jsonl")
    monkeypatch.setattr("repro_agent.tavily_usage.TAVILY_LOG_PATH", tmp_path / "tavily_usage.jsonl")
    for k in ("BUDGET_SINCE", "LIVE_MIN_BUDGET_USD", "LIVE_MIN_TAVILY_CREDITS", "TRUST_PROXY", "MAX_DAILY_COST_USD", "LLM_PROVIDER",
              "MAX_DAILY_DISCOVER_PER_IP", "ALLOW_FULL_RUNS", "RATE_LIMIT_POST_PER_MIN", "RATE_LIMIT_PER_MIN", "MAX_BODY_BYTES"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BUDGET_TOTAL_USD", "30")
    monkeypatch.setenv("TAVILY_CREDITS_TOTAL", "1000")
    monkeypatch.setattr(api, "daily_limit", {"date": None, "count": 0})
    monkeypatch.setattr(api, "repro_daily", {"date": None, "count": 0})
    monkeypatch.setattr(api, "ip_limits", live_gate.IpLimits())
    monkeypatch.setattr(api, "MAX_DAILY_JOBS", 5)
    monkeypatch.setattr(api, "job_semaphore", asyncio.Semaphore(1))
    api.jobs.clear()
    api._held.clear()
    api.current_job.clear()
    return tmp_path


def spend(tmp_path, net_usd, ts=None):
    with (tmp_path / "costs.jsonl").open("a") as f:
        f.write(json.dumps({"ts": ts or time.time(), "model": "m", "cost_usd": net_usd}) + "\n")


# ---- Redaktion ----
def test_redact_secrets_and_paths(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "abcd1234efgh5678")
    text = ("key abcd1234efgh5678 tvly-ABCDEFGHIJKL1234 Authorization: Bearer abcdefghijklmnop1234 "
            'api_key="supersecretvalue" at /home/maxmustermann/proj/x.py and /Users/erika/y')
    out = web_guard.redact_text(text)
    for leak in ("abcd1234efgh5678", "tvly-ABCDEFGHIJKL1234", "abcdefghijklmnop1234", "supersecretvalue", "maxmustermann", "erika"):
        assert leak not in out
    assert "~/proj/x.py" in out


def test_username_is_not_replaced_inside_words(monkeypatch):
    monkeypatch.setattr("getpass.getuser", lambda: "app")          # so heisst der Nutzer im Container
    out = web_guard.redact_text("Happy application mapped at /srv/app/data and app", "/nowhere")
    assert "Happy application mapped" in out and out.endswith(" and app")
    assert "/srv/user/data" in out


def test_redact_tree_keeps_json_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "v1.supersecretnebiuskey")
    (tmp_path / "d.json").write_text(json.dumps({"err": "failed with v1.supersecretnebiuskey in /home/someone/x", "n": 1}))
    (tmp_path / "e.jsonl").write_text(json.dumps({"a": "Bearer abcdefghijklmnop1234"}) + "\n")
    assert web_guard.redact_tree(tmp_path) == 2
    assert "supersecret" not in (tmp_path / "d.json").read_text()
    assert json.loads((tmp_path / "d.json").read_text())["n"] == 1
    assert json.loads((tmp_path / "e.jsonl").read_text().splitlines()[0])


def test_live_export_is_redacted(tmp_path, monkeypatch):
    import dataclasses

    from repro_agent import catalog as cat
    from repro_agent.dossier import Attempt
    from test_catalog import cand, dossier_with, entry

    monkeypatch.setenv("TAVILY_API_KEY", "abcd1234efgh5678")
    e = entry(cand())
    d = dossier_with("nicht reproduziert", (0.78, 0.79))
    d.paper_id = e["id"]
    d.attempts = [Attempt(claim_id=d.claims[0].id, branch_id="b0", parent_snapshot="s", code_hash="h", seed=0, status="abweichung",
                          measured=0.78, error_summary=None)]
    run = tmp_path / "runs" / e["id"] / "20261001-120000"
    run.mkdir(parents=True)
    (run / "dossier.json").write_text(json.dumps(dataclasses.asdict(d)))
    (run / "events.jsonl").write_text(json.dumps({"type": "error", "message": "boom abcd1234efgh5678 /home/x/y"}) + "\n")
    catalog = tmp_path / "catalog.yaml"
    cat.save_catalog([e], catalog)
    out = tmp_path / "out"
    ref = site_export.export_live_run(run, catalog, out, "n", 3)
    text = (out / ref["paper_id"] / ref["timestamp"] / "events.jsonl").read_text()
    assert "abcd1234efgh5678" not in text and "/home/x" not in text


# ---- Header, Doku, Gesundheit ----
def test_security_headers_and_no_docs():
    c = TestClient(api.app)
    r = c.get("/api/health")
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"] and r.headers["cache-control"] == "no-store"
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(path).status_code == 404
    assert "access-control-allow-origin" not in c.get("/api/status", headers={"origin": "https://evil.example"}).headers   # kein CORS


def test_body_limit_413_and_rate_limit(monkeypatch):
    c = TestClient(api.app)
    big = {"datasets": ["Cora"], "pad": "x" * 20000}
    r = c.post("/api/discover", json=big)
    assert r.status_code == 413 and r.json()["reason"] == "too_large"
    web_guard.HITS.clear()
    monkeypatch.setenv("RATE_LIMIT_POST_PER_MIN", "2")
    codes = [c.post("/api/discover", json={"datasets": ["Nope"]}).status_code for _ in range(3)]
    assert codes == [400, 400, 429]
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    web_guard.HITS.clear()
    codes = [c.get("/api/health").status_code for _ in range(3)]
    assert codes[-1] == 429


# ---- Discovery: Eingaben und Sperre ----
def test_discovery_datasets_are_deduplicated_and_capped(monkeypatch):
    seen = []

    class Agent:
        def discover(self, datasets, on_event, **_):
            seen.append(datasets)
    monkeypatch.setattr(api, "DiscoveryAgent", Agent)
    with TestClient(api.app) as c:
        assert c.post("/api/discover", json={"datasets": ["Cora"] * 8}).status_code == 200
        for _ in range(50):
            if seen:
                break
            time.sleep(0.05)
        assert seen == [["Cora"]]
        assert c.post("/api/discover", json={"datasets": ["Cora"] * 9}).status_code == 422    # mehr als 8 Eintraege
        assert c.post("/api/discover", json={"datasets": []}).status_code == 400


def test_discovery_constructor_failure_frees_the_lock(monkeypatch):
    def boom():
        raise RuntimeError("no key abcd1234efgh5678 at /home/someone/x")
    monkeypatch.setenv("TAVILY_API_KEY", "abcd1234efgh5678")
    monkeypatch.setattr(api, "DiscoveryAgent", boom)
    with TestClient(api.app) as c:
        job = c.post("/api/discover", json={"datasets": ["Cora"]}).json()["job_id"]
        with c.stream("GET", f"/api/discover/{job}/stream") as r:
            body = "".join(r.iter_text())
        assert "RuntimeError" in body and "abcd1234efgh5678" not in body and "someone" not in body
        assert not api.job_semaphore.locked() and api.current_job == {}


# ---- Tageskosten, Fail-closed ----
def test_daily_cost_cap(gate, monkeypatch):
    monkeypatch.setenv("MAX_DAILY_COST_USD", "1.00")
    spend(gate, 0.80 / 1.19)                                  # brutto 0,80 < 1,00
    c = TestClient(api.app)
    assert c.get("/api/status").json()["live"] is True
    spend(gate, 0.30 / 1.19)                                  # brutto 1,10
    st = c.get("/api/status").json()
    assert st["live"] is False and st["reason"] == "daily_limit"
    r = c.post("/api/discover", json={"datasets": ["Cora"]})
    assert r.status_code == 503 and r.json()["reason"] == "daily_limit"


def test_daily_cost_cap_ignores_yesterday(gate, monkeypatch):
    monkeypatch.setenv("MAX_DAILY_COST_USD", "1.00")
    spend(gate, 5.0, ts=time.time() - 2 * 86400)
    assert TestClient(api.app).get("/api/status").json()["live"] is True


def test_tokenfactory_without_budget_config_is_closed(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "tokenfactory")
    monkeypatch.delenv("BUDGET_TOTAL_USD")
    c = TestClient(api.app)
    assert c.get("/api/status").json()["reason"] == "budget_floor"
    monkeypatch.setenv("BUDGET_TOTAL_USD", "30")
    assert c.get("/api/status").json()["reason"] == "budget_floor"       # BUDGET_SINCE fehlt noch
    monkeypatch.setenv("BUDGET_SINCE", "2026-10-09T11:11Z")
    assert c.get("/api/status").json()["live"] is True


# ---- Persistente Zaehler ----
def test_counters_survive_restart(gate, monkeypatch):
    live_gate.record_start("reproduction", "1.2.3.4")
    live_gate.record_start("discovery", "1.2.3.4")
    live_gate.record_start("reproduction", "5.6.7.8")
    old = gate / "live_jobs.jsonl"
    old.write_text(old.read_text() + json.dumps({"ts": time.time() - 3 * 86400, "kind": "reproduction", "ip": "x"}) + "\n")   # gestern zaehlt nicht
    assert "1.2.3.4" not in old.read_text()                                # nur Hash im Log
    api.restore_counters()                                                  # wie nach dem Neustart
    assert api.repro_daily["count"] == 2 and api.daily_limit["count"] == 1
    assert api.ip_limits.used_up("reproduction", "1.2.3.4") and not api.ip_limits.used_up("reproduction", "9.9.9.9")
    monkeypatch.setattr(api, "MAX_DAILY_REPRO_JOBS", 2)
    assert api.refusal("reproduction", "9.9.9.9").reason == "daily_limit"


def test_started_jobs_are_logged(gate, monkeypatch):
    class Agent:
        def discover(self, datasets, on_event, **_):
            pass
    monkeypatch.setattr(api, "DiscoveryAgent", Agent)
    with TestClient(api.app) as c:
        assert c.post("/api/discover", json={"datasets": ["Cora"]}).status_code == 200
    (e,) = live_gate.todays_starts()
    assert e["kind"] == "discovery" and len(e["ip"]) == 16


def test_ipv6_counts_per_64():
    class Req:
        headers = {}
        client = type("C", (), {"host": "2001:db8:1:2:aaaa:bbbb:cccc:dddd"})()
    a = live_gate.client_ip(Req())
    Req.client = type("C", (), {"host": "2001:db8:1:2:1111:2222:3333:4444"})()
    assert a == live_gate.client_ip(Req()) == "2001:db8:1:2::/64"


# ---- SSE-Plaetze ----
def test_stream_slots_limits_and_expiry(monkeypatch):
    monkeypatch.setenv("MAX_SSE_PER_IP", "2")
    monkeypatch.setenv("MAX_SSE_STREAMS", "3")
    s = api.StreamSlots()
    a, b = s.acquire("1.1.1.1"), s.acquire("1.1.1.1")
    assert a and b and s.acquire("1.1.1.1") is None                        # je IP
    assert s.acquire("2.2.2.2") and s.acquire("3.3.3.3") is None            # global
    s.release(a)
    assert s.acquire("1.1.1.1")
    monkeypatch.setattr(api, "STREAM_MAX_S", -100)                         # verlorene Plaetze verfallen von selbst
    s2 = api.StreamSlots()
    assert s2.acquire("1.1.1.1")
    time.sleep(0.01)
    assert len(s2.slots) == 1 and s2.acquire("1.1.1.1") and len(s2.slots) == 1
