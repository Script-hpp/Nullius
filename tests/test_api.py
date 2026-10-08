import pytest
from fastapi.testclient import TestClient
from repro_agent import api, live_gate

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def open_gate(tmp_path, monkeypatch):
    """Offene Sperre mit leeren Logs in tmp_path (nie das echte logs/). Einzelne Tests ziehen Werte nach."""
    monkeypatch.setattr("repro_agent.llm.COST_LOG_PATH", tmp_path / "costs.jsonl")
    monkeypatch.setattr("repro_agent.tavily_usage.TAVILY_LOG_PATH", tmp_path / "tavily_usage.jsonl")
    for k in ("BUDGET_TOTAL_USD", "BUDGET_SINCE", "LIVE_MIN_BUDGET_USD", "TAVILY_CREDITS_TOTAL", "TAVILY_CREDITS_SINCE",
              "LIVE_MIN_TAVILY_CREDITS", "TRUST_PROXY", "MAX_DAILY_DISCOVER_PER_IP", "MAX_DAILY_REPRO_PER_IP"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BUDGET_TOTAL_USD", "30")
    monkeypatch.setenv("TAVILY_CREDITS_TOTAL", "1000")
    monkeypatch.setattr(api, "daily_limit", {"date": None, "count": 0})
    monkeypatch.setattr(api, "repro_daily", {"date": None, "count": 0})
    monkeypatch.setattr(api, "ip_limits", live_gate.IpLimits())
    monkeypatch.setattr(api, "MAX_DAILY_JOBS", 5)
    api.jobs.clear()
    monkeypatch.setattr(api, "MAX_DAILY_REPRO_JOBS", 3)
    monkeypatch.setattr(api, "job_semaphore", __import__("asyncio").Semaphore(1))
    return tmp_path


def spend_budget(tmp_path, net_usd):
    import json, time
    (tmp_path / "costs.jsonl").write_text(json.dumps({"ts": time.time(), "model": "m", "cost_usd": net_usd}) + "\n")


def spend_tavily(tmp_path, credits):
    import json, time
    (tmp_path / "tavily_usage.jsonl").write_text(json.dumps({"ts": time.time(), "endpoint": "search", "credits": credits}) + "\n")


def test_api_400_unknown_dataset():
    res = client.post("/api/discover", json={"datasets": ["UnknownDataset"]})
    assert res.status_code == 400


def test_start_without_access_code_works(monkeypatch):
    monkeypatch.delenv("ACCESS_CODE", raising=False)
    started = []

    class FakeAgent:
        def discover(self, datasets, on_event):
            started.append(datasets)

    monkeypatch.setattr(api, "DiscoveryAgent", FakeAgent)
    with TestClient(api.app) as c:
        res = c.post("/api/discover", json={"datasets": ["Cora"]})
        assert res.status_code == 200 and "job_id" in res.json()


def test_api_429_running_job(monkeypatch):
    class FakeSemaphore:
        def locked(self): return True

    monkeypatch.setattr("repro_agent.api.job_semaphore", FakeSemaphore())
    res = client.post("/api/discover", json={"datasets": ["Cora"]})
    assert res.status_code == 429 and res.json()["reason"] == "busy"


def test_budget_floor_503_no_job_no_call(open_gate, monkeypatch):
    spend_budget(open_gate, (30 - 2.9) / 1.19)          # Rest 2,90 $ brutto < 3,00
    monkeypatch.setattr(api, "DiscoveryAgent", lambda: pytest.fail("kein Aufruf unter der Schwelle"))
    res = client.post("/api/discover", json={"datasets": ["Cora"]})
    assert res.status_code == 503
    body = res.json()
    assert body["reason"] == "budget_floor" and "message" in body
    assert not api.jobs and api.daily_limit["count"] == 0
    assert "29" not in str(body) and "2.9" not in str(body)      # kein Kontostand


def test_budget_just_above_floor_is_open(open_gate):
    spend_budget(open_gate, (30 - 3.1) / 1.19)
    assert client.get("/api/status").json()["live"] is True


def test_tavily_floor_503(open_gate):
    spend_tavily(open_gate, 750)                          # Rest 250 < 300
    res = client.post("/api/discover", json={"datasets": ["Cora"]})
    assert res.status_code == 503 and res.json()["reason"] == "tavily_floor"


def test_tavily_since_ignores_old_entries(open_gate, monkeypatch):
    spend_tavily(open_gate, 990)
    monkeypatch.setenv("TAVILY_CREDITS_SINCE", "2999-01-01T00:00Z")
    assert client.get("/api/status").json()["live"] is True


def test_daily_limit_429_with_reset(monkeypatch):
    monkeypatch.setattr(api, "MAX_DAILY_JOBS", 0)
    res = client.post("/api/discover", json={"datasets": ["Cora"]})
    assert res.status_code == 429 and res.json()["reason"] == "daily_limit" and res.json()["reset_at"].endswith("Z")


def test_ip_limit_429_per_ip(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY", "1")
    api.ip_limits.add("discovery", "1.2.3.4")
    res = client.post("/api/discover", json={"datasets": ["Cora"]}, headers={"x-forwarded-for": "1.2.3.4"})
    assert res.status_code == 429 and res.json()["reason"] == "daily_limit"
    st = client.get("/api/status", headers={"x-forwarded-for": "1.2.3.4"}).json()
    assert st["kinds"]["discovery"]["reason"] == "daily_limit" and st["kinds"]["reproduction"]["live"] is True and st["live"] is True
    other = client.get("/api/status", headers={"x-forwarded-for": "9.9.9.9"}).json()
    assert other["kinds"]["discovery"]["live"] is True


def test_forwarded_for_ignored_without_trust_proxy():
    api.ip_limits.add("discovery", "1.2.3.4")
    st = client.get("/api/status", headers={"x-forwarded-for": "1.2.3.4"}).json()
    assert st["kinds"]["discovery"]["live"] is True          # Header gilt nur mit TRUST_PROXY=1; Peer ist "testclient"


def test_forwarded_for_takes_last_entry(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY", "1")
    api.ip_limits.add("discovery", "5.5.5.5")
    st = client.get("/api/status", headers={"x-forwarded-for": "1.1.1.1, 5.5.5.5"}).json()   # vorderer Eintrag ist faelschbar
    assert st["kinds"]["discovery"]["reason"] == "daily_limit"


def test_status_open():
    st = client.get("/api/status").json()
    assert st["live"] is True and st["reason"] is None and st["reset_at"] is None


def test_status_reasons(open_gate, monkeypatch):
    spend_budget(open_gate, 30)
    assert client.get("/api/status").json()["reason"] == "budget_floor"
    (open_gate / "costs.jsonl").unlink()
    spend_tavily(open_gate, 800)
    assert client.get("/api/status").json()["reason"] == "tavily_floor"
    (open_gate / "tavily_usage.jsonl").unlink()
    monkeypatch.setattr(api, "MAX_DAILY_JOBS", 0)
    monkeypatch.setattr(api, "MAX_DAILY_REPRO_JOBS", 0)
    st = client.get("/api/status").json()
    assert st["live"] is False and st["reason"] == "daily_limit" and st["reset_at"]

    class Locked:
        def locked(self): return True

    monkeypatch.setattr(api, "MAX_DAILY_JOBS", 5)
    api.jobs.clear()
    monkeypatch.setattr(api, "MAX_DAILY_REPRO_JOBS", 3)
    monkeypatch.setattr(api, "job_semaphore", Locked())
    assert client.get("/api/status").json()["reason"] == "busy"


def test_status_has_no_secrets(open_gate):
    spend_budget(open_gate, 1)
    text = client.get("/api/status").text
    assert "usd" not in text.lower() and "credits" not in text.lower()
