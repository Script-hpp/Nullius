"""Discovery -> persisted exact claim -> guarded experiment -> exported evidence.

Only fake Discovery / sandbox / orchestrator: no network or paid calls.
"""

import json

import pytest

from repro_agent import api, catalog as cat, live_catalog
from test_api_repro import env, FakeOrch, read_stream, start  # noqa: F401 (pytest fixture)
from test_catalog import cand, cora_entry, evidence, tavily_orch


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(live_catalog, "PATH", tmp_path / "logs" / "live_candidates.json")
    monkeypatch.setattr(live_catalog, "PREFLIGHT_DIR", tmp_path / "logs" / "preflight")


def test_exact_claims_survive_store_reload_without_collisions_or_curated_mutations(env):
    original = env.cat.read_bytes()
    first = live_catalog.save_candidate({**cand(), "ts": 10})
    second = live_catalog.save_candidate({**cand(), "dataset": "CiteSeer", "ts": 11})
    third = live_catalog.save_candidate({**cand(), "claim": {**cand()["claim"], "target": 70}, "ts": 12})
    assert len({e["catalog_id"] for e in (first, second, third)}) == 3
    assert len(live_catalog.load_entries()) == 3
    assert live_catalog.save_candidate({**cand(), "ts": 13, "cached": True})["catalog_id"] == first["catalog_id"]
    saved = env.client.get("/api/discover/candidates").json()
    assert [e["ts"] for e in saved] == [11, 12, 13]
    assert live_catalog.find_entry(first["catalog_id"])["candidate"]["claim"]["target"] == 82.92
    assert env.cat.read_bytes() == original
    assert FakeOrch.instances == []


def test_live_discovery_stream_saves_before_publishing_and_can_start_the_same_claim(env, monkeypatch):
    monkeypatch.setattr(api, "daily_limit", {"date": None, "count": 0})
    monkeypatch.setattr(api, "MAX_DAILY_JOBS", 5)
    monkeypatch.setenv("MAX_DAILY_DISCOVER_PER_IP", "100")

    class Discovery:
        def discover(self, datasets, on_event, **_):
            on_event({**cand(), "ts": 10, "community": [{"url": "https://example.com", "title": "Note", "snippet": "Source"}]})

    monkeypatch.setattr(api, "DiscoveryAgent", Discovery)
    job = env.client.post("/api/discover", json={"datasets": ["Cora"]}).json()["job_id"]
    response = env.client.get(f"/api/discover/{job}/stream")
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    replay = env.client.get("/api/discover/replay").json()
    assert replay == events and replay[-1]["type"] == "done"
    assert live_catalog.REPLAY_PATH.exists()
    event = next(e for e in events if e["type"] == "candidate")
    assert event["runnable"] is True and events[-1]["type"] == "done"
    stored = live_catalog.find_entry(event["catalog_id"])
    assert stored["candidate"]["community"][0]["title"] == "Note"
    check = start(env, catalog_id=event["catalog_id"], claim={"target": 0}, dataset="Cora").json()["job_id"]
    run = read_stream(env, check)
    assert run[0]["entry"]["claim"]["target"] == 82.92  # browser overrides are never used
    assert run[0]["entry"]["task"] == "Cora-LCC"
    assert run[-1]["type"] == "finished" and FakeOrch.instances[0].seeds == 3
    assert live_catalog.find_entry(event["catalog_id"])["status"] == cat.CHECKED
    index = json.loads((env.tmp / "data" / "index.json").read_text())
    assert index[0]["catalog_id"] == event["catalog_id"] and index[0]["title"] == cand()["title"]
    rows = json.loads((env.tmp / "data" / "catalog.json").read_text())
    row = next(e for e in rows if e["id"] == event["catalog_id"])
    assert row["status"] == cat.CHECKED and row["run_id"] == run[-1]["run_id"]
    assert (env.tmp / "data" / index[0]["paper_id"] / index[0]["timestamp"] / "dossier.json").exists()
    # Re-discovering the same cached claim preserves the actual experiment outcome.
    live_catalog.save_candidate({**cand(), "ts": 20, "cached": True})
    assert live_catalog.find_entry(event["catalog_id"])["status"] == cat.CHECKED
    assert all(e["id"] != event["catalog_id"] for e in cat.load_catalog(env.cat))
    # A later run of a curated entry must not remove live Discovery evidence.
    read_stream(env, start(env).json()["job_id"])
    rows = json.loads((env.tmp / "data" / "catalog.json").read_text())
    assert next(e for e in rows if e["id"] == event["catalog_id"])["run_id"] == run[-1]["run_id"]


def test_unknown_and_unrunnable_claims_are_refused_before_any_experiment_or_start_count(env):
    unknown = start(env, catalog_id="disc_fabricated")
    assert unknown.status_code == 404
    event = live_catalog.save_candidate({**cand(), "claim": {**cand()["claim"], "matched_task": None}})
    assert event["runnable"] is False and "Cora split" in event["run_reason"]
    response = start(env, catalog_id=event["catalog_id"])
    assert response.status_code == 400 and "Cora split" in response.json()["detail"]
    eligible = env.client.get("/api/reproduce/eligibility").json()
    assert eligible[event["catalog_id"]]["runnable"] is False
    assert eligible[env.etth]["runnable"] is False
    assert eligible[env.id]["runnable"] is True
    assert FakeOrch.instances == [] and api.repro_daily["count"] == 0 and not api.job_semaphore.locked()


def test_dynamic_claim_uses_the_same_daily_and_ip_limits(env, monkeypatch):
    event = live_catalog.save_candidate(cand())
    monkeypatch.setattr(api, "MAX_DAILY_REPRO_JOBS", 0)
    assert start(env, catalog_id=event["catalog_id"]).json()["reason"] == "daily_limit"
    monkeypatch.setattr(api, "MAX_DAILY_REPRO_JOBS", 3)
    monkeypatch.setenv("MAX_DAILY_REPRO_PER_IP", "0")
    assert start(env, catalog_id=event["catalog_id"]).json()["reason"] == "daily_limit"
    assert FakeOrch.instances == []


def test_storage_error_is_visible_and_does_not_start_a_paid_experiment(env):
    live_catalog.PATH.parent.mkdir(parents=True)
    live_catalog.PATH.write_text("broken JSON")
    assert env.client.get("/api/discover/candidates").status_code == 503
    assert env.client.get("/api/reproduce/eligibility").status_code == 503
    assert start(env, catalog_id="disc_missing").status_code == 503
    assert FakeOrch.instances == [] and api.repro_daily["count"] == 0


def test_real_orchestrator_adapter_loads_live_claim_and_preserves_target_and_task(tmp_path, monkeypatch):
    candidate = cora_entry()["candidate"]
    event = live_catalog.save_candidate({**candidate, "arxiv_id": "2511.08094", "title": "Live claim", "ts": 10})
    orch = tavily_orch(tmp_path, monkeypatch, [], evidence(0.8292))
    dossier = orch.run(event["catalog_id"], source="live_catalog")
    assert dossier.paper_id == event["catalog_id"]
    assert dossier.claims[0].target == pytest.approx(0.8292)
    assert dossier.claims[0].target_evidence["quote"] == "SL-GCN 82.92"
    assert (tmp_path / "runs" / event["catalog_id"]).exists()


def test_forecasting_keeps_unchecked_columns_and_missing_input_length_as_caveats(env):
    event = live_catalog.save_candidate(cand("2505.20716", "ETTh1", claim={"method": "Times2D", "metric": "mse", "target": .378,
        "horizon": 96, "matched_task": "ETTh1", "column_check": "unchecked", "quote": "Times2D 0.378"}, protocol="unklar"))
    assert event["runnable"] is True
    warnings = event["run_warnings"]
    assert any("Tabellenspalte" in warning for warning in warnings)
    assert any("Protokoll" in warning for warning in warnings)
    assert any("Eingabelaenge" in warning for warning in warnings)
    assert event["claim"]["target"] == .378


def test_previous_validated_discovery_cache_is_imported_once_without_calls_or_mutations(env):
    candidates = {"2511.08094": {**cand(), "status": cat.UNCHECKED, "cache_version": 3},
                  "rejected": {"status": "rejected", "cache_version": 3},
                  "obsolete": {**cand("2401.00001"), "status": cat.UNCHECKED, "cache_version": 1}}
    live_catalog.CACHE_PATH.write_text(json.dumps(candidates))
    before = live_catalog.CACHE_PATH.read_bytes()
    live_catalog.import_existing_cache()
    (event,) = live_catalog.candidate_events()
    assert event["cached"] is True and event["runnable"] is True
    assert live_catalog.find_entry(event["catalog_id"])["candidate"]["claim"]["target"] == 82.92
    live_catalog.import_existing_cache()
    assert len(live_catalog.load_entries()) == 1 and FakeOrch.instances == []
    assert live_catalog.CACHE_PATH.read_bytes() == before
    assert env.client.get("/api/discover/replay").json() == []  # claims alone cannot reconstruct the old search


def test_latest_server_replay_is_persisted_and_redacted_without_spending(env, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-test-key-123456")
    recording = [{"type": "discovery_start", "ts": 100}, {"type": "error", "ts": 101,
        "message": "secret-test-key-123456 at /home/someone/project"}, {"type": "done", "ts": 102}]
    live_catalog.save_replay(recording)
    response = env.client.get("/api/discover/replay")
    assert response.status_code == 200 and response.json()[-1]["type"] == "done"
    assert "secret-test-key-123456" not in response.text and "someone" not in response.text
    assert FakeOrch.instances == [] and api.repro_daily["count"] == 0


def test_replay_storage_failure_does_not_hide_completion_or_hold_worker_slot(env, monkeypatch):
    monkeypatch.setattr(api, "daily_limit", {"date": None, "count": 0})
    monkeypatch.setattr(api, "MAX_DAILY_JOBS", 5)
    monkeypatch.setenv("MAX_DAILY_DISCOVER_PER_IP", "100")

    class Discovery:
        def discover(self, datasets, on_event, **_):
            on_event({"type": "pending", "ts": 100})

    def fail(events):
        raise OSError("read-only volume")

    monkeypatch.setattr(api, "DiscoveryAgent", Discovery)
    monkeypatch.setattr(live_catalog, "save_replay", fail)
    job = env.client.post("/api/discover", json={"datasets": ["Cora"]}).json()["job_id"]
    response = env.client.get(f"/api/discover/{job}/stream")
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["type"] == "done"
    assert any(event["type"] == "warning" and "replay" in event["message"] for event in events)
    assert not api.job_semaphore.locked()
