"""Readiness check (2026-10-15): a found claim gets a start button only after every step before the first paid code
generation passed, and the run continues from exactly that checked state. Fakes, no network."""

import json

import pytest

from repro_agent import api, live_catalog
from repro_agent import loop as loop_mod
from repro_agent.repo_source import RepoDump
from repro_agent.spec import Spec, SpecItem
from test_author_code_mode import cg_orch
from test_catalog import evidence, tavily_orch


def with_repo(monkeypatch, files=None):
    files = files or {"models/gcn.py": "class GCN: pass"}
    monkeypatch.setattr(loop_mod, "find_repo_url", lambda t: "https://github.com/x/y")
    monkeypatch.setattr(loop_mod, "fetch_readme", lambda *a, **k: "")
    monkeypatch.setattr(loop_mod, "read_repo", lambda *a, **k: RepoDump(url="https://github.com/x/y", commit="abc1234567",
                                                                       files=files, n_tracked=len(files)))
    spec = Spec(model="GCN", family="graph", items=[SpecItem("lr", "0.01", "lr 0.01", "paper", "paper text", 0.01, enforce=True)],
                excerpt="=== repo:models/gcn.py@abc1234 ===\nclass GCN: pass\n", excerpt_docs=["repo:models/gcn.py@abc1234"])
    monkeypatch.setattr(loop_mod, "build_spec", lambda *a, **k: spec)


def test_ready_claim_is_saved_and_the_run_reuses_the_checked_state(tmp_path, monkeypatch):
    o, e, _ = cg_orch(tmp_path, monkeypatch)
    with_repo(monkeypatch)
    saved = tmp_path / "preflight" / "x.pkl"
    result = o.preflight(e["id"], "catalog", save_path=saved)
    assert result["ready"] and result["stage"] == "ready" and result["mode"] == loop_mod.MODE_AUTHOR_CODE
    assert result["repo"] == "https://github.com/x/y" and saved.exists()

    # The run must not read sources or the repository again: it continues from the saved check.
    def never(*a, **k):
        raise AssertionError("source or repository read again")
    for name in ("find_repo_url", "read_repo", "fetch_paper_text", "build_spec", "extract_target_cached"):
        monkeypatch.setattr(loop_mod, name, never)
    run, _, seen = cg_orch(tmp_path, monkeypatch)
    d = run.run(e["id"], source="catalog", prepared_path=saved)
    assert d.mode == loop_mod.MODE_AUTHOR_CODE and d.repo["url"] == "https://github.com/x/y"
    assert d.claims[0].target == pytest.approx(0.8292) and d.verdicts
    replayed = [t for t, kw in seen if kw.get("from_preflight")]
    assert "target_resolved" in replayed and "mode" in replayed
    assert run.last_run_dir != o.last_run_dir


@pytest.mark.parametrize("change", ["code", "config", "claim"])
def test_saved_check_from_other_code_config_or_claim_is_not_reused(tmp_path, monkeypatch, change):
    o, e, _ = cg_orch(tmp_path, monkeypatch)
    with_repo(monkeypatch)
    saved = tmp_path / "x.pkl"
    assert o.preflight(e["id"], "catalog", save_path=saved)["ready"]
    run, _, _ = cg_orch(tmp_path, monkeypatch)
    if change == "code":
        monkeypatch.setattr(loop_mod, "code_version", lambda: "other")
    elif change == "config":
        run.independent_selection = not getattr(run, "independent_selection", False)
    else:
        monkeypatch.setattr(run, "_load_paper", lambda *a: ({"title": "changed"}, []))
    assert run._resume_prepared(saved, e["id"], "catalog") is None
    assert run._resume_prepared(tmp_path / "missing.pkl", e["id"], "catalog") is None


def test_claim_without_authors_code_is_not_ready(tmp_path, monkeypatch):
    o, e, _ = cg_orch(tmp_path, monkeypatch)   # no repository found
    saved = tmp_path / "x.pkl"
    result = o.preflight(e["id"], "catalog", save_path=saved)
    assert not result["ready"] and result["stage"] == "authors' code" and "no repository" in result["reason"]
    assert not saved.exists()


def test_claim_whose_target_cannot_be_verified_is_not_ready(tmp_path, monkeypatch):
    from test_catalog import cora_entry
    e = cora_entry()
    e["candidate"]["claim"]["require_verified_target"] = True
    o = tavily_orch(tmp_path, monkeypatch, [e], evidence(0.75))
    result = o.preflight(e["id"], "catalog")
    assert not result["ready"] and result["stage"] == "target" and "stopped" in result["reason"]


def test_authors_model_on_a_gpu_only_kernel_is_not_ready(tmp_path, monkeypatch):
    # SST (2026-10-15) imported mamba_ssm; the CPU sandbox cannot run it and the run spent its budget without a measurement.
    o, e, _ = cg_orch(tmp_path, monkeypatch)
    with_repo(monkeypatch, {"models/gcn.py": "import torch\nfrom mamba_ssm import Mamba\nclass GCN: pass\n"})
    saved = tmp_path / "x.pkl"
    result = o.preflight(e["id"], "catalog", save_path=saved)
    assert not result["ready"] and result["stage"] == "authors' code"
    assert "mamba_ssm (models/gcn.py)" in result["reason"] and "GPU-only" in result["reason"]
    assert not saved.exists()


def test_gpu_kernel_imports_ignore_fallbacks_relative_imports_and_files_outside_the_code_prompt(tmp_path, monkeypatch):
    from repro_agent.method_guard import gpu_kernel_imports
    guarded = "try:\n    from flash_attn import flash_attn_func\nexcept ImportError:\n    flash_attn_func = None\n"
    files = {"a.py": guarded, "b.py": "from .triton import kernel\nimport torch_geometric, dgl\n", "c.sh": "import triton",
             "d.py": "def f(:\nimport causal_conv1d.cuda\n", "e.py": "import xformers.ops as xops\n"}
    assert gpu_kernel_imports(files) == {"causal_conv1d": ["d.py"], "xformers": ["e.py"]}
    # Only the authors' files that go into the code prompt count when a spec names them.
    o, e, _ = cg_orch(tmp_path, monkeypatch)
    with_repo(monkeypatch, {"models/gcn.py": "class GCN: pass", "tools/profile.py": "import triton\n"})
    assert o.preflight(e["id"], "catalog")["ready"]


# ---- live catalog and API ----------------------------------------------------------------------------------------------------
def test_found_claim_is_offered_only_after_a_passed_readiness_check(tmp_path, monkeypatch):
    monkeypatch.setattr(live_catalog, "PATH", tmp_path / "live.json")
    entry = {"id": "c1", "discovery_event": {}}
    monkeypatch.setattr(live_catalog, "eligibility", lambda e: {"runnable": True, "reason": None, "warnings": []})
    (tmp_path / "live.json").write_text(json.dumps([entry]))
    assert live_catalog.offer(entry)["runnable"] is False
    live_catalog.save_readiness("c1", {"ready": False, "stage": "authors' code", "reason": "no repository"})
    stored = live_catalog.find_entry("c1")
    assert live_catalog.offer(stored) == {"runnable": False, "reason": "no repository", "warnings": [], "stage": "authors' code"}
    assert stored["discovery_event"]["readiness"]["ready"] is False
    live_catalog.save_readiness("c1", {"ready": True, "stage": "ready"})
    assert live_catalog.offer(live_catalog.find_entry("c1"))["runnable"] is True
    assert live_catalog.preflight_path("../c1").name == ".._c1.pkl" and live_catalog.preflight_path("../c1").parent == live_catalog.PREFLIGHT_DIR


def test_discovery_checks_each_found_claim_and_reports_readiness(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from test_api import open_gate
    open_gate.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(live_catalog, "PATH", tmp_path / "live.json")
    monkeypatch.setattr(live_catalog, "REPLAY_PATH", tmp_path / "replay.json")
    monkeypatch.setattr(live_catalog, "PREFLIGHT_DIR", tmp_path / "preflight")
    monkeypatch.setattr(live_catalog, "save_candidate", lambda evt: {**evt, "catalog_id": evt["arxiv_id"], "runnable": evt["ok"]})
    saved = {}
    monkeypatch.setattr(live_catalog, "save_readiness", lambda cid, r: saved.setdefault(cid, {"ready": bool(r.get("ready")),
                        "stage": r.get("stage"), "reason": r.get("reason"), "cost_gross_usd": r.get("cost_gross_usd")}))
    checked = []

    class FakeOrch:
        def preflight(self, cid, source, save_path=None):
            checked.append((cid, source, save_path.name))
            if cid == "boom":
                raise RuntimeError("sandbox down")
            return {"ready": cid == "good", "stage": "ready" if cid == "good" else "authors' code",
                    "reason": None if cid == "good" else "no repository", "cost_gross_usd": .03}
        def close(self): pass

    class FakeAgent:
        def discover(self, datasets, on_event, **_):
            for cid, ok in (("good", True), ("norepo", True), ("boom", True), ("static", False)):
                on_event({"type": "candidate", "arxiv_id": cid, "ok": ok})

    monkeypatch.setattr(api, "DiscoveryAgent", FakeAgent)
    monkeypatch.setattr(api, "make_preflight_orchestrator", FakeOrch)
    with TestClient(api.app) as c:
        job = c.post("/api/discover", json={"datasets": ["Cora"]}).json()["job_id"]
        text = c.get(f"/api/discover/{job}/stream").text
    events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
    final = {e["catalog_id"]: e for e in events if e["type"] == "readiness" and e["state"] != "checking"}
    assert [c[0] for c in checked] == ["good", "norepo", "boom"] and checked[0][1] == "live_catalog"
    assert final["good"]["state"] == "ready" and final["norepo"]["state"] == "blocked"
    assert final["boom"]["state"] == "blocked" and final["boom"]["reason"] == api.PREFLIGHT_UNFINISHED
    assert "static" not in final and events[-1]["type"] == "done" and not any(e["type"] == "error" for e in events)


def test_saved_claims_without_a_check_are_listed_and_checked_only_on_request(monkeypatch, capsys):
    from repro_agent import preflight_saved
    entries = [{"id": "old", "title": "Old find"}, {"id": "done", "title": "Checked", "readiness": {"ready": True}}]
    monkeypatch.setattr(live_catalog, "load_entries", lambda: entries)
    monkeypatch.setattr(live_catalog, "eligibility", lambda e: {"runnable": True})
    calls = []
    monkeypatch.setattr(api, "preflight_claim", lambda cid: calls.append(cid) or {"ready": True, "stage": "ready", "reason": None})
    preflight_saved.main([])
    assert calls == [] and "old" in capsys.readouterr().out
    preflight_saved.main(["--run"])
    assert calls == ["old"]


def test_curated_live_checks_only_for_featured_entries_of_the_real_catalog():
    from repro_agent import catalog as cat
    offers = {e["id"]: live_catalog.curated_offer(e) for e in cat.load_catalog()}
    offered = {i for i, o in offers.items() if o["runnable"]}
    assert offered == {"disc_1609_02907", "disc_2307_03756", "disc_2501_01087"}
    assert "10-minute CPU limit" in offers["disc_2303_06053"]["reason"]          # TSMixer: recorded result stays, no live re-run
    assert "does not name the model variant" in offers["disc_2405_00946"]["reason"]   # SparseTSF
    assert "80 minutes" in offers["disc_2410_09348_d8991277dd33d43f"]["reason"]      # BANGS: recorded result, no live re-run
    unchecked = next(i for i, e in ((e["id"], e) for e in cat.load_catalog()) if e.get("status") == cat.UNCHECKED and offers[i]["reason"] == live_catalog.CURATED_NOT_OFFERED)
    assert unchecked
