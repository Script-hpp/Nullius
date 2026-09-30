"""Live-Hook des Orchestrators (nur API) und gemeinsame Kandidaten-Logik (run_candidate). Attrappen, kein Netz."""

import json
from types import SimpleNamespace

from repro_agent import catalog as cat
from repro_agent.costs import BudgetExceeded
from repro_agent.dossier import Dossier
from repro_agent.events import EventLog
from repro_agent.llm import BillingBlocked
from repro_agent.loop import CostLimitExceeded
from test_catalog import cand, dossier_with, entry
from test_loop_fakes import FakeSandbox, make_orch


def test_eventlog_write_returns_record_with_and_without_path(tmp_path):
    rec = EventLog(None).write("x", a=1)
    assert rec["type"] == "x" and rec["a"] == 1 and "ts" in rec
    p = tmp_path / "e.jsonl"
    rec2 = EventLog(p).write("y", b=2)
    assert json.loads(p.read_text()) == rec2


def test_emit_calls_hook_and_file_but_live_events_only_the_hook(tmp_path):
    o, _ = make_orch(FakeSandbox())
    seen = []
    o.events = EventLog(tmp_path / "events.jsonl")
    o.on_event = seen.append
    o._emit("plan", text="t")
    o._emit_live("llm_call", role="code", cost_net_usd=0.1, total_net_usd=0.1)
    assert [e["type"] for e in seen] == ["plan", "llm_call"]
    assert [json.loads(line)["type"] for line in (tmp_path / "events.jsonl").read_text().splitlines()] == ["plan"]   # nie llm_call


def test_without_hook_nothing_changes(tmp_path):
    o, _ = make_orch(FakeSandbox())
    o.events = EventLog(tmp_path / "events.jsonl")
    o._emit("plan", text="t")
    o._emit_live("llm_call", role="code")                      # kein Hook: keine Wirkung, kein Fehler
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 1


def test_ask_reports_running_cost_to_the_hook_only():
    o, _ = make_orch(FakeSandbox())
    seen = []
    o.on_event = seen.append
    o._lock = __import__("threading").Lock()
    o.llm = SimpleNamespace(total_cost_usd=0.0, gross=lambda x: x * 1.19,
                            complete=lambda role, msgs, **k: SimpleNamespace(content="c", cost_usd=0.002))
    o._ask("code", "s", "u")
    (evt,) = seen
    assert evt["type"] == "llm_call" and evt["role"] == "code" and evt["cost_net_usd"] == 0.002


class FakeRunOrch:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error
        self.last_run_dir, self.spent_gross_usd = None, 0.05

    def run(self, paper_id, source="benchmark"):
        if self.error:
            raise self.error
        return self.result


def test_run_candidate_outcomes():
    e = entry(cand())
    ok = cat.run_candidate(FakeRunOrch(dossier_with("reproduziert")), e)
    assert ok.status == cat.CHECKED and ok.dossier is not None and ok.cost_gross_usd == 0.05 and not ok.abort
    bad = entry(cand(protocol="unklar", claim={**cand()["claim"], "matched_task": None}))
    nr = cat.run_candidate(FakeRunOrch(error=AssertionError("never run")), bad)
    assert nr.status == cat.NOT_RUNNABLE and "matched_task" in nr.reason
    for exc in (BillingBlocked("x"), BudgetExceeded("y")):
        ab = cat.run_candidate(FakeRunOrch(error=exc), e)
        assert ab.status == cat.UNCHECKED and ab.abort and "Abbruch" in ab.reason
    fail = cat.run_candidate(FakeRunOrch(error=CostLimitExceeded("limit")), e)
    assert fail.status == cat.FAILED and not fail.abort and fail.cost_gross_usd == 0.05 and "CostLimitExceeded" in fail.reason


def test_apply_outcome_writes_entry_like_curation():
    e = entry(cand())
    cat.apply_outcome(e, cat.run_candidate(FakeRunOrch(dossier_with("nicht reproduziert", (0.78,))), e))
    assert e["status"] == "geprueft" and e["verdict"]["urteil"] == "nicht reproduziert" and e["cost_gross_usd"] == 0.05
    cat.apply_outcome(e, cat.run_candidate(FakeRunOrch(error=BillingBlocked("x")), e))
    assert e["status"] == "ungeprueft" and "verdict" not in e and "cost_gross_usd" in e   # Kosten des letzten Laufs bleiben stehen


def test_plan_and_code_text_go_to_the_hook_only_and_once_per_version(tmp_path):
    from repro_agent.dossier import Claim
    from repro_agent.tasks import cora_task

    sb = FakeSandbox()
    o, _ = make_orch(sb, seeds=2, repairs=["print('v2')\n"], write="print('v1')\n")
    o.plan = lambda *a, **k: "1. step one\n2. step two"
    seen = []
    o.on_event = seen.append
    o.events = EventLog(tmp_path / "events.jsonl")
    claim = Claim("c", "GCN", "accuracy", 0.83, 0.01, "s", "paper")
    d = Dossier(paper_id="p", claims=[claim])
    o.run_claim({"seeds": 2}, claim, cora_task(), "snap", d, tmp_path)
    plan = [e for e in seen if e["type"] == "plan_text"]
    code = [e for e in seen if e["type"] == "code_text"]
    assert len(plan) == 1 and plan[0]["text"] == "1. step one\n2. step two"
    assert [c["code"] for c in code] == ["print('v1')\n"]                     # eine Version, einmal, obwohl mehrere Seeds/Versuche
    written = {json.loads(line)["type"] for line in (tmp_path / "events.jsonl").read_text().splitlines()}
    assert "plan_text" not in written and "code_text" not in written and "plan" in written


def test_export_copies_every_code_version(tmp_path):
    import importlib.util, shutil
    from repro_agent import site_export
    from test_catalog import dossier_with
    import dataclasses
    from repro_agent.dossier import Attempt

    run = tmp_path / "runs" / "disc_x" / "20261001-000000"
    run.mkdir(parents=True)
    d = dossier_with("reproduziert", (0.8,))
    d.attempts = [Attempt(claim_id=d.claims[0].id, branch_id="b", parent_snapshot="s", code_hash="bbb222", seed=0, status="stabil", measured=0.8, error_summary=None)]
    (run / "dossier.json").write_text(json.dumps(dataclasses.asdict(d)))
    (run / "disc_x_accuracy_aaa111.py").write_text("v1")
    (run / "disc_x_accuracy_bbb222.py").write_text("v2")
    (run / "disc_x_accuracy_plan.md").write_text("plan")
    out = tmp_path / "data"
    site_export.export_run(run, out, {})
    dest = out / "disc_x" / "20261001-000000"
    assert (dest / "code_aaa111.py").read_text() == "v1" and (dest / "code_bbb222.py").read_text() == "v2"
    assert (dest / "code.py").read_text() == "v2" and (dest / "plan.md").read_text() == "plan"
