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
