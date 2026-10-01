"""A3/M1: Zufallsniveau auf der Validierung -> fester Reparaturhinweis ohne Zahlen. Attrappen, kein Netz."""

import re

from repro_agent.dossier import Claim, Dossier
from repro_agent.harness.repair_hints import CHECKLIST_CLASSIFICATION, val_fault_message
from repro_agent.tasks import cora_task
from test_candidates import C0, C1, ValSandbox
from test_loop_fakes import make_orch


def make(val_by_code, repairs):
    sb = ValSandbox({C0: 0.80, C1: 0.80}, val_by_code)
    o, calls = make_orch(sb, seeds=2, repairs=repairs, write=C0)
    o.graph_repair, o.n_candidates = True, 1
    events = []
    o._emit = lambda type, **kw: events.append((type, kw))
    claim = Claim("c", "GCN", "accuracy", 0.83, 0.01, "T", "paper")
    d = Dossier(paper_id="p", claims=[claim])
    return o, calls, events, claim, d, sb


def test_message_has_no_numbers_and_names_the_indexing_rule():
    msg = val_fault_message()
    assert not re.search(r"\d\.\d|\d{2,}", msg)   # keine Messzahl (argmax(1) ist Code, keine Statistik)
    assert "pred[test_ids]" in msg and "Never zip" in msg
    assert "pred[test_ids]" in CHECKLIST_CLASSIFICATION


def test_val_below_majority_triggers_fixed_repair(tmp_path):
    o, calls, events, claim, d, sb = make({C0: 0.10, C1: 0.80}, [C1])   # Test 0.80 ist ueber den Baselines, nur Val verraet es
    o.run_claim({"seeds": 2}, claim, cora_task(), "snap", d, tmp_path)
    assert len(calls["repair"]) == 1 and calls["repair"][0] == val_fault_message()
    assert [kw["result"] for t, kw in events if t == "fault_repair"] == ["repariert"]
    assert d.fault_repairs == {"c": 1} and {a.code_hash for a in d.attempts if a.seed > 0} == {d.attempts[-1].code_hash}


def test_val_at_or_above_majority_no_repair(tmp_path):
    o, calls, events, claim, d, sb = make({C0: 0.60}, [])
    o.run_claim({"seeds": 2}, claim, cora_task(), "snap", d, tmp_path)
    assert calls["repair"] == [] and d.fault_repairs == {}


def test_no_val_value_no_repair(tmp_path):
    o, calls, events, claim, d, sb = make({}, [])
    o.run_claim({"seeds": 2}, claim, cora_task(), "snap", d, tmp_path)
    assert calls["repair"] == []


def test_forecast_metric_never_val_checked(tmp_path):
    o, calls, events, claim, d, sb = make({C0: 0.10}, [])
    claim.metric = "mse"
    assert o._val_fault_check(claim, "x", 0) is None
