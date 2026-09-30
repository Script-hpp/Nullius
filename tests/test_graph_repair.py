"""Graph-Reparatur (nur mit graph_repair=True) und Jury-Filter. Attrappen, kein Netz, kein Geld."""

import hashlib
import importlib.util
import inspect
from pathlib import Path

import pytest

from repro_agent import catalog as cat
from repro_agent.dossier import Claim, Dossier
from repro_agent.loop import MAX_GRAPH_REPAIRS, CostLimitExceeded, Orchestrator
from repro_agent.report import render_markdown
from repro_agent.sandbox.base import RunResult
from repro_agent.tasks import cora_task
from test_catalog import cand, dossier_with, entry
from test_loop_fakes import FakeSandbox, make_orch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("curate_catalog", ROOT / "scripts" / "curate_catalog.py")
curate_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curate_mod)

V1, V2, V3 = "print('v1')\n", "print('v2')\n", "print('v3')\n"


def h(code):
    return hashlib.sha256(code.encode()).hexdigest()[:12]


class CrashSandbox(FakeSandbox):
    """Code mit CRASH im Text scheitert an der Formpruefung."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.bad = set()

    def put_file(self, branch, path, content):
        if b"CRASH" in content:
            self.bad.add(branch)
        super().put_file(branch, path, content)

    def run(self, branch, cmd, timeout_s):
        if "SMOKE=1" in cmd and branch in self.bad:
            return RunResult(1, "", "RuntimeError: boom", False)
        return super().run(branch, cmd, timeout_s)


def setup(full, ablated, repairs=(), flag=True, seeds=3, sandbox_cls=FakeSandbox, write=V1):
    """full/ablated: Genauigkeit je Anzahl bisheriger Reparaturen (Index, letzter Wert gilt weiter)."""
    holder = {}

    def acc(seed, is_ablated):
        table = ablated if is_ablated else full
        return table[min(len(holder["calls"]["repair"]), len(table) - 1)]

    sb = sandbox_cls(acc=acc)
    o, calls = make_orch(sb, seeds=seeds, repairs=list(repairs), write=write)
    holder["calls"] = calls
    o.graph_repair = flag
    events = []
    o._emit = lambda type, **kw: events.append((type, kw))
    return o, calls, events


def run(o, tmp_path):
    claim = Claim("c", "GCN", "accuracy", 0.83, 0.01, "Tabelle", "paper")
    d = Dossier(paper_id="p", claims=[claim])
    o.run_claim({"seeds": o.seeds}, claim, cora_task(), "snap", d, tmp_path)
    return d


def types(events, name):
    return [kw for t, kw in events if t == name]


def test_flag_default_off():
    assert inspect.signature(Orchestrator.__init__).parameters["graph_repair"].default is False


def test_without_flag_no_repair_and_ablation_after_verdict(tmp_path):
    o, calls, events = setup([0.55], [0.55], flag=False)
    d = run(o, tmp_path)
    assert calls["repair"] == [] and types(events, "graph_repair") == []
    order = [t for t, _ in events]
    assert order.index("ablation") > max(i for i, t in enumerate(order) if t == "attempt")   # nach allen Seeds wie bisher
    assert "runde" not in types(events, "ablation")[0]
    assert d.verdicts[0].methode == "nein" and d.graph_repairs == {}
    assert "Graph-Reparaturen" not in render_markdown(d)


def test_repair_then_graph_used(tmp_path):
    o, calls, events = setup([0.55, 0.83], [0.55, 0.60], repairs=[V2])
    d = run(o, tmp_path)
    assert len(calls["repair"]) == 1
    assert calls["repair"][0].startswith("Your code does not use the graph") and "0.5500" in calls["repair"][0]
    (gr,) = types(events, "graph_repair")
    assert gr["runde"] == 1 and gr["from_hash"] == h(V1) and gr["to_hash"] == h(V2) and gr["result"] == "repariert"
    assert gr["acc_full"] == 0.55 and gr["acc_ablated"] == 0.55
    order = [(t, kw.get("seed")) for t, kw in events if t in ("attempt", "ablation")]
    first_rest = next(i for i, (t, s) in enumerate(order) if t == "attempt" and s == 1)
    assert [t for t, _ in order[:first_rest]].count("ablation") == 2            # beide Ablationen vor den Restseeds
    assert {a.code_hash for a in d.attempts if a.seed > 0} == {h(V2)}          # Restseeds mit dem reparierten Code
    v = d.verdicts[0]
    assert v.measured_mean == pytest.approx(0.83)                              # Seed 0 des verworfenen Codes (0.55) zaehlt nicht
    assert d.ablations["c"]["verdict"] == "Graph genutzt" and v.methode == "nicht geprueft"
    assert d.graph_repairs == {"c": 1} and "Graph-Reparatur Runde 1" in v.explanation
    assert any(a.code_hash == h(V1) and a.measured == 0.55 for a in d.attempts)   # verworfener Versuch bleibt sichtbar
    assert "Graph-Reparaturen: 1" in render_markdown(d)


def test_max_two_repairs_then_continue_with_last_code(tmp_path):
    assert MAX_GRAPH_REPAIRS == 2
    o, calls, events = setup([0.55], [0.55], repairs=[V2, V3])
    d = run(o, tmp_path)
    assert len(calls["repair"]) == 2 and len(types(events, "graph_repair")) == 2
    assert len(types(events, "ablation")) == 3                                 # Runde 0, 1, 2
    assert {a.code_hash for a in d.attempts if a.seed > 0} == {h(V3)}
    assert d.verdicts[0].methode == "nein" and d.graph_repairs == {"c": 2}
    assert d.ablations["c"]["verdict"] == "Graph nicht genutzt"


def test_crash_after_repair_falls_back_visibly(tmp_path):
    o, calls, events = setup([0.55], [0.55], repairs=["CRASH1", "CRASH2", "CRASH3"], sandbox_cls=CrashSandbox)
    d = run(o, tmp_path)
    (gr,) = types(events, "graph_repair")
    assert gr["result"] == "crash" and gr["fallback_hash"] == h(V1)
    assert {a.code_hash for a in d.attempts if a.seed > 0} == {h(V1)}          # weiter mit dem letzten lauffaehigen Code
    assert d.ablations["c"]["accuracy_full"] == 0.55 and d.verdicts[0].measured_mean == pytest.approx(0.55)
    assert "Rueckfall auf Code " + h(V1) in d.verdicts[0].explanation
    assert len(types(events, "ablation")) == 1                                 # Ablation des lauffaehigen Codes bleibt


def test_cost_limit_in_repair_is_not_caught(tmp_path):
    o, calls, events = setup([0.55], [0.55], repairs=[CostLimitExceeded("MAX_COST_PER_RUN_USD erreicht")])
    with pytest.raises(CostLimitExceeded):
        run(o, tmp_path)


def test_other_ablation_verdicts_trigger_no_repair(tmp_path):
    o, calls, events = setup([0.82], [0.50])                                   # Graph genutzt
    d = run(o, tmp_path)
    assert calls["repair"] == [] and types(events, "graph_repair") == [] and len(types(events, "ablation")) == 1
    assert d.graph_repairs == {} and d.ablations["c"]["verdict"] == "Graph genutzt"
    o2, calls2, ev2 = setup([0.82], [0.76])                                    # unklar (Abfall 0.06)
    run(o2, tmp_path)
    assert calls2["repair"] == [] and types(ev2, "graph_repair") == []


# ---- Jury-Filter -----------------------------------------------------------------
def checked(gesamt, status=cat.CHECKED):
    e = entry(cand())
    e["status"] = status
    e["verdict"] = {"gesamtaussage": gesamt}
    return e


@pytest.mark.parametrize("gesamt, ready", [
    ("vorlaeufig", True), ("Paper reproduziert", True), ("nicht reproduziert", True),
    ("nicht bewertbar (Methode nicht umgesetzt)", False), ("Agenten-Fehler vermutet", False),
    ("nicht reproduzierbar (Setup)", False),
])
def test_jury_ready_by_gesamtaussage(gesamt, ready):
    ok, reason = cat.jury_ready(checked(gesamt))
    assert ok is ready and gesamt in reason


@pytest.mark.parametrize("status", [cat.UNCHECKED, cat.FAILED, cat.NOT_RUNNABLE])
def test_jury_not_ready_unless_checked(status):
    ok, reason = cat.jury_ready(checked("vorlaeufig", status))
    assert ok is False and status in reason


def test_record_result_writes_jury_and_graph_repairs():
    e = entry(cand())
    d = dossier_with("reproduziert")
    d.graph_repairs = {d.claims[0].id: 2}
    cat.record_result(e, cat.CHECKED, dossier=d)
    assert e["jury"] == {"ready": True, "reason": "Gesamtaussage: vorlaeufig"} and e["verdict"]["graph_repairs"] == 2
    cat.record_result(e, cat.CHECKED, dossier=dossier_with("nicht reproduzierbar (Setup)", ()))
    assert e["jury"]["ready"] is False
    cat.record_result(e, cat.FAILED, reason="boom")
    assert e["jury"] == {"ready": False, "reason": "Status lauf_fehlgeschlagen"}


def test_list_reads_catalog_only_and_writes_nothing(tmp_path, capsys):
    a, b = entry(cand("2001.00001")), entry(cand("2001.00002"))
    cat.record_result(a, cat.CHECKED, dossier=dossier_with("reproduziert"))
    del a["jury"]                                                              # alter Eintrag ohne jury-Feld
    f = tmp_path / "catalog.yaml"
    cat.save_catalog([a, b], f)
    before = f.read_text()
    rc = curate_mod.main(["--list", "--catalog", str(f)], orchestrator_factory=lambda: pytest.fail("Aufruf bei --list"))
    out = capsys.readouterr().out
    assert rc == 0 and f.read_text() == before
    assert "2001.00001" in out and "geprueft" in out and "urteil=reproduziert" in out and "jury=ja" in out
    assert "2001.00002" in out and "jury=nein (Status ungeprueft)" in out


def test_curate_prints_overview_and_enables_graph_repair(tmp_path, capsys):
    from test_catalog import FakeOrch, run_curate
    entries = [entry(cand("2001.00001"))]
    rc, saved = run_curate(tmp_path, entries, FakeOrch(["reproduziert"]))
    out = capsys.readouterr().out
    assert rc == 0 and "Uebersicht:" in out and "jury=ja" in out and saved[0]["jury"]["ready"] is True
    src = (ROOT / "scripts" / "curate_catalog.py").read_text()
    assert "graph_repair=True" in src
