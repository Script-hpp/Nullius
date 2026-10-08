"""Steckbrief-Modus im Loop (docs/recall_steckbrief_plan.md): Flag aus = Prompts unveraendert, Auswahl konform vor Validierung,
S4-Einbindung (Fixture je Einheit, Ereignis model_file_missing). Attrappen, kein Netz, kein Geld."""

import pytest

from repro_agent import loop as loop_mod
from repro_agent import spec_check
from repro_agent.dossier import Claim
from repro_agent.loop import Orchestrator
from repro_agent.repo_source import RepoDump
from repro_agent.setup_details import Doc
from repro_agent.spec import Spec, SpecItem, format_spec
from repro_agent.tasks import cora_task, etth1_task
from test_candidates import C0, C1, C2, run, setup


def _spec(enforced: bool = True) -> Spec:
    items = [SpecItem("lr", "0.005", "learning rate 0.005", "paper", "paper text", 0.005, enforce=enforced),
             SpecItem("feature_norm", "row-normalised", "features are row-normalised", "paper", "paper text")]
    return Spec(model="GCN", family="graph", items=items)


def _prompts(spec_mode: bool, with_spec: bool, task) -> dict:
    o = Orchestrator.__new__(Orchestrator)
    o.details, o.graph_repair = [], False
    if spec_mode is not None:
        o.spec_mode = spec_mode
    if with_spec:
        o.spec = _spec()
    seen = {}
    o._ask = lambda role, system, user, **kw: seen.setdefault("plan", user) and type("R", (), {"content": "plan", "finish_reason": "stop"})()
    o._code_from_llm = lambda system, user, **kw: user
    o.plan({"title": "T"}, Claim("c", "GCN erreicht 81,5 %", "accuracy", 0.815, 0.01, "T", "paper"), task)
    return {"plan": seen["plan"], "write": o.write_code("PLAN", task), "repair": o.repair_code("PLAN", task, "print(1)", "boom")}


@pytest.mark.parametrize("task", [cora_task(), etth1_task()])
def test_flag_off_prompts_identical_even_with_spec_present(task):
    legacy = _prompts(None, False, task)          # Objekt ohne Attribut spec_mode (wie vor dem Steckbrief)
    off = _prompts(False, True, task)             # Flag aus, Steckbrief vorhanden: darf nie in einen Prompt
    assert legacy == off
    assert all("CONFIG" not in p and "Reproduction spec" not in p for p in off.values())


def test_flag_on_changes_prompts():
    on = _prompts(True, True, cora_task())
    assert all("Reproduction spec" in on[k] for k in ("plan", "write", "repair"))
    assert "CONFIG = {" in on["write"]


def test_selection_prefers_spec_conform_over_better_validation(tmp_path):
    o, sb, temps, events = setup({C0: 0.83, C1: 0.60, C2: 0.70}, {C0: 0.90, C1: 0.70, C2: 0.65})
    o.spec_mode, o.spec = True, _spec(enforced=False)
    o.repair_code = lambda plan, task, code, msg: code           # Reparatur aendert nichts
    o._enforce_spec = lambda code: code
    asked = []

    def check(ask, code, spec):
        asked.append(code)
        return [{"key": "feature_norm", "value": "x", "ok": code == C1, "lines": "", "reason": "nein"}]

    spec_check.check_items = check
    try:
        run(o, tmp_path)
    finally:
        spec_check.check_items = ORIG_CHECK
    (ev,) = [kw for t, kw in events if t == "candidates"]
    assert ev["chosen"] == 1 and ev["reason"].startswith("spec (1)")     # C0 hat die beste Validierung, ist aber nicht konform
    assert [kw["round"] for t, kw in events if t == "spec_check" and kw["candidate"] == 0] == [0, 1, 2]   # hoechstens 2 Reparaturrunden


ORIG_CHECK = spec_check.check_items


class _Orch(Orchestrator):
    def __init__(self):
        self.paper_only, self.tavily_mode, self.spec_mode, self.setup_v2 = False, "live", True, True
        self.events, self._warnings, self._paper_id, self._tavily_calls = [], [], "zeng_nlinear_2023", 0
        self.sandbox = self.llm = None

    def _emit(self, type, **kw):
        self.events.append((type, kw))

    def _tavily_getter(self):
        return lambda: None

    def _record_source(self, *a, **k):
        pass


@pytest.fixture
def gather(monkeypatch):
    seen = {}
    monkeypatch.setattr(loop_mod, "find_repo_url", lambda text: "https://github.com/x/LTSF-Linear")
    monkeypatch.setattr(loop_mod, "fetch_readme", lambda *a, **k: "")

    def fake_read(sandbox, llm, url, claim_text, dataset, **kw):
        seen.update(kw)
        return RepoDump(url=url, commit="abc1234", files=seen.get("_files", {"models/DLinear.py": "class Model: pass"}), n_tracked=1)

    monkeypatch.setattr(loop_mod, "read_repo", fake_read)
    o = _Orch()
    o.paper_text = type("P", (), {"text": "paper", "arxiv_id": "1"})()
    return o, seen


def test_gather_uses_unit_key_and_reports_missing_model_file(gather):
    o, seen = gather
    o._gather_docs("NLinear erreicht MSE 0,4 auf ETTh1", "ETTh1")
    assert seen["unit_key"] == "zeng_nlinear_2023" and seen["model"] == "NLinear" and seen["mode"] == "live"
    (ev,) = [kw for t, kw in o.events if t == "model_file_missing"]
    assert ev["model"] == "NLinear" and "models/DLinear.py" in ev["files"]        # falsche Datei (DLinear) zaehlt nicht als Modelldatei


def test_gather_no_event_when_model_file_present(gather):
    o, seen = gather
    seen["_files"] = {"models/NLinear.py": "class Model: pass"}
    o._gather_docs("NLinear erreicht MSE 0,4 auf ETTh1", "ETTh1")
    assert not [t for t, _ in o.events if t == "model_file_missing"]


def test_gather_flag_off_keeps_repo_key(gather):
    o, seen = gather
    o.spec_mode = False
    o._gather_docs("NLinear erreicht MSE 0,4 auf ETTh1", "ETTh1")
    assert seen["unit_key"] is None and seen["model"] is None


def test_single_code_path_also_checks_and_repairs_spec(tmp_path, monkeypatch):
    o, sb, temps, events = setup({C0: 0.83, C1: 0.83}, {}, flag=False)
    o.write_code = lambda plan, task: C0
    o.spec_mode, o.spec = True, _spec(enforced=False)
    o._enforce_spec = lambda code: code
    o.repair_code = lambda plan, task, code, msg: C1
    monkeypatch.setattr(spec_check, "check_items",
                        lambda ask, code, spec: [{"key": "feature_norm", "value": "x", "ok": code == C1, "lines": "", "reason": "nein"}])
    run(o, tmp_path)
    rounds = [(kw["round"], kw["candidate"], kw["score"]) for t, kw in events if t == "spec_check"]
    assert rounds == [(0, None, (0, 1)), (1, None, (1, 1))]
