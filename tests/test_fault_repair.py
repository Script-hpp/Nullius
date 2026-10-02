"""Agenten-Fehler-Reparatur nach Seed 0, Plan-Validator, Methodentreue-Hinweis, Cache aus (nur Katalog/API-Flag). Attrappen, kein Netz."""

import inspect
import json
import re
from types import SimpleNamespace

import pytest

from repro_agent import api, catalog as cat
from repro_agent import loop as loop_mod
from repro_agent.claims_source import PaperText, extract_target_cached
from repro_agent.config import LLMConfig, Settings
from repro_agent.dossier import Claim, Dossier
from repro_agent.harness import repair_hints
from repro_agent.loop import MAX_FAULT_REPAIRS, MAX_GRAPH_REPAIRS, MAX_SEED0_REPAIRS, Orchestrator, plan_needs_retry
from repro_agent.report import render_markdown
from repro_agent.sandbox.base import RunResult
from repro_agent.tasks import cora_task
from test_graph_repair import V1, V2, h, run, setup, types
from test_loop_fakes import FakeSandbox, make_orch


class DiagSandbox(FakeSandbox):
    """Harness liefert zusaetzlich diagnostics mit Teststatistik (target_std, corr), die nie in einen Prompt duerfen."""

    def run(self, branch, cmd, timeout_s):
        res = super().run(branch, cmd, timeout_s)
        if "cora_eval.py" in cmd and res.stdout:
            out = json.loads(res.stdout)
            out["diagnostics"] = {"pred_std": 2.5, "target_std": 0.987654, "corr": 0.123456}
            return RunResult(res.exit_code, json.dumps(out), res.stderr, res.timed_out)
        return res


def test_fault_repair_fixes_code_below_trivial_baseline(tmp_path):
    # erster Code liefert 0.12 (< Baseline no_graph 0.50), repariert 0.81; Graph-Check danach bestanden
    o, calls, events = setup([0.12, 0.81], [0.10, 0.50], repairs=[V2], sandbox_cls=DiagSandbox)
    d = run(o, tmp_path)
    assert len(calls["repair"]) == 1
    msg = calls["repair"][0]
    assert msg.startswith("Your script ran, but its result is worse than a trivial baseline")
    assert "Measured ACCURACY: 0.1200" in msg and "baselines: majority, no_graph" in msg and "The paper reports 0.83" in msg
    assert "0.1300" not in msg and "0.5000" not in msg
    assert "copy.deepcopy(model.state_dict())" in msg and "Keep the method described in the plan" in msg
    (fr,) = types(events, "fault_repair")
    assert fr["runde"] == 1 and fr["from_hash"] == h(V1) and fr["to_hash"] == h(V2) and fr["result"] == "repariert" and fr["measured"] == 0.12
    assert fr["baselines"] == {"majority": 0.13, "no_graph": 0.5}
    assert d.fault_repairs == {"c": 1} and d.graph_repairs == {}
    assert d.verdicts[0].measured_mean == pytest.approx(0.81)            # Seed 0 des verworfenen Codes zaehlt nicht
    assert d.ablations["c"]["verdict"] == "Graph genutzt"
    assert "Fehler-Reparaturen: 1" in render_markdown(d) and "Fehler-Reparatur Runde 1" in d.verdicts[0].explanation


def test_feedback_never_contains_test_statistics(tmp_path):
    o, calls, events = setup([0.12, 0.81], [0.10, 0.50], repairs=[V2], sandbox_cls=DiagSandbox)
    d = run(o, tmp_path)
    msg = calls["repair"][0]
    assert "0.987654" not in msg and "0.123456" not in msg and "target_std" not in msg
    assert not re.search(r"\bcorr\b|correlation", msg.lower())
    a = next(x for x in d.attempts if x.seed == 0)
    assert a.diagnostics == {"pred_std": 2.5, "target_std": 0.987654, "corr": 0.123456}   # fuer uns im Dossier
    ev = next(e for t, e in events if t == "attempt")
    assert ev["diagnostics"]["corr"] == 0.123456 and "stdout_tail" in ev


def test_forecast_message_has_prediction_spread_and_no_target_statistics():
    msg = repair_hints.fault_message("mse", 2.3191, 0.35, {"zero": {"mse": 1.1099, "mae": 0.8}, "repeat": {"mse": 1.2944, "mae": 0.7}}, pred_std=2.5)
    assert "Measured MSE: 2.3191" in msg and "baselines: zero, repeat" in msg
    assert "1.1099" not in msg and "1.2944" not in msg   # Baseline-Werte sind Teststatistik
    assert "standard deviation of your predictions is 2.500; the training data (standardised) has 1.0" in msg
    assert "'{window}_{step}_{channel}'" in msg and "copy.deepcopy(model.state_dict())" in msg and "normalise each input window" in msg and "if the plan or the paper uses input normalisation" in msg
    assert "target_std" not in msg and "correlation" not in msg.lower()
    assert list(inspect.signature(repair_hints.fault_message).parameters) == ["metric", "measured", "target", "baselines", "pred_std"]   # keine Teststatistik als Parameter


def test_classification_message_has_label_checklist():
    msg = repair_hints.fault_message("accuracy", 0.12, 0.83, {"majority": {"accuracy": 0.13}})
    assert "one class id per test node id" in msg and "standard deviation" not in msg and "copy.deepcopy" in msg


def test_fault_and_graph_rounds_share_a_total_limit(tmp_path):
    assert (MAX_FAULT_REPAIRS, MAX_GRAPH_REPAIRS, MAX_SEED0_REPAIRS) == (2, 2, 3)
    o, calls, events = setup([0.12], [0.12], repairs=[f"print('v{i}')\\n" for i in range(2, 9)])   # bleibt schlecht und ohne Graph
    d = run(o, tmp_path)
    assert len(calls["repair"]) == MAX_SEED0_REPAIRS
    assert len(types(events, "fault_repair")) == 2 and len(types(events, "graph_repair")) == 1
    assert d.fault_repairs == {"c": 2} and d.graph_repairs == {"c": 1}


def test_fault_repair_crash_falls_back_to_last_runnable_code(tmp_path):
    from test_graph_repair import CrashSandbox
    o, calls, events = setup([0.12], [0.12], repairs=["CRASH1", "CRASH2", "CRASH3"], sandbox_cls=CrashSandbox)
    d = run(o, tmp_path)
    (fr,) = types(events, "fault_repair")
    assert fr["result"] == "crash" and fr["fallback_hash"] == h(V1)
    assert {a.code_hash for a in d.attempts if a.seed > 0} == {h(V1)}
    assert "Rueckfall auf Code " + h(V1) in d.verdicts[0].explanation


def test_without_flag_nothing_changes(tmp_path):
    o, calls, events = setup([0.12], [0.10], flag=False)
    d = run(o, tmp_path)
    assert calls["repair"] == [] and types(events, "fault_repair") == [] and d.fault_repairs == {}
    assert d.verdicts[0].urteil == "Agenten-Fehler vermutet"
    order = [t for t, _ in events]
    assert order.index("ablation") > max(i for i, t in enumerate(order) if t == "attempt")


def test_good_result_triggers_no_fault_repair(tmp_path):
    o, calls, events = setup([0.82], [0.50])
    run(o, tmp_path)
    assert calls["repair"] == [] and types(events, "fault_repair") == []


# ---- Plan-Validator und Methodentreue-Hinweis (nur mit Flag) -----------------------------------------
@pytest.mark.parametrize("text, retry", [
    ("```python\nimport numpy as np\n```", True),
    ("import numpy as np\nimport torch\nx = 1", True),
    ("1. Patch the series\n2. Mixture of experts\n3. Train with Huber loss", False),
    ("\n".join(f"step {i}" for i in range(60)), True),
])
def test_plan_needs_retry(text, retry):
    assert plan_needs_retry(text) is retry


def plan_orch(flag, answers):
    o, _ = make_orch(FakeSandbox())
    del o.plan                      # make_orch ersetzt plan; wir wollen die echte Methode
    o.graph_repair = flag
    o.details = []
    asked, events = [], []
    def ask(role, system, user, max_tokens=None, temperature=0.2):
        asked.append((role, user, temperature))
        return SimpleNamespace(content=answers[min(len(asked) - 1, len(answers) - 1)])
    o._ask = ask
    o._emit = lambda type, **kw: events.append((type, kw))
    return o, asked, events


def test_plan_retry_only_with_flag_and_only_once():
    code = "```python\nimport numpy as np\nimport os\n```"
    o, asked, events = plan_orch(True, [code, "1. real plan"])
    claim = Claim("c", "t", "mse", 0.35, 0.01, "s", "paper")
    assert o.plan({"title": "T"}, claim, cora_task()) == "1. real plan"
    assert len(asked) == 2 and "no code block" in asked[1][1] and asked[1][2] == 0.4 and events[0][0] == "plan_retry"
    o, asked, events = plan_orch(True, [code])                     # auch die zweite Antwort ist Code: nicht endlos
    assert o.plan({"title": "T"}, claim, cora_task()) == code and len(asked) == 2
    o, asked, events = plan_orch(False, [code])                    # Benchmark: unveraendert
    assert o.plan({"title": "T"}, claim, cora_task()) == code and len(asked) == 1 and events == []


def test_fidelity_note_only_with_flag():
    for flag in (True, False):
        o, _ = make_orch(FakeSandbox())
        del o.write_code, o.repair_code
        o.graph_repair, o.details = flag, []
        seen = []
        o._code_from_llm = lambda system, user: seen.append(user) or "code"
        o.write_code("plan", cora_task())
        o.repair_code("plan", cora_task(), "old", "boom")
        assert all(("Plan step -> implemented in" in u) is flag and ("core mechanism" in u) is flag for u in seen) and len(seen) == 2


# ---- Cache aus fuer Live-Laeufe, Zielwert aus gespeicherter Extraktion sichtbar -------------------------------
def settings(cache=True):
    return Settings(llm=LLMConfig("tokenfactory", "http://x", "k", "m", "m", "m"), sandbox_backend="local", tavily_api_key="",
                    max_cost_per_run_usd=0.5, max_attempts_per_claim=3, llm_cache=cache)


def make_real(monkeypatch, cache=True, **kw):
    monkeypatch.setattr(loop_mod, "load_settings", lambda: settings(cache))
    monkeypatch.setattr(loop_mod, "LLMClient", lambda s: SimpleNamespace(total_cost_usd=0.0, settings=s))
    monkeypatch.setattr(loop_mod, "get_sandbox", lambda backend: object())
    return Orchestrator(**kw)


def test_llm_cache_override(monkeypatch):
    assert make_real(monkeypatch).settings.llm_cache is True                       # CLI/Benchmark: Default aus .env
    assert make_real(monkeypatch, llm_cache=False).settings.llm_cache is False
    assert make_real(monkeypatch, cache=False, llm_cache=True).settings.llm_cache is True
    assert make_real(monkeypatch, fresh=True).settings.llm_cache is False          # fresh bleibt Alias
    assert make_real(monkeypatch).llm.settings.llm_cache is True


def test_api_live_runs_switch_the_cache_off(monkeypatch):
    got = {}
    monkeypatch.setattr(loop_mod, "Orchestrator", lambda **kw: got.update(kw) or SimpleNamespace())
    api.make_orchestrator(True)
    assert got == {"graph_repair": True, "setup_v2": True, "seeds": 3, "llm_cache": False}
    got.clear()
    api.make_orchestrator(False)
    assert got["llm_cache"] is False and got["seeds"] is None


def test_llm_call_event_reports_cached_and_model():
    o, _ = make_orch(FakeSandbox())
    seen = []
    o.on_event = seen.append
    o._lock = __import__("threading").Lock()
    o.llm = SimpleNamespace(total_cost_usd=0.0, gross=lambda x: x * 1.19,
                            complete=lambda role, msgs, **k: SimpleNamespace(content="c", cost_usd=0.0, cached=True, model="nvidia/x"))
    o._ask("code", "s", "u")
    assert seen[0]["cached"] is True and seen[0]["model"] == "nvidia/x"


def test_target_from_stored_extraction_is_flagged(tmp_path):
    from pathlib import Path
    fixture = Path(__file__).resolve().parents[1] / "benchmark" / "fixtures" / "1609.02907.json"
    paper = PaperText(**json.loads(fixture.read_text()))
    ev = extract_target_cached(None, paper, "GCN (this paper)", "Cora", "accuracy")   # liegt als Fixture im Repo
    assert ev.from_store is True


def test_dossier_roundtrip_and_catalog_verdict_carry_fault_repairs(tmp_path, monkeypatch):
    monkeypatch.setattr("repro_agent.dossier.REPO_ROOT", tmp_path)
    d = Dossier(paper_id="p", fault_repairs={"c": 2})
    assert Dossier.load(d.save("t")).fault_repairs == {"c": 2}
    from test_catalog import dossier_with, entry, cand
    dd = dossier_with("reproduziert")
    dd.fault_repairs = {dd.claims[0].id: 1}
    assert cat.verdict_dict(entry(cand()), dd)["fault_repairs"] == 1


def test_prediction_diagnostics_and_harness_file_list():
    import numpy as np
    from repro_agent.harness.diagnostics import prediction_diagnostics
    from repro_agent.tasks import HARNESS_DIR, etth1_task
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(4, 3, 2))
    d = prediction_diagnostics(truth * 2 + 0.0, truth)
    assert d["pred_std"] == pytest.approx(2 * truth.std()) and d["target_std"] == pytest.approx(truth.std()) and d["corr"] == pytest.approx(1.0)
    assert prediction_diagnostics(np.ones((2, 2)), truth[:2, :2])["corr"] is None       # konstante Vorhersage: keine Korrelation
    assert "diagnostics.py" in etth1_task().harness_files and (HARNESS_DIR / "diagnostics.py").exists()


def test_fault_on_mean_over_seeds_triggers_one_repair_and_rerun(tmp_path):
    # Seed 0 liegt ueber der Baseline (0.81), die anderen Seeds weit darunter: Mittel 0.32 < no_graph 0.50
    o, calls, events = setup([0.81], [0.10], repairs=[V2], sandbox_cls=DiagSandbox)
    sb = o.sandbox
    sb.acc = lambda seed, abl: 0.10 if abl else (0.81 if seed == 0 or calls["repair"] else 0.08)
    d = run(o, tmp_path)
    assert len(calls["repair"]) == 1 and calls["repair"][0].startswith("Your script ran, but its result is worse")
    assert d.verdicts[0].measured_mean == pytest.approx(0.81) and d.verdicts[0].urteil != "Agenten-Fehler vermutet"
    assert d.fault_repairs == {"c": 1}
    assert [e["runde"] for e in types(events, "fault_repair")] == ["mittelwert"]


def test_repair_hints_for_forbidden_library_and_one_class():
    from repro_agent.loop import repair_hint
    assert "plain PyTorch" in repair_hint("Statische Pruefung: Dataset-Bibliothek: 'torch_geometric' in Zeile 5")
    assert "copy.deepcopy" in repair_hint("Nur eine einzige Vorhersage-Klasse ueber alle IDs -- verdaechtig")
    assert repair_hint("RuntimeError: boom") == ""
