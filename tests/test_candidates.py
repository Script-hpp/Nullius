"""A1: mehrere Code-Kandidaten, Auswahl ueber Validierung (nur Produktpfad). Attrappen, kein Netz, kein Geld."""

import csv
import json
import threading

import pytest

from repro_agent import llm as llm_mod
from repro_agent.config import LLMConfig, Settings
from repro_agent.dossier import Claim, Dossier
from repro_agent.harness import etth1_data
from repro_agent.loop import CostLimitExceeded, Orchestrator, select_candidate
from repro_agent.sandbox.base import RunResult
from repro_agent.tasks import ETTH1_VAL_WINDOWS, cora_task, etth1_task
from test_loop_fakes import FakeSandbox, make_orch

C0, C1, C2 = "print('c0')\n", "print('c1')\n", "print('c2')\n"


class ValSandbox(FakeSandbox):
    """Test- und Val-Genauigkeit je Code; ohne Kanten (Ablation) 0.3."""

    def __init__(self, test_by_code, val_by_code):
        super().__init__()
        self.test_by_code, self.val_by_code, self.code_of = test_by_code, val_by_code, {}

    def put_file(self, branch, path, content):
        if path.endswith("train.py"):
            self.code_of[branch] = content.decode()
        super().put_file(branch, path, content)

    def run(self, branch, cmd, timeout_s):
        code = self.code_of.get(branch)
        if "cora_val_eval.py" in cmd:
            v = self.val_by_code.get(code)
            if v is None:
                return RunResult(0, json.dumps({"status": "ungueltig", "reason": "x"}), "", False)
            self.cmds.append((branch, cmd))
            return RunResult(0, json.dumps({"status": "ok", "metric": "accuracy", "value": v, "baselines": {"majority": 0.3}}), "", False)
        if "cora_eval.py" in cmd:
            self.cmds.append((branch, cmd))
            acc = 0.3 if branch in self.ablated else self.test_by_code[code]
            out = {"status": "ok", "metrics": {"accuracy": acc},
                   "baselines": {"majority": {"accuracy": 0.13}, "no_graph": {"accuracy": 0.2}}}
            return RunResult(0, json.dumps(out), "", False)
        return super().run(branch, cmd, timeout_s)


def setup(test_by_code, val_by_code, flag=True, k=3):
    sb = ValSandbox(test_by_code, val_by_code)
    o, calls = make_orch(sb, seeds=3)
    temps, codes = [], [C0, C1, C2]

    def write_code(plan, task, temperature=0.2):
        temps.append(temperature)
        return codes[len(temps) - 1]

    o.write_code = write_code
    o.graph_repair, o.n_candidates = flag, k
    events = []
    o._emit = lambda type, **kw: events.append((type, kw))
    return o, sb, temps, events


def run(o, tmp_path):
    claim = Claim("c", "GCN", "accuracy", 0.83, 0.01, "Tabelle", "paper")
    d = Dossier(paper_id="p", claims=[claim])
    o.run_claim({"seeds": 3}, claim, cora_task(), "snap", d, tmp_path)
    return d


def test_select_candidate_rules():
    assert select_candidate([{"metric": "accuracy", "value": 0.7}, {"metric": "accuracy", "value": 0.8}, None]) == (1, "val")
    assert select_candidate([{"metric": "mse", "value": 0.7}, {"metric": "mse", "value": 0.4}]) == (1, "val")
    assert select_candidate([{"metric": "accuracy", "value": 0.8}, {"metric": "accuracy", "value": 0.8}]) == (0, "val")   # Gleichstand: kleiner Index
    assert select_candidate([None, {"metric": "accuracy", "value": 0.1}]) == (1, "val")
    assert select_candidate([None, None]) == (0, "fallback")


def test_best_val_wins_even_if_another_has_best_test(tmp_path):
    # Test: C0 am besten (0.83), Val: C1 am besten -> gewaehlt wird C1
    o, sb, temps, events = setup({C0: 0.83, C1: 0.60, C2: 0.70}, {C0: 0.70, C1: 0.80, C2: 0.75})
    d = run(o, tmp_path)
    assert temps == pytest.approx([0.2, 0.5, 0.8])
    (ev,) = [kw for t, kw in events if t == "candidates"]
    assert ev["chosen"] == 1 and ev["reason"] == "val" and [c["val"]["value"] for c in ev["candidates"]] == [0.70, 0.80, 0.75]
    assert "measured" not in json.dumps(ev) and "0.83" not in json.dumps(ev)          # keine Testwerte im Ereignis
    rest = [a for a in d.attempts if a.seed > 0]
    assert {a.candidate for a in rest} == {1} and len({a.code_hash for a in rest}) == 1
    seed0 = [a for a in d.attempts if a.seed == 0]
    assert sorted((a.candidate, a.chosen) for a in seed0) == [(0, False), (1, True), (2, False)]
    assert all(a.chosen for a in d.attempts if a.candidate == 1)
    assert d.verdicts[0].measured_mean == pytest.approx(0.60)                           # nur der Sieger zaehlt
    (selected,) = [kw for t, kw in events if t == "measurements_selected"]
    assert [a["seed"] for a in selected["attempts"]] == [0, 1, 2]
    assert [a["measured"] for a in selected["attempts"]] == pytest.approx([0.60] * 3)
    assert {a["code_hash"] for a in selected["attempts"]} == {rest[0].code_hash}
    assert "gewaehlt Kandidat 1" in d.verdicts[0].explanation


def test_ablation_and_rest_seeds_only_for_winner(tmp_path):
    o, sb, temps, events = setup({C0: 0.83, C1: 0.60, C2: 0.70}, {C0: 0.70, C1: 0.80, C2: 0.75})
    run(o, tmp_path)
    abl = [kw for t, kw in events if t == "ablation"]
    assert len(abl) == 1 and abl[0]["accuracy_full"] == 0.60                            # Seed 0 des Siegers wiederverwendet
    # Val nur Seed 0 je Kandidat (3) plus der B1-Lauf mit maskierten Val-Labels je Kandidat (3), nicht bei Ablation/Rest
    assert sum(1 for _, c in sb.cmds if "cora_val_eval.py" in c) == 6


def test_fallback_without_val_values_takes_first_runnable(tmp_path):
    o, sb, temps, events = setup({C0: 0.83, C1: 0.60, C2: 0.70}, {})
    d = run(o, tmp_path)
    (ev,) = [kw for t, kw in events if t == "candidates"]
    assert ev["chosen"] == 0 and ev["reason"] == "fallback" and "Rueckfall ohne Val-Wert" in d.verdicts[0].explanation


def test_repair_counters_sum_over_candidates(tmp_path):
    o, sb, temps, events = setup({C0: 0.83, C1: 0.60, C2: 0.70}, {C0: 0.7, C1: 0.8, C2: 0.75})
    d = Dossier(paper_id="p", claims=[])
    d.fault_repairs["c"] = 1
    claim = Claim("c", "GCN", "accuracy", 0.83, 0.01, "T", "paper")
    d.claims = [claim]
    o.run_claim({"seeds": 3}, claim, cora_task(), "snap", d, tmp_path)
    assert d.fault_repairs["c"] == 1                                                    # keine Reparaturen -> unveraendert


def test_cost_limit_in_candidate_phase_propagates(tmp_path):
    o, sb, temps, events = setup({C0: 0.8}, {})

    def write(plan, task, temperature=0.2):
        if temps:
            raise CostLimitExceeded("MAX_COST_PER_RUN_USD erreicht")
        temps.append(temperature)
        return C0

    o.write_code = write
    with pytest.raises(CostLimitExceeded):
        run(o, tmp_path)


def test_without_flag_single_code_no_val(tmp_path):
    o, sb, temps, events = setup({C0: 0.83}, {C0: 0.7}, flag=False)
    o.write_code = lambda plan, task: C0       # alte Signatur: ohne temperature
    d = run(o, tmp_path)
    assert not any("val_eval" in c for _, c in sb.cmds) and not [t for t, _ in events if t == "candidates"]
    assert all(a.candidate is None and a.val is None for a in d.attempts)


def test_prompts_val_notes_only_in_product_path():
    prompts = {}
    for flag in (False, True):
        o = Orchestrator.__new__(Orchestrator)
        o.details, o.graph_repair = [], flag
        o._code_from_llm = lambda system, user, **kw: user
        prompts[flag] = o.write_code("plan", cora_task())
    assert "val_predictions.csv" not in prompts[False] and "val_predictions.csv" in prompts[True]
    assert prompts[True].startswith(prompts[False].split(" Write train.py")[0])


def test_val_smoke_file_checked_only_in_product_path():
    t = cora_task()
    ids = list(range(20))
    ok = ("id,prediction\n" + "".join(f"{i},1\n" for i in ids)).encode()

    class SB:
        def __init__(self):
            self.files, self.cmds = {"/output/predictions.csv": ok}, []

        def run(self, branch, cmd, timeout_s):
            self.cmds.append(cmd)
            if "print(*sorted" in cmd:
                return RunResult(0, " ".join(map(str, ids)), "", False)
            return RunResult(0, "", "", False)

        def get_file(self, branch, path):
            if path not in self.files:
                raise FileNotFoundError(path)
            return self.files[path]

    for flag, expect_error in ((False, False), (True, True)):
        o = Orchestrator.__new__(Orchestrator)
        o.sandbox, o._smoke_ok, o.graph_repair = SB(), set(), flag
        err = o._smoke_check("b", "code", t)
        assert (err is not None and "val_predictions.csv" in err) == expect_error
    o = Orchestrator.__new__(Orchestrator)
    o.sandbox, o._smoke_ok, o.graph_repair = SB(), set(), True
    o.sandbox.files["/output/val_predictions.csv"] = ok
    assert o._smoke_check("b", "code2", t) is None


def test_etth1_val_windows_stay_inside_val(tmp_path):
    import numpy as np

    rows = np.arange(14400, dtype=float)[:, None] + np.zeros((1, 7))
    rows += np.arange(7)[None, :] * 0.001
    path = tmp_path / "ETTh1.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + [f"c{i}" for i in range(7)])
        for i, r in enumerate(rows):
            w.writerow([i] + list(r))
    assert etth1_data.n_val_windows() == ETTH1_VAL_WINDOWS == 117 and etth1_data.VAL_STRIDE == 24
    z = etth1_data.load_standardized(path)
    vt = etth1_data.val_targets(path)
    assert vt.shape == (117, 96, 7)
    assert np.array_equal(vt[0], z[8640:8640 + 96]) and np.array_equal(vt[116], z[8640 + 24 * 116:8640 + 24 * 116 + 96])
    assert 8640 + 24 * 116 + 95 < etth1_data.VAL_END                                    # letztes Zielfenster endet in Val
    assert not np.isin(vt[:, :, 0], z[etth1_data.VAL_END:, 0]).any()                    # kein Testzeilen-Ziel
    t = etth1_task()
    assert len(t.val_smoke_ids) == 8 * 96 * 7 and "0..116" in t.val_notes


def test_cache_key_contains_temperature():
    s = Settings(llm=LLMConfig("openrouter", "http://x", "k", "m", "m", "m"), sandbox_backend="local",
                 tavily_api_key="", max_cost_per_run_usd=1.0, max_attempts_per_claim=1, llm_cache=True)
    c = llm_mod.LLMClient(s)
    msgs = [{"role": "user", "content": "x"}]
    keys = {c._cache_key("code", msgs, t) for t in (0.2, 0.5, 0.8)}
    assert len(keys) == 3


def test_cost_log_lines_not_interleaved_under_threads(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_mod, "COST_LOG_PATH", tmp_path / "costs.jsonl")
    s = Settings(llm=LLMConfig("openrouter", "http://x", "k", "m", "m", "m"), sandbox_backend="local",
                 tavily_api_key="", max_cost_per_run_usd=1.0, max_attempts_per_claim=1, llm_cache=False)
    c = llm_mod.LLMClient(s)
    resp = llm_mod.LLMResponse("x" * 2000, "m", 10, 10, 0.001, False)
    ts = [threading.Thread(target=lambda: [c._book(resp) for _ in range(30)]) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    lines = (tmp_path / "costs.jsonl").read_text().splitlines()
    assert len(lines) == 240 and all(json.loads(x) for x in lines)
    assert c.total_cost_usd == pytest.approx(0.24)


def test_forbidden_library_hint_in_first_prompt_and_repair_prompt_product_path_only():
    """M3: Hinweis schon im Code-Prompt (write_code) und im Reparatur-Prompt; Pfad ohne Flag bleibt wie bisher."""
    seen = {}
    for flag in (False, True):
        o = Orchestrator.__new__(Orchestrator)
        o.details, o.graph_repair = [], flag
        o._code_from_llm = lambda system, user, **kw: user
        t = cora_task()
        seen[flag] = (o.write_code("plan", t), o.repair_code("plan", t, "x = 1", "boom"))
    for text in seen[True]:
        assert "are NOT installed and are forbidden" in text and "dgl" in text and "ogb" in text and "plain PyTorch" in text
    for text in seen[False]:
        assert "are NOT installed and are forbidden" not in text and "val_predictions" not in text
    # Pfad ohne Flag: Prompt exakt wie vor A1/A3
    t = cora_task()
    assert seen[False][0] == (
        f"{t.data_description}\n\n\n\nReproduction plan:\nplan\n\nThe script runs WITHOUT network access and must not download "
        f"anything, spawn processes or use eval/exec. {t.code_notes} Write train.py implementing this plan.")
    assert seen[False][1] == (
        f"{t.data_description}\n\n\n\nReproduction plan:\nplan\n\nThis script failed:\n```python\nx = 1\n```\nProblem: boom\n\n"
        "Return the corrected full script.")


def test_candidate_without_code_is_skipped(tmp_path):
    """Liefert ein Kandidat keinen Code (ValueError/RuntimeError aus dem LLM-Pfad), laeuft der Lauf mit den uebrigen weiter;
    der Index des Kandidaten bleibt erhalten."""
    o, sb, temps, events = setup({C0: 0.80, C2: 0.70}, {C0: 0.70, C2: 0.75})

    def write_code(plan, task, temperature=0.2):
        temps.append(temperature)
        if len(temps) == 2:
            raise ValueError("kein Code-Block")
        return {1: C0, 3: C2}[len(temps)]

    o.write_code = write_code
    d = run(o, tmp_path)
    (ev,) = [kw for t, kw in events if t == "candidates"]
    assert [c["index"] for c in ev["candidates"]] == [0, 2] and ev["chosen"] == 2
    assert [kw["candidate"] for t, kw in events if t == "code_missing"] == [1]
    assert d.verdicts[0].measured_mean == pytest.approx(0.70)


def test_all_candidates_without_code_end_run(tmp_path):
    o, sb, temps, events = setup({C0: 0.8}, {C0: 0.7})

    def write_code(plan, task, temperature=0.2):
        raise ValueError("kein Code-Block")

    o.write_code = write_code
    with pytest.raises(RuntimeError, match="Kein Code-Kandidat"):
        run(o, tmp_path)
