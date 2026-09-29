"""Loop-Tests mit Attrappen (Sandbox und LLM). Kein Netz, kein Geld. Fangen Pipeline-Fehler vor teuren Laeufen ab."""

import json
import re
import threading
from types import SimpleNamespace

import pytest

from repro_agent.config import LLMConfig, Settings
from repro_agent.dossier import Claim, Dossier
from repro_agent.llm import BillingBlocked
from repro_agent.loop import CostLimitExceeded, Orchestrator
from repro_agent.sandbox.base import RunResult
from repro_agent.tasks import cora_task

SMOKE_IDS = list(range(20))
GOOD_CODE = "print('ok')\n"


def csv_ids(ids):
    return ("id,prediction\n" + "".join(f"{i},1\n" for i in ids)).encode()


class FakeSandbox:
    """Antwortet je nach Kommando; merkt sich alle Kommandos."""

    def __init__(self, smoke="ok", acc=lambda seed, ablated: 0.81, train_exit=0):
        self.cmds, self.n, self.smoke, self.acc, self.train_exit = [], 0, smoke, acc, train_exit
        self.seed, self.ablated, self.lock = {}, set(), threading.Lock()
        self.smoke_results = []

    def base_snapshot(self, image, setup_cmds, label=None):
        return "snap"

    def branch(self, snapshot):
        with self.lock:
            self.n += 1
            return f"b{self.n}"

    def put_file(self, branch, path, content):
        pass

    def put_files(self, branch, files):
        pass

    def get_file(self, branch, path):
        return csv_ids(SMOKE_IDS)

    def run(self, branch, cmd, timeout_s):
        with self.lock:
            self.cmds.append((branch, cmd))
        if "print(*sorted" in cmd:
            return RunResult(0, " ".join(map(str, SMOKE_IDS)), "", False)
        if "SMOKE=1" in cmd:
            res = self.smoke if isinstance(self.smoke, RunResult) else (
                self.smoke_results.pop(0) if self.smoke_results else RunResult(0, "", "", False))
            return res
        if "import coverage" in cmd:
            return RunResult(1, "", "", False)
        if "np.zeros((0, 2)" in cmd:
            self.ablated.add(branch)
            return RunResult(0, "", "", False)
        if "train.py" in cmd:
            self.seed[branch] = int(re.search(r"SEED=(\d+)", cmd).group(1))
            return RunResult(self.train_exit, "", "boom" if self.train_exit else "", False)
        if "cora_eval.py" in cmd:
            acc = self.acc(self.seed.get(branch, 0), branch in self.ablated)
            out = {"status": "ok", "metrics": {"accuracy": acc},
                   "baselines": {"majority": {"accuracy": 0.13}, "no_graph": {"accuracy": 0.5}}}
            return RunResult(0, json.dumps(out), "", False)
        return RunResult(0, "", "", False)


def make_orch(sandbox, seeds=3, max_attempts=3, repairs=None, write=GOOD_CODE):
    o = Orchestrator.__new__(Orchestrator)
    o.sandbox, o.seeds, o._smoke_ok, o._lock = sandbox, seeds, set(), threading.Lock()
    o.details, o._cost_base, o._snapshots = [], 0.0, {}
    o.llm = SimpleNamespace(total_cost_usd=0.0, gross=lambda x: x * 1.19)
    o.settings = Settings(
        llm=LLMConfig("tokenfactory", "http://x", "k", "m", "m", "m"), sandbox_backend="local",
        tavily_api_key="", max_cost_per_run_usd=0.5, max_attempts_per_claim=max_attempts, llm_cache=False,
    )
    calls = {"write": 0, "repair": []}
    repairs = list(repairs or [])

    def write_code(plan, task):
        calls["write"] += 1
        return write

    def repair_code(plan, task, code, summary):
        calls["repair"].append(summary)
        r = repairs.pop(0) if repairs else GOOD_CODE + "# repaired\n"
        if isinstance(r, Exception):
            raise r
        return r

    o.plan, o.write_code, o.repair_code = (lambda *a, **k: "plan"), write_code, repair_code
    return o, calls


def run_claim(o, tmp_path):
    claim = Claim("c", "GCN 81 %", "accuracy", 0.815, 0.01, "Tabelle", "paper")
    dossier = Dossier(paper_id="p", claims=[claim])
    o.run_claim({"seeds": 3}, claim, cora_task(), "snap", dossier, tmp_path)
    return dossier


def train_cmds(sb):
    return [c for _, c in sb.cmds if "train.py" in c and "SMOKE" not in c]


def test_main_run_never_sees_smoke(tmp_path):
    sb = FakeSandbox()
    d = run_claim(make_orch(sb)[0], tmp_path)
    smoke = [c for _, c in sb.cmds if "SMOKE=1" in c]
    assert len(smoke) == 1 and "SEED=0" in smoke[0]        # Formpruefung einmal je Code
    assert train_cmds(sb) and all("SMOKE" not in c for c in train_cmds(sb))
    assert len(d.attempts) == 3 and {a.seed for a in d.attempts} == {0, 1, 2}


def test_smoke_crash_goes_to_repair_then_succeeds(tmp_path):
    sb = FakeSandbox()
    sb.smoke_results = [RunResult(1, "", "RuntimeError: mat1 and mat2 shapes cannot be multiplied", False)]
    o, calls = make_orch(sb, seeds=1)
    d = run_claim(o, tmp_path)
    assert len(calls["repair"]) == 1 and "mat1 and mat2" in calls["repair"][0]
    assert [a.status for a in d.attempts][0] == "fehler" and d.attempts[-1].status in ("stabil", "abweichung")
    assert d.verdicts[0].urteil != "nicht reproduzierbar (Setup)"


def test_attempt_limit_gives_setup_verdict(tmp_path):
    sb = FakeSandbox(smoke=RunResult(1, "", "always broken", False))
    o, calls = make_orch(sb, seeds=2, max_attempts=3)
    d = run_claim(o, tmp_path)
    assert len(calls["repair"]) == 2 and len(d.attempts) == 3         # max_attempts Versuche, davor Reparaturen
    assert d.verdicts[0].urteil == "nicht reproduzierbar (Setup)"
    assert train_cmds(sb) == []                                      # nie trainiert, Restseeds nicht gestartet


def test_other_seeds_reuse_code_of_seed_zero(tmp_path):
    sb = FakeSandbox()
    o, calls = make_orch(sb, seeds=4)
    d = run_claim(o, tmp_path)
    assert calls["write"] == 1 and calls["repair"] == []
    assert len({a.code_hash for a in d.attempts}) == 1 and len(d.attempts) == 4


def test_billing_block_in_repair_aborts_the_whole_run(tmp_path):
    sb = FakeSandbox(smoke=RunResult(1, "", "broken", False))
    o, calls = make_orch(sb, seeds=3, repairs=[BillingBlocked("Nebius-Abrechnung blockiert")])
    with pytest.raises(BillingBlocked):
        run_claim(o, tmp_path)
    assert len(calls["repair"]) == 1 and train_cmds(sb) == []          # kein zweiter Versuch


def test_static_violation_is_invalid_and_never_executed(tmp_path):
    sb = FakeSandbox()
    o, _ = make_orch(sb, seeds=1, max_attempts=1, write="import requests\nrequests.get('http://x')\n")
    d = run_claim(o, tmp_path)
    assert d.attempts[0].status == "ungueltig" and not [c for _, c in sb.cmds if "train.py" in c]


def test_ablation_unused_graph_sets_method_no(tmp_path):
    sb = FakeSandbox(acc=lambda seed, ablated: 0.81)                  # ohne Kanten gleich gut
    d = run_claim(make_orch(sb, seeds=2)[0], tmp_path)
    assert d.ablations["c"]["verdict"] == "Graph nicht genutzt" and d.verdicts[0].methode == "nein"
    assert d.verdicts[0].gesamtaussage != "Paper reproduziert"


def test_ablation_used_graph_keeps_method_unchecked(tmp_path):
    sb = FakeSandbox(acc=lambda seed, ablated: 0.5 if ablated else 0.81)
    d = run_claim(make_orch(sb, seeds=2)[0], tmp_path)
    assert d.ablations["c"]["verdict"] == "Graph genutzt" and d.verdicts[0].methode == "nicht geprueft"


def test_cost_limit_is_checked_gross():
    o, _ = make_orch(FakeSandbox())
    o.llm = SimpleNamespace(total_cost_usd=0.45, gross=lambda x: x * 1.19)   # netto 0,45 -> brutto 0,5355 >= 0,5
    with pytest.raises(CostLimitExceeded):
        o._ask("code", "s", "u")
