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
        self.puts = getattr(self, "puts", [])
        self.puts.append((branch, path, content))
        # Evaluation now receives only predictions in a clean branch. The fake scorer
        # uses their originating run's metadata instead of executing an actual model.
        if hasattr(content, 'run_metadata'):
            meta = content.run_metadata
            self.seed[branch] = meta['seed']
            for name in ('ablated', 'masked'):
                if meta.get(name):
                    getattr(self, name).add(branch)
            if hasattr(self, 'code_of'):
                self.code_of[branch] = meta.get('code')

    def put_files(self, branch, files):
        pass

    def get_file(self, branch, path):
        class Predictions(bytes):
            pass
        data = Predictions(csv_ids(SMOKE_IDS))
        data.run_metadata = {'seed': self.seed.get(branch, 0), 'code': getattr(self, 'code_of', {}).get(branch),
                             'ablated': branch in self.ablated, 'masked': branch in getattr(self, 'masked', set())}
        return data

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
    o.allow_test_feedback = True  # explicit reproduction of the historical diagnostic policy
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
    assert len(smoke) == 3 and {int(re.search(r'SEED=(\d+)', c).group(1)) for c in smoke} == {0, 1, 2}
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


def test_events_are_written_with_run_facts(tmp_path):
    from repro_agent.events import EventLog

    sb = FakeSandbox(acc=lambda seed, ablated: 0.5 if ablated else 0.81)
    sb.smoke_results = [RunResult(1, "", "shape error", False)]
    o, _ = make_orch(sb, seeds=2)
    o.events = EventLog(tmp_path / "events.jsonl")
    d = run_claim(o, tmp_path)
    ev = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    types = [e["type"] for e in ev]
    assert types[:3] == ["claim_start", "plan", "code_written"] and types[-1] == "verdict"
    assert "repair" in types and types.count("attempt") == len(d.attempts)
    att = [e for e in ev if e["type"] == "attempt"]
    assert att[0]["status"] == "fehler" and att[0]["branch_id"].startswith("b") and att[0]["parent_snapshot"] == "snap"
    assert next(e for e in ev if e["type"] == "verdict")["methode"] == "nicht geprueft"


# ---- Infrastrukturfehler der Sandbox: Versuch auf neuem Branch wiederholen ----
from repro_agent.sandbox.base import SandboxUnavailable


class FlakySandbox(FakeSandbox):
    """Wirft beim n-ten Aufruf von `run` einen Infrastrukturfehler."""

    def __init__(self, fail_calls, **kw):
        super().__init__(**kw)
        self.fail_calls, self.calls = set(fail_calls), 0

    def run(self, branch, cmd, timeout_s):
        self.calls += 1
        if self.calls in self.fail_calls:
            raise SandboxUnavailable("Sandbox-API-Timeout bei run")
        return super().run(branch, cmd, timeout_s)


def test_infra_error_repeats_attempt_on_new_branch_and_gives_no_agent_failure(tmp_path, monkeypatch):
    import repro_agent.loop as loop_mod

    monkeypatch.setattr(loop_mod.time, "sleep", lambda s: None)
    sb = FlakySandbox({1, 2})                      # die ersten beiden Aufrufe scheitern: alter Branch verworfen
    o, calls = make_orch(sb, seeds=1)
    d = run_claim(o, tmp_path)
    assert calls["repair"] == []                   # keine Reparatur: der Agenten-Code war nie schuld
    assert len(d.attempts) == 1 and d.attempts[0].status in ("stabil", "abweichung")
    assert d.attempts[0].branch_id not in ("b1", "b2")   # gewertet wurde ein spaeterer, frischer Branch


def test_infra_error_after_all_retries_aborts_instead_of_judging(tmp_path, monkeypatch):
    import repro_agent.loop as loop_mod

    monkeypatch.setattr(loop_mod.time, "sleep", lambda s: None)
    sb = FlakySandbox(set(range(1, 1000)))
    o, _ = make_orch(sb, seeds=1)
    with pytest.raises(SandboxUnavailable):
        run_claim(o, tmp_path)


def test_nebius_guard_converts_api_timeout():
    from contree_sdk.sdk.exceptions.api import ApiTimeoutError

    from repro_agent.sandbox.nebius import NebiusSandbox

    def boom():
        raise ApiTimeoutError(request=None, response=None, timeout_type="read")

    with pytest.raises(SandboxUnavailable, match="Sandbox-API-Timeout bei run"):
        NebiusSandbox._guard("run", boom)
    assert NebiusSandbox._guard("run", lambda: 42) == 42


def test_infra_retry_reruns_the_same_code_with_the_same_seed(tmp_path, monkeypatch):
    import repro_agent.loop as loop_mod

    monkeypatch.setattr(loop_mod.time, "sleep", lambda s: None)
    sb = FlakySandbox({4})                          # Fehler mitten im ersten Versuch (nach put_file, Vorbereitung, Smoke)
    o, calls = make_orch(sb, seeds=1)
    run_claim(o, tmp_path)
    codes = [c for (_, p, c) in sb.puts if p.endswith("train.py")]
    assert len(codes) >= 2 and len(set(codes)) == 1     # derselbe Code auf beiden Branches
    assert calls["write"] == 1 and calls["repair"] == []  # nichts neu generiert
    branches = {b for (b, p, _) in sb.puts if p.endswith("train.py")}
    assert len(branches) >= 2                           # wirklich ein neuer Branch
    seeds = {re.search(r"SEED=(\d+)", c).group(1) for _, c in sb.cmds if "SEED=" in c}
    assert seeds == {"0"}                               # derselbe Seed
