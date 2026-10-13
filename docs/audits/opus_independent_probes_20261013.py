"""Independent offline probes (Opus analyst review, 2026-10-13). Assertions describe CURRENT behaviour.

Run: uv run python docs/audits/opus_independent_probes_20261013.py
No provider, Tavily, sandbox or network calls (socket.connect is patched to fail). Fakes only where stated; every probe names
which product function it drives unmodified. A green run confirms a mechanism; it is not a product acceptance test.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from repro_agent import spec_check  # noqa: E402
from repro_agent.dossier import Claim, Dossier  # noqa: E402
from repro_agent.harness import ett_data, verify  # noqa: E402
from repro_agent.harness.code_check import find_violations  # noqa: E402
from repro_agent.harness.training_check import checkpoint_issues  # noqa: E402
from repro_agent.harness.val_use_check import static_val_use  # noqa: E402
from repro_agent.loop import select_candidate  # noqa: E402
from repro_agent.method_guard import placeholder_issues  # noqa: E402
from repro_agent.sandbox.base import RunResult  # noqa: E402
from repro_agent.setup_details import Doc  # noqa: E402
from repro_agent.spec import Spec, SpecItem, drop_run_script_defaults, raw_items  # noqa: E402
from repro_agent.tasks import cora_task  # noqa: E402
from test_candidates import C1, C2, ValSandbox, setup  # noqa: E402
from test_val_label_check import LEAK_CAT, MaskSandbox  # noqa: E402


def h(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:12]


def snapshot() -> dict[str, str]:
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files
            if p.startswith(("src/", "web/src/", "tests/", "benchmark/", "deploy/")) or p == "Dockerfile"}


# ---- N03: repo-diff repair replaces the selected code without a fresh validation-label control -----------------------
HONEST = "print('honest c0')\n"
REPO_FILE = "class GCN:\n    def forward(self, x, adj):\n        return adj @ x\n"


def repo_diff_adopts_unchecked_code():
    """Drives Orchestrator.run_claim -> _run_candidates -> _repo_diff_round -> run_rest unmodified (Fake sandbox/LLM only)."""
    with tempfile.TemporaryDirectory() as temp:
        o, sb, _, events = setup({HONEST: 0.80, C1: 0.79, C2: 0.78, LEAK_CAT: 0.80}, {HONEST: 0.80, C1: 0.78, C2: 0.77, LEAK_CAT: 0.95})
        msb = MaskSandbox(sb.test_by_code, sb.val_by_code, {LEAK_CAT: 0.25})
        o.sandbox, o.allow_test_feedback = msb, False
        codes = iter([HONEST, C1, C2])
        o.write_code = lambda plan, task, temperature=0.2: next(codes)
        o._repo_files = {"models/GCN.py": REPO_FILE}
        deviation = [{"aspect": "propagation", "file": "models/GCN.py", "repo_quote": "return adj @ x",
                      "our_quote": "print('honest c0')", "difference": "no propagation"}]
        o._ask = lambda *a, **k: SimpleNamespace(content=json.dumps(deviation), finish_reason="stop")
        o.repair_code = lambda plan, task, code, msg: LEAK_CAT
        claim = Claim("c", "GCN", "accuracy", 0.80, 0.01, "Tabelle", "paper")
        d = Dossier(paper_id="p", claims=[claim])
        o.run_claim({"seeds": 3}, claim, cora_task(), "snap", d, Path(temp))
    (rd,) = [kw for t, kw in events if t == "repo_diff"]
    chosen = sorted({a.code_hash for a in d.attempts if a.chosen})
    checked = {v["code_hash"] for v in d.val_label_check.get("c", {}).values()}
    masked_runs_for_new = [b for b in msb.masked if msb.code_of.get(b) == LEAK_CAT]
    ablation_before_adoption = [kw for t, kw in events if t == "ablation" and kw.get("runde") == "repo_diff"]
    assert rd["result"] == "adopted" and rd["to_hash"] == h(LEAK_CAT)
    assert chosen == [h(LEAK_CAT)] and h(LEAK_CAT) not in checked and not masked_runs_for_new
    assert static_val_use(LEAK_CAT)["verdict"] == "training"
    assert ablation_before_adoption == [] and "Ablation" in rd["reason"]
    return {"repo_diff_result": rd["result"], "reason_text": rd["reason"], "val_old": rd["val_old"], "val_new": rd["val_new"],
            "final_chosen_hashes": chosen, "new_hash_static_val_use": "training",
            "val_label_checked_hashes": sorted(checked), "masked_run_for_new_code": False,
            "ablation_before_adoption": False, "verdict": d.verdicts[0].urteil}


# ---- E13 extension: spec score is taken before _run_seed crash repairs --------------------------------------------------
GOOD = "CONFIG={'lr':0.001}\nrate=CONFIG['lr']\nprint('good')\n"
REPAIRED = "CONFIG={'lr':0.001}\nrate=0.5\nprint('repaired, CONFIG lr never read')\n"
OTHER = "CONFIG={'lr':0.001}\nrate=CONFIG['lr']\nprint('other')\n"


class CrashOnce(ValSandbox):
    def run(self, branch, cmd, timeout_s):
        if "SMOKE=1" in cmd and self.code_of.get(branch) == GOOD:
            return RunResult(1, "", "RuntimeError: shape mismatch", False)
        return super().run(branch, cmd, timeout_s)


def crash_repair_keeps_stale_spec_score():
    spec = Spec(model="Toy", family="graph", items=[SpecItem("lr", "0.001", "q", "paper", "paper", 0.001, enforce=True)])
    with tempfile.TemporaryDirectory() as temp:
        o, sb, _, events = setup({GOOD: 0.8, OTHER: 0.8, REPAIRED: 0.8}, {GOOD: 0.8, OTHER: 0.79, REPAIRED: 0.8}, k=2)
        csb = CrashOnce(sb.test_by_code, sb.val_by_code)
        o.sandbox, o.allow_test_feedback, o.spec_mode, o.spec = csb, False, True, spec
        codes = iter([GOOD, OTHER])
        o.write_code = lambda plan, task, temperature=0.2: next(codes)
        o.repair_code = lambda plan, task, code, msg: REPAIRED
        claim = Claim("c", "Toy", "accuracy", 0.80, 0.01, "Tabelle", "paper")
        d = Dossier(paper_id="p", claims=[claim])
        o.run_claim({"seeds": 3}, claim, cora_task(), "snap", d, Path(temp))
    (sel,) = [kw for t, kw in events if t == "candidates"]
    spec_events = [kw for t, kw in events if t == "spec_check"]
    _, final_report = spec_check.config_check(REPAIRED, spec)
    assert sel["chosen_hash"] == h(REPAIRED) and sel["reason"].startswith("spec (1)")
    assert spec_check.score(spec, final_report, []) == (0, 1) and final_report["unused"] == ["lr"]
    assert not any(e["code_hash"] == h(REPAIRED) for e in spec_events)
    return {"selected_hash": sel["chosen_hash"], "selection_reason": sel["reason"], "scored_hashes": sorted({e["code_hash"] for e in spec_events}),
            "selected_code_actual_score": [0, 1], "repair_trigger": "smoke crash in _run_seed (not quality loop)"}


# ---- E03 realistic variant: clamping all visible labels (label propagation style) -------------------------------------
CLAMP = """
import numpy as np, torch
d = np.load('/data/cora_public.npz')
labels = torch.from_numpy(d['labels'])
logits = model(X)
known = labels >= 0
pred = logits.argmax(1)
pred[known] = labels[known]
"""


def clamped_visible_labels_win_selection():
    with tempfile.TemporaryDirectory() as temp:
        o, sb, _, events = setup({CLAMP: 0.80, C1: 0.81, C2: 0.80}, {CLAMP: 1.0, C1: 0.80, C2: 0.79})
        msb = MaskSandbox(sb.test_by_code, sb.val_by_code, {CLAMP: 0.86})
        o.sandbox, o.allow_test_feedback = msb, False
        codes = iter([CLAMP, C1, C2])
        o.write_code = lambda plan, task, temperature=0.2: next(codes)
        claim = Claim("c", "GCN", "accuracy", 0.80, 0.01, "Tabelle", "paper")
        d = Dossier(paper_id="p", claims=[claim])
        o.run_claim({"seeds": 3}, claim, cora_task(), "snap", d, Path(temp))
    (sel,) = [kw for t, kw in events if t == "candidates"]
    control = d.val_label_check["c"]["0"]
    static = static_val_use(CLAMP)["verdict"]
    assert static != "training"   # observed: "none" (the mask over all visible labels is no sink without backward/fit)
    assert sel["chosen"] == 0 and control["verdict"] == "Validierungslabels genutzt" and control["excluded"] is False
    return {"static_verdict": static, "val_full": control["val_full"], "val_masked": control["val_masked"], "chosen": 0}


# ---- E04 precise form: same validation windows used for fitting AND for candidate selection ---------------------------
def ett_trainval_fit_wins_selection():
    """Uses ett_data.load_standardized/val_targets/targets (product harness code) on a synthetic series; select_candidate unmodified."""
    rng = np.random.default_rng(3)
    n, ch = 14400, 7
    x = np.zeros((n, ch))
    for t in range(1, n):
        x[t] = 0.97 * x[t - 1] + rng.normal(size=ch)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "toy.csv"
        path.write_text("date," + ",".join(f"c{i}" for i in range(ch)) + "\n" + "\n".join("d," + ",".join(repr(float(v)) for v in r) for r in x) + "\n")
        z = ett_data.load_standardized("etth1", path)
        val_truth, test_truth = ett_data.val_targets("etth1", path), ett_data.targets("etth1", path)
    L, H, train_end, val_end = 336, 96, 8640, 11520
    stride = ett_data.val_stride("etth1")

    def windows(lo, hi, step=1):
        starts = np.arange(lo + L, hi - H + 1, step)
        X = np.stack([z[s - L:s].T for s in starts]).reshape(-1, L)
        Y = np.stack([z[s:s + H].T for s in starts]).reshape(-1, H)
        return X, Y

    def fit(X, Y, lam=1e-3):
        return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)

    def predict(W, starts):
        return np.stack([(z[s - L:s].T @ W).T for s in starts])

    val_starts = train_end + stride * np.arange(ett_data.n_val_windows("etth1"))
    test_starts = val_end + np.arange(ett_data.n_windows("etth1"))
    Xa, Ya = windows(0, train_end, 4)
    Xb, Yb = windows(0, val_end, 4)
    Wa, Wb = fit(Xa, Ya), fit(Xb, Yb)
    res = {}
    for name, W in (("train_only", Wa), ("train_plus_val", Wb)):
        res[name] = {"val_mse": float(((predict(W, val_starts) - val_truth) ** 2).mean()),
                     "test_mse": float(((predict(W, test_starts) - test_truth) ** 2).mean())}
    pick, why = select_candidate([{"metric": "mse", "value": res[k]["val_mse"]} for k in ("train_only", "train_plus_val")])
    val_gain = res["train_only"]["val_mse"] - res["train_plus_val"]["val_mse"]
    test_gain = res["train_only"]["test_mse"] - res["train_plus_val"]["test_mse"]
    assert pick == 1 and why == "val" and val_gain > test_gain
    return {**res, "selected": "train_plus_val", "val_advantage": val_gain, "test_advantage": test_gain,
            "note": "val advantage exceeds test advantage: validation is no longer an independent selection signal"}


# ---- E08/E09 on the real SARAF run-script lines (commit 7b49aa8, copied verbatim) ------------------------------------
REAL_ARG = ("    parser.add_argument('--time_aware_weight', type=float, default=0.5, \n"
            "                        help='time-aware bonus weight for temporal retrieval (0.0 = pure temporal, 1.0 = pure time-aware)')\n")
REAL_SH = """for seed in 2021 0 42
do
    python -u run.py \\
      --model_id ETTh1_720_96 \\
      --data ETTh1 \\
      --seq_len 720 \\
      --pred_len 96 \\
      --learning_rate 0.001 \\
      --topm 20

    python -u run.py \\
      --model_id ETTh1_720_336 \\
      --data ETTh1 \\
      --pred_len 336 \\
      --learning_rate 0.0001 \\
      --topm 20

    python -u run.py \\
      --model_id ETTm1_720_96 \\
      --data ETTm1 \\
      --pred_len 96 \\
      --time_aware_weight 0.1 \\
      --topm 20
done
"""


def real_script_defaults_and_rates():
    excerpt = f"=== repo:run.py@7b49aa8 ===\n{REAL_ARG}\n=== repo:scripts/ETTh_720.sh@7b49aa8 ===\n{REAL_SH}"
    changed, dropped = drop_run_script_defaults(excerpt)
    sh = Doc("repo:scripts/ETTh_720.sh@7b49aa8", REAL_SH)
    items = raw_items([sh], [sh.name], "SARAF", "forecast")
    assert "time_aware_weight" in dropped and "default=0.5" not in changed
    assert not any(i.key == "lr" for i in items)
    return {"dropped_defaults": dropped, "etth1_96_call_sets_time_aware_weight": False,
            "lr_in_raw_items": False, "etth1_96_lr_in_script": "0.001",
            "stored_live_event": "spec.dropped_defaults of job 175724ef contains time_aware_weight and learning_rate"}


# ---- E11 variants not covered by the first audit ---------------------------------------------------------------------
def config_variants_undetected():
    spec = Spec(model="Toy", family="forecast", items=[SpecItem("lr", "0.001", "q", "paper", "paper", 0.001, enforce=True)])
    cases = {"augassign": "CONFIG={'lr':0.001}\nCONFIG['lr'] *= 1000\nopt=Adam(lr=CONFIG['lr'])\n",
             "dict_merge": "CONFIG={'lr':0.001}\nCONFIG={**CONFIG,'lr':1.0}\nopt=Adam(lr=CONFIG['lr'])\n",
             "setitem_call": "CONFIG={'lr':0.001}\nCONFIG.__setitem__('lr', 1.0)\nopt=Adam(lr=CONFIG['lr'])\n"}
    out = {}
    for name, code in cases.items():
        new, report = spec_check.config_check(code, spec)
        env = {"Adam": lambda **kw: kw}
        exec(compile(new, name, "exec"), env)
        assert not report["conflicts"] and env["opt"]["lr"] != 0.001
        out[name] = {"score": spec_check.score(spec, report, []), "effective_lr": env["opt"]["lr"]}
    return out


# ---- E17/E18/E19 own variants -----------------------------------------------------------------------------------------
def guard_variants():
    ref = {"layers/r.py": "class Retrieval:\n    def retrieve(self, x):\n        return x.mean(0)\n"}
    placeholder = "class Retrieval:\n    def retrieve(self, x):\n        return 0 * x\n"
    shallow = ("for epoch in range(3):\n    optimizer.step()\n    best = {k: v for k, v in model.state_dict().items()}\n"
               "model.load_state_dict(best)\n")
    harmless = '"""Unlike socket-based loaders we read a local npz."""\nimport numpy as np\n'
    assert placeholder_issues(placeholder, ref) == []
    assert checkpoint_issues(shallow) == []
    assert find_violations(harmless)
    return {"zero_times_input_placeholder_detected": False, "dict_comprehension_alias_detected": False,
            "docstring_word_rejected": find_violations(harmless)}


# ---- E14 order: first or last duplicate wins ---------------------------------------------------------------------------
def duplicate_last_wins():
    with tempfile.TemporaryDirectory() as temp:
        p = Path(temp) / "p.csv"
        p.write_text("id,prediction\n5,1\n5,2\n6,0\n")
        preds = verify.load_predictions(str(p))
    assert preds == {"5": "2", "6": "0"} and verify.check_plausibility(preds, {"5", "6"}) is None
    return {"accepted": preds, "rows": 3}


PROBES = [repo_diff_adopts_unchecked_code, crash_repair_keeps_stale_spec_score, clamped_visible_labels_win_selection,
          ett_trainval_fit_wins_selection, real_script_defaults_and_rates, config_variants_undetected, guard_variants,
          duplicate_last_wins]


if __name__ == "__main__":
    before = snapshot()
    results = []
    with patch("socket.socket.connect", side_effect=RuntimeError("network disabled during audit")):
        for fn in PROBES:
            results.append({"probe": fn.__name__, "confirmed": True, "observed": fn()})
            print(fn.__name__ + ": confirmed", flush=True)
    assert before == snapshot(), "product files changed during audit"
    out = {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "network_disabled": True, "product_files_unchanged": True, "results": results}
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n")
    print("Evidence:", path)
