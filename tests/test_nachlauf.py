"""Modus Nachlauf M1: P1 (check_command), P2 (independence), P3 (self_check) und der Ablauf mit Fake-Sandbox. Kein Sandbox-, LLM- oder Tavily-Aufruf."""
import dataclasses
import io
import json
import re

import numpy as np
import pytest

from repro_agent import nachlauf as nl
from repro_agent.dossier import Dossier
from repro_agent.harness import independence as ind
from repro_agent.sandbox.base import RunResult

# ---- Beispieldateien -------------------------------------------------------------------------------------------------------

SCRIPT_MD = """<!-- ETTh1 -->

python -u run.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --channel 7 --itr 1 --batch_size 128 --learning_rate 0.005 --model Linear --rev --seq_len 336 --pred_len 96 --gpu 3

<!-- ETTm1 -->

python -u run.py --is_training 1 --data_path ETTm1.csv --data ETTm1 --features M --channel 7 --itr 1 --batch_size 128 --learning_rate 0.005 --model Linear --rev --seq_len 336 --pred_len 96 --gpu 3
"""
RUN_PY = """fix_seed = 1024
import argparse
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--use_all', action='store_true', help='train on the entire dataset incl. test')
parser.add_argument('--drop', type=float, default=0.1, help='dropout ratio')
args = parser.parse_args()
"""
ETTH1_SH = """
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
seq_len=336
model_name=DLinear

python -u run_longExp.py \\
  --is_training 1 \\
  --root_path ./dataset/ \\
  --data_path ETTh1.csv \\
  --model_id ETTh1_$seq_len'_'96 \\
  --model $model_name \\
  --data ETTh1 \\
  --features M \\
  --seq_len $seq_len \\
  --pred_len 96 \\
  --enc_in 7 \\
  --des 'Exp' \\
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'96.log

python -u run_longExp.py \\
  --is_training 1 \\
  --root_path ./dataset/ \\
  --data_path ETTh1.csv \\
  --model_id ETTh1_$seq_len'_'192 \\
  --model $model_name \\
  --data ETTh1 \\
  --features M \\
  --seq_len $seq_len \\
  --pred_len 192 \\
  --enc_in 7 \\
  --des 'Exp' \\
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'192.log
"""
RTSF_CMD = ("python -u run.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --channel 7 --itr 1 --batch_size 128 "
            "--learning_rate 0.005 --model Linear --rev --seq_len 336 --pred_len 96 --gpu 3")
LTSF_CMD = ("python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_96 --model DLinear "
            "--data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --des Exp --itr 1 --batch_size 32 --learning_rate 0.005")
FILES = {"script.md": SCRIPT_MD, "run.py": RUN_PY, "scripts/etth1.sh": ETTH1_SH}


# ---- P1 ---------------------------------------------------------------------------------------------------------------------

def test_p1_verbatim_documented_with_allowed_deviations_silent():
    r = nl.check_command(RTSF_CMD, FILES, {"file": "script.md", "line": 3})
    assert r["documented"] and r["diff"] == [] and r["source"] == {"file": "script.md", "line": 3} and r["valid"]
    other = RTSF_CMD.replace("--gpu 3", "--gpu 0 --use_gpu False").replace("--data_path ETTh1.csv", "--data_path /x/ETTh1.csv")
    other += " > run.log 2>&1 &"
    assert nl.check_command(other, FILES, {"file": "script.md", "line": 3})["documented"]


def test_p1_resolves_shell_variables_and_continuation_lines():
    r = nl.check_command(LTSF_CMD, FILES, {"file": "scripts/etth1.sh", "line": 7})
    assert r["documented"] and r["diff"] == [] and r["source"]["line"] == 8   # Startzeile der 96er Zeile
    r192 = nl.check_command(LTSF_CMD.replace("--pred_len 96", "--pred_len 192").replace("ETTh1_336_96", "ETTh1_336_192"),
                            FILES, {"file": "scripts/etth1.sh"})
    assert r192["documented"] and r192["source"]["line"] > r["source"]["line"]


def test_p1_corrected_model_name_is_listed_not_rejected():
    r = nl.check_command(RTSF_CMD.replace("--model Linear", "--model RLinear"), FILES, {"file": "script.md", "line": 3})
    assert not r["documented"] and r["valid"]
    assert r["diff"] == [{"kind": "changed", "flag": "--model", "doc": "Linear", "command": "RLinear"}]


def test_p1_other_deviations_are_listed():
    r = nl.check_command(LTSF_CMD.replace("0.005", "0.01") + " --extra 1", FILES, {"file": "scripts/etth1.sh"})
    kinds = {(d["kind"], d["flag"]) for d in r["diff"]}
    assert ("changed", "--learning_rate") in kinds and ("added", "--extra") in kinds and r["valid"]


def test_p1_train_only_and_help_text_flags_are_rejected():
    r = nl.check_command(LTSF_CMD + " --train_only True", FILES, {"file": "scripts/etth1.sh"})
    assert not r["valid"] and any("--train_only" in h for h in r["forbidden"])
    r2 = nl.check_command(RTSF_CMD + " --use_all", {**FILES}, {"file": "script.md", "line": 3})
    assert not r2["valid"] and any("--use_all" in h and "entire" in h for h in r2["forbidden"])
    # ein Schalter, der nicht im Befehl steht, loest nichts aus
    assert nl.forbidden_flags(RTSF_CMD, FILES) == []


def test_p1_command_not_in_file():
    r = nl.check_command("python other.py --x 1", FILES, {"file": "script.md", "line": 3})
    assert not r["documented"] and r["diff"][0]["kind"] == "missing"


# ---- P3 ---------------------------------------------------------------------------------------------------------------------

DROPLAST = {"128": {"mse": 0.36596}, "32": {"mse": 0.3841}}


def test_p3_three_cases_with_and_without_drop_last():
    tl = {"drop_last": True, "batch": 128, "source": "data_factory.py:18"}
    ok = nl.self_check(0.3660, {"mse": 0.3698}, DROPLAST, tl)
    assert ok["status"] == "geprueft" and ok["compared_with"] == "drop_last_128" and ok["distance"] < 0.001
    bad = nl.self_check(0.3698 + 0.002, {"mse": 0.3698}, DROPLAST, tl)   # all-windows value is NOT the comparison basis with drop_last
    assert bad["status"] == "ungueltig" and "do not measure the same" in bad["reason"]
    assert nl.self_check(None, {"mse": 0.37}, DROPLAST, tl)["status"] == "nicht geprueft"
    nodrop = {"drop_last": False, "batch": 32, "source": "data_factory.py:20"}
    ok2 = nl.self_check(0.3841, {"mse": 0.38412}, DROPLAST, nodrop)
    assert ok2["status"] == "geprueft" and ok2["compared_with"] == "all_windows"
    assert nl.self_check(0.40, {"mse": 0.38412}, DROPLAST, nodrop)["status"] == "ungueltig"
    assert nl.self_check(None, {"mse": 0.38}, None, nodrop)["status"] == "nicht geprueft"
    assert nl.self_check(0.37, {"mse": 0.37}, DROPLAST, {"drop_last": True, "batch": 77})["status"] == "nicht geprueft"


# ---- Umgebung, Seed-Kopie, Dossier ------------------------------------------------------------------------------------------

def test_env_diff():
    d = nl.env_diff("torch==1.9.0\nnumpy\npandas>=1.0  # c\nscikit_learn==1.0\n", "torch==2.14.1+cpu\nnumpy==1.26.4\nscikit-learn==1.0\n")
    assert {x["package"]: x for x in d} == {
        "torch": {"package": "torch", "required": "==1.9.0", "installed": "2.14.1+cpu"},
        "pandas": {"package": "pandas", "required": ">=1.0", "installed": None},
    }


def test_seed_copy_changes_exactly_one_line():
    out = nl.make_seed_copy(RUN_PY, r"fix_seed = 1024", "fix_seed = {seed}", 3)
    assert out.splitlines()[0] == "fix_seed = 3" and out.splitlines()[1:] == RUN_PY.splitlines()[1:]
    with pytest.raises(ValueError):
        nl.make_seed_copy(RUN_PY, r"fix_seed = 7", "fix_seed = {seed}", 3)
    with pytest.raises(ValueError):
        nl.make_seed_copy("fix_seed = 1\nfix_seed = 1\n", r"fix_seed = 1", "fix_seed = {seed}", 3)


def test_old_dossier_without_nachlauf_field_loads(tmp_path):
    d = Dossier(paper_id="p")
    data = dataclasses.asdict(d)
    data.pop("nachlauf")
    (tmp_path / "dossier.json").write_text(json.dumps(data))
    loaded = Dossier.load(tmp_path / "dossier.json")
    assert loaded.nachlauf == {} and loaded.mode == "nachbau"
    d.nachlauf = {"c": {"command": "x"}}
    d.mode = "nachlauf"
    (tmp_path / "d2.json").write_text(json.dumps(dataclasses.asdict(d)))
    assert Dossier.load(tmp_path / "d2.json").nachlauf == {"c": {"command": "x"}}


# ---- P2 ---------------------------------------------------------------------------------------------------------------------

def public(n=900, steps=16, seed=0):
    rng = np.random.default_rng(seed)
    return {"train": rng.normal(size=(20, 7)).astype(np.float32), "val": rng.normal(size=(10, 7)).astype(np.float32),
            "test_inputs": rng.normal(size=(n, steps, 7)).astype(np.float32)}


def honest(x, scale=0.9):
    return np.repeat(x[:, -1:, :], 96, axis=1) * scale


def cheater(x, scale=0.9):
    """Schreibt Ziele aus Nachbarfenstern ab: Schritt s von Fenster i = letzter Wert von Fenster i+s+1 (dort steht das Ziel)."""
    n = x.shape[0]
    idx = np.minimum(np.arange(n)[:, None] + np.arange(96)[None, :] + 1, n - 1)
    return x[idx, -1, :]


def test_offsets_never_overlap_property():
    for n in (900, 2785):
        for r in range(ind.STRIDE):
            idx = ind.window_indices(r, n)
            assert all(b - a >= 816 for a, b in zip(idx, idx[1:]))
            assert all(0 <= i < n for i in idx)
    assert len(set(ind.random_offsets(8))) == 8
    assert all(0 <= r < 816 for r in ind.random_offsets(8))


def test_make_probe_shape_mapping_and_no_mapping_in_arrays():
    pub = public()
    arrays, mapping = ind.make(pub, 5, np.random.default_rng(1))
    assert set(arrays) == {"train", "val", "test_inputs"}            # keine Zuordnung in der Datei
    assert np.array_equal(arrays["train"], pub["train"]) and np.array_equal(arrays["val"], pub["val"])
    assert arrays["test_inputs"].shape[0] == len(mapping) == 4 and mapping.count(None) == 2
    for pos, w in enumerate(mapping):
        if w is not None:
            assert np.array_equal(arrays["test_inputs"][pos], pub["test_inputs"][w])
    back = ind.load_npz_bytes(ind.to_npz_bytes(arrays))
    assert set(back) == {"train", "val", "test_inputs"}


def run_offsets(adapter, offsets=(5, 100, 500, 800)):
    pub = public()
    full = adapter(pub["test_inputs"])
    results = []
    for r in offsets:
        arrays, mapping = ind.make(pub, r, np.random.default_rng(r))
        results.append({"offset": r, **ind.compare(full, adapter(arrays["test_inputs"]), mapping)})
    return ind.summarize(results)


def test_honest_adapter_passes_cheater_fails():
    assert run_offsets(honest)["ok"]
    bad = run_offsets(cheater)
    assert not bad["ok"] and bad["reason"] == ind.REASON


def test_noise_windows_are_not_compared_and_nan_fails():
    pub = public()
    full = honest(pub["test_inputs"])
    arrays, mapping = ind.make(pub, 5, np.random.default_rng(2))
    probe = honest(arrays["test_inputs"])
    noise_pos = [p for p, w in enumerate(mapping) if w is None]
    probe[noise_pos] = 123.0                                         # Rauschfenster beliebig: egal
    assert ind.compare(full, probe, mapping)["ok"]
    real_pos = next(p for p, w in enumerate(mapping) if w is not None)
    probe[real_pos, 0, 0] += 1e-3
    assert not ind.compare(full, probe, mapping)["ok"]
    probe[real_pos, 0, 0] = np.nan
    assert ind.compare(full, probe, mapping)["reason"] == "non-finite predictions on real windows"


def test_parse_predictions_roundtrip():
    pred = np.random.default_rng(0).normal(size=(3, 96, 7))
    assert np.allclose(ind.parse_predictions(csv_bytes(pred)), pred, atol=1e-6)


# ---- Fake-Sandbox -----------------------------------------------------------------------------------------------------------

def csv_bytes(pred) -> bytes:
    n = pred.shape[0]
    w, s, c = np.meshgrid(np.arange(n), np.arange(96), np.arange(7), indexing="ij")
    rows = np.column_stack([w.ravel(), s.ravel(), c.ravel(), pred.ravel()])
    buf = io.StringIO()
    np.savetxt(buf, rows, fmt="%d_%d_%d,%.8g", header="id,prediction", comments="")
    return buf.getvalue().encode()


EVAL = {"status": "ok", "metrics": {"mse": 0.3698, "mae": 0.39}, "baselines": {"repeat": {"mse": 1.29}},
        "metrics_droplast": {"128": {"mse": 0.36596, "mae": 0.39}}, "diagnostics": {"corr": 0.8}}


class FakeSandbox:
    """Zustand je Branch (Dateien); protokolliert alle Aufrufe in `ops` (Reihenfolge)."""

    def __init__(self, pub, adapter=honest, git_status="", own="mse:0.3660, mae:0.39", raw_in_predict=False, eval_out=EVAL):
        self.pub, self.adapter, self.git_status, self.own, self.eval_out = pub, adapter, git_status, own, eval_out
        self.raw_in_predict = raw_in_predict
        self.ops, self.snaps, self.branches, self.kind, self.n = [], {}, {}, {}, 0
        self._lock = __import__("threading").Lock()   # P2 laeuft parallel

    def base_snapshot(self, image, setup_cmds, label=None):
        self.ops.append(("snapshot", label, list(setup_cmds)))
        kind = "predict" if label.endswith("predict") else "train"
        files = {"/data/etth1_public.npz": ind.to_npz_bytes(self.pub)} if kind == "predict" else {
            "/work/repo/script.md": SCRIPT_MD.encode(), "/work/repo/run.py": RUN_PY.encode(), "/work/repo/dataset/ETTh1.csv": b"raw"}
        if kind == "predict" and self.raw_in_predict:
            files["/tmp/ETTh1.csv"] = b"raw"
        self.snaps[label] = (kind, files)
        return label

    def branch(self, snap):
        kind, files = self.snaps[snap]
        with self._lock:
            self.n += 1
            b = f"b{self.n}"
            self.branches[b] = dict(files)
            self.kind[b] = kind
            self.ops.append(("branch", b, kind))
        return b

    def put_files(self, b, files):
        self.ops.append(("put_files", b, sorted(files)))
        self.branches[b].update(files)

    def get_file(self, b, path):
        self.ops.append(("get_file", b, path))
        return self.branches[b][path]

    def run(self, b, cmd, timeout_s):
        self.ops.append(("run", b, cmd, timeout_s))
        f, kind = self.branches[b], self.kind[b]
        ok = lambda out="": RunResult(0, out, "", False)  # noqa: E731
        if cmd.startswith("cat /work/repo/"):
            p = cmd.split()[1]
            return ok(f[p].decode()) if p in f else RunResult(1, "", "no such file", False)
        if cmd == "pip freeze":
            return ok("torch==2.14.1+cpu\nnumpy==1.26.4\n")
        if "rev-parse HEAD" in cmd and "unshare" not in cmd:
            return ok("COMMIT\n")
        if cmd.startswith("find / -xdev -name 'ETTh1*'"):
            return ok("".join(p + "\n" for p in f if p.rsplit("/", 1)[-1].startswith("ETTh1")) + "__FIND_DONE__\n")
        if "grep -c" in cmd:
            return ok("2\n")
        if "git status --short" in cmd:
            return ok(f"__GIT_STATUS__[{self.git_status}]\n__GIT_DIFFSTAT__[]\n")
        if cmd.startswith("ls -1"):
            return ok("/work/repo/checkpoints/x/checkpoint.pth\n") if "/work/repo/checkpoints/x/checkpoint.pth" in f else RunResult(2, "", "", False)
        if "/opt/nachlauf/predict.py" in cmd:
            assert kind == "predict", "Adapter darf nur im Vorhersage-Branch laufen"
            assert "unshare -n" in cmd and not [p for p in f if p.rsplit("/", 1)[-1].startswith("ETTh1")]
            x = ind.load_npz_bytes(f["/data/etth1_public.npz"])["test_inputs"]
            if "SMOKE=1" in cmd:
                x = x[:8]
            f["/output/predictions.csv"] = csv_bytes(self.adapter(x, float(f[nl.CKPT_PATH].decode().split("=")[1])))
            return ok("__WALL_MS__ 2000\n")
        if "unshare -n" in cmd and "run.py" in cmd:
            assert kind == "train" and "/opt/nachlauf/predict.py" not in f
            f["/work/repo/checkpoints/x/checkpoint.pth"] = b"scale=0.9"
            return ok(self.own + "\n__WALL_MS__ 25000\n")
        if "etth1_public.npz')['test_inputs'].shape[0]" in cmd:
            m = ind.load_npz_bytes(f["/data/etth1_public.npz"])["test_inputs"].shape[0]
            return ok(f"{m}\n1\n")
        if cmd.startswith("python /opt/harness-") and cmd.endswith("etth1_eval.py"):
            assert any(p.endswith("/etth1_eval.py") for p in f), "Harness muss vorher kopiert sein"
            assert "/output/predictions.csv" in f
            return ok(json.dumps(self.eval_out) + "\n")
        return ok()


def spec(**kw):
    base = dict(
        unit="rlinear", paper_id="nachlauf_manual/rlinear", claim_text="RLinear 0.366", target=0.366, tolerance=0.01,
        repo_url="https://example.invalid/RTSF.git", commit="COMMIT", command=RTSF_CMD.replace("--model Linear", "--model RLinear"),
        command_source={"file": "script.md", "line": 3}, script="run.py",
        seed={"kind": "copy", "repo_seed": 1024, "regex": r"fix_seed = 1024", "template": "fix_seed = {seed}"},
        adapter_source="import numpy as np\n# honest fake adapter source\n", test_loader={"drop_last": True, "batch": 128, "source": "data_factory.py:18"},
        argparse_files=["run.py"], requirements_files=[],
    )
    base.update(kw)
    return nl.UnitSpec(**base)


def go(sb, seeds=(1024,), offsets=(5, 100, 500, 800), out_dir=None, sp=None):
    return nl.run_unit(sb, sp or spec(), list(seeds), offsets=list(offsets), out_dir=out_dir,
                       public_loader=lambda: sb.pub, np_rng=np.random.default_rng(0), log=lambda *_: None)


def pos(sb, pred):
    return next(i for i, o in enumerate(sb.ops) if pred(o))


def test_flow_order_and_isolation(tmp_path):
    sb = FakeSandbox(public())
    d = go(sb, seeds=(1024, 0), out_dir=tmp_path)
    assert d.mode == "nachlauf" and [a.status for a in d.attempts] == ["stabil", "stabil"]
    assert d.verdicts[0].urteil == "reproduziert" and d.verdicts[0].measured_mean == pytest.approx(0.3698)
    train_branches = [o[1] for o in sb.ops if o[0] == "branch" and o[2] == "train"]
    # Trainings-Branches: kein Adapter, kein Harness
    for o in sb.ops:
        if o[0] == "put_files" and o[1] in train_branches:
            assert not any(nl.ADAPTER_DIR in p or "harness" in p for p in o[2]), o
    # Vorhersage-Snapshot ohne Rohdaten-Kommando; mit Rohdaten-Pruefung
    pred_cmds = next(o[2] for o in sb.ops if o[0] == "snapshot" and o[1].endswith("predict"))
    assert not any(f"{nl.REPO_DIR}/dataset" in c or "ETTh1.csv" in c and "urlretrieve" in c and "/work/" in c for c in pred_cmds)
    assert nl.NO_RAW_CHECK in pred_cmds
    # Harness erst nach dem Adapter-Lauf, im selben Branch, unveraendertes etth1_eval.py ohne unshare
    full = next(o[1] for o in sb.ops if o[0] == "run" and "NACHLAUF_CKPT" in o[2] and "SMOKE" not in o[2]
                and o[1] not in [x[1] for x in sb.ops if x[0] == "run" and "etth1_public.npz')['test_inputs']" in x[2]])
    p_adapter = pos(sb, lambda o: o[0] == "run" and o[1] == full and "predict.py" in o[2])
    p_harness = pos(sb, lambda o: o[0] == "put_files" and o[1] == full and any(p.endswith("etth1_eval.py") for p in o[2]))
    p_eval = pos(sb, lambda o: o[0] == "run" and o[1] == full and o[2].endswith("etth1_eval.py"))
    assert p_adapter < p_harness < p_eval
    assert "unshare" not in sb.ops[p_eval][2]
    # alle Adapter-Laeufe unter unshare -n, Training mit 600 s Timeout
    assert all("unshare -n" in o[2] for o in sb.ops if o[0] == "run" and "predict.py" in o[2])
    assert all(o[3] == 600 for o in sb.ops if o[0] == "run" and "run.py" in o[2] and "unshare" in o[2])
    # Seed-Kopie nur fuer Seed 0 (nicht Repo-Seed), nur im Trainings-Branch
    assert sum(1 for o in sb.ops if o[0] == "put_files" and any(p == "/opt/seed/run.py" for p in o[2])) == 1
    # Smoke in eigenem Branch vor dem vollen Lauf
    smoke_b = next(o[1] for o in sb.ops if o[0] == "run" and "SMOKE=1" in o[2])
    assert smoke_b != full
    # Adapter hat im Vorhersage-Branch nie Rohdaten gesehen (assert im Fake); Proben-Zuordnung steht in keiner Datei
    for o in sb.ops:
        if o[0] == "put_files":
            assert all(p.startswith(("/opt/nachlauf", "/opt/seed", "/opt/harness", "/data/etth1_public.npz")) for p in o[2]), o


def test_dossier_fields_and_events(tmp_path):
    sb = FakeSandbox(public())
    d = go(sb, seeds=(1024, 0), out_dir=tmp_path)
    info = d.nachlauf["rlinear_etth1_96_mse"]
    assert info["commit"] == "COMMIT" and info["command_documented"] is False
    assert info["command_diff"] == [{"kind": "changed", "flag": "--model", "doc": "Linear", "command": "RLinear"}]
    assert info["own_metric"] == {"1024": 0.366, "0": 0.366}
    assert info["self_check"]["1024"]["status"] == "geprueft" and info["self_check"]["1024"]["compared_with"] == "drop_last_128"
    assert info["independence"]["0"]["ok"] and info["independence"]["0"]["n_offsets"] == 4
    assert info["predict_snapshot_raw_files"] == [] and info["test_loader"]["batch"] == 128
    assert d.repo == {"url": "https://example.invalid/RTSF.git", "commit": "COMMIT"}
    assert any("Startbefehl weicht von der Dokumentation ab" in w for w in d.warnings)
    assert any("nachlauf" in s.lower() for s in d.isolation) and len(d.isolation) == len(nl.NACHLAUF_ISOLATION)
    assert d.attempts[0].metrics_droplast["128"]["mse"] == 0.36596
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert {e["type"] for e in events} == {"nachlauf_train", "nachlauf_predict", "nachlauf_independence", "nachlauf_self_check"}
    dataclasses.asdict(d)   # serialisierbar


def test_cheating_adapter_fails_p2_in_the_flow():
    sb = FakeSandbox(public(), adapter=cheater)
    d = go(sb)
    assert d.attempts[0].status == "ungueltig" and "P2" in d.attempts[0].error_summary
    assert d.verdicts[0].urteil == "nicht reproduzierbar (Setup)"
    assert d.nachlauf["rlinear_etth1_96_mse"]["independence"]["1024"]["reason"] == ind.REASON


def test_raw_data_in_predict_snapshot_aborts():
    sb = FakeSandbox(public(), raw_in_predict=True)
    d = go(sb)
    assert d.attempts[0].status == "ungueltig" and not any(o[0] == "run" and "run.py" in o[2] for o in sb.ops)
    assert d.nachlauf["rlinear_etth1_96_mse"]["predict_snapshot_raw_files"]


def test_forbidden_flag_blocks_before_any_training():
    sb = FakeSandbox(public())
    d = go(sb, sp=spec(command=spec().command + " --train_only True"))
    assert d.attempts[0].status == "ungueltig" and "--train_only" in d.attempts[0].error_summary
    assert not any(o[0] == "run" and "unshare" in o[2] for o in sb.ops)


def test_changed_repo_files_make_attempt_invalid_and_stop_before_predict():
    sb = FakeSandbox(public(), git_status=" M run.py")
    d = go(sb)
    assert d.attempts[0].status == "ungueltig" and "repo files changed" in d.attempts[0].error_summary
    assert not any(o[0] == "run" and "predict.py" in o[2] for o in sb.ops)


def test_p3_mismatch_invalidates_and_missing_own_metric_warns():
    d = go(FakeSandbox(public(), own="mse:0.4100, mae:0.4"))
    assert d.attempts[0].status == "ungueltig" and d.verdicts[0].urteil == "nicht reproduzierbar (Setup)"
    d2 = go(FakeSandbox(public(), own="nothing printed"))
    assert d2.attempts[0].status == "stabil" and d2.nachlauf["rlinear_etth1_96_mse"]["own_metric"]["1024"] is None
    assert any("Zuordnung nicht geprueft" in w for w in d2.warnings)


def test_static_adapter_check_rejects_before_any_snapshot():
    sb = FakeSandbox(public())
    d = go(sb, sp=spec(adapter_source="import subprocess\n"))
    assert d.attempts[0].status == "ungueltig" and not sb.ops


def test_without_seed_method_only_repo_seed_runs():
    sb = FakeSandbox(public())
    d = go(sb, seeds=(1024, 0, 1), sp=spec(seed={"kind": "none", "repo_seed": 1024}))
    assert [a.seed for a in d.attempts] == [1024] and "eine Messung, keine Streuung" in d.warnings


def test_hand_specs_pass_static_check_and_p1():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_nachlauf_manual.py"
    mod = importlib.util.spec_from_file_location("run_nachlauf_manual", path)
    m = importlib.util.module_from_spec(mod)
    mod.loader.exec_module(m)
    from repro_agent.harness.code_check import find_violations
    for name, make in m.SPECS.items():
        s = make()
        assert find_violations(s.adapter_source) == [], name
        assert "NACHLAUF_CKPT" in s.adapter_source and "etth1_public.npz" in s.adapter_source
        assert "dataset" not in s.adapter_source.lower() or "dataset" not in re.findall(r"open\(.*\)", s.adapter_source)
    assert m.main(["--unit", "dlinear", "--dry-run"], sandbox_factory=lambda: (_ for _ in ()).throw(AssertionError("kein Client"))) == 0


def test_nachlauf_module_builds_no_prompt_from_test_values():
    """M1 hat keinen LLM-Pfad; metrics_droplast (Testwerte) bleibt in Dossier und Pruefung. Diese Zeilen duerfen nie einen Prompt bauen."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "repro_agent" / "nachlauf.py").read_text()
    assert not re.search(r"\b(llm|LLMClient|chat|messages|system=|user=)\b", src)
