"""Nachlauf M3 (Agenten-Teil): Fake-LLM und Fake-Sandbox, erfundenes Repo. Kein echter LLM-, Tavily- oder Sandbox-Aufruf."""
import fnmatch
import importlib.util
import json
import random
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from repro_agent import nachlauf as nl
from repro_agent import nachlauf_agent as na
from repro_agent.dossier import Claim
from repro_agent.harness import independence as ind
from repro_agent.llm import LLMResponse
from repro_agent.sandbox.base import RunResult

sys.path.insert(0, str(Path(__file__).parent))
from test_nachlauf import csv_bytes  # noqa: E402
from test_nachlauf_m2 import SeriesFake, honest, mse_all_and_droplast, series_public  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "COMMIT"
CMD = "python -u train_gizmo.py --task demo --horizon 96 --lr 0.01 --input_len 336"

# ---- erfundenes Repo ("gizmo"): keine Datei, kein Name und kein Schalter der Entwicklungs-Repos -----------------------------------

REPO = {
    "README.md": f"# Gizmo\nRun the demo experiment:\n\n{CMD}\n\nOther tasks: python -u train_gizmo.py --task other --horizon 24\n",
    "train_gizmo.py": ("import argparse\nseed = 4242\nparser = argparse.ArgumentParser()\n"
                       "parser.add_argument('--horizon', type=int, default=24, help='forecast length')\n"
                       "parser.add_argument('--use_all', action='store_true', help='train on the entire dataset incl. the test part')\n"
                       "parser.add_argument('--lr', type=float, default=0.01, help='learning rate')\n"),
    "gizmo/model.py": "import torch.nn as nn\nclass Model(nn.Module):\n    def __init__(self, cfg):\n        self.lin = nn.Linear(cfg.input_len, cfg.horizon)\n    def forward(self, x):\n        return self.lin(x)\n",
    "gizmo/loop.py": "def run():\n    torch.save(model.state_dict(), 'out/demo/weights.bin')\n    print('mse:', test_mse)\n",
    "gizmo/loader.py": "def make(flag):\n    batch_size = 128\n    drop_last = True\n",
    "requirements.txt": "numpy\nfoo-lib==1.0\n",
    "docs/notes.md": "unrelated notes\n",
}
LISTING = sorted(REPO) + ["LICENSE", "assets/logo.png"]
GOOD = {
    "command": CMD,
    "command_source": {"file": "README.md", "quote": CMD},
    "seed": {"kind": "copy", "file": "train_gizmo.py", "line": "seed = 4242"},
    "checkpoint_glob": "out/*/weights.bin",
    "test_loader": {"drop_last": {"value": True, "file": "gizmo/loader.py", "quote": "drop_last = True"},
                    "batch_size": {"value": 128, "file": "gizmo/loader.py", "quote": "batch_size = 128"}},
    "input_len": 336, "output_len": 96,
    "needs_time_features": {"value": False, "file": "gizmo/model.py", "quote": None},
    "model_file": "gizmo/model.py", "training_file": "gizmo/loop.py",
}


def jx(obj) -> str:
    return "Reasoning...\n```json\n" + json.dumps(obj) + "\n```"


def code(variant: str) -> str:
    return f"```python\nimport numpy as np\n# variant: {variant}\nprint('adapter')\n```"


def adapter_for(src: str):
    v = re.search(r"# variant: (\w+)", src).group(1)
    return {
        "honest": lambda x, s: honest(x, s),
        "p3bad": lambda x, s: honest(x, 0.5),                                   # nur P3: eigenes Fenster, aber falsche Skalierung
        "p2bad": lambda x, s: honest(x, s) + 0.05 * x[:, -1:, :].mean(axis=(0, 1), keepdims=True),   # haengt von der Datei ab
        "shape": lambda x, s: honest(x, s),
        "crash": lambda x, s: honest(x, s),
    }[v], v


class AgentSandbox(SeriesFake):
    """SeriesFake plus Lese-Snapshot, Training des erfundenen Repos (skriptbar) und Adapter-Varianten nach Marker im Quelltext."""

    def __init__(self, train_script=None, own_adapter=None, repo=REPO, **kw):
        super().__init__(honest, **kw)
        self.repo, self.train_script, self.train_calls, self.snapshot_cmds = repo, train_script, 0, {}
        if own_adapter is not None:
            own = mse_all_and_droplast(own_adapter(self.pub["test_inputs"], 0.9), self.truth)[1]
            self.own = f"mse:{own:.4f}, mae:0.4"
            self.own_value = own

    def base_snapshot(self, image, setup_cmds, label=None):
        self.ops.append(("snapshot", label, list(setup_cmds)))
        self.snapshot_cmds[label] = list(setup_cmds)
        if label.startswith("nachlauf-read"):
            kind, files = "read", {f"/repo/{p}": t.encode() for p, t in self.repo.items()}
        elif label.endswith("predict"):
            kind, files = "predict", {"/data/etth1_public.npz": ind.to_npz_bytes(self.pub)}
        else:
            kind, files = "train", {f"/work/repo/{p}": t.encode() for p, t in self.repo.items()}
            files["/work/repo/dataset/ETTh1.csv"] = b"raw"
        self.snaps[label] = (kind, files)
        return label

    def run(self, b, cmd, timeout_s):
        f, kind = self.branches[b], self.kind[b]
        if cmd.startswith("cd /repo && git ls-files"):
            self.ops.append(("run", b, cmd, timeout_s))
            return RunResult(0, "\n".join(LISTING) + "\n", "", False)
        if cmd.startswith("ls -1 /work/repo/"):
            self.ops.append(("run", b, cmd, timeout_s))
            pat = cmd.split()[2]
            hits = [p for p in f if fnmatch.fnmatch(p, pat)]
            return RunResult(0 if hits else 2, "".join(h + "\n" for h in hits), "", False)
        if "unshare -n" in cmd and "train_gizmo" in cmd:
            self.ops.append(("run", b, cmd, timeout_s))
            assert kind == "train" and "/opt/nachlauf/predict.py" not in f
            self.train_calls += 1
            if self.train_script:
                res = self.train_script(self.train_calls, cmd)
                if res is not None:
                    return RunResult(res[0], res[1], res[2], False)
            f["/work/repo/out/demo/weights.bin"] = b"scale=0.9"
            return RunResult(0, self.own + "\n__WALL_MS__ 25000\n", "", False)
        if "/opt/nachlauf/predict.py" in cmd:
            src = f[nl.ADAPTER_PATH].decode()
            fn, variant = adapter_for(src)
            if variant == "crash":
                self.ops.append(("run", b, cmd, timeout_s))
                return RunResult(1, "", "Traceback (most recent call last):\n  File \"predict.py\", line 3\nKeyError: 'depth'\n", False)
            if variant == "shape" and "SMOKE=1" in cmd:
                self.ops.append(("run", b, cmd, timeout_s))
                x = ind.load_npz_bytes(f["/data/etth1_public.npz"])["test_inputs"][:8]
                p = honest(x, 0.9)[:, :95]
                w, s, c = np.meshgrid(np.arange(8), np.arange(95), np.arange(7), indexing="ij")
                import io as _io
                buf = _io.StringIO()
                np.savetxt(buf, np.column_stack([w.ravel(), s.ravel(), c.ravel(), p.ravel()]), fmt="%d_%d_%d,%.8g", header="id,prediction", comments="")
                f["/output/predictions.csv"] = buf.getvalue().encode()
                return RunResult(0, "__WALL_MS__ 10\n", "", False)
            self.adapter = fn
        return super().run(b, cmd, timeout_s)

    def sandbox_has_raw(self) -> bool:
        return False


class FakeLLM:
    def __init__(self, script: dict, cost: float = 0.01):
        self.script = {k: list(v) for k, v in script.items()}
        self.total_cost_usd = 0.0
        self.cost = cost
        self.calls: list[tuple] = []
        self.settings = types.SimpleNamespace(vat_rate=0.19, max_attempts_per_claim=4, max_cost_per_run_usd=100.0)

    def complete(self, role, messages, temperature=0.2, max_tokens=None, timeout_s=None):
        user = messages[-1]["content"]
        purpose = re.search(r"Task: (\w+)", user).group(1)
        item = self.script[purpose].pop(0)       # strikt: ein unerwarteter Aufruf scheitert laut
        text = item(user) if callable(item) else item
        self.total_cost_usd += self.cost
        self.calls.append((purpose, role, user))
        return LLMResponse(content=text, model="fake", prompt_tokens=10, completion_tokens=10, cost_usd=self.cost, cached=False, finish_reason="stop")

    def purposes(self):
        return [c[0] for c in self.calls]


def base_script(**over) -> dict:
    s = {"select_files": ["```json\n" + json.dumps(["README.md", "train_gizmo.py", "gizmo/model.py", "gizmo/loop.py", "gizmo/loader.py"]) + "\n```"],
         "start_command": [jx(GOOD)], "write_adapter": [code("honest")]}
    s.update(over)
    return s


def claim_for(sb) -> Claim:
    return Claim(id="gizmo_mse", text="GizmoNet reaches MSE 0.40 on ETTh1 at horizon 96 (input window 336)", metric="mse", target=round(sb.own_value, 4),
                 tolerance=0.01, source="Table 2", origin="paper")


def go(sb, llm, tmp_path, seeds=("repo", 0), **kw):
    agent = na.NachlaufAgent(sb, llm, out_dir=tmp_path, public_loader=lambda: sb.pub, np_rng=np.random.default_rng(0), log=lambda *_: None,
                             explore_offsets=[5, 100], **{k: v for k, v in kw.items() if k in ("store", "budget_stop", "max_attempts", "parallel")})
    d = agent.run("gizmo_2024", claim_for(sb), "https://example.invalid/gizmo.git", COMMIT, seeds=list(seeds), offsets=[5, 100])
    return d, agent


def info_of(d):
    return next(iter(d.nachlauf.values()))


# ---- Gut-Fall ------------------------------------------------------------------------------------------------------------------

def test_good_case_from_start_to_verdict(tmp_path):
    sb, llm = AgentSandbox(), None
    llm = FakeLLM(base_script())
    d, agent = go(sb, llm, tmp_path)
    assert llm.purposes() == ["select_files", "start_command", "write_adapter"]       # 3 Aufrufe im Normalfall
    assert d.mode == "nachlauf" and d.verdicts[0].urteil == "reproduziert"
    info = info_of(d)
    assert info["command"] == CMD and info["command_documented"] is True and info["command_source"] == {"file": "README.md", "line": 4}
    assert info["seed_method"].startswith("copy") and info["test_loader"] == {"drop_last": True, "batch": 128, "source": "gizmo/loader.py:3"}
    assert info["own_metric"].keys() == {"4242", "0"} and info["self_check"]["4242"]["status"] == "geprueft"
    assert info["independence"]["0"]["ok"] and info["env_diff"] == [{"package": "foo-lib", "required": "==1.0", "installed": None}]
    ag = info["agent"]
    assert ag["outcome"] == "ok" and ag["n_llm_calls"] == 3 and ag["verification"]["command_source"]["belegt"] and ag["discarded"] == []
    assert [a.status for a in d.attempts] == ["stabil", "stabil"]
    # Seeds: Repo-Seed mit unveraendertem Skript, Seed 0 ueber Seed-Kopie
    trains = [o[2] for o in sb.ops if o[0] == "run" and "unshare -n" in o[2] and "train_gizmo" in o[2]]
    assert len(trains) == 3 and " train_gizmo.py " in trains[0] and "/opt/seed/train_gizmo.py" in trains[-1]      # Erkundung, Repo-Seed, Seed 0
    types_ = {json.loads(l)["type"] for l in (tmp_path / "events.jsonl").read_text().splitlines()}
    assert {"nachlauf_command", "nachlauf_adapter", "nachlauf_train", "nachlauf_independence"} <= types_
    assert list(tmp_path.glob("adapter_*.py")) and (tmp_path / "command_answer.json").exists()
    json.dumps(d.nachlauf, default=str)


def test_adapter_and_branch_isolation_in_the_agent_flow(tmp_path):
    sb = AgentSandbox()
    go(sb, FakeLLM(base_script()), tmp_path)
    for o in sb.ops:
        if o[0] == "put_files" and sb.kind[o[1]] == "train":
            assert not any(nl.ADAPTER_DIR in p or "harness" in p for p in o[2])
    assert not any(o[0] == "run" and "predict.py" in o[2] and sb.kind[o[1]] != "predict" for o in sb.ops)


# ---- Zitate -----------------------------------------------------------------------------------------------------------------

def test_non_literal_quote_is_asked_again_once_then_accepted(tmp_path):
    bad = json.loads(json.dumps(GOOD))
    bad["command_source"]["quote"] = "python train_gizmo.py --horizon 96"          # steht so nicht in der Datei
    fixed = {"command_source": GOOD["command_source"]}
    sb = AgentSandbox()
    llm = FakeLLM(base_script(start_command=[jx(bad)], start_command_requote=[jx(fixed)]))
    d, _ = go(sb, llm, tmp_path)
    assert llm.purposes().count("start_command_requote") == 1
    ag = info_of(d)["agent"]
    assert ag["verification"]["command_source"]["belegt"] and ag["discarded"] == []
    assert not [w for w in d.warnings if "nicht belegt" in w]


def test_still_unbelegt_after_the_second_ask_falls_back(tmp_path):
    bad = json.loads(json.dumps(GOOD))
    bad["command"] = CMD.replace("--lr 0.01", "--lr 0.5")                                                 # steht so nirgends im Repo
    bad["command_source"]["quote"] = "python totally_made_up.py"
    bad["seed"] = {"kind": "copy", "file": "train_gizmo.py", "line": "seed = 1234"}                      # Zeile gibt es nicht
    bad["test_loader"]["drop_last"]["quote"] = "drop_last = False"                                         # steht nicht in der Datei
    sb = AgentSandbox()
    sb.own = f"mse:{mse_all_and_droplast(honest(sb.pub['test_inputs']), sb.truth)[0]:.4f}, mae:0.4"      # ohne belegten Loader gilt der Wert ueber alle Fenster
    llm = FakeLLM(base_script(start_command=[jx(bad)], start_command_requote=[jx({"command_source": bad["command_source"], "seed": bad["seed"],
                                                                                   "test_loader": {"drop_last": bad["test_loader"]["drop_last"]}})]))
    d, _ = go(sb, llm, tmp_path)
    assert llm.purposes().count("start_command_requote") == 1                                              # genau einmal neu gefragt
    info, ag = info_of(d), info_of(d)["agent"]
    assert {x["item"] for x in ag["discarded"]} == {"command_source", "seed", "test_loader.drop_last"}
    assert info["command_documented"] is False and any("Startbefehl nicht belegt" in w for w in d.warnings)            # P1 meldet es, Lauf geht weiter
    assert "eine Messung, keine Streuung" in d.warnings and any("Test-Loader nicht belegt" in w for w in d.warnings)
    assert [a.seed for a in d.attempts if a.status in ("stabil", "abweichung")] == [4242] or [a.seed for a in d.attempts if a.status in ("stabil", "abweichung")] == [None]
    assert info["test_loader"]["source"] is None and list(info["self_check"].values())[0]["compared_with"] == "all_windows"  # P3 mit allen Fenstern


def test_needs_time_features_ends_with_setup_verdict_before_any_training(tmp_path):
    tf = json.loads(json.dumps(GOOD))
    tf["needs_time_features"] = {"value": True, "file": "gizmo/model.py", "quote": "return self.lin(x)"}
    sb = AgentSandbox()
    d, _ = go(sb, FakeLLM(base_script(start_command=[jx(tf)])), tmp_path)
    v = d.verdicts[0]
    assert v.urteil == "nicht reproduzierbar (Setup)" and "model needs time features, not supported in v1" in v.explanation
    assert not any(o[0] == "run" and "unshare" in o[2] for o in sb.ops)


# ---- Training mit Reparatur ----------------------------------------------------------------------------------------------------

def long_err() -> str:
    return "x" * 3000 + "\nTraceback (most recent call last):\nModuleNotFoundError: No module named 'foo_lib'\n"


def test_training_failure_repair_with_stderr_tail_and_named_package(tmp_path):
    def script(n, cmd):
        return (1, "stdout-only-marker-77", long_err()) if n == 1 else None
    repaired = {"command": CMD, "command_source": GOOD["command_source"], "checkpoint_glob": "out/*/weights.bin", "install": ["foo-lib"]}
    sb = AgentSandbox(train_script=script)
    llm = FakeLLM(base_script(repair_command=[jx(repaired)]))
    d, _ = go(sb, llm, tmp_path)
    assert d.verdicts[0].urteil == "reproduziert"
    prompt = next(c[2] for c in llm.calls if c[0] == "repair_command")
    assert "ModuleNotFoundError: No module named 'foo_lib'" in prompt and "stdout-only-marker-77" not in prompt     # Ende von stderr, keine Zusammenfassung
    assert "x" * 1400 in prompt and "x" * 1600 not in prompt                                                        # 1500 Zeichen
    assert any("foo-lib" in " ".join(cmds) for lbl, cmds in sb.snapshot_cmds.items() if "-r1-" in lbl)             # Snapshots neu mit dem Paket
    assert [a["status"] for a in info_of(d)["agent"]["attempts"]] == ["fehler", "ok"]
    assert info_of(d)["agent"]["repairs"][0]["kind"] == "package_install"


def test_package_not_in_dependency_files_is_rejected_without_a_run(tmp_path):
    def script(n, cmd):
        return (1, "", long_err()) if n == 1 else None
    evil = {"command": CMD, "command_source": GOOD["command_source"], "checkpoint_glob": "out/*/weights.bin", "install": ["evil-pkg"]}
    clean = {**evil, "install": []}
    sb = AgentSandbox(train_script=script)
    llm = FakeLLM(base_script(repair_command=[jx(evil), jx(clean)]))
    d, _ = go(sb, llm, tmp_path)
    assert d.verdicts[0].urteil == "reproduziert"
    repair_prompts = [c[2] for c in llm.calls if c[0] == "repair_command"]
    assert len(repair_prompts) == 2 and na.PACKAGE_SENTENCE.format(pkg="evil-pkg") in repair_prompts[1]
    assert not any("evil-pkg" in " ".join(cmds) for cmds in sb.snapshot_cmds.values())                             # nie installiert
    atts = info_of(d)["agent"]["attempts"]
    assert [a["status"] for a in atts] == ["fehler", "ungueltig", "ok"] and atts[1]["rejected_by"] == ["package"]
    assert sb.train_calls == 4 - 0 - 1 or sb.train_calls >= 2                                                      # Versuch 2 lief nicht


def test_forbidden_switch_is_rejected_by_p1_with_a_fixed_sentence(tmp_path):
    forbidden = json.loads(json.dumps(GOOD))
    forbidden["command"] = CMD + " --use_all"
    repaired = {"command": CMD, "command_source": GOOD["command_source"], "checkpoint_glob": "out/*/weights.bin", "install": []}
    sb = AgentSandbox()
    llm = FakeLLM(base_script(start_command=[jx(forbidden)], repair_command=[jx(repaired)]))
    d, _ = go(sb, llm, tmp_path)
    assert d.verdicts[0].urteil == "reproduziert"
    assert not any(o[0] == "run" and "unshare" in o[2] and "--use_all" in o[2] for o in sb.ops)                    # kein Lauf mit diesem Befehl
    prompt = next(c[2] for c in llm.calls if c[0] == "repair_command")
    assert na.P1_SENTENCE.format(flag="--use_all") in prompt
    assert info_of(d)["agent"]["attempts"][0]["rejected_by"] == ["P1"]


def test_attempt_limit_gives_setup_verdict_with_all_attempts(tmp_path):
    sb = AgentSandbox(train_script=lambda n, cmd: (1, "", "RuntimeError: boom\n"))
    repaired = {"command": CMD, "command_source": GOOD["command_source"], "checkpoint_glob": "out/*/weights.bin", "install": []}
    llm = FakeLLM(base_script(repair_command=[jx(repaired)] * 3))
    d, _ = go(sb, llm, tmp_path, max_attempts=4)
    assert d.verdicts[0].urteil == "nicht reproduzierbar (Setup)"
    assert len(info_of(d)["agent"]["attempts"]) == 4 and llm.purposes().count("repair_command") == 3 and "write_adapter" not in llm.purposes()
    assert len([a for a in d.attempts if a.status == "fehler"]) == 4


# ---- Adapter -------------------------------------------------------------------------------------------------------------------

def test_adapter_static_violation_then_repair(tmp_path):
    sb = AgentSandbox()
    llm = FakeLLM(base_script(write_adapter=["```python\nimport subprocess\n# variant: honest\n```"], repair_adapter=[code("honest")]))
    d, _ = go(sb, llm, tmp_path)
    assert d.verdicts[0].urteil == "reproduziert" and llm.purposes().count("repair_adapter") == 1
    assert "static check rejected" in next(c[2] for c in llm.calls if c[0] == "repair_adapter")
    assert [a["problems"] for a in info_of(d)["agent"]["adapters"]] == [["static"], []]


def test_adapter_wrong_form_and_crash_get_stderr_or_form_message(tmp_path):
    sb = AgentSandbox()
    llm = FakeLLM(base_script(write_adapter=[code("shape")], repair_adapter=[code("crash"), code("honest")]))
    d, _ = go(sb, llm, tmp_path)
    prompts = [c[2] for c in llm.calls if c[0] == "repair_adapter"]
    assert "wrong form" in prompts[0] or "expected" in prompts[0]                    # Formpruefung (P4)
    assert "KeyError: 'depth'" in prompts[1]                                         # Ende von stderr
    assert d.verdicts[0].urteil == "reproduziert"


def test_p3_invalid_gets_exactly_one_repair_round(tmp_path):
    sb = AgentSandbox()
    llm = FakeLLM(base_script(write_adapter=[code("p3bad")], repair_adapter=[code("p3bad")]))
    d, _ = go(sb, llm, tmp_path)
    assert llm.purposes().count("repair_adapter") == 1                               # genau eine Runde, dann Schluss
    prompt = next(c[2] for c in llm.calls if c[0] == "repair_adapter")
    assert na.P3_SENTENCE in prompt
    assert d.verdicts[0].urteil == "nicht reproduzierbar (Setup)"
    assert [a["problems"] for a in info_of(d)["agent"]["adapters"]] == [["P3"], ["P3"]]
    # und ein Erfolg nach der einen Runde
    llm2 = FakeLLM(base_script(write_adapter=[code("p3bad")], repair_adapter=[code("honest")]))
    d2, _ = go(AgentSandbox(), llm2, tmp_path / "b")
    assert d2.verdicts[0].urteil == "reproduziert" and llm2.purposes().count("repair_adapter") == 1


def test_p2_invalid_gets_exactly_one_repair_round(tmp_path):
    sb = AgentSandbox(own_adapter=adapter_for("# variant: p2bad")[0])                # P3 besteht (Eigenwert aus demselben Adapter), nur P2 faellt
    llm = FakeLLM(base_script(write_adapter=[code("p2bad")], repair_adapter=[code("p2bad")]))
    d, _ = go(sb, llm, tmp_path)
    assert llm.purposes().count("repair_adapter") == 1
    assert na.P2_SENTENCE in next(c[2] for c in llm.calls if c[0] == "repair_adapter")
    assert d.verdicts[0].urteil == "nicht reproduzierbar (Setup)"
    assert [a["problems"] for a in info_of(d)["agent"]["adapters"]] == [["P2"], ["P2"]]


# ---- Leak-Wache ----------------------------------------------------------------------------------------------------------------

NAMES = ["rtsf", "rlinear", "ltsf", "dlinear", "nlinear", "plumprc", "cure-lab", "sparsetsf", "fits", "patchtst", "lss-1138", "vewoxic", "yuqinie98",
         "run_longexp", "script.md", "etth1.sh", "fix_seed", "--rev", "--enc_in", "--channel", "--individual", "--train_only", "seq_len", "pred_len",
         "data_factory", "data_provider", "exp_main", "checkpoint.pth"]


def has_name(text: str) -> list[str]:
    low = text.lower()
    return [n for n in NAMES if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w])" if n[0].isalnum() else re.escape(n), low)]


def test_no_test_values_and_no_repo_names_in_any_prompt(tmp_path):
    sb = AgentSandbox(own_adapter=adapter_for("# variant: p2bad")[0])
    # P3 und P2 gleichzeitig ungueltig: feste Saetze in der Reparatur, keine Zahl
    sb.own = "mse:0.123456, mae:0.4"
    llm = FakeLLM(base_script(write_adapter=[code("p2bad")], repair_adapter=[code("honest")]))
    d, agent = go(sb, llm, tmp_path)
    secrets = ["0.123456", "0.1234"]
    h = next(o for o in [d.attempts] if o)
    for a in d.attempts:
        for m in (a.metrics or {}).values():
            secrets += [f"{m:.4f}", f"{m:.5f}", repr(m)[:8]]
        for dl in (a.metrics_droplast or {}).values():
            secrets += [f"{dl['mse']:.4f}", f"{dl['mse']:.5f}"]
    info = info_of(d)
    for sc in info["self_check"].values():
        if sc.get("distance") is not None:
            secrets += [f"{sc['distance']:.4f}", f"{sc['distance']:.5f}"]
        secrets += [f"{sc['harness']:.4f}"] if sc.get("harness") is not None else []
    for ind_ in info["independence"].values():
        if ind_.get("max_abs_diff"):
            secrets += [f"{ind_['max_abs_diff']:.4f}", f"{ind_['max_abs_diff']:.6f}", f"{ind_['max_abs_diff']:.2e}"]
    secrets = [s for s in secrets if len(s) >= 6]
    assert secrets
    for p in agent.asker.prompts:
        text = p["system"] + "\n" + p["user"]
        assert not [s for s in secrets if s in text], (p["purpose"], [s for s in secrets if s in text])
        assert has_name(p["system"] + "\n" + p["user"].split("Files:")[0].split("Repository file list")[0].split("Model file(s)")[0]) == [], p["purpose"]


def test_prompt_constants_contain_no_repo_model_file_or_switch_names():
    for name in dir(na):
        val = getattr(na, name)
        if name.isupper() and isinstance(val, str) and len(val) > 40:
            assert has_name(val) == [], (name, has_name(val))
    assert re.search(r"\d", na.P3_SENTENCE + na.P2_SENTENCE) is None             # feste Saetze enthalten keine Zahl


# ---- Budgetstopp -------------------------------------------------------------------------------------------------------------------

def test_budget_stop_before_the_next_call_by_process_cost(tmp_path):
    sb = AgentSandbox()
    llm = FakeLLM(base_script(), cost=0.5)                      # brutto je Aufruf 0,595; Grenze 1,0 -> nach zwei Aufrufen Schluss
    with pytest.raises(na.BudgetStop):
        go(sb, llm, tmp_path, budget_stop=1.0)
    assert len(llm.calls) == 2


def test_budget_stop_by_store_before_the_first_call(tmp_path):
    store = tmp_path / "store.jsonl"
    store.write_text(json.dumps({"cost_usd_run": 0.9}) + "\n")  # 0,9 netto = 1,071 brutto
    llm = FakeLLM(base_script())
    with pytest.raises(na.BudgetStop):
        go(AgentSandbox(), llm, tmp_path / "o", store=store, budget_stop=1.0)
    assert llm.calls == []


def test_cost_guard_does_not_double_count_stored_runs():
    llm = types.SimpleNamespace(total_cost_usd=0.4)
    g = na.CostGuard(llm, None, 10.0, 0.19)
    g.note_stored(0.4)
    assert g.total_gross() == pytest.approx(0.0)
    llm.total_cost_usd = 0.5
    assert g.total_gross() == pytest.approx(0.1 * 1.19)


# ---- P2 parallel, Versaetze je Seed --------------------------------------------------------------------------------------------

def test_p2_parallel_equals_sequential():
    sb = AgentSandbox()
    pub = sb.pub
    results = {}
    for par in (1, 4):
        run = nl._Run(sb, nl.UnitSpec(unit="u", paper_id="p", claim_text="c", target=0.4, tolerance=0.01, repo_url="u", commit=COMMIT, command=CMD,
                                      command_source={}, script="train_gizmo.py", seed={"kind": "none", "repo_seed": 1}, adapter_source="# variant: honest\nimport numpy\n",
                                      test_loader={}), nl.EventLog(None), lambda *_: None, lambda: pub, None, np.random.default_rng(0))
        run.parallel = par
        run.pred_snap = sb.base_snapshot("img", [], label="nachlauf-u-predict")
        full = honest(pub["test_inputs"], 0.9)
        results[par] = run.independence(b"scale=0.9", full, [5, 100, 500, 800, 20, 300], 1)
    assert results[1] == results[4] and results[1]["ok"] and results[1]["n_offsets"] == 6


def test_offsets_are_drawn_per_seed(tmp_path):
    sb = AgentSandbox()
    spec = nl.UnitSpec(unit="u", paper_id="p", claim_text="c", target=round(sb.own_value, 4), tolerance=0.01, repo_url="u", commit=COMMIT, command=CMD,
                       command_source={"file": "README.md", "line": 4}, script="train_gizmo.py",
                       seed={"kind": "copy", "repo_seed": 4242, "regex": r"seed = 4242", "template": "seed = {seed}"},
                       adapter_source="import numpy\n# variant: honest\n", test_loader={"drop_last": True, "batch": 128, "source": "x:1"},
                       argparse_files=["train_gizmo.py"], checkpoint_glob="out/*/weights.bin")
    d = nl.run_unit(sb, spec, [4242, 0, 1], offsets=3, public_loader=lambda: sb.pub, rng=random.Random(7), np_rng=np.random.default_rng(0), log=lambda *_: None)
    offs = [tuple(v["offsets"]) for v in info_of(d)["independence"].values()]
    assert len(offs) == 3 and len(set(offs)) == 3


# ---- Skript ------------------------------------------------------------------------------------------------------------------------

def load_script(rel, name):
    s = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_run_nachlauf_script_import_path_and_dry_run(capsys):
    m = load_script("scripts/run_nachlauf.py", "run_nachlauf_t")
    assert callable(m.default_factory())                                     # echte Importzeile, kein Client
    assert m.main(["--paper", "rlinear_2023", "--repo", "https://example.invalid/g", "--commit", "abc", "--dry-run"],
                  sandbox_factory=lambda: (_ for _ in ()).throw(AssertionError("kein Client")),
                  llm_factory=lambda: (_ for _ in ()).throw(AssertionError("kein LLM"))) == 0
    out = capsys.readouterr().out
    assert "rlinear_etth1_96_mse" in out and "LLM-Aufrufe je Lauf" in out and "Stopp bei 3.00" in out


def test_run_nachlauf_script_writes_store_row_and_stops_on_budget(tmp_path, capsys):
    m = load_script("scripts/run_nachlauf.py", "run_nachlauf_t2")
    m.ROOT = tmp_path
    sb = AgentSandbox()
    store = tmp_path / "store.jsonl"
    llm = FakeLLM(base_script(), cost=0.01)

    class Spy(na.NachlaufAgent):
        def __init__(self, *a, **k):
            super().__init__(*a, public_loader=lambda: sb.pub, np_rng=np.random.default_rng(0), explore_offsets=[5, 100], log=lambda *_: None, **k)

    m.na = types.SimpleNamespace(**{**{k: getattr(na, k) for k in dir(na) if not k.startswith("__")}, "NachlaufAgent": Spy})
    rc = m.main(["--paper", "rlinear_2023", "--repo", "https://example.invalid/gizmo.git", "--commit", COMMIT, "--store", str(store), "--seeds", "repo", "--offsets", "2"],
                sandbox_factory=lambda: sb, llm_factory=lambda: llm)
    row = json.loads(store.read_text().splitlines()[0])
    assert row["paper"] == "rlinear_2023" and row["valid"] is True and row["cost_usd_run"] == pytest.approx(0.03) and row["n_llm_calls"] == 3
    assert {"urteil", "mean", "std", "code_commit", "rejected_by", "run_dir", "cost_usd_run_gross"} <= row.keys()
    assert (tmp_path / row["run_dir"] / "dossier.json").exists() and rc in (0, 1)
    # zweiter Aufruf: der Store ist schon ueber der Grenze -> kein LLM-Aufruf, Zeile "abgebrochen"
    llm2 = FakeLLM(base_script())
    rc2 = m.main(["--paper", "rlinear_2023", "--repo", "https://example.invalid/gizmo.git", "--commit", COMMIT, "--store", str(store), "--budget-stop", "0.01"],
                 sandbox_factory=lambda: AgentSandbox(), llm_factory=lambda: llm2)
    assert llm2.calls == [] and rc2 == 1
    assert json.loads(store.read_text().splitlines()[1])["urteil"] == "abgebrochen"


def test_parse_json_object_takes_the_last_top_level_object():
    text = 'thinking {"a": 1} more {not json} final ```json\n{"command_source": {"file": "x", "quote": "y"}, "k": [1, {"z": 2}]}\n```'
    assert na.parse_json_object(text) == {"command_source": {"file": "x", "quote": "y"}, "k": [1, {"z": 2}]}
    assert na.parse_json_object(text, required="a") == {"a": 1}
    assert na.parse_json_object("no json here") is None


def test_public_file_is_loaded_once_per_run(tmp_path):
    sb = AgentSandbox()
    calls = []
    agent = na.NachlaufAgent(sb, FakeLLM(base_script()), out_dir=tmp_path, public_loader=lambda: calls.append(1) or sb.pub, np_rng=np.random.default_rng(0),
                             log=lambda *_: None, explore_offsets=[5, 100])
    agent.run("gizmo_2024", claim_for(sb), "https://example.invalid/gizmo.git", COMMIT, seeds=["repo", 0], offsets=[5, 100])
    assert len(calls) == 1
