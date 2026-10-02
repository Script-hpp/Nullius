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
from repro_agent import nachlauf_candidates as cand
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

RUN_SH = """model=gizmonet
len=336
for h in 24 96; do
python -u train_gizmo.py --task demo --model $model --horizon $h --input_len $len --lr 0.01 \\
  --batch_size 128 > logs/h$h.log 2>&1
done

python -u train_gizmo.py --task other --horizon 24
"""
CMD = "python -u train_gizmo.py --task demo --model gizmonet --horizon 96 --input_len 336 --lr 0.01 --batch_size 128"
REPO = {
    "README.md": "# Gizmo\nSee scripts/ for the commands of each experiment.\n",
    "scripts/run_demo.sh": RUN_SH,
    "train_gizmo.py": ("import argparse\nseed = 4242\nparser = argparse.ArgumentParser()\n"
                       "parser.add_argument('--horizon', type=int, default=24, help='forecast length')\n"
                       "parser.add_argument('--use_all', action='store_true', help='train on the entire dataset incl. the test part')\n"
                       "parser.add_argument('--lr', type=float, default=0.01, help='learning rate')\n"
                       "parser.add_argument('--batch_size', type=int, default=64, help='batch size')\n"
                       "parser.add_argument('--seed', type=int, default=7, help='random seed')\n"),
    "gizmo/model.py": "import torch.nn as nn\nclass Model(nn.Module):\n    def __init__(self, cfg):\n        self.lin = nn.Linear(cfg.input_len, cfg.horizon)\n    def forward(self, x):\n        return self.lin(x)\n",
    "gizmo/loop.py": "def run():\n    torch.save(model.state_dict(), 'out/demo/weights.bin')\n    print('mse:', test_mse)\n",
    "gizmo/loader.py": "def make(flag, args):\n    batch_size = args.batch_size\n    drop_last = True\n",
    "requirements.txt": "numpy\nfoo-lib==1.0\n",
    "docs/notes.md": "unrelated notes\n",
}
LISTING = sorted(REPO) + ["LICENSE", "assets/logo.png"]


def pick(user: str, needle: str) -> int | None:
    """Nummer der Zeile `[k] ...` im Prompt, die `needle` enthaelt (das Modell waehlt per Nummer)."""
    m = re.search(rf"^\[(\d+)\][^\n]*{re.escape(needle)}", user, re.M)
    return int(m.group(1)) if m else None


def good_answer(user: str) -> str:
    return jx({"command_candidate": pick(user, "h=96"), "command": "python evil_invented.py --x 1",     # frei erfundener Befehl: wird ignoriert
               "checkpoint_glob": "out/*/weights.bin",
               "test_loader": {"drop_last": {"candidate": pick(user, "drop_last = True"), "value": True}, "batch_size": {"switch": "--batch_size"}},
               "input_len": 336, "output_len": 96, "needs_time_features": {"value": False, "file": "gizmo/model.py", "quote": None},
               "model_file": "gizmo/model.py", "training_file": "gizmo/loop.py"})


def seed_answer(needle: str = "seed = 4242"):
    return lambda user: jx({"seed_candidate": pick(user, needle)})


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
        if cmd.startswith("cd /repo && git grep -lIE"):
            self.ops.append(("run", b, cmd, timeout_s))
            import shlex as _sh
            ere = _sh.split(cmd)[6]
            rx = re.compile(ere.replace("[[:space:]]", r"\s"))
            hits = sorted(p for p, t in self.repo.items() if rx.search(t) and not p.endswith((".csv", ".json", ".ipynb")))
            return RunResult(0, "".join(h + "\n" for h in hits), "", False)
        if cmd.startswith("cd /repo && git grep -hE"):
            self.ops.append(("run", b, cmd, timeout_s))
            out = "".join(l + "\n" for t in self.repo.values() for l in t.splitlines() if re.match(r"^\s*(import|from)\s+[A-Za-z_]", l))
            return RunResult(0, out, "", False)
        if cmd.startswith("cd /repo && git ls-files"):
            self.ops.append(("run", b, cmd, timeout_s))
            return RunResult(0, "\n".join(sorted(self.repo) + ["LICENSE", "assets/logo.png"]) + "\n", "", False)
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
         "start_command": [good_answer], "choose_seed": [seed_answer()], "write_adapter": [code("honest")]}
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

def ok_train_script(fail_first=0, stderr="RuntimeError: boom\n"):
    return lambda n, cmd: (1, "stdout-only-marker-77", stderr) if n <= fail_first else None


def test_good_case_from_start_to_verdict(tmp_path):
    sb = AgentSandbox()
    llm = FakeLLM(base_script())
    d, agent = go(sb, llm, tmp_path)
    assert llm.purposes() == ["select_files", "start_command", "choose_seed", "write_adapter"]      # 4 Aufrufe im Normalfall
    assert d.mode == "nachlauf" and d.verdicts[0].urteil == "reproduziert"
    info = info_of(d)
    # Befehl und Quelle kommen aus der Kandidatenliste (Datei, Zeile), nicht aus der Antwort des Modells ("evil_invented.py" wird ignoriert)
    assert info["command"] == CMD and info["command_documented"] is True and info["command_source"] == {"file": "scripts/run_demo.sh", "line": 4}
    assert "evil_invented" not in json.dumps(info["command"]) and info["seed_method"].startswith("copy")
    assert info["test_loader"] == {"drop_last": True, "batch": 128, "source": "gizmo/loader.py:3", "batch_source": "command"}
    assert info["own_metric"].keys() == {"4242", "0"} and info["self_check"]["4242"]["status"] == "geprueft"
    assert info["independence"]["0"]["ok"] and info["env_diff"] == [{"package": "foo-lib", "required": "==1.0", "installed": None}]
    ag = info["agent"]
    assert ag["outcome"] == "ok" and ag["n_llm_calls"] == 4 and ag["verification"]["command_candidate"]["belegt"] and ag["discarded"] == []
    assert ag["candidates"]["n_total"] == 3 and ag["candidates"]["n_shown"] == 3 and ag["candidates"]["n_dropped"] == 0
    assert ag["seed_choice"]["text"] == "seed = 4242" and any("line 2 of train_gizmo.py" in x for x in ag["seed_candidates"])
    assert [a.status for a in d.attempts] == ["stabil", "stabil"]
    trains = [o[2] for o in sb.ops if o[0] == "run" and "unshare -n" in o[2] and "train_gizmo" in o[2]]
    assert len(trains) == 3 and " train_gizmo.py " in trains[0] and "/opt/seed/train_gizmo.py" in trains[-1]      # Erkundung, Repo-Seed, Seed 0
    assert all("evil_invented" not in t for t in trains)
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


# ---- Kandidaten (38c) ------------------------------------------------------------------------------------------------------------

def test_candidates_are_found_although_the_model_does_not_select_the_command_file(tmp_path):
    sel = "```json\n" + json.dumps(["train_gizmo.py", "gizmo/model.py"]) + "\n```"          # run_demo.sh fehlt in der Auswahl des Modells
    sb = AgentSandbox()
    llm = FakeLLM(base_script(select_files=[sel]))
    d, agent = go(sb, llm, tmp_path)
    prompt = next(c[2] for c in llm.calls if c[0] == "start_command")
    assert "scripts/run_demo.sh:4:" in prompt and "(command with h=96:" in prompt
    assert "scripts/run_demo.sh" in info_of(d)["agent"]["reading"]["files"]                  # Dateien mit Kandidaten gehoeren immer zu den gelesenen
    assert info_of(d)["command_source"] == {"file": "scripts/run_demo.sh", "line": 4} and info_of(d)["command"] == CMD
    assert "gizmo/loader.py" in info_of(d)["agent"]["reading"]["files"]                       # Loader-Dateien ebenso (DataLoader/drop_last per grep)


def test_extract_candidates_from_shell_loop_with_variables_and_continuation_lines():
    cs, skipped = cand.extract_candidates({"scripts/run_demo.sh": RUN_SH + 'python -u train_gizmo.py --x $nowhere\n'}, LISTING)
    assert skipped == 1
    got = [(c.file, c.line, c.note, c.command) for c in cs]
    assert got == [
        ("scripts/run_demo.sh", 4, "h=24", CMD.replace("--horizon 96", "--horizon 24")),
        ("scripts/run_demo.sh", 4, "h=96", CMD),
        ("scripts/run_demo.sh", 8, "", "python -u train_gizmo.py --task other --horizon 24"),
    ]
    assert cs[0].text.endswith("--batch_size 128 > logs/h$h.log 2>&1") and cs[0].script == "train_gizmo.py"
    # Aufrufe von Dateien, die nicht im Repo liegen, und `python -m` sind keine Kandidaten
    none, _ = cand.extract_candidates({"a.md": "run `python -u not_in_repo.py --x 1` or python -m pkg.mod --y\n"}, LISTING)
    assert none == []
    md, _ = cand.extract_candidates({"README.md": "Train:\n\n    python train_gizmo.py --task demo && echo done\n"}, LISTING)
    assert [c.command for c in md] == ["python train_gizmo.py --task demo"]


def test_check_command_resolves_loop_variables():
    r = nl.check_command(CMD, {"scripts/run_demo.sh": RUN_SH}, {"file": "scripts/run_demo.sh", "line": 4})
    assert r["documented"] is True and r["diff"] == [] and r["source"]["line"] == 4


def test_no_candidate_gives_setup_verdict_without_any_training(tmp_path):
    repo = {k: v for k, v in REPO.items() if k != "scripts/run_demo.sh"}
    sb = AgentSandbox(repo=repo)
    llm = FakeLLM(base_script())
    d, _ = go(sb, llm, tmp_path)
    v = d.verdicts[0]
    assert v.urteil == "nicht reproduzierbar (Setup)" and na.NO_COMMAND_REASON in v.explanation
    assert "start_command" not in llm.purposes() and not any(o[0] == "run" and "unshare" in o[2] for o in sb.ops)    # kein Modellaufruf, kein Training
    assert not any(o[0] == "snapshot" and not o[1].startswith("nachlauf-read") for o in sb.ops)
    assert info_of(d)["agent"]["outcome"] == "no_command"


def test_model_chooses_none_gives_the_same_setup_verdict(tmp_path):
    none = lambda user: jx({"command_candidate": None, "checkpoint_glob": "out/*/weights.bin"})  # noqa: E731
    sb = AgentSandbox()
    d, _ = go(sb, FakeLLM(base_script(start_command=[none])), tmp_path)
    assert d.verdicts[0].urteil == "nicht reproduzierbar (Setup)" and na.NO_COMMAND_REASON in d.verdicts[0].explanation
    assert not any(o[0] == "run" and "unshare" in o[2] for o in sb.ops)


def test_unknown_candidate_number_is_no_command(tmp_path):
    bad = lambda user: jx({"command_candidate": 99})  # noqa: E731
    d, _ = go(AgentSandbox(), FakeLLM(base_script(start_command=[bad])), tmp_path)
    assert na.NO_COMMAND_REASON in d.verdicts[0].explanation
    assert info_of(d)["agent"]["discarded"][0]["item"] == "command_candidate"


def many_candidates(n):
    lines = [f"python -u train_gizmo.py --task t{i} --horizon 24" for i in range(n)]
    lines += ["python -u train_gizmo.py --task demo --model gizmonet --horizon 96"]
    return "\n".join(lines) + "\n"


def test_narrowing_prefers_claim_words_and_counts_the_rest():
    text = many_candidates(60)
    cs, _ = cand.extract_candidates({"scripts/many.sh": text}, LISTING)
    shown, info = cand.narrow(cs, cand.claim_words("GizmoNet reaches MSE 0.40 on ETTh1 at horizon 96 (input window 336)", "ETTh1"))
    assert info["n_total"] == 61 and info["n_shown"] == cand.MAX_CANDIDATES_PROMPT == len(shown) and info["n_dropped"] == 31
    assert any("--horizon 96" in c.command for c in shown) and shown[0].k == 1 and [c.k for c in shown] == list(range(1, 31))
    assert info["words"]["model"] == "GizmoNet" and info["dropped_sample"]


def test_narrowing_is_recorded_in_the_dossier_and_prompt(tmp_path):
    repo = {**REPO, "scripts/many.sh": many_candidates(60)}
    sb = AgentSandbox(repo=repo)
    llm = FakeLLM(base_script(select_files=["[]"]))
    d, _ = go(sb, llm, tmp_path)
    ci = info_of(d)["agent"]["candidates"]
    assert ci["n_total"] >= 63 and ci["n_shown"] == 30 and ci["n_dropped"] == ci["n_total"] - 30 and len(ci["shown"]) == 30
    assert "30 of" in next(c[2] for c in llm.calls if c[0] == "start_command") and "more not shown" in next(c[2] for c in llm.calls if c[0] == "start_command")


# ---- Test-Loader und Seed aus dem Code (38c) ---------------------------------------------------------------------------------------

def ctx_for(command_needle="h=96"):
    cs, _ = cand.extract_candidates({"scripts/run_demo.sh": RUN_SH}, LISTING)
    shown, _ = cand.narrow(cs, cand.claim_words("GizmoNet reaches MSE", "ETTh1"))
    ll = cand.loader_lines({"gizmo/loader.py": REPO["gizmo/loader.py"]})
    return shown, ll, next(c.k for c in shown if command_needle in c.note or command_needle in c.command)


def verify(ans_part, k=None, command_needle="h=96"):
    shown, ll, kk = ctx_for(command_needle)
    ans = {"command_candidate": k or kk, **ans_part}
    return na.verify_answer(ans, REPO, shown, ll)


def test_batch_size_from_command_from_default_and_not_belegt():
    dl = {"drop_last": {"candidate": 1, "value": True}}
    v = verify({"test_loader": {**dl, "batch_size": {"switch": "--batch_size"}}})["test_loader.batch_size"]
    assert v["belegt"] and v["value"] == 128 and v["source"] == "command"                                # Wert des Schalters im gewaehlten Befehl
    v = verify({"test_loader": {**dl, "batch_size": {"switch": "--batch_size"}}}, command_needle="--task other")["test_loader.batch_size"]
    assert v["belegt"] and v["value"] == 64 and v["source"] == "train_gizmo.py:7" and "default=64" in v["quote"]   # default= der Argumentdefinition
    v = verify({"test_loader": {**dl, "batch_size": {"switch": "--bs"}}})["test_loader.batch_size"]
    assert v["belegt"] is False and "no value" in v["reason"]                                              # Schalter weder im Befehl noch definiert
    v = verify({"test_loader": {**dl, "batch_size": {"file": "gizmo/loader.py", "quote": "batch_size = args.batch_size"}}})["test_loader.batch_size"]
    assert v["belegt"] is False                                                                           # nur ein Verweis auf ein Argument: kein Wert im Zitat
    v = verify({"test_loader": {**dl, "batch_size": {"file": "gizmo/loader.py", "quote": "batch_size = 32"}}})["test_loader.batch_size"]
    assert v["belegt"] is False                                                                           # Zitat nicht in der Datei
    code_py = {**REPO, "gizmo/loader.py": "batch_size = 32\ndrop_last = False\n"}
    v = na.verify_answer({"test_loader": {"batch_size": {"file": "gizmo/loader.py", "quote": "batch_size = 32"}}}, code_py, [], [], only={"test_loader.batch_size"})
    assert v["test_loader.batch_size"]["belegt"] and v["test_loader.batch_size"]["value"] == 32           # woertliches Zitat mit Zahl aus dem Code


def test_drop_last_line_and_value_come_from_the_listed_line():
    v = verify({"test_loader": {"drop_last": {"candidate": 1, "value": False}, "batch_size": {"switch": "--batch_size"}}})["test_loader.drop_last"]
    assert v["belegt"] and v["value"] is True and v["model_value"] is False and v["line"] == 3        # die Zeile gilt, nicht die Behauptung
    ref = {**REPO, "gizmo/loader.py": "drop_last = flag\n"}
    ll = cand.loader_lines({"gizmo/loader.py": ref["gizmo/loader.py"]})
    v = na.verify_answer({"test_loader": {"drop_last": {"candidate": 1, "value": True}}}, ref, [], ll, only={"test_loader.drop_last"})["test_loader.drop_last"]
    assert v["belegt"] is False and "literal" in v["reason"]


def test_seed_candidates_assignment_switch_and_none():
    sc = cand.seed_candidates("train_gizmo.py", REPO["train_gizmo.py"])
    assert [(c.kind, c.line, c.value) for c in sc] == [("copy", 2, 4242), ("flag", 8, 7)] and sc[1].flag == "--seed"
    assert cand.seed_spec(sc[0], REPO["train_gizmo.py"]) == {"kind": "copy", "repo_seed": 4242, "regex": "seed\\ =\\ 4242", "template": "seed = {seed}"}
    assert cand.seed_spec(sc[1], REPO["train_gizmo.py"]) == {"kind": "flag", "flag": "--seed", "repo_seed": 7}
    assert cand.seed_candidates("x.py", "import numpy\nrandom.seed(seed)\nx = 3\n") == []                  # kein Seed mit Zahl, keine Schalterdefinition
    call = cand.seed_candidates("x.py", "np.random.seed(11)\n")
    assert [(c.kind, c.value) for c in call] == [("copy", 11)]
    dup = "seed = 5\nseed = 5\n"
    assert cand.seed_candidates("x.py", dup) == []                                                        # nicht eindeutig: gar nicht angeboten
    assert cand.seed_spec(cand.SeedCandidate(1, "copy", "x.py", 1, "seed = 5", value=5), dup) is None    # und seed_spec lehnt sie ab


def test_seed_choice_switch_runs_seeds_through_the_flag(tmp_path):
    sb = AgentSandbox()
    d, _ = go(sb, FakeLLM(base_script(choose_seed=[seed_answer("--seed")])), tmp_path, seeds=("repo", 0))
    info = info_of(d)
    assert info["seed_method"] == "command-line switch of the repository" and info["agent"]["seed_choice"]["kind"] == "flag"
    trains = [o[2] for o in sb.ops if o[0] == "run" and "unshare -n" in o[2] and "train_gizmo" in o[2]]
    assert any(" --seed 0" in t for t in trains) and not any("--seed 7" in t for t in trains)               # Repo-Seed (default 7): Lauf ohne Schalter
    assert [a.seed for a in d.attempts if a.status in ("stabil", "abweichung")] == [7, 0]


def test_no_seed_candidates_means_no_seed_call_and_a_warning(tmp_path):
    repo = {**REPO, "train_gizmo.py": "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--horizon', type=int, default=24)\n"}
    sb = AgentSandbox(repo=repo)
    llm = FakeLLM(base_script(choose_seed=[]))
    d, _ = go(sb, llm, tmp_path)
    assert "choose_seed" not in llm.purposes() and "eine Messung, keine Streuung" in d.warnings
    assert [a.seed for a in d.attempts if a.status in ("stabil", "abweichung")] == [None]


def test_seed_choice_null_or_wrong_form_falls_back_to_a_single_measurement(tmp_path):
    for answer in (jx({"seed_candidate": None}), jx({"seed_candidate": "copy"}), "no json", jx({"seed_candidate": 77}), '["copy"]'):
        d, _ = go(AgentSandbox(), FakeLLM(base_script(choose_seed=[lambda u, a=answer: a, lambda u, a=answer: a, lambda u, a=answer: a])), tmp_path / str(abs(hash(answer))))
        assert "eine Messung, keine Streuung" in d.warnings or d.verdicts[0].urteil == "nicht reproduzierbar (Setup)"


# ---- needs_time_features ------------------------------------------------------------------------------------------------------------

def test_needs_time_features_ends_with_setup_verdict_before_any_training(tmp_path):
    def tf(user):
        return jx({**json.loads(good_answer(user).split("```json\n")[1].split("\n```")[0]),
                   "needs_time_features": {"value": True, "file": "gizmo/model.py", "quote": "return self.lin(x)"}})
    sb = AgentSandbox()
    d, _ = go(sb, FakeLLM(base_script(start_command=[tf])), tmp_path)
    v = d.verdicts[0]
    assert v.urteil == "nicht reproduzierbar (Setup)" and "model needs time features, not supported in v1" in v.explanation
    assert not any(o[0] == "run" and "unshare" in o[2] for o in sb.ops)


def test_needs_time_features_quote_is_asked_again_once(tmp_path):
    def bad(user):
        return jx({**json.loads(good_answer(user).split("```json\n")[1].split("\n```")[0]),
                   "needs_time_features": {"value": True, "file": "gizmo/model.py", "quote": "not literally there"}})
    llm = FakeLLM(base_script(start_command=[bad], start_command_requote=[jx({"needs_time_features": {"value": False}})]))
    d, _ = go(AgentSandbox(), llm, tmp_path)
    assert llm.purposes().count("start_command_requote") == 1 and d.verdicts[0].urteil == "reproduziert"
    llm2 = FakeLLM(base_script(start_command=[bad], start_command_requote=[jx({"needs_time_features": {"value": True, "file": "gizmo/model.py", "quote": "still wrong"}})]))
    d2, _ = go(AgentSandbox(), llm2, tmp_path / "b")
    assert llm2.purposes().count("start_command_requote") == 1                                      # nur einmal; danach "nicht belegt", Lauf geht weiter
    assert d2.verdicts[0].urteil == "reproduziert" and info_of(d2)["agent"]["discarded"][0]["item"] == "needs_time_features"


# ---- Training mit Reparatur ----------------------------------------------------------------------------------------------------

def long_err() -> str:
    return "x" * 3000 + "\nTraceback (most recent call last):\nModuleNotFoundError: No module named 'foo_lib'\n"


def repair(**kw):
    """Reparatur-Antwort: waehlt den Kandidaten h=96 per Nummer (der Prompt nennt ihn in der Liste), Rest wie angegeben."""
    def answer(user):
        base = {"command_candidate": pick(user, "h=96"), "checkpoint_glob": "out/*/weights.bin", "install": []}
        return jx({**base, **kw})
    return answer


def test_training_failure_repair_with_stderr_tail_and_named_package(tmp_path):
    sb = AgentSandbox(train_script=lambda n, cmd: (1, "stdout-only-marker-77", long_err()) if n == 1 else None)
    llm = FakeLLM(base_script(repair_command=[repair(install=["foo-lib"])]))
    d, _ = go(sb, llm, tmp_path)
    assert d.verdicts[0].urteil == "reproduziert"
    prompt = next(c[2] for c in llm.calls if c[0] == "repair_command")
    assert "ModuleNotFoundError: No module named 'foo_lib'" in prompt and "stdout-only-marker-77" not in prompt     # Ende von stderr, keine Zusammenfassung
    assert "x" * 1400 in prompt and "x" * 1600 not in prompt                                                        # 1500 Zeichen
    assert any("foo-lib" in " ".join(cmds) for lbl, cmds in sb.snapshot_cmds.items() if "-r1-" in lbl)             # Snapshots neu mit dem Paket
    assert [a["status"] for a in info_of(d)["agent"]["attempts"]] == ["fehler", "ok"]
    assert info_of(d)["agent"]["repairs"][0]["kind"] == "package_install"


def test_package_not_in_dependency_files_is_rejected_without_a_run(tmp_path):
    sb = AgentSandbox(train_script=lambda n, cmd: (1, "", long_err()) if n == 1 else None)
    llm = FakeLLM(base_script(repair_command=[repair(install=["evil-pkg"]), repair()]))
    d, _ = go(sb, llm, tmp_path)
    assert d.verdicts[0].urteil == "reproduziert"
    repair_prompts = [c[2] for c in llm.calls if c[0] == "repair_command"]
    assert len(repair_prompts) == 2 and na.PACKAGE_SENTENCE.format(pkg="evil-pkg") in repair_prompts[1]
    assert not any("evil-pkg" in " ".join(cmds) for cmds in sb.snapshot_cmds.values())                             # nie installiert
    atts = info_of(d)["agent"]["attempts"]
    assert [a["status"] for a in atts] == ["fehler", "ungueltig", "ok"] and atts[1]["rejected_by"] == ["package"]


def test_repair_may_modify_the_chosen_command_and_p1_lists_the_deviation(tmp_path):
    variant = CMD.replace("--lr 0.01", "--lr 0.02")
    sb = AgentSandbox(train_script=lambda n, cmd: (1, "", "RuntimeError: boom\n") if n == 1 else None)
    llm = FakeLLM(base_script(repair_command=[repair(command=variant)]))
    d, _ = go(sb, llm, tmp_path)
    info = info_of(d)
    assert info["command"] == variant and info["command_documented"] is False
    assert info["command_diff"] == [{"kind": "changed", "flag": "--lr", "doc": "0.01", "command": "0.02"}]
    assert any("Startbefehl weicht von der Dokumentation ab" in w for w in d.warnings)


def test_repair_may_choose_another_candidate_by_number(tmp_path):
    other = lambda user: jx({"command_candidate": pick(user, "--task other"), "checkpoint_glob": "out/*/weights.bin", "install": []})  # noqa: E731
    sb = AgentSandbox(train_script=lambda n, cmd: (1, "", "RuntimeError: boom\n") if n == 1 else None)
    d, _ = go(sb, FakeLLM(base_script(repair_command=[other])), tmp_path)
    info = info_of(d)
    assert info["command"] == "python -u train_gizmo.py --task other --horizon 24" and info["command_source"] == {"file": "scripts/run_demo.sh", "line": 8}
    assert info["command_documented"] is True


def test_repair_command_without_candidate_number_is_rejected(tmp_path):
    invented = lambda user: jx({"command": "python -u train_gizmo.py --task made_up --horizon 96", "checkpoint_glob": "out/*/weights.bin", "install": []})  # noqa: E731
    sb = AgentSandbox(train_script=lambda n, cmd: (1, "", "RuntimeError: boom\n") if n == 1 else None)
    llm = FakeLLM(base_script(repair_command=[invented, repair()]))
    d, _ = go(sb, llm, tmp_path)
    prompts = [c[2] for c in llm.calls if c[0] == "repair_command"]
    assert len(prompts) == 2 and na.NO_CANDIDATE_SENTENCE in prompts[1]                                      # fester Satz in der naechsten Reparatur
    assert not any(o[0] == "run" and "made_up" in o[2] for o in sb.ops)                                      # nie ausgefuehrt
    atts = info_of(d)["agent"]["attempts"]
    assert [a["status"] for a in atts] == ["fehler", "ungueltig", "ok"] and atts[1]["rejected_by"] == ["package"]
    assert info_of(d)["agent"]["repairs"][0]["kind"] == "command_rejected"
    assert d.verdicts[0].urteil == "reproduziert"


def test_repair_variant_with_a_different_script_is_rejected(tmp_path):
    other_script = lambda user: jx({"command_candidate": pick(user, "h=96"), "command": "python -u other_script.py --x 1", "install": []})  # noqa: E731
    sb = AgentSandbox(train_script=lambda n, cmd: (1, "", "RuntimeError: boom\n") if n == 1 else None)
    llm = FakeLLM(base_script(repair_command=[other_script, repair()]))
    go(sb, llm, tmp_path)
    assert na.SAME_SCRIPT_SENTENCE in [c[2] for c in llm.calls if c[0] == "repair_command"][1]
    assert not any(o[0] == "run" and "other_script" in o[2] for o in sb.ops)


def test_forbidden_switch_in_a_repair_variant_is_rejected_by_p1_with_a_fixed_sentence(tmp_path):
    sb = AgentSandbox(train_script=lambda n, cmd: (1, "", "RuntimeError: boom\n") if n == 1 else None)
    llm = FakeLLM(base_script(repair_command=[repair(command=CMD + " --use_all"), repair()]))
    d, _ = go(sb, llm, tmp_path)
    assert d.verdicts[0].urteil == "reproduziert"
    assert not any(o[0] == "run" and "unshare" in o[2] and "--use_all" in o[2] for o in sb.ops)                    # kein Lauf mit diesem Befehl
    assert na.P1_SENTENCE.format(flag="--use_all") in [c[2] for c in llm.calls if c[0] == "repair_command"][1]
    assert info_of(d)["agent"]["attempts"][1]["rejected_by"] == ["P1"]


def test_attempt_limit_gives_setup_verdict_with_all_attempts(tmp_path):
    sb = AgentSandbox(train_script=lambda n, cmd: (1, "", "RuntimeError: boom\n"))
    llm = FakeLLM(base_script(repair_command=[repair()] * 3))
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
    assert row["paper"] == "rlinear_2023" and row["valid"] is True and row["cost_usd_run"] == pytest.approx(0.04) and row["n_llm_calls"] == 4
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


# ---- 38b: robuste Antworten, Rohantworten, unerwartete Fehler -----------------------------------------------------------------------

WRONG_FORMS = [None, "text", 5, 1.5, True, [], ["x"], [{"a": 1}], {}, {"x": 1}, {"file": 5, "quote": 5}, {"file": ["a"], "quote": ["b"]}]
GOOD_PARTS = {"test_loader": {"drop_last": {"candidate": 1, "value": True}, "batch_size": {"switch": "--batch_size"}},
              "needs_time_features": {"value": False, "file": "gizmo/model.py", "quote": None}}


def mutated(key, value, sub=None):
    ans = json.loads(json.dumps(GOOD_PARTS))
    if sub:
        ans[key][sub] = value
    else:
        ans[key] = value
    return ans


def verify_raw(ans_part, only=None, k="good"):
    shown, ll, kk = ctx_for()
    ans = {"command_candidate": kk if k == "good" else k, **ans_part}
    return na.verify_answer(ans, REPO, shown, ll, only=only)


MALFORMED = (
    [("command_candidate", None, v, "command_candidate") for v in WRONG_FORMS + [0, -1, 99, "abc", [1], {"k": 1}]]
    + [("test_loader", "drop_last", v, "test_loader.drop_last") for v in WRONG_FORMS + [{"candidate": 99, "value": True}, {"candidate": "x"}, {"candidate": [1]}, {"value": True},
                                                                                   {"candidate": None, "value": True}]]
    + [("test_loader", "batch_size", v, "test_loader.batch_size") for v in WRONG_FORMS + ["128", 128, {"switch": 5}, {"switch": "bs"}, {"switch": "--nowhere"},
                                                                                    {"file": "gizmo/loader.py"}, {"quote": "batch_size = 8"},
                                                                                    {"file": "gizmo/loader.py", "quote": "batch_size = args.batch_size"},
                                                                                    {"file": ["gizmo/loader.py"], "quote": "batch_size = 8"}]]
)


@pytest.mark.parametrize("key,sub,value,item", MALFORMED)
def test_malformed_field_is_unbelegt_and_never_crashes(key, sub, value, item):
    part = mutated(key, value, sub) if key != "command_candidate" else dict(GOOD_PARTS)
    ver = verify_raw(part, k=value) if key == "command_candidate" else verify_raw(part)
    assert set(ver) == {"command_candidate", "test_loader.drop_last", "test_loader.batch_size", "needs_time_features"}
    assert ver[item]["belegt"] is False, (item, value, ver[item])


def test_malformed_answer_that_is_not_an_object():
    for ans in (None, "text", 5, [], ["command_candidate"]):
        ver = na.verify_answer(ans, REPO, *ctx_for()[:2])
        assert ver["command_candidate"]["belegt"] is False


@pytest.mark.parametrize("value", ["yes", "no", ["true"], 5, {}, {"value": "maybe"}, {"value": True}, {"value": True, "file": "gizmo/model.py", "quote": "not in file"}])
def test_needs_time_features_malformed_is_never_a_yes(value):
    ver = verify_raw(mutated("needs_time_features", value), only={"needs_time_features"})["needs_time_features"]
    assert ver["value"] is False                                  # nie ein "ja" ohne belegtes Zitat


@pytest.mark.parametrize("value", [None, False, {"value": False}, {"value": None}])
def test_needs_time_features_no_forms_are_ok(value):
    assert verify_raw(mutated("needs_time_features", value), only={"needs_time_features"})["needs_time_features"] == {"belegt": True, "reason": "", "value": False}


def test_rescues_that_make_sense():
    assert verify_raw(mutated("needs_time_features", False), only={"needs_time_features"})["needs_time_features"]["belegt"] is True
    v = verify_raw(mutated("test_loader", "128", "batch_size") | {"test_loader": {**GOOD_PARTS["test_loader"], "batch_size": {"switch": "--batch_size"}}},
                   only={"test_loader.batch_size"})
    assert v["test_loader.batch_size"]["belegt"] is True and v["test_loader.batch_size"]["value"] == 128
    assert na._intval("128") == 128 and na._intval(True) is None and na._intval(1.5) is None
    assert verify_raw({}, k="2")["command_candidate"]["belegt"] is True                                 # Nummer als Ziffernfolge


def test_wrong_typed_answers_in_the_flow_do_not_crash(tmp_path):
    """Falsche Typen in der Antwort (wie im echten Lauf mit seed als String): Rueckfall, nie ein Absturz."""
    good = lambda user: json.loads(good_answer(user).split("```json\n")[1].split("\n```")[0])  # noqa: E731
    answers = [
        lambda u: jx({**good(u), "test_loader": "128"}),
        lambda u: jx({**good(u), "test_loader": {"drop_last": "true", "batch_size": 128}}),
        lambda u: jx({**good(u), "checkpoint_glob": 5, "input_len": "abc", "model_file": ["x"], "training_file": None}),
        lambda u: jx({**good(u), "needs_time_features": "no"}),
    ]
    for i, ans in enumerate(answers):
        sb = AgentSandbox()
        sb.own = f"mse:{mse_all_and_droplast(honest(sb.pub['test_inputs']), sb.truth)[0]:.4f}, mae:0.4"
        d, _ = go(sb, FakeLLM(base_script(start_command=[ans], start_command_requote=[jx({})], choose_seed=[seed_answer()])), tmp_path / str(i))
        assert d.mode == "nachlauf"


def test_seed_answer_as_string_does_not_crash(tmp_path):
    reported = lambda u: jx({"seed_candidate": "copy", "other": {"x": 1}})  # noqa: E731
    d, _ = go(AgentSandbox(), FakeLLM(base_script(choose_seed=[reported])), tmp_path)
    assert info_of(d)["agent"]["outcome"] == "ok" and "eine Messung, keine Streuung" in d.warnings
    assert [x["item"] for x in info_of(d)["agent"]["discarded"]] == ["seed"]


def test_repair_command_with_wrong_shapes_does_not_crash(tmp_path):
    sb = AgentSandbox(train_script=lambda n, cmd: (1, "", "RuntimeError: boom\n") if n <= 1 else None)
    wrong = lambda u: jx({"command_candidate": "x", "command": None, "command_source": "x", "install": "foo-lib", "checkpoint_glob": 5})  # noqa: E731
    llm = FakeLLM(base_script(repair_command=[wrong, repair()]))
    d, _ = go(sb, llm, tmp_path)
    assert d.verdicts[0].urteil == "reproduziert"
    assert [a["status"] for a in info_of(d)["agent"]["attempts"]] == ["fehler", "ungueltig", "ok"]          # falsche Form: Versuch verbraucht, nichts erfunden


@pytest.mark.parametrize("answer", ["no list at all", "[1, 2, {\"a\": 1}]", "[\"does/not/exist.py\"]", "{\"files\": [\"README.md\"]}", ""])
def test_unusable_file_selection_falls_back_to_name_based_choice(tmp_path, answer):
    sb = AgentSandbox()
    d, agent = go(sb, FakeLLM(base_script(select_files=[answer], start_command_requote=[jx({})])), tmp_path)
    assert info_of(d)["agent"]["outcome"] == "ok"
    files = set(info_of(d)["agent"]["reading"]["files"])
    assert "README.md" in files and "scripts/run_demo.sh" in files and "gizmo/loader.py" in files   # Kandidaten- und Loader-Dateien immer, unabhaengig von der Auswahl


def test_raw_answers_are_written_before_parsing_and_never_prompted(tmp_path):
    sel = base_script()["select_files"][0]
    sb = AgentSandbox()
    llm = FakeLLM(base_script())
    d, agent = go(sb, llm, tmp_path)
    assert (tmp_path / "llm_select_files_1.txt").read_text() == sel
    raw = (tmp_path / "llm_start_command_1.txt").read_text()
    assert raw.startswith("Reasoning...") and '"command_candidate"' in raw and (tmp_path / "llm_choose_seed_1.txt").exists()
    assert "variant: honest" in (tmp_path / "llm_write_adapter_1.txt").read_text()
    assert not any("llm_" in p["user"] or "llm_" in p["system"] for p in agent.asker.prompts)


def test_unusable_model_answer_ends_with_setup_verdict_not_a_traceback(tmp_path):
    sb = AgentSandbox()
    llm = FakeLLM(base_script(start_command=["I cannot answer in JSON", "still not json {", "nope"]))
    d, agent = go(sb, llm, tmp_path)
    v = d.verdicts[0]
    assert v.urteil == "nicht reproduzierbar (Setup)" and "AgentAnswerError" in v.explanation
    assert all((tmp_path / f"llm_start_command_{i}.txt").exists() for i in (1, 2, 3))      # Rohantworten bleiben nachvollziehbar
    assert (tmp_path / "error.txt").exists() and info_of(d)["agent"]["error"]["type"] == "AgentAnswerError"
    assert d.mode == "nachlauf" and agent.guard.run_net() == pytest.approx(0.01 * 4)         # Kosten der Aufrufe stehen fest


def test_unexpected_error_ends_with_setup_verdict_and_reason(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom in verification")
    monkeypatch.setattr(na, "verify_answer", boom)
    d, agent = go(AgentSandbox(), FakeLLM(base_script()), tmp_path)
    assert d.verdicts[0].urteil == "nicht reproduzierbar (Setup)" and "boom in verification" in d.verdicts[0].explanation
    assert info_of(d)["agent"]["error"]["type"] == "RuntimeError" and "Traceback" in (tmp_path / "error.txt").read_text()
    assert agent.guard.run_net() > 0                                                        # Kosten bis zum Absturz


@pytest.mark.parametrize("exc", [na.BudgetStop("stop"), na.BillingBlocked("billing"), na.BudgetExceeded("budget")])
def test_budget_and_billing_errors_still_propagate(tmp_path, monkeypatch, exc):
    def raiser(*a, **k):
        raise exc
    monkeypatch.setattr(na, "verify_answer", raiser)
    with pytest.raises(type(exc)):
        go(AgentSandbox(), FakeLLM(base_script()), tmp_path)


def test_script_writes_store_row_with_costs_after_an_unexpected_error(tmp_path, monkeypatch):
    m = load_script("scripts/run_nachlauf.py", "run_nachlauf_t3")
    m.ROOT = tmp_path
    sb = AgentSandbox()
    store = tmp_path / "store.jsonl"
    llm = FakeLLM(base_script(), cost=0.01)

    class Spy(na.NachlaufAgent):
        def __init__(self, *a, **k):
            super().__init__(*a, public_loader=lambda: sb.pub, np_rng=np.random.default_rng(0), explore_offsets=[5, 100], log=lambda *_: None, **k)

    m.na = types.SimpleNamespace(**{**{k: getattr(na, k) for k in dir(na) if not k.startswith("__")}, "NachlaufAgent": Spy})
    monkeypatch.setattr(na, "verify_answer", lambda *a, **k: (_ for _ in ()).throw(ValueError("unexpected shape")))
    rc = m.main(["--paper", "rlinear_2023", "--repo", "https://example.invalid/gizmo.git", "--commit", COMMIT, "--store", str(store), "--seeds", "repo"],
                sandbox_factory=lambda: sb, llm_factory=lambda: llm)
    row = json.loads(store.read_text().splitlines()[0])
    assert rc == 1 and row["urteil"] == "nicht reproduzierbar (Setup)" and row["valid"] is False
    assert row["cost_usd_run"] == pytest.approx(0.02) and row["n_llm_calls"] == 2 and (tmp_path / row["run_dir"] / "dossier.json").exists()


# ---- 38d: Seed nur aus nutzbaren Zeilen, Adapter-Vertrag, Nachvollziehbarkeit, Umgebungs-Warnung --------------------------------------------------

READS_ONLY = ("import argparse\nimport random, torch\nfix_seed = 1024\nrandom.seed(fix_seed)\ntorch.manual_seed(fix_seed)\nnp.random.seed(args.seed_value)\n"
              "parser = argparse.ArgumentParser()\nparser.add_argument('--horizon', type=int, default=24)\n")
TWO_USABLE = READS_ONLY + "parser.add_argument('--seed', type=int, default=3)\n"


def test_only_usable_seed_lines_are_offered():
    sc = cand.seed_candidates("run.py", READS_ONLY)
    assert [(c.kind, c.line, c.value) for c in sc] == [("copy", 3, 1024)]                        # Zeilen, die den Seed nur benutzen, fehlen
    assert "[1] (line 3 of run.py) fix_seed = 1024" in sc[0].line_text()
    assert [(c.kind, c.line) for c in cand.seed_candidates("run.py", TWO_USABLE)] == [("copy", 3), ("flag", 9)]
    assert cand.seed_candidates("x.py", "seed = 5\nseed = 5\n") == []                             # nicht eindeutig: nicht nutzbar


def test_exactly_one_usable_seed_line_is_taken_without_an_llm_call(tmp_path):
    repo = {**REPO, "train_gizmo.py": READS_ONLY}
    llm = FakeLLM(base_script(choose_seed=[]))
    d, _ = go(AgentSandbox(repo=repo), llm, tmp_path)
    assert "choose_seed" not in llm.purposes()
    ag = info_of(d)["agent"]
    assert ag["seed_choice"]["how"] == "only_candidate" and ag["seed_choice"]["text"] == "fix_seed = 1024" and ag["discarded"] == []
    assert [a.seed for a in d.attempts if a.status in ("stabil", "abweichung")] == [1024, 0]      # fuenf Seeds waeren moeglich: kein "eine Messung"
    assert "eine Messung, keine Streuung" not in d.warnings


def test_seed_choice_resolution_by_line_number_and_by_variable():
    sc = cand.seed_candidates("run.py", TWO_USABLE)
    assert cand.resolve_seed_choice(1, sc, TWO_USABLE) == (sc[0], "number")
    assert cand.resolve_seed_choice(9, sc, TWO_USABLE) == (sc[1], "line_number")                   # Zeilennummer der Schalterdefinition
    assert cand.resolve_seed_choice(4, sc, TWO_USABLE) == (sc[0], "resolved_variable")             # `random.seed(fix_seed)` liest nur: Zuweisung der Variablen
    assert cand.resolve_seed_choice(5, sc, TWO_USABLE) == (sc[0], "resolved_variable")             # torch.manual_seed(fix_seed)
    assert cand.resolve_seed_choice(6, sc, TWO_USABLE)[0] is None                                  # liest eine andere Variable (args.seed_value): keine Aufloesung
    assert cand.resolve_seed_choice(None, sc, TWO_USABLE) == (None, "none") and cand.resolve_seed_choice(500, sc, TWO_USABLE)[0] is None


def test_model_choosing_a_reading_line_is_resolved_in_the_flow(tmp_path):
    repo = {**REPO, "train_gizmo.py": TWO_USABLE}
    pick_reader = lambda user: jx({"seed_candidate": 4})      # noqa: E731   (Zeilennummer von random.seed(fix_seed))
    d, _ = go(AgentSandbox(repo=repo), FakeLLM(base_script(choose_seed=[pick_reader])), tmp_path)
    ag = info_of(d)["agent"]
    assert ag["seed_choice"]["how"] == "resolved_variable" and ag["seed_choice"]["text"] == "fix_seed = 1024" and "eine Messung, keine Streuung" not in d.warnings


def test_seed_and_command_prompts_say_the_numbers_are_the_bracket_numbers(tmp_path):
    llm = FakeLLM(base_script(choose_seed=[seed_answer()]))
    go(AgentSandbox(), llm, tmp_path)
    seed_prompt = next(c[2] for c in llm.calls if c[0] == "choose_seed")
    assert "square brackets (not a line number" in seed_prompt and "(line 2 of train_gizmo.py)" in seed_prompt
    assert "never line numbers of files" in next(c[2] for c in llm.calls if c[0] == "start_command")


def test_adapter_contract_names_the_working_directory_and_forbids_file_assumptions():
    assert "working directory is the repository root" in na.ADAPTER_CONTRACT and "os.getcwd()" in na.ADAPTER_CONTRACT and "__file__" in na.ADAPTER_CONTRACT
    assert "stored somewhere else" in na.ADAPTER_CONTRACT
    assert f"PYTHONPATH={nl.REPO_DIR}" in nl.predict_cmd() and f"cd {nl.REPO_DIR}" in nl.predict_cmd()      # Repo auch ohne Hilfe des Skripts importierbar


def test_failed_adapter_versions_keep_the_problem_text_in_event_dossier_and_attempt(tmp_path):
    sb = AgentSandbox()
    llm = FakeLLM(base_script(write_adapter=[code("crash")], repair_adapter=[code("p3bad"), code("honest")]))
    d, agent = go(sb, llm, tmp_path)
    ad = info_of(d)["agent"]["adapters"]
    assert [a["problems"] for a in ad] == [["smoke"], ["P3"], []]
    assert "KeyError: 'depth'" in ad[0]["problem_text"] and ad[1]["problem_text"] == na.P3_SENTENCE and ad[2]["problem_text"] == ""
    repair_prompts = [c[2] for c in llm.calls if c[0] == "repair_adapter"]
    assert ad[0]["problem_text"] in repair_prompts[0] and ad[1]["problem_text"] in repair_prompts[1]      # exakt der Text, der in den Prompt ging
    assert ad[1]["detail"]["p3_distance"] is not None and ad[1]["detail"]["p3_compared_with"]           # Zahlen nur im Dossier-Detail
    ev = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines() if '"nachlauf_adapter"' in l]
    assert [e["problem_text"] for e in ev][0].endswith("KeyError: 'depth'\n") or "KeyError: 'depth'" in ev[0]["problem_text"]
    errs = [(a.status, a.error_summary) for a in d.attempts if a.code_hash in {x["hash"] for x in ad[:2]}]
    assert [e[0] for e in errs] == ["fehler", "ungueltig"] and "KeyError: 'depth'" in errs[0][1] and errs[1][1] == na.P3_SENTENCE


def test_smoke_problem_text_is_recorded(tmp_path):
    llm = FakeLLM(base_script(write_adapter=[code("shape")], repair_adapter=[code("honest")]))
    d, _ = go(AgentSandbox(), llm, tmp_path)
    first = info_of(d)["agent"]["adapters"][0]
    assert first["problems"] == ["smoke"] and "expected" in first["problem_text"] and first["problem_text"] in next(c[2] for c in llm.calls if c[0] == "repair_adapter")


def test_every_prompt_is_stored_next_to_the_raw_answer(tmp_path):
    llm = FakeLLM(base_script())
    d, agent = go(AgentSandbox(), llm, tmp_path)
    for p, n in (("select_files", 1), ("start_command", 1), ("choose_seed", 1), ("write_adapter", 1)):
        text = (tmp_path / f"llm_{p}_{n}_prompt.txt").read_text()
        rec = next(x for x in agent.asker.prompts if x["purpose"] == p)
        assert text == f"### system\n{rec['system']}\n\n### user\n{rec['user']}\n" and (tmp_path / f"llm_{p}_{n}.txt").exists()


def test_prompt_is_written_before_the_call_even_if_the_call_crashes(tmp_path):
    class Crash(FakeLLM):
        def complete(self, *a, **k):
            raise RuntimeError("network down")
    d, _ = go(AgentSandbox(), Crash(base_script()), tmp_path)
    assert (tmp_path / "llm_select_files_1_prompt.txt").exists() and d.verdicts[0].urteil == "nicht reproduzierbar (Setup)"


def test_environment_warning_is_compact_and_lists_only_imported_packages(tmp_path):
    reqs = "\n".join(f"pkg{i}==1.{i}" for i in range(40)) + "\nnumpy\nfoo-lib==1.0\ntorch==1.9.0\n"
    repo = {**REPO, "requirements.txt": reqs, "gizmo/model.py": "import torch\nimport foo_lib\n" + REPO["gizmo/model.py"]}
    d, _ = go(AgentSandbox(repo=repo), FakeLLM(base_script()), tmp_path)
    warn = next(w for w in d.warnings if w.startswith("Umgebung weicht"))
    assert len(warn) < 400 and "42 Pakete" in warn and "foo-lib fehlt statt ==1.0" in warn and "torch" in warn and "pkg7" not in warn
    assert len(info_of(d)["env_diff"]) == 42                                                  # volle Liste bleibt im Dossier


def test_env_warning_without_import_information_shows_the_first_packages_only():
    diffs = [{"package": f"p{i}", "required": "==1", "installed": None} for i in range(30)]
    w = nl.env_warning(diffs, None)
    assert "30 Pakete" in w and "p0 fehlt" in w and "und 20 weitere" in w and "p29" not in w
    assert nl.env_warning(diffs, {"p3"}).endswith("p3 fehlt statt ==1")
    assert nl.env_warning(diffs, set()).endswith("keines")
    assert nl.env_warning([{"package": "scikit-learn", "required": "==1.0", "installed": "1.5"}], {"sklearn"}).endswith("scikit-learn 1.5 statt ==1.0")


# ---- 38e: vorgegebene Schreibfunktion im Adapter-Vertrag ----------------------------------------------------------------------------------------

def load_writer():
    ns: dict = {}
    exec(na.WRITE_PREDICTIONS_CODE, ns)      # noqa: S102  (eigene Konstante)
    return ns["write_predictions"]


def test_writer_small_array_roundtrip_header_ids_and_directory(tmp_path):
    write = load_writer()
    pred = np.random.default_rng(1).normal(size=(3, 96, 7)).astype(np.float32)
    path = tmp_path / "new" / "dir" / "predictions.csv"                      # legt das Verzeichnis an
    write(pred, str(path))
    raw = path.read_bytes()
    lines = raw.decode().splitlines()
    assert lines[0] == "id,prediction" and len(lines) == 1 + 3 * 96 * 7 and lines[1].startswith("0_0_0,") and lines[-1].startswith("2_95_6,")
    assert all(re.fullmatch(r"\d+_\d+_\d+,-?[\d.e+-]+", l) for l in lines[1:])
    assert [l.split(",")[0] for l in lines[1:8]] == [f"0_0_{c}" for c in range(7)]          # Kanal laeuft am schnellsten
    back = ind.parse_predictions(raw)
    assert back.shape == (3, 96, 7) and np.allclose(back, pred, rtol=1e-6, atol=1e-7)


def test_writer_output_passes_the_form_check_for_the_smoke_windows(tmp_path):
    from repro_agent.harness.smoke import check_smoke_output
    write = load_writer()
    task = nl.etth1_task()
    write(np.random.default_rng(2).normal(size=(8, 96, 7)), str(tmp_path / "p.csv"))
    assert check_smoke_output((tmp_path / "p.csv").read_bytes(), task.smoke_ids, task.smoke_dims, task.smoke_dim_names) is None


def test_writer_accepts_float32_lists_and_rejects_wrong_rank(tmp_path):
    write = load_writer()
    write(np.zeros((1, 96, 7), dtype=np.float32) + 0.5, str(tmp_path / "a.csv"))
    assert (tmp_path / "a.csv").read_text().splitlines()[1] == "0_0_0,0.5"
    with pytest.raises(ValueError):
        write(np.zeros((96, 7)), str(tmp_path / "b.csv"))
    write(np.zeros((0, 96, 7)), str(tmp_path / "c.csv"))
    assert (tmp_path / "c.csv").read_text() == "id,prediction\n"


def test_writer_full_array_roundtrip_and_runtime(tmp_path, capsys):
    import time
    write = load_writer()
    pred = np.random.default_rng(3).normal(size=(2785, 96, 7)).astype(np.float32)        # 1 871 520 Zeilen
    path = tmp_path / "full.csv"
    t0 = time.perf_counter()
    write(pred, str(path))
    secs = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n[write_predictions] full array 2785x96x7 ({pred.size} rows): {secs:.2f} s, {path.stat().st_size / 1e6:.1f} MB")
    assert secs < 30
    back = ind.parse_predictions(path.read_bytes())
    assert back.shape == (2785, 96, 7) and np.allclose(back, pred, rtol=1e-6, atol=1e-7) and not np.isnan(back).any()


def test_writer_passes_the_static_check_and_has_no_numpy_string_arithmetic():
    assert na.find_violations(na.WRITE_PREDICTIONS_CODE) == []
    code_ = na.WRITE_PREDICTIONS_CODE
    assert "astype(str)" not in code_ and "np.char" not in code_ and "def write_predictions(pred, path=" in code_


def test_contract_gives_the_writer_verbatim_and_forbids_own_writing_logic():
    c = na.ADAPTER_CONTRACT
    assert na.WRITE_PREDICTIONS_CODE.strip() in c                                    # woertlich im Vertrag
    assert "UNCHANGED" in c and "Do NOT write your own code for the file" in c and "no `+` on string arrays" in c
    assert "Write it efficiently" not in c and "numpy to text" not in c               # der alte, missverstaendliche Satz ist weg


def test_adapter_prompts_contain_the_writer(tmp_path):
    llm = FakeLLM(base_script(write_adapter=[code("crash")], repair_adapter=[code("honest")]))
    go(AgentSandbox(), llm, tmp_path)
    for purpose in ("write_adapter", "repair_adapter"):
        assert na.WRITE_PREDICTIONS_CODE.strip() in next(c[2] for c in llm.calls if c[0] == purpose)


def test_hand_written_adapters_use_an_equivalent_format():
    """Die Hand-Adapter (M1) schreiben dasselbe Format; die Schreibfunktion des Vertrags erzeugt dieselben Zeilen wie der Task-Rueckleser erwartet."""
    pred = np.arange(2 * 96 * 7, dtype=np.float64).reshape(2, 96, 7) / 7.0
    ns: dict = {}
    exec(na.WRITE_PREDICTIONS_CODE, ns)
    import tempfile
    p = Path(tempfile.mkdtemp()) / "p.csv"
    ns["write_predictions"](pred, str(p))
    assert np.allclose(ind.parse_predictions(p.read_bytes()), pred, rtol=1e-7)
