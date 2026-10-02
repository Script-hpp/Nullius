"""Ablauf von scripts/prestudy_nachlauf/sandbox_run.py mit Fake-Sandbox (kein Sandbox-Aufruf)."""
import importlib.util
import json
from pathlib import Path

import pytest

from repro_agent.sandbox.base import RunResult

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prestudy_nachlauf" / "sandbox_run.py"
spec = importlib.util.spec_from_file_location("sandbox_run", SCRIPT)
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)

EVAL_OUT = json.dumps({"status": "ok", "metrics": {"mse": 0.37, "mae": 0.39}, "metrics_droplast": {"128": {"mse": 0.366}}})
MAPPING = json.dumps({"ok": True, "max_abs_diff": 1e-14})


class FakeSandbox:
    def __init__(self, git_status=""):
        self.ops = []
        self.git_status = git_status
        self.n = 0

    def base_snapshot(self, image, setup_cmds, label=None):
        self.ops.append(("snapshot", label))
        return "snap"

    def branch(self, snapshot):
        self.n += 1
        self.ops.append(("branch", snapshot))
        return f"b{self.n}"

    def put_files(self, branch, files):
        self.ops.append(("put_files", branch, sorted(files)))

    def run(self, branch, cmd, timeout_s):
        self.ops.append(("run", branch, cmd, timeout_s))
        out = ""
        if "__DIFFLINES__" in cmd:
            out = "__DIFFLINES__ 2\n1\n"
        elif "adapter_" in cmd:
            out = "mapping ok\n__WALL_MS__ 6000\n"
        elif "unshare -n python" in cmd:
            out = "mse:0.3660, mae:0.3909\n__WALL_MS__ 25000\n"
        elif cmd.startswith("cat /output/mapping.json"):
            out = MAPPING
        elif "__GIT_STATUS__" in cmd:
            out = f"__GIT_STATUS__[{self.git_status}]\n__GIT_DIFFSTAT__[]\n"
        elif "etth1_eval.py" in cmd:
            out = EVAL_OUT + "\n"
        return RunResult(0, out, "", False)


def kinds(sb):
    return [o[0] for o in sb.ops]


def test_order_and_isolation(tmp_path):
    sb = FakeSandbox()
    res = sr.run_unit(sb, "snap", "rlinear", [1024, 0], tmp_path)
    assert [r["status"] for r in res] == ["ok", "ok"]
    # je Seed ein eigener Branch (plus Probe-Branch davor), streng nacheinander
    assert kinds(sb).count("branch") == 3
    runs = [o for o in sb.ops if o[0] == "run" and o[1] == "b2"]       # Repo-Seed
    cmds = [o[2] for o in runs]
    i_train = next(i for i, c in enumerate(cmds) if "unshare -n python" in c and "run.py" in c)
    i_adapt = next(i for i, c in enumerate(cmds) if "adapter_rtsf.py" in c)
    i_git = next(i for i, c in enumerate(cmds) if "git status --short" in c)
    i_eval = next(i for i, c in enumerate(cmds) if "etth1_eval.py" in c)
    assert i_train < i_adapt < i_git < i_eval
    assert "unshare -n" in cmds[i_adapt] and "unshare" not in cmds[i_eval]
    assert all(o[3] == 600 for o in runs if "unshare" in o[2])
    # Harness wird erst nach dem Git-Check kopiert, nie vor dem Training
    pos = lambda pred: next(n for n, o in enumerate(sb.ops) if pred(o))  # noqa: E731
    p_train = pos(lambda o: o[0] == "run" and o[1] == "b2" and "unshare -n python" in o[2])
    p_git = pos(lambda o: o[0] == "run" and o[1] == "b2" and "git status --short" in o[2])
    p_put = pos(lambda o: o[0] == "put_files" and o[1] == "b2" and any("etth1_eval.py" in p for p in o[2]))
    p_eval = pos(lambda o: o[0] == "run" and o[1] == "b2" and "etth1_eval.py" in o[2])
    assert p_train < p_git < p_put < p_eval
    assert sum(1 for o in sb.ops if o[0] == "put_files" and o[1] == "b2" and any("etth1_eval.py" in p for p in o[2])) == 1
    first_put = next(o for o in sb.ops if o[0] == "put_files" and o[1] == "b2")
    assert not any("etth1_eval.py" in p for p in first_put[2])
    assert any(p.endswith("adapter_rtsf.py") for p in first_put[2])


def test_seed_copy_only_for_non_repo_seed(tmp_path):
    sb = FakeSandbox()
    sr.run_unit(sb, "snap", "dlinear", [2021, 3], tmp_path)
    trains = [o[2] for o in sb.ops if o[0] == "run" and "unshare -n python" in o[2]]
    assert " run_longExp.py " in trains[0] and "/opt/seed" not in trains[0]
    assert "/opt/seed/ltsf_run_seed3.py" in trains[1]


def test_json_line_and_report(tmp_path):
    sb = FakeSandbox()
    res = sr.run_unit(sb, "snap", "rlinear", [1024], tmp_path)[0]
    assert res["repo_mse"] == 0.3660 and res["harness_mse_all"] == 0.37 and res["train_wall_s"] == 25.0
    assert res["metrics_droplast"]["128"]["mse"] == 0.366 and res["mapping"]["ok"] and res["train_exit"] == 0
    assert (tmp_path / "nachlauf_prestudy_sandbox_rlinear.txt").read_text().count("JSON") == 1


def test_changed_repo_blocks_harness(tmp_path):
    sb = FakeSandbox(git_status=" M run.py")
    res = sr.run_unit(sb, "snap", "rlinear", [1024], tmp_path)[0]
    assert res["status"] == "repo_changed"
    assert not any("etth1_eval.py" in o[2] for o in sb.ops if o[0] == "run")
    assert not any(o[0] == "put_files" and any("etth1_eval.py" in p for p in o[2]) for o in sb.ops)


def test_dry_run_creates_no_client(capsys):
    def boom():
        raise AssertionError("Sandbox-Client darf im Dry-Run nicht entstehen")

    assert sr.main(["--unit", "rlinear", "--dry-run", "--seeds", "repo,0"], sandbox_factory=boom) == 0
    out = capsys.readouterr().out
    assert "unshare -n" in out and "etth1_eval.py" in out and "git clone" in out


def test_setup_has_sha_check_and_commits():
    joined = "\n".join(sr.setup_cmds())
    assert sr.ETTH1_SHA256 in joined and "sha256sum -c" in joined and "numpy<2" in joined
    for _, commit in sr.REPOS.values():
        assert commit in joined


def test_seed_copy_files_exist():
    for u in sr.UNITS.values():
        for s in (0, 1, 2, 3):
            assert sr.seed_files(u, s)
