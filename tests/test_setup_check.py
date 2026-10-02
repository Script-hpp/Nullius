"""Basis-Snapshot: Exit-Code jedes Setup-Befehls, eine Wiederholung, SandboxSetupError, Daten-Check als letzter Befehl, Infrastrukturfehler im Loop. Attrappen, keine Sandbox."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from repro_agent import loop as loop_mod
from repro_agent.sandbox import local as local_mod
from repro_agent.sandbox.base import SandboxSetupError
from repro_agent.sandbox.nebius import NebiusSandbox
from repro_agent.tasks import TASKS, cora_task, data_check_cmd, etth1_task
from test_catalog import cora_entry, evidence, tavily_orch


class FakeImage:
    """Wie ein contree-Image: run().wait() liefert ein NEUES Image mit .result (exit_code, stdout, stderr)."""

    def __init__(self, log, script, cmd=None, result=None):
        self.log, self.script, self.cmd, self._result = log, script, cmd, result

    def run(self, shell, disposable=False):
        return FakeImage(self.log, self.script, cmd=shell)

    def wait(self):
        code, out, err = self.script(self.cmd)
        self.log.append((self.cmd, code))
        self._result = SimpleNamespace(exit_code=code, stdout=out, stderr=err)
        return self

    @property
    def result(self):
        return self._result

    def tag_as(self, tag):
        self.log.append(("TAG", tag))
        return self


def nebius(script):
    log = []
    sb = NebiusSandbox.__new__(NebiusSandbox)
    sb._sessions = {}
    sb._client = SimpleNamespace(images=SimpleNamespace(oci=lambda image: FakeImage(log, script)))
    return sb, log


def test_all_commands_ok_gives_a_snapshot_tagged_after_the_last_command():
    sb, log = nebius(lambda cmd: (0, "", ""))
    snap = sb.base_snapshot("img", ["a", "b"], label="x")
    assert snap == "repro-snapshot-x" and [c for c, _ in log] == ["a", "b", "TAG"]


def test_a_failing_command_is_retried_once_and_then_the_setup_goes_on():
    attempts = {"dl": 0}

    def script(cmd):
        if cmd == "download":
            attempts["dl"] += 1
            return (1, "", "curl: network unreachable") if attempts["dl"] == 1 else (0, "", "")
        return (0, "", "")

    sb, log = nebius(script)
    assert sb.base_snapshot("img", ["pip", "download", "check"], label="x") == "repro-snapshot-x"
    assert [c for c, _ in log if c == "download"] == ["download", "download"] and log[-1][0] == "TAG"


def test_two_failures_raise_setup_error_with_the_end_of_stderr_and_no_snapshot():
    long_err = "x" * 3000 + "FileNotFoundError: cora.npz"
    sb, log = nebius(lambda cmd: (2, "", long_err) if cmd == "download" else (0, "", ""))
    with pytest.raises(SandboxSetupError) as e:
        sb.base_snapshot("img", ["pip", "download", "never"], label="x")
    msg = str(e.value)
    assert "Exit-Code 2" in msg and "download" in msg and msg.endswith("FileNotFoundError: cora.npz") and len(msg) < 1900
    assert [c for c, _ in log] == ["pip", "download", "download"]     # "never" and the tag were not reached


def test_an_exception_of_the_sdk_counts_as_a_failure_too():
    class Boom(FakeImage):
        def run(self, shell, disposable=False):
            return Boom(self.log, self.script, cmd=shell)

        def wait(self):
            raise RuntimeError("image state FAILED")
    sb, _ = nebius(lambda cmd: (0, "", ""))
    sb._client.images.oci = lambda image: Boom([], None, cmd="c")
    with pytest.raises(SandboxSetupError, match="RuntimeError"):
        sb.base_snapshot("img", ["a"], label="x")


def test_local_sandbox_checks_exit_codes_retries_once_and_raises(monkeypatch):
    calls = []

    def fake_run(args, **kw):
        calls.append(args)
        if args[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(args, 7, stdout="", stderr="no such file /data/cora_public.npz")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(local_mod.subprocess, "run", fake_run)
    with pytest.raises(SandboxSetupError, match="Exit-Code 7"):
        local_mod.LocalSandbox().base_snapshot("img", ["python prep.py"], label="x")
    assert len([c for c in calls if c[:2] == ["docker", "exec"]]) == 2                    # one retry
    assert calls[-1][:3] == ["docker", "rm", "-f"] and not [c for c in calls if c[:2] == ["docker", "commit"]]   # container removed, nothing committed


def test_data_check_is_the_last_setup_command_of_each_task_and_fails_on_a_missing_file(tmp_path):
    assert cora_task().setup_cmds[-1] == data_check_cmd("/data/cora_public.npz")
    assert etth1_task().setup_cmds[-1] == data_check_cmd("/data/etth1_public.npz")
    assert all(TASKS[k]().setup_cmds for k in TASKS)
    f = tmp_path / "data.npz"
    miss = subprocess.run(data_check_cmd(str(f)), shell=True, capture_output=True, text=True)
    assert miss.returncode == 1 and "fehlt oder ist leer" in miss.stderr
    f.write_bytes(b"")
    assert subprocess.run(data_check_cmd(str(f)), shell=True).returncode == 1            # empty file counts as missing
    f.write_bytes(b"x")
    assert subprocess.run(data_check_cmd(str(f)), shell=True).returncode == 0


def test_loop_ends_a_run_with_a_failed_setup_as_infrastructure_error_not_as_a_verdict(tmp_path, monkeypatch):
    e = cora_entry()
    o = tavily_orch(tmp_path, monkeypatch, [e], evidence(0.8292))
    emitted = []
    real = o._emit
    o._emit = lambda type, **kw: (emitted.append((type, kw)), real(type, **kw))[1]

    def broken(*a, **k):
        raise SandboxSetupError("Setup-Befehl fehlgeschlagen (Exit-Code 1): python /opt/prep/cora_data.py\nFileNotFoundError: planetoid download")

    o.sandbox.base_snapshot = broken
    with pytest.raises(SandboxSetupError):
        o.run(e["id"], source="catalog")
    (ev,) = [kw for t, kw in emitted if t == "infrastructure_error"]
    assert ev["stage"] == "sandbox_setup" and "planetoid download" in ev["message"]
    assert emitted[-1][0] == "run_end" and emitted[-1][1]["error"] == "sandbox_setup"
    saved = sorted((tmp_path / "runs" / "disc_2511_08094").iterdir())[-1]
    d = json.loads((saved / "dossier.json").read_text())
    assert d["verdicts"] == [] and d["attempts"] == []                                    # no verdict at all, in particular not "nicht reproduzierbar (Setup)"
    assert any("Sandbox-Setup fehlgeschlagen (Infrastruktur" in w for w in d["warnings"])
    assert "nicht reproduzierbar" not in (saved / "zeugnis.md").read_text()


def test_a_failing_repo_clone_is_a_warning_and_the_run_goes_on_without_the_repo(tmp_path, monkeypatch):
    e = cora_entry()
    o = tavily_orch(tmp_path, monkeypatch, [e], evidence(0.8292))
    monkeypatch.setattr(loop_mod, "find_repo_url", lambda t: "https://github.com/dead/link")
    monkeypatch.setattr(loop_mod, "fetch_readme", lambda *a, **k: "")

    def dead(*a, **k):
        raise SandboxSetupError("Setup-Befehl fehlgeschlagen (Exit-Code 128): git clone\nfatal: repository not found")

    monkeypatch.setattr(loop_mod, "read_repo", dead)
    d = o.run(e["id"], source="catalog")
    assert d.repo is None and d.verdicts
    assert any("Repo nicht gelesen" in w and "Exit-Code 128" in w for w in d.warnings)
