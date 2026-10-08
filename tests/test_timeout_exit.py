"""F2 (Timeout mit Stack und Ausgabe, Pflicht-Hinweis danach), F3 (Exit-Code im Fehlertext, OOM bei 137, leerer Text nie leer), F8 (Versuchsprotokoll)."""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "review"))

from repro_agent.loop import (
    OOM_HINT, STAY_FAST_NOTE, TIMEOUT_REPAIR_MESSAGE, describe_exit, repair_hint, timeout_report)
from repro_agent.sandbox.base import RunResult
from repro_agent.tasks import cora_task
from test_loop_fakes import FakeSandbox, make_orch, run_claim, train_cmds

STACK = ('Fatal Python error: Aborted\n\nCurrent thread 0x00007f (most recent call first):\n'
         '  File "/work/train.py", line 5 in attention\n  File "/work/train.py", line 9 in <module>\n')
LOOP_CODE = "import torch\n\ndef attention(x):\n    out = []\n    for i in range(len(x)):\n        out.append(x[i])\n    return out\n\nattention([1])\n"


def test_describe_exit_puts_exit_code_last_and_never_returns_empty():
    assert describe_exit(1, "Traceback...\nValueError: boom\n") == "Traceback...\nValueError: boom\n(exit code 1)"
    empty = describe_exit(137, "")
    assert "printed nothing" in empty and empty.endswith("(exit code 137)")
    assert describe_exit(2, "   \n").startswith("The script printed nothing")


def test_repair_hint_oom_by_exit_code_without_text():
    assert repair_hint(describe_exit(137, "")) == OOM_HINT
    assert repair_hint(describe_exit(-9, "")) == OOM_HINT
    assert repair_hint(describe_exit(137, "Traceback (most recent call last):\n  File x\nRuntimeError: boom")) == OOM_HINT   # Exit-Code gilt auch mit Text
    assert repair_hint(describe_exit(1, "ValueError: boom")) == ""
    assert repair_hint("exit code 137 was mentioned in the middle\nValueError: x") == ""   # nur das Ende zaehlt
    # bestehende Regeln unveraendert
    assert repair_hint("Smoke run (SMOKE=1) crashed:\nKilled") == OOM_HINT
    assert repair_hint("IndexError: index 1708 is out of bounds") == ""


def test_timeout_report_has_message_stack_source_line_and_output():
    res = {"stack": STACK, "stdout_tail": "epoch 3 loss 1.2\nepoch 4 loss 1.1\n"}
    text = timeout_report(res, LOOP_CODE, 600)
    assert text.startswith(TIMEOUT_REPAIR_MESSAGE.format(limit=600))
    assert "in `attention` at train.py line 5: `for i in range(len(x)):`" in text
    assert "Python stack at the time limit" in text and 'line 9 in <module>' in text
    assert "Last output of the script before the limit:\nepoch 3 loss 1.2" in text


def test_timeout_report_without_stack_or_output():
    text = timeout_report({}, "x = 1\n", 600)
    assert text.startswith(TIMEOUT_REPAIR_MESSAGE.format(limit=600)) and "printed nothing before the limit" in text and "stack" not in text.lower()


def test_timeout_report_ignores_frames_outside_train_py():
    stack = 'Fatal Python error: Aborted\n\nCurrent thread 0x1 (most recent call first):\n  File "/usr/lib/python3/torch/nn.py", line 7 in forward\n'
    assert "at train.py line" not in timeout_report({"stack": stack}, "x = 1\n", 600)


class TimeoutSandbox(FakeSandbox):
    """Training: erster Lauf mit weichem Timeout (Exit 124 + Stack), danach Fehler oder Erfolg."""

    def __init__(self, train_results):
        super().__init__()
        self.train_results = list(train_results)

    def run(self, branch, cmd, timeout_s):
        if "train.py" in cmd and "SMOKE=1" not in cmd and "import coverage" not in cmd:
            with self.lock:
                self.cmds.append((branch, cmd))
            self.seed[branch] = 0
            return self.train_results.pop(0) if self.train_results else RunResult(0, "", "", False)
        return super().run(branch, cmd, timeout_s)


def _product(o):
    o.graph_repair = True
    o.train_timeout_s = 600
    return o


def test_training_command_uses_abrt_timeout_before_the_sandbox_limit(tmp_path):
    sb = FakeSandbox()
    _product(make_orch(sb, seeds=1)[0])
    o, _ = make_orch(sb, seeds=1)
    o.train_timeout_s = 600
    run_claim(o, tmp_path)
    cmd = train_cmds(sb)[0]
    assert "PYTHONFAULTHANDLER=1" in cmd and "PYTHONUNBUFFERED=1" in cmd and "unshare -n timeout -s ABRT 585 python train.py" in cmd


def test_soft_timeout_becomes_timeout_status_with_stack(tmp_path):
    soft = RunResult(124, "epoch 1\n", STACK, False)
    sb = TimeoutSandbox([soft, soft])
    o, calls = make_orch(sb, seeds=1, max_attempts=2, write=LOOP_CODE, repairs=[LOOP_CODE + "# fast\n"])
    _product(o)
    d = run_claim(o, tmp_path)
    assert [a.status for a in d.attempts] == ["timeout", "timeout"]
    assert len(calls["repair"]) == 1
    msg = calls["repair"][0]
    assert msg.startswith(TIMEOUT_REPAIR_MESSAGE.format(limit=600))
    assert "train.py line 5" in msg and "epoch 1" in msg


def test_sticky_stay_fast_note_after_a_timeout(tmp_path):
    soft = RunResult(124, "", STACK, False)
    crash = RunResult(1, "", "RuntimeError: shapes", False)
    sb = TimeoutSandbox([soft, crash, RunResult(0, "", "", False)])
    o, calls = make_orch(sb, seeds=1, max_attempts=3, write=LOOP_CODE, repairs=[LOOP_CODE + "# a\n", LOOP_CODE + "# b\n"])
    _product(o)
    run_claim(o, tmp_path)
    assert STAY_FAST_NOTE not in calls["repair"][0]             # die Reparatur nach dem Timeout traegt die Timeout-Meldung selbst
    assert "RuntimeError: shapes" in calls["repair"][1] and STAY_FAST_NOTE in calls["repair"][1]   # die naechste Reparatur bekommt den Pflicht-Hinweis


def test_no_sticky_note_without_timeout_and_not_in_plain_path(tmp_path):
    crash = RunResult(1, "", "RuntimeError: shapes", False)
    sb = TimeoutSandbox([crash, RunResult(0, "", "", False)])
    o, calls = make_orch(sb, seeds=1, max_attempts=2, write=LOOP_CODE, repairs=[LOOP_CODE + "# a\n"])
    _product(o)
    run_claim(o, tmp_path)
    assert STAY_FAST_NOTE not in calls["repair"][0] and calls["repair"][0].rstrip().endswith("(exit code 1)") or "(exit code 1)" in calls["repair"][0]


def test_plain_path_timeout_has_no_repair(tmp_path):
    soft = RunResult(124, "", STACK, False)
    sb = TimeoutSandbox([soft])
    o, calls = make_orch(sb, seeds=1, max_attempts=3, write=LOOP_CODE)
    run_claim(o, tmp_path)
    assert calls["repair"] == []   # ohne graph_repair bleibt ein Timeout ohne Reparatur (wie bisher)


def test_hard_sandbox_timeout_still_a_timeout(tmp_path):
    sb = TimeoutSandbox([RunResult(-1, "", "", True)])
    o, calls = make_orch(sb, seeds=1, max_attempts=2, write=LOOP_CODE, repairs=[LOOP_CODE + "# fast\n"])
    _product(o)
    d = run_claim(o, tmp_path)
    assert d.attempts[0].status == "timeout" and "printed nothing" in calls["repair"][0]


def test_run_result_exit_code_reaches_attempt_log(tmp_path):
    """F8: _attempt-Ergebnis traegt exit_code; der Gate-Wrapper schreibt eine Zeile je Versuch."""
    import cg_gate
    sb = SimpleNamespace(begin_attempt=lambda: None, release_attempt=lambda: None, times={})
    log = tmp_path / "attempts.jsonl"
    o = cg_gate.make_orchestrator(sb, attempt_log=log)

    def fake_attempt(snapshot, code, seed, task, metric, **kw):
        return "b1", {"status": "fehler", "detail": "x" * 3000 + "\nValueError: boom\n(exit code 1)", "exit_code": 1}

    o._attempt.impl[0] = fake_attempt
    o._attempt("snap", "print(1)\n", 3, None, "accuracy")
    row = json.loads(log.read_text().splitlines()[0])
    assert row["seed"] == 3 and row["status"] == "fehler" and row["exit_code"] == 1 and len(row["detail"]) == 1500 and row["detail"].endswith("(exit code 1)")
    assert len(row["code_hash"]) == 12 and "ts" in row and "wall_s" in row


def test_attempt_log_records_exception(tmp_path):
    import cg_gate
    log = tmp_path / "a.jsonl"
    cg_gate.log_attempt(log, threading.Lock(), "code", 0, 1.23, None, RuntimeError("sandbox down"))
    row = json.loads(log.read_text())
    assert row["status"] == "exception" and "RuntimeError: sandbox down" in row["detail"]
