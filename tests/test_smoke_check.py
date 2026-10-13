import pytest

from repro_agent.harness.smoke import check_smoke_output
from repro_agent.loop import Orchestrator
from repro_agent.sandbox.base import RunResult
from repro_agent.tasks import etth1_task, cora_task

IDS = tuple(f"{w}_{s}_{c}" for w in range(2) for s in range(3) for c in range(4))
DIMS = (2, 3, 4)
NAMES = ("windows", "steps", "channels")


def csv_for(ids, val="0.5"):
    return ("id,prediction\n" + "\n".join(f"{i},{val}" for i in ids) + "\n").encode()


def test_correct_form_passes():
    assert check_smoke_output(csv_for(IDS), IDS, DIMS, NAMES) is None


def test_transposed_shape_gives_precise_message():
    swapped = tuple(f"{w}_{c}_{s}" for w in range(2) for s in range(3) for c in range(4))  # (2, 4, 3)
    msg = check_smoke_output(csv_for(swapped), IDS, DIMS, NAMES)
    assert "(2, 3, 4)" in msg and "(2, 4, 3)" in msg


def test_wrong_count_header_and_values():
    assert "expected 24 unique ids, got 23" in check_smoke_output(csv_for(IDS[:-1]), IDS, DIMS, NAMES)
    assert "header" in check_smoke_output(b"a,b\n1,2\n", IDS, DIMS, NAMES)
    assert "non-finite" in check_smoke_output(csv_for(IDS, "nan"), IDS, DIMS, NAMES)
    assert "non-numeric" in check_smoke_output(csv_for(IDS, "abc"), IDS, DIMS, NAMES)
    assert "expected 24" in check_smoke_output(csv_for(IDS + IDS[:1]), IDS, DIMS, NAMES)   # Duplikat


def test_task_smoke_definitions():
    t = etth1_task()
    assert len(t.smoke_ids) == 8 * 96 * 7 and t.smoke_dims == (8, 96, 7) and "SMOKE=1" in t.code_notes
    c = cora_task()
    assert c.smoke_ids_cmd and "SMOKE=1" in c.code_notes


class FakeSandbox:
    def __init__(self, results, out=b""):
        self.results, self.out, self.cmds = list(results), out, []

    def run(self, branch, cmd, timeout_s):
        self.cmds.append(cmd)
        return self.results.pop(0)

    def get_file(self, branch, path):
        return self.out


def _orch(sb):
    o = Orchestrator.__new__(Orchestrator)
    o.sandbox, o._smoke_ok = sb, set()
    return o


def ok(stdout=""):
    return RunResult(0, stdout, "", False)


def _task():
    from dataclasses import replace
    return replace(etth1_task(), smoke_ids=IDS, smoke_dims=DIMS, smoke_dim_names=NAMES)


def test_smoke_crash_goes_to_repair_message_and_not_cached():
    sb = FakeSandbox([RunResult(1, "", "RuntimeError: mat1 and mat2 shapes cannot be multiplied", False)])
    o = _orch(sb)
    msg = o._smoke_check("b", "code", _task())
    assert "crashed" in msg and "mat1 and mat2" in msg and not o._smoke_ok


def test_smoke_timeout_and_pass_is_cached():
    o = _orch(FakeSandbox([RunResult(-1, "", "", True)]))
    assert "ignores SMOKE=1" in o._smoke_check("b", "code", _task())
    sb = FakeSandbox([ok(), ok()], out=csv_for(IDS))
    o = _orch(sb)
    assert o._smoke_check("b", "code", _task()) is None
    assert o._smoke_check("b", "code", _task()) is None      # zweites Mal ohne Lauf
    assert len(sb.cmds) == 2                                 # Smoke + rm, nicht wiederholt


def test_smoke_cache_is_scoped_to_task_and_seed():
    from dataclasses import replace
    sb = FakeSandbox([ok(), ok(), ok(), ok(), ok(), ok()], out=csv_for(IDS))
    o = _orch(sb)
    t = _task()
    assert o._smoke_check('b', 'code', t, 0) is None
    assert o._smoke_check('b', 'code', t, 1) is None
    assert o._smoke_check('b', 'code', replace(t, data_description=t.data_description+'different task'), 1) is None
    assert len(sb.cmds)==6


def test_smoke_wrong_shape_message():
    swapped = tuple(f"{w}_{c}_{s}" for w in range(2) for s in range(3) for c in range(4))
    o = _orch(FakeSandbox([ok()], out=csv_for(swapped)))
    msg = o._smoke_check("b", "code", _task())
    assert "wrong form" in msg and "(2, 4, 3)" in msg


def test_task_without_smoke_is_skipped():
    from dataclasses import replace
    sb = FakeSandbox([])
    assert _orch(sb)._smoke_check("b", "code", replace(etth1_task(), smoke_ids=())) is None
    assert sb.cmds == []
