"""repair_result: nach jeder Fehler-Reparatur nennt das Ereignis den Hash des neuen Codes (die Evidence-Ansicht erzaehlt damit die Kette). Attrappen, kein Netz."""

from repro_agent.sandbox.base import RunResult
from test_graph_repair import V1, V2, h, run, setup, types
from test_loop_fakes import FakeSandbox


def test_repair_result_names_the_fixed_code_and_follows_the_repair(tmp_path):
    sb = FakeSandbox()
    sb.smoke_results = [RunResult(1, "", "RuntimeError: mat1 and mat2 shapes cannot be multiplied", False)]
    o, calls, events = setup([0.81], [0.5], repairs=[V2], sandbox_cls=lambda **kw: sb)
    run(o, tmp_path)
    names = [t for t, _ in events]
    rep, res = types(events, "repair"), types(events, "repair_result")
    assert len(rep) == 1 and len(res) == 1
    assert res[0]["from_hash"] == rep[0]["from_hash"] == h(V1) and res[0]["to_hash"] == h(V2)
    assert names.index("repair") < names.index("repair_result") < names.index("attempt", names.index("repair"))


def test_no_repair_result_without_a_repair(tmp_path):
    o, calls, events = setup([0.81], [0.5])
    run(o, tmp_path)
    assert types(events, "repair") == [] and types(events, "repair_result") == []
