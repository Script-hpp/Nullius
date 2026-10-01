"""Produktpfad (graph_repair): Code-Extraktion tolerant, Timeout wird repariert, fehlender Code fuehrt zu Neuschreiben.
Ohne Flag bleibt das Verhalten von Benchmark v1/v2 unveraendert."""

import threading
import types as pytypes

import pytest

from repro_agent.loop import Orchestrator, extract_code

CODE = "import numpy as np\nprint(1)\n"


def test_strict_extraction_unchanged():
    assert extract_code(f"x\n```python\n{CODE}```") == CODE
    for text in (f"```py\n{CODE}```", f"```python\n{CODE}", CODE):
        with pytest.raises(ValueError):
            extract_code(text)


@pytest.mark.parametrize("text", [f"```py\n{CODE}```", f"```Python\n{CODE}```", f"Here:\n```python\n{CODE}", CODE])
def test_lenient_extraction(text):
    assert extract_code(text, lenient=True).strip() == CODE.strip()


def test_lenient_rejects_prose_and_non_python():
    for text in ("I cannot do that.", "```json\n{\"a\": [1,\n```"):
        with pytest.raises(ValueError):
            extract_code(text, lenient=True)


def _orch(flag, results, repair_raises=False):
    o = Orchestrator.__new__(Orchestrator)
    o.graph_repair = flag
    o.settings = pytypes.SimpleNamespace(max_attempts_per_claim=4)
    o._lock, o.events, o._cost_base = threading.Lock(), None, 0.0
    o.llm = pytypes.SimpleNamespace(total_cost_usd=0.0)
    o.calls = []
    it = iter(results)
    o._attempt = lambda *a, **k: ("b", next(it))
    o._announce_code = lambda *a, **k: None
    o.triage = lambda t: t

    def repair(plan, task, code, summary):
        o.calls.append(("repair", summary))
        if repair_raises:
            raise ValueError("Keine Code-Block in der Antwort des Modells")
        return CODE + "# fixed\n"

    def write(plan, task):
        o.calls.append(("write", None))
        return CODE + "# rewritten\n"
    o.repair_code, o.write_code = repair, write
    return o


def _run(o, tmp_path):
    claim = pytypes.SimpleNamespace(id="c", target=0.8, tolerance=0.01, metric="accuracy")
    dossier = pytypes.SimpleNamespace(attempts=[], baselines=None)
    return o._run_seed(claim, None, "snap", dossier, tmp_path, "plan", CODE, 0)


OK = {"status": "ok", "measured": 0.8}
TIMEOUT = {"status": "timeout", "detail": "Training > 600s"}
CRASH = {"status": "fehler", "detail": "RuntimeError: boom"}


def test_timeout_without_flag_stops(tmp_path):
    o = _orch(False, [TIMEOUT])
    value, note, _ = _run(o, tmp_path)
    assert value is None and "kein Reparaturversuch" in note and o.calls == []


def test_timeout_with_flag_is_repaired_with_speed_hint(tmp_path):
    o = _orch(True, [TIMEOUT, OK])
    value, note, code = _run(o, tmp_path)
    assert value == 0.8 and note is None and code.endswith("# fixed\n")
    assert o.calls[0][0] == "repair" and "faster" in o.calls[0][1]


def test_missing_code_without_flag_stops(tmp_path):
    o = _orch(False, [CRASH], repair_raises=True)
    value, note, _ = _run(o, tmp_path)
    assert value is None and "Reparatur ohne Code-Block" in note


def test_missing_code_with_flag_rewrites_from_plan(tmp_path):
    o = _orch(True, [CRASH, OK], repair_raises=True)
    value, note, code = _run(o, tmp_path)
    assert value == 0.8 and code.endswith("# rewritten\n")
    assert [c[0] for c in o.calls] == ["repair", "write"]
