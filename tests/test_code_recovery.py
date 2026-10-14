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
    o.use_tavily, o.tavily_mode = False, "fixture"
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


def test_repeated_timeouts_stop_the_candidate(tmp_path):
    # TSMixer local catalog check 2026-10-14: every attempt ran ~10 min into the limit with the enforced paper settings
    o = _orch(True, [TIMEOUT, TIMEOUT, OK])
    value, note, _ = _run(o, tmp_path)
    assert value is None and "CPU limit" in note and len(o.calls) == 1


def test_timeouts_across_candidates_stop_the_claim(tmp_path):
    o = _orch(True, [TIMEOUT, TIMEOUT, TIMEOUT, OK])
    o._train_timeout = lambda: 600
    _run(o, tmp_path)                      # first candidate: two timeouts
    value, note, _ = _run(o, tmp_path)     # next candidate: one more reaches the claim limit
    assert value is None and "CPU limit" in note
    value, note, _ = _run(o, tmp_path)     # later candidates do not start another 10-minute attempt
    assert value is None and "CPU limit" in note


def test_missing_code_without_flag_stops(tmp_path):
    o = _orch(False, [CRASH], repair_raises=True)
    value, note, _ = _run(o, tmp_path)
    assert value is None and "Reparatur ohne Code-Block" in note


def test_missing_code_with_flag_rewrites_from_plan(tmp_path):
    o = _orch(True, [CRASH, OK], repair_raises=True)
    value, note, code = _run(o, tmp_path)
    assert value == 0.8 and code.endswith("# rewritten\n")
    assert [c[0] for c in o.calls] == ["repair", "write"]


# ---- Rolle 5: Tavily-Fehlersuche ------------------------------------------
from repro_agent.loop import MAX_ERROR_SEARCHES, error_query  # noqa: E402

TB = 'Traceback (most recent call last):\n  File "/work/train.py", line 61, in forward\nRuntimeError: The size of tensor a (2708) must match the size of tensor b (13264)\n'


def test_error_query_takes_last_exception_line_without_paths_and_ids():
    assert error_query(TB) == "RuntimeError: The size of tensor a (N) must match the size of tensor b (N)"
    assert error_query("no exception here") is None


class FakeTavily:
    def __init__(self):
        self.queries = []

    def resolve_error(self, q):
        self.queries.append(q)
        return {"results": [{"title": "t", "url": "https://stackoverflow.com/q/1", "content": "use .view(-1)"}]}


def _hint_orch(mode="live", use_tavily=True):
    o = _orch(True, [])
    o.use_tavily, o.tavily_mode, o.fake = use_tavily, mode, FakeTavily()
    o._tavily_getter = lambda: (lambda: o.fake)
    o._warn = lambda t: None
    return o


def test_error_hints_live_search_once_per_error_and_limited():
    o = _hint_orch()
    claim = pytypes.SimpleNamespace(id="c")
    h1 = o._error_hints(claim, TB)
    assert "stackoverflow.com" in h1 and "use .view(-1)" in h1
    assert o._error_hints(claim, TB) == h1 and len(o.fake.queries) == 1   # gleiche Zeile: kein zweiter Aufruf
    for i in range(MAX_ERROR_SEARCHES + 2):
        o._error_hints(claim, f"ValueError: case {chr(97 + i)}")
    assert len(o.fake.queries) == MAX_ERROR_SEARCHES


@pytest.mark.parametrize("mode,use", [("fixture", True), ("live", False)])
def test_error_hints_off_in_fixture_mode_and_without_tavily(mode, use):
    o = _hint_orch(mode, use)
    assert o._error_hints(pytypes.SimpleNamespace(id="c"), TB) == "" and o.fake.queries == []


def test_crash_repair_gets_web_hints_in_product_path(tmp_path):
    o = _hint_orch()
    it = iter([{"status": "fehler", "detail": TB}, OK])
    o._attempt = lambda *a, **k: ("b", next(it))
    value, _, _ = _run(o, tmp_path)
    assert value == 0.8 and "Web search results" in o.calls[0][1]


def test_error_hints_live_in_fixture_mode_when_benchmark_v4_flag_set():
    o = _hint_orch("fixture")
    o.error_search_live = True
    assert "stackoverflow.com" in o._error_hints(pytypes.SimpleNamespace(id="c"), TB)


# ---- Repo-Funde (Rolle 1, Link ohne Hinweiswort, Importe) ---------------------
from repro_agent.repo_source import find_repo_url, imported_files, search_repo_url  # noqa: E402


def test_single_repo_link_without_hint_is_used():
    assert find_repo_url("matrix where aij \n\n1https://github.com/Tiiiger/SGC \n\nvi and") == "https://github.com/Tiiiger/SGC"
    assert find_repo_url("see https://github.com/a/b and https://github.com/c/d") is None


def test_imported_local_modules_are_added():
    files = {"models/gat.py": "from utils import layers\nimport numpy as np\n", "execute_cora.py": "from models import GAT\n"}
    listing = ["models/gat.py", "execute_cora.py", "utils/layers.py", "models/__init__.py", "utils/process.py"]
    got = imported_files(files, listing)
    assert "utils/layers.py" in got and "utils/process.py" not in got


class FakeSearch:
    def __init__(self, results):
        self.results, self.n = results, 0

    def find_material(self, title, domains=None):
        self.n += 1
        assert domains == ["github.com"]
        return {"results": self.results}


def test_repo_search_accepts_only_hits_that_name_the_paper(tmp_path):
    t = "Pitfalls of Graph Neural Network Evaluation"
    fake = FakeSearch([{"url": "https://github.com/x/other", "content": "some GNN code"},
                       {"url": "https://github.com/shchur/gnn-benchmark/tree/master", "content": "Code for Pitfalls of Graph Neural Network Evaluation"}])
    assert search_repo_url(t, "1811.05868", lambda: fake, fixture_dir=tmp_path) == "https://github.com/shchur/gnn-benchmark"
    assert search_repo_url(t, "1811.05868", lambda: fake, fixture_dir=tmp_path) == "https://github.com/shchur/gnn-benchmark"
    assert fake.n == 1   # zweiter Aufruf aus der Fixture
    none = FakeSearch([{"url": "https://github.com/x/other", "content": "unrelated"}])
    assert search_repo_url(t, "0000.00000", lambda: none, fixture_dir=tmp_path) is None
