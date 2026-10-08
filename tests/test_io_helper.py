"""Ausgabe-Helfer (docs/recall_analysis_3.md Hebel 4): Agent liefert Arrays, nullius_io schreibt predictions.csv/val_predictions.csv und die
SMOKE-Teilmengen. Flag aus = Prompts und Branch unveraendert. Attrappen, kein Netz, kein Docker."""

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from repro_agent.harness.smoke import check_smoke_output
from repro_agent.dossier import Claim
from repro_agent.loop import Orchestrator
from repro_agent.tasks import TASKS, cora_lcc_task, cora_task, etth1_task
from repro_agent.timeouts import train_timeout

CLAIM = Claim("c", "X erreicht 81,5 % auf Cora", "accuracy", 0.815, 0.01, "T", "paper")


def _load_helper(task, tmp_path, monkeypatch, data: dict, smoke: bool):
    """Helfer-Quelltext des Tasks als Modul laden, Datenpfad und Ausgabeordner auf tmp_path umbiegen."""
    npz = tmp_path / "data.npz"
    np.savez(npz, **data)
    src = task.io_helper_source()
    path = tmp_path / "nullius_io.py"
    path.write_text(src)
    monkeypatch.setenv("SMOKE", "1" if smoke else "0")
    spec = importlib.util.spec_from_file_location("nullius_io_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nullius_io_test"] = mod
    spec.loader.exec_module(mod)
    mod.OUT_DIR = str(tmp_path / "out")
    mod.CONFIG = {**mod.CONFIG, "data": str(npz)}
    return mod


def _graph(n=60, f=5):
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    return {"features": rng.random((n, f)).astype(np.float32), "edges": np.zeros((0, 2), np.int64), "labels": np.zeros(n, np.int64),
            "train_idx": idx[:10], "val_idx": idx[10:40], "test_idx": idx[40:]}


@pytest.mark.parametrize("smoke", [True, False])
def test_node_helper_writes_ids_of_the_task(tmp_path, monkeypatch, smoke):
    d = _graph()
    mod = _load_helper(cora_task(), tmp_path, monkeypatch, d, smoke)
    pred = np.arange(60) % 7                                  # [N], indexierbar mit Knoten-Id
    mod.write_predictions(pred)
    rows = (tmp_path / "out" / "predictions.csv").read_text().splitlines()
    assert rows[0] == "id,prediction"
    ids = [int(r.split(",")[0]) for r in rows[1:]]
    expected = sorted(int(i) for i in d["test_idx"])[: 20 if smoke else None]
    assert ids == expected
    assert [int(r.split(",")[1]) for r in rows[1:]] == [i % 7 for i in ids]
    vids = [int(r.split(",")[0]) for r in (tmp_path / "out" / "val_predictions.csv").read_text().splitlines()[1:]]
    assert vids == sorted(int(i) for i in d["val_idx"])[: 20 if smoke else None]
    # die Formpruefung des Orchestrators akzeptiert die Datei
    assert check_smoke_output((tmp_path / "out" / "predictions.csv").read_bytes(), tuple(str(i) for i in expected)) is None


def test_node_helper_accepts_logits_and_torch_like_tensor(tmp_path, monkeypatch):
    mod = _load_helper(cora_task(), tmp_path, monkeypatch, _graph(), False)

    class T:   # nur .detach().cpu().numpy() wie bei torch.Tensor
        def __init__(self, a): self.a = a
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.a

    logits = np.zeros((60, 7)); logits[:, 3] = 1.0
    mod.write_predictions(T(logits))
    assert {r.split(",")[1] for r in (tmp_path / "out" / "predictions.csv").read_text().splitlines()[1:]} == {"3"}


def test_node_helper_rejects_test_only_predictions_with_clear_message(tmp_path, monkeypatch):
    d = _graph()
    mod = _load_helper(cora_task(), tmp_path, monkeypatch, d, False)
    with pytest.raises(ValueError, match=r"WHOLE graph, shape \[60\]"):
        mod.write_predictions(np.zeros(len(d["test_idx"]), int))


@pytest.mark.parametrize("smoke", [True, False])
def test_window_helper_writes_ids_and_subsets(tmp_path, monkeypatch, smoke):
    task = etth1_task()
    n_win, n_val = 2785, 117
    data = {"train": np.zeros((10, 7), np.float32), "val": np.zeros((10, 7), np.float32), "test_inputs": np.zeros((3, 4, 7), np.float32)}
    mod = _load_helper(task, tmp_path, monkeypatch, data, smoke)
    mod.write_predictions(np.full((n_win, 96, 7), 0.5), np.full((n_val, 96, 7), 0.25))
    rows = (tmp_path / "out" / "predictions.csv").read_text().splitlines()
    n = 8 if smoke else n_win
    assert rows[0] == "id,prediction" and len(rows) == 1 + n * 96 * 7
    assert rows[1] == "0_0_0,0.5" and rows[-1].startswith(f"{n - 1}_95_6,")
    vrows = (tmp_path / "out" / "val_predictions.csv").read_text().splitlines()
    assert len(vrows) == 1 + (8 if smoke else n_val) * 96 * 7
    if smoke:   # genau die Ids, die die Formpruefung erwartet
        assert check_smoke_output((tmp_path / "out" / "predictions.csv").read_bytes(), task.smoke_ids, task.smoke_dims, task.smoke_dim_names) is None


def test_window_helper_rejects_wrong_shape(tmp_path, monkeypatch):
    mod = _load_helper(etth1_task(), tmp_path, monkeypatch, {"train": np.zeros((2, 7))}, False)
    with pytest.raises(ValueError, match=r"shape \[n_windows, 96, 7\]"):
        mod.write_predictions(np.zeros((2785, 7, 96)), np.zeros((117, 96, 7)))
    with pytest.raises(ValueError, match="NaN or inf"):
        mod.write_predictions(np.full((2785, 96, 7), np.nan), np.zeros((117, 96, 7)))


@pytest.mark.parametrize("key", list(TASKS))
def test_every_task_has_helper_variant_and_keeps_semantics(key):
    task = TASKS[key]()
    io = task.with_io_helper()
    assert "nullius_io" in io.data_description and "nio.SMOKE" in io.code_notes
    assert "write /output/predictions.csv with header" not in io.data_description    # alter Ausgabeabschnitt ersetzt
    assert io.eval_cmd == task.eval_cmd and io.harness_files == task.harness_files    # Messung unveraendert
    assert io.smoke_ids == task.smoke_ids and io.smoke_ids_cmd == task.smoke_ids_cmd
    assert task.io_helper_source().count("CONFIG = {") == 1


def _prompts(o: Orchestrator, task) -> dict:
    o._code_from_llm = lambda system, user, **kw: user
    return {"write": o.write_code("PLAN", task), "repair": o.repair_code("PLAN", task, "print(1)", "boom")}


def _orch(io=None):
    o = Orchestrator.__new__(Orchestrator)
    o.details, o.graph_repair = [], True
    if io is not None:
        o.io_helper = io
    return o


@pytest.mark.parametrize("task", [cora_task(), etth1_task(), cora_lcc_task()])
def test_flag_off_prompts_byte_identical_and_on_changes_only_the_contract(task):
    legacy, off, on = _prompts(_orch(), task), _prompts(_orch(False), task), _prompts(_orch(True), task)
    assert legacy == off
    assert all("nullius_io" not in p for p in off.values())
    for k in ("write", "repair"):
        assert "nullius_io" in on[k] and "Do not write these files yourself" in on[k]
        assert "Dataset:" in on[k] and str(task.eval_cmd) not in on[k]


def test_helper_is_copied_next_to_train_py_only_with_flag():
    class SB:
        def __init__(self): self.files = {}
        def put_file(self, b, path, content): self.files[path] = content
        def run(self, b, cmd, t): return SimpleNamespace(exit_code=1, stdout="", stderr="x", timed_out=False)
        def put_files(self, b, files): pass

    task = cora_task()
    for flag, expected in ((False, False), (True, True)):
        o = _orch(flag)
        o.sandbox, o._smoke_ok = SB(), set()
        o._smoke_check = lambda *a, **k: "stop"
        o._run_and_eval("b", "print(1)", 0, task, "accuracy")
        assert ("/work/nullius_io.py" in o.sandbox.files) is expected


def test_train_timeout_from_reference_runtime():
    assert train_timeout(30) == 600                  # nie unter dem Standard
    assert train_timeout(120) == 1200                # 10 x Referenz
    assert train_timeout(101) == 1020                # 1010 s, auf volle Minuten aufgerundet
    assert train_timeout(5000) == 3600               # Deckel


def test_train_timeout_attribute_is_used():
    o = _orch(False)
    assert o._train_timeout() == 600
    o.train_timeout_s = 1500
    assert o._train_timeout() == 1500


def test_repair_hint_for_out_of_memory():
    from repro_agent.loop import OOM_HINT, repair_hint
    assert repair_hint("Smoke run (SMOKE=1) crashed:\nKilled") == OOM_HINT
    assert repair_hint("RuntimeError: [enforce fail at alloc_cpu.cpp] DefaultCPUAllocator: can't allocate memory") == OOM_HINT
    assert repair_hint("Traceback (most recent call last):\n  File x\nValueError: boom\nKilled by a handler") == ""   # Traceback: kein Speicherproblem
    assert repair_hint("IndexError: index 1708 is out of bounds") == ""
