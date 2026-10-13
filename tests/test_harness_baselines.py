"""Integrationstest: Harness liefert Baselines, die zur unabhaengigen Hostrechnung passen."""
from pathlib import Path

import pytest

from repro_agent.loop import Orchestrator
from repro_agent.tasks import cora_task, etth1_task

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def orch(nebius):   # nebius-Fixture ueberspringt ohne Zugangsdaten
    return Orchestrator(use_tavily=False)


def test_etth1_baselines_match_paper_repeat(orch):
    task = etth1_task()
    snap = orch.sandbox.base_snapshot(task.image, task.setup_cmds, label="devtest-baselines-ett")
    code = (ROOT / "tests" / "reference" / "dlinear_reference.py").read_text()
    res = orch._run_and_eval(orch.sandbox.branch(snap), code, 0, task, "mse", eval_snapshot=snap)
    assert res["status"] == "ok"
    rep, zero = res["baselines"]["repeat"], res["baselines"]["zero"]
    assert rep["mse"] == pytest.approx(1.2944, abs=5e-4) and rep["mae"] == pytest.approx(0.7132, abs=5e-4)
    assert zero["mse"] == pytest.approx(1.1099, abs=5e-4) and zero["mae"] == pytest.approx(0.7960, abs=5e-4)
    assert res["metrics"]["mae"] <= res["metrics"]["mse"] ** 0.5   # MAE <= RMSE


def test_cora_baselines_and_faithful_gcn(orch):
    task = cora_task()
    snap = orch.sandbox.base_snapshot(task.image, task.setup_cmds, label="devtest-baselines-cora")
    code = next((ROOT / "runs" / "kipf_gcn_2017" / "20260928-211537").glob("gcn_cora_acc_*.py")).read_text()
    res = orch._run_and_eval(orch.sandbox.branch(snap), code, 0, task, "accuracy", eval_snapshot=snap)
    assert res["status"] == "ok"
    majority = res["baselines"]["majority"]["accuracy"]
    no_graph = res["baselines"]["no_graph"]["accuracy"]
    print("majority", majority, "no_graph", no_graph, "gcn", res["measured"])
    assert 0.25 < majority < 0.45
    assert 0.45 < no_graph < 0.70            # MLP-Niveau ohne Graph
    assert res["measured"] > no_graph + 0.10  # getreues GCN nutzt den Graphen
