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
    res = orch._run_and_eval(orch.sandbox.branch(snap), code, 0, task, "mse")
    assert res["status"] == "ok"
    rep, zero = res["baselines"]["repeat"], res["baselines"]["zero"]
    assert rep["mse"] == pytest.approx(1.2944, abs=5e-4) and rep["mae"] == pytest.approx(0.7132, abs=5e-4)
    assert zero["mse"] == pytest.approx(1.1099, abs=5e-4) and zero["mae"] == pytest.approx(0.7960, abs=5e-4)
    assert res["metrics"]["mae"] <= res["metrics"]["mse"] ** 0.5   # MAE <= RMSE
