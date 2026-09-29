import pytest

from repro_agent.harness.ablation import classify_ablation
from repro_agent.loop import Orchestrator
from repro_agent.tasks import cora_task, etth1_task


@pytest.mark.parametrize("full,abl,expected", [
    (0.80, 0.51, "Graph genutzt"),
    (0.80, 0.70, "Graph genutzt"),        # Abfall genau 0,10
    (0.80, 0.71, "unklar"),
    (0.80, 0.78, "Graph nicht genutzt"),   # Abfall genau 0,02
    (0.80, 0.80, "Graph nicht genutzt"),
    (0.80, 0.85, "Graph nicht genutzt"),   # besser ohne Kanten
    (0.80, None, "nicht messbar"),
])
def test_classify(full, abl, expected):
    assert classify_ablation(full, abl) == expected


def test_only_cora_has_ablation():
    assert cora_task().ablation_cmd and "np.zeros((0, 2)" in cora_task().ablation_cmd
    assert not etth1_task().ablation_cmd


def _orch(results):
    o = Orchestrator.__new__(Orchestrator)
    o.sandbox = type("S", (), {"branch": lambda self, s: "b"})()
    calls = []

    def run(branch, code, seed, task, metric, **kw):
        calls.append(kw)
        return results.pop(0)

    o._run_and_eval = run
    return o, calls


def test_ablation_check_uses_ablate_flag_and_classifies():
    o, calls = _orch([{"status": "ok", "measured": 0.52}])
    ab = o.ablation_check("snap", "code", 0, cora_task(), "accuracy", 0.80)
    assert calls == [{"ablate": True}] and ab["verdict"] == "Graph genutzt" and ab["accuracy_ablated"] == 0.52
    o, _ = _orch([{"status": "ok", "measured": 0.79}])
    assert o.ablation_check("s", "c", 0, cora_task(), "accuracy", 0.80)["verdict"] == "Graph nicht genutzt"


def test_crash_without_edges_is_not_measurable_not_used():
    o, _ = _orch([{"status": "fehler", "detail": "IndexError"}])
    ab = o.ablation_check("s", "c", 0, cora_task(), "accuracy", 0.80)
    assert ab["verdict"] == "nicht messbar" and "IndexError" in ab["detail"]
