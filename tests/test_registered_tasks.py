"""Registrierte Tasks: Verdrahtung mit Katalog, Discovery, Anzeige und Loop (Attrappen, kein Netz, kein Geld)."""

import pytest

from repro_agent import catalog, discovery, paper_head
from repro_agent.tasks import HARNESS_DIR, TASKS

from test_loop_fakes import FakeSandbox, make_orch, run_claim, train_cmds


@pytest.mark.parametrize("name", sorted(TASKS))
def test_task_is_wired_everywhere(name):
    t = TASKS[name]()
    for fname in t.harness_files + list(t.val_harness_files):
        assert (HARNESS_DIR / fname).exists(), fname
    assert name in catalog.TASK_SEEDS and name in catalog.TASK_METRICS and name in catalog.TASK_SUFFIX
    assert name in discovery.TASK_KEYWORDS
    if name != "Cora-LCC":   # Cora-LCC ist nur Protokoll-Variante von Cora, kein eigener Datensatz in der Suche
        assert name in discovery.ALLOWED_METRICS and name in discovery.DISCOVERY_QUERIES
        assert name in discovery.EXTRACT_QUERIES and name in paper_head.DATASETS


class CiteSeerSandbox(FakeSandbox):
    """Wie FakeSandbox, der Harness heisst aber citeseer_eval.py."""

    def run(self, branch, cmd, timeout_s):
        self.eval_seen = getattr(self, "eval_seen", False) or "citeseer_eval.py" in cmd
        return super().run(branch, cmd.replace("citeseer_eval.py", "cora_eval.py"), timeout_s)


def test_citeseer_claim_runs_through_the_loop(tmp_path):
    from repro_agent.dossier import Claim, Dossier

    sb = CiteSeerSandbox(acc=lambda seed, ablated: 0.71)
    o, _ = make_orch(sb)
    claim = Claim("c", "GCN 70.3 %", "accuracy", 0.703, 0.01, "Tabelle 2", "paper")
    dossier = Dossier(paper_id="p", claims=[claim])
    o.run_claim({"seeds": 3}, claim, TASKS["CiteSeer"](), "snap", dossier, tmp_path)
    assert sb.eval_seen
    assert train_cmds(sb) and len(dossier.attempts) == 3
    assert dossier.verdicts[0].urteil == "reproduziert"
