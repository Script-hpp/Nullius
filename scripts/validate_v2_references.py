"""Misst die neuen menschlichen Referenzloesungen fuer Benchmark v2 durch die Pipeline (Harness, ohne LLM, nur Sandbox-Zeit).

  uv run python scripts/validate_v2_references.py gat_lcc [n=30]   # Shchur-GAT, Cora-LCC, 30 Splits x 1 Initialisierung
  uv run python scripts/validate_v2_references.py nlinear [n=5]    # NLinear, ETTh1, Horizont 96, 5 Seeds
Ausgabe: Mittel/Std, Referenzurteil gegen Zielwert und Toleranz aus benchmark/papers.yaml (Regel: <= Tol. reproduziert, <= 3 x Tol. teilweise,
sonst nicht reproduziert). Kein Agentenergebnis, nur Selbsttest der Pipeline.
"""

from __future__ import annotations

import hashlib
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml

from repro_agent.dossier import REPO_ROOT
from repro_agent.loop import Orchestrator
from repro_agent.sandbox.base import get_sandbox
from repro_agent.tasks import cora_lcc_task, etth1_task

REFS = {   # Name -> (Datei, Task, Metrik, Claim-ID, Standard-Seeds)
    "gat_lcc": ("gat_lcc_reference.py", cora_lcc_task, "accuracy", "shchur_gat_cora_acc", 30),
    "nlinear": ("nlinear_reference.py", etth1_task, "mse", "nlinear_etth1_96_mse", 5),
}


def verdict(mean: float, target: float, tol: float) -> str:
    d = abs(mean - target)
    return "reproduziert" if d <= tol else "teilweise" if d <= 3 * tol else "nicht reproduziert"


def main() -> None:
    name = sys.argv[1]
    file, task_fn, metric, claim_id, default_n = REFS[name]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else default_n
    code = (REPO_ROOT / "tests" / "reference" / file).read_text()
    claim = next(c for p in yaml.safe_load((REPO_ROOT / "benchmark" / "papers.yaml").read_text())["papers"]
                 for c in p.get("claims", []) if c["id"] == claim_id)
    orch = Orchestrator.__new__(Orchestrator)
    orch.sandbox, orch._smoke_ok = get_sandbox("nebius"), {hashlib.sha256(code.encode()).hexdigest()}   # Referenz ist verifiziert
    task = task_fn()
    snap = orch.sandbox.base_snapshot(task.image, task.setup_cmds, label=f"v2-ref-{name}")

    def run(seed):
        return seed, orch._run_and_eval(orch.sandbox.branch(snap), code, seed, task, metric)

    with ThreadPoolExecutor(5) as pool:
        results = list(pool.map(run, range(n)))
    ok = [r["measured"] for _, r in results if r["status"] == "ok"]
    for s, r in results:
        if r["status"] != "ok":
            print("seed", s, r["status"], str(r.get("detail"))[-200:])
    if len(ok) < 2:
        print(f"{len(ok)}/{n} gueltig: kein Referenzurteil")
        return
    mean = statistics.mean(ok)
    print(f"{name}: {len(ok)}/{n} gueltig; {metric} Mittel {mean:.4f}, Std {statistics.stdev(ok):.4f}; "
          f"Ziel {claim['target']} +- {claim['tolerance']}, Abstand {abs(mean - claim['target']):.4f} -> Referenzurteil "
          f"{verdict(mean, claim['target'], claim['tolerance'])}")
    print("Baselines Seed 0:", results[0][1].get("baselines"))


if __name__ == "__main__":
    main()
