"""Misst die menschliche DLinear-Referenz neu durch die Pipeline (Harness, ohne LLM, nur Sandbox-Zeit).

  uv run python scripts/validate_etth1_harness.py [seeds=5]
Zwei Varianten: konstante Lernrate und Halbierung pro Epoche (LR_HALVE=1, wie im Original-Repo). Dazu die Baselines des Harness.
Kein Reproduktionsurteil, nur Selbsttest der Pipeline.
"""

from __future__ import annotations

import hashlib
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

from repro_agent.dossier import REPO_ROOT
from repro_agent.loop import Orchestrator
from repro_agent.sandbox.base import get_sandbox
from repro_agent.tasks import etth1_task


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    base = (REPO_ROOT / "tests" / "reference" / "dlinear_reference.py").read_text()
    variants = {"konstante Lernrate": base, "Lernrate halbiert pro Epoche": 'import os\nos.environ["LR_HALVE"] = "1"\n' + base}
    orch = Orchestrator.__new__(Orchestrator)
    orch.sandbox = get_sandbox("nebius")
    orch._smoke_ok = {hashlib.sha256(c.encode()).hexdigest() for c in variants.values()}   # Referenz ist verifiziert
    task = etth1_task()
    snap = orch.sandbox.base_snapshot(task.image, task.setup_cmds, label="etth1-validate")
    jobs = [(name, s) for name in variants for s in range(n)]

    def run(job):
        name, seed = job
        return name, orch._run_and_eval(orch.sandbox.branch(snap), variants[name], seed, task, "mse")

    with ThreadPoolExecutor(5) as pool:
        results = list(pool.map(run, jobs))
    for name in variants:
        rs = [r for nm, r in results if nm == name]
        ok = [r for r in rs if r["status"] == "ok"]
        mse = [r["metrics"]["mse"] for r in ok]
        mae = [r["metrics"]["mae"] for r in ok]
        print(f"{name}: {len(ok)}/{len(rs)} gueltig; MSE Mittel {statistics.mean(mse):.4f} Median {statistics.median(mse):.4f} "
              f"Std {statistics.stdev(mse):.4f}; MAE Mittel {statistics.mean(mae):.4f}")
        for r in rs:
            if r["status"] != "ok":
                print("  ", r["status"], str(r.get("detail"))[-200:])
    first = next(r for _, r in results if r["status"] == "ok")
    print("Baselines:", first.get("baselines"))


if __name__ == "__main__":
    main()
