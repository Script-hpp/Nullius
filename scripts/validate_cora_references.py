"""Misst die menschlichen Referenzloesungen GCN (Kipf) und SGC (Wu) auf Cora durch die Pipeline (ohne LLM).

  uv run python scripts/validate_cora_references.py [seeds=10]
"""

from __future__ import annotations

import hashlib
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

from repro_agent.dossier import REPO_ROOT
from repro_agent.loop import Orchestrator
from repro_agent.sandbox.base import get_sandbox
from repro_agent.tasks import cora_task


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    refs = {name: (REPO_ROOT / "tests" / "reference" / f"{name}_cora_reference.py").read_text() for name in ("gcn", "sgc")}
    orch = Orchestrator.__new__(Orchestrator)
    orch.sandbox = get_sandbox("nebius")
    orch._smoke_ok = {hashlib.sha256(c.encode()).hexdigest() for c in refs.values()}
    task = cora_task()
    snap = orch.sandbox.base_snapshot(task.image, task.setup_cmds, label="cora-refs")
    jobs = [(name, s) for name in refs for s in range(n)]

    def run(job):
        name, seed = job
        return name, orch._run_and_eval(orch.sandbox.branch(snap), refs[name], seed, task, "accuracy")

    with ThreadPoolExecutor(5) as pool:
        results = list(pool.map(run, jobs))
    for name in refs:
        rs = [r for nm, r in results if nm == name]
        acc = [r["measured"] for r in rs if r["status"] == "ok"]
        for r in rs:
            if r["status"] != "ok":
                print("  ", name, r["status"], str(r.get("detail"))[-300:])
        if not acc:
            continue
        print(f"{name}: {len(acc)}/{len(rs)} gueltig; Accuracy Mittel {statistics.mean(acc):.4f} Median {statistics.median(acc):.4f} "
              f"Std {statistics.stdev(acc):.4f}")
        for r in rs:
            if r["status"] != "ok":
                print("  ", r["status"], str(r.get("detail"))[-200:])


if __name__ == "__main__":
    main()
