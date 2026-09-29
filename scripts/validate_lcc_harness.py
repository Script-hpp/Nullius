"""Validiert die Cora-LCC-Pipeline (Datenvorbereitung, Harness) mit der menschlichen Referenzloesung, ohne LLM.

  uv run python scripts/validate_lcc_harness.py [n_seeds]     (Standard 30 = 10 Splits x 3 Initialisierungen)
Ausgabe: Mittel/Std der Accuracy ueber alle Laeufe, je Split-Mittel, Baselines. Kein Paper-Reproduktionsurteil.
"""

from __future__ import annotations

import hashlib
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

from repro_agent.dossier import REPO_ROOT
from repro_agent.loop import Orchestrator
from repro_agent.sandbox.base import get_sandbox
from repro_agent.tasks import cora_lcc_task


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    code = (REPO_ROOT / "tests" / "reference" / "gcn_lcc_reference.py").read_text()
    orch = Orchestrator.__new__(Orchestrator)
    orch.sandbox, orch._smoke_ok = get_sandbox("nebius"), {hashlib.sha256(code.encode()).hexdigest()}
    task = cora_lcc_task()
    snap = orch.sandbox.base_snapshot(task.image, task.setup_cmds, label="lcc-validate")

    def run(seed):
        return seed, orch._run_and_eval(orch.sandbox.branch(snap), code, seed, task, "accuracy")

    with ThreadPoolExecutor(5) as pool:
        results = list(pool.map(run, range(n)))
    ok = [(s, r["measured"]) for s, r in results if r["status"] == "ok"]
    for s, r in results:
        if r["status"] != "ok":
            print("seed", s, r["status"], str(r.get("detail"))[-200:])
    accs = [a for _, a in ok]
    print(f"{len(ok)}/{n} gueltig; Accuracy Mittel {statistics.mean(accs):.4f}, Std {statistics.stdev(accs):.4f}")
    for k in range(n // task.seeds_per_split):
        vals = [a for s, a in ok if task.split_for(s) == k]
        print(f"  Split {k}: {statistics.mean(vals):.4f} (n={len(vals)})")
    b = results[0][1].get("baselines")
    print("Baselines Split 0:", b)


if __name__ == "__main__":
    main()
