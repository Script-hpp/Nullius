"""Ablation ohne LLM-Aufrufe: welcher Unterschied im Agenten-Code kostet Genauigkeit?

Nimmt einen vorhandenen train.py, erzeugt Varianten per Textersetzung und misst jede in
eigenen Branches (5 Seeds) ueber denselben unabhaengigen Harness wie der Loop.

  uv run python scripts/ablate_gcn.py runs/kipf_gcn_2017/<ts>/gcn_cora_acc_<hash>.py
"""

from __future__ import annotations

import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from repro_agent.loop import Orchestrator
from repro_agent.tasks import cora_task

SEEDS = range(5)
ROW_NORM = "    X = X / np.maximum(X.sum(axis=1, keepdims=True), 1e-12)   # Zeilennormalisierung (ablation)\n"


def variants(code: str) -> dict[str, str]:
    out = {"baseline": code}
    marker = "X = data['features'].astype(np.float32)"
    line_end = code.index("\n", code.index(marker)) + 1
    with_norm = code[:line_end] + ROW_NORM + code[line_end:]
    out["+row-normalize"] = with_norm
    import re

    pat = re.compile(r"patience_limit\s*=\s*\d+")
    if pat.search(code):
        out["patience=20"] = pat.sub("patience_limit = 20", code)
        out["+row-normalize +patience=20"] = pat.sub("patience_limit = 20", with_norm)
    return out


def main() -> None:
    code = Path(sys.argv[1]).read_text()
    orch = Orchestrator(use_tavily=False)
    task = cora_task()
    snap = orch.sandbox.base_snapshot(task.image, task.setup_cmds, label="ablation-gcn")
    jobs = [(name, c, s) for name, c in variants(code).items() for s in SEEDS]

    def run(job):
        name, c, seed = job
        res = orch._run_and_eval(orch.sandbox.branch(snap), c, seed, task, "accuracy")
        return name, res.get("measured"), res

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(run, jobs))
    for name in variants(code):
        vals = [m for n, m, _ in results if n == name and m is not None]
        bad = [r["status"] for n, m, r in results if n == name and m is None]
        if vals:
            print(f"{name:32s} mean={statistics.mean(vals):.4f} std={statistics.stdev(vals):.4f} n={len(vals)} {bad or ''}")
        else:
            print(f"{name:32s} keine gueltigen Laeufe {bad}")


if __name__ == "__main__":
    main()
