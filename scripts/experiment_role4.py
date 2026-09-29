"""Rolle-4-Experiment (Regeln: benchmark/experiments/role4_gcn.md). Vorab festgelegt, nicht aendern.

  uv run python scripts/experiment_role4.py [generierungen=5] [seeds=5]
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from repro_agent.dossier import REPO_ROOT
from repro_agent.loop import Orchestrator

PAPER = "kipf_gcn_2017"
NORM = re.compile(r"F\.normalize|row.?sum|\.sum\((?:axis|dim)\s*=\s*1[^)]*keepdim", re.I)


def main() -> None:
    n_gen = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    only = sys.argv[3] if len(sys.argv) > 3 else "AB"   # z. B. "B": nur Arm B
    # Ergebnisse stehen in einer JSONL-Datei; ein Neustart macht dort weiter (Tageslimit der
    # Free-Endpunkte: 50 Anfragen/Tag). Abgebrochene Generierungen zaehlen nicht.
    store = REPO_ROOT / "runs" / "experiment_role4.jsonl"
    rows = [json.loads(x) for x in store.read_text().splitlines()] if store.exists() else []
    snapshots: dict = {}
    for arm, with_details in (("A_ohne_rolle4", False), ("B_mit_rolle4", True)):
        if arm[0] not in only:
            continue
        orch = Orchestrator(tavily_mode="fixture", use_tavily=True, fresh=True, setup_details=with_details, seeds=n_seeds)
        orch._snapshots = snapshots            # Basis-Snapshot ueber Arme teilen
        done = sum(r["arm"] == arm for r in rows)
        for g in range(done, n_gen):
            try:
                d = orch.run(PAPER)
            except RuntimeError as exc:         # z. B. 429 Tageslimit
                print(f"ABBRUCH bei {arm} Generierung {g}: {exc}", flush=True)
                break
            v = d.verdicts[0]
            run_dir = orch.last_run_dir
            code = next(run_dir.glob("gcn_cora_acc_*.py")).read_text()
            row = {
                "arm": arm, "generation": g, "urteil": v.urteil, "mean": v.measured_mean,
                "std": v.measured_std, "code_has_normalization": bool(NORM.search(code)),
                "n_details": len(d.setup_details), "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            }
            rows.append(row)
            with store.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print("ERGEBNIS", json.dumps(row), flush=True)
    for arm in ("A_ohne_rolle4", "B_mit_rolle4"):
        r = [x for x in rows if x["arm"] == arm]
        ok = sum(x["urteil"] == "reproduziert" for x in r)
        norm = sum(x["code_has_normalization"] for x in r)
        print(f"{arm}: {ok}/{len(r)} reproduziert, Normalisierung im Code: {norm}/{len(r)}")


if __name__ == "__main__":
    main()
