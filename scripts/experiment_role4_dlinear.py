"""Rolle-4-Experiment DLinear (Regeln: benchmark/experiments/role4_dlinear.md). Nicht aendern.

Arme abwechselnd, Ergebnisse als JSONL, Neustart macht weiter.
  LLM_PROVIDER=tokenfactory uv run python scripts/experiment_role4_dlinear.py [generierungen=8] [seeds=5]
"""

from __future__ import annotations

import json
import re
import sys

from repro_agent.dossier import REPO_ROOT
from repro_agent.loop import Orchestrator

PAPER = "zeng_dlinear_2023"
CLAIM = "dlinear_etth1_96_mse"
LR_DECAY = re.compile(r"0\.5\s*\*\*|gamma\s*=\s*0\.5|ExponentialLR|StepLR|lr_lambda|LambdaLR", re.I)
WINDOW_336 = re.compile(r"\b336\b")
STORE = REPO_ROOT / "runs" / "experiment_role4_dlinear.jsonl"


def main() -> None:
    n_gen = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rows = [json.loads(x) for x in STORE.read_text().splitlines()] if STORE.exists() else []
    snapshots: dict = {}
    arms = {
        "A_ohne_rolle4": Orchestrator(tavily_mode="fixture", use_tavily=True, fresh=True, setup_details=False, seeds=n_seeds,
                                      claim_ids=[CLAIM]),
        "B_mit_rolle4": Orchestrator(tavily_mode="fixture", use_tavily=True, fresh=True, setup_details=True, seeds=n_seeds,
                                     claim_ids=[CLAIM]),
    }
    for orch in arms.values():
        orch._snapshots = snapshots
    for g in range(n_gen):
        for arm, orch in arms.items():
            if any(r["arm"] == arm and r["generation"] == g for r in rows):
                continue
            try:
                d = orch.run(PAPER)
            except RuntimeError as exc:
                print(f"ABBRUCH {arm} Generierung {g}: {exc}", flush=True)
                return
            v = d.verdicts[0]
            code = next(orch.last_run_dir.glob(f"{CLAIM}_*.py")).read_text()
            row = {
                "arm": arm, "generation": g, "urteil": v.urteil, "mse": v.measured_mean,
                "std": v.measured_std, "cost_usd": orch.spent_usd, "n_details": len(d.setup_details),
                "code_lr_decay": bool(LR_DECAY.search(code)), "code_uses_336": bool(WINDOW_336.search(code)),
                "run_dir": str(orch.last_run_dir.relative_to(REPO_ROOT)),
            }
            rows.append(row)
            with STORE.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print("ERGEBNIS", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
