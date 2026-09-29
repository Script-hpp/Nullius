"""Rolle-4-Experiment Shchur (Regeln: benchmark/experiments/role4_shchur.md). Nicht aendern.

Arme abwechselnd, Ergebnisse als JSONL, Neustart macht weiter.
  LLM_PROVIDER=tokenfactory uv run python scripts/experiment_role4_shchur.py [generierungen=6]
Arm A: ohne Rolle 4. Arm B: Rolle 4 nur aus dem Paper-Text (Anhang C), kein README, kein Repo.
"""

from __future__ import annotations

import json
import re
import sys

from repro_agent.dossier import REPO_ROOT
from repro_agent.loop import Orchestrator

PAPER = "shchur_pitfalls_2018"
CLAIM = "shchur_gcn_cora_acc"
FEATURES = {   # Regex-Heuristiken fuer Anhang-C-Werte im Code
    "hidden_64": re.compile(r"\b64\b"),
    "dropout_08": re.compile(r"\b0?\.8\b"),
    "lr_001": re.compile(r"\b0?\.01\b|1e-2"),
    "patience_50": re.compile(r"\b50\b"),
    "l2_1e3": re.compile(r"\b1e-3\b|0?\.001\b"),
}
STORE = REPO_ROOT / "runs" / "experiment_role4_shchur.jsonl"


def main() -> None:
    n_gen = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rows = [json.loads(x) for x in STORE.read_text().splitlines()] if STORE.exists() else []
    snapshots: dict = {}
    arms = {
        "A_ohne_rolle4": Orchestrator(use_tavily=True, fresh=True, setup_details=False, claim_ids=[CLAIM]),
        "B_rolle4_nur_paper": Orchestrator(use_tavily=True, fresh=True, setup_details=True, paper_only=True,
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
                "arm": arm, "generation": g, "urteil": v.urteil, "gesamt": v.gesamtaussage,
                "accuracy": v.measured_mean, "std": v.measured_std, "cost_usd": orch.spent_usd,
                "n_details": len(d.setup_details), "ablation": d.ablations.get(CLAIM, {}).get("verdict"),
                "features": {k: bool(rx.search(code)) for k, rx in FEATURES.items()},
                "run_dir": str(orch.last_run_dir.relative_to(REPO_ROOT)),
            }
            rows.append(row)
            with STORE.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print("ERGEBNIS", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
