"""Benchmark-Lauf (Regeln: benchmark/experiments/benchmark_protocol.md). Nicht aendern.

Pro Paper und Arm 3 frische Generierungen; je Generierung der echte Claim und der Placebo-Claim. Arme abwechselnd, Ergebnisse als JSONL,
Neustart macht weiter.
  LLM_PROVIDER=tokenfactory uv run python scripts/run_benchmark.py <paper_id> [<paper_id> ...]
Arm A: ohne Tavily (`use_tavily=False`, keine Rollen 2 und 4). Arm B: mit Tavily (Rolle 2 Zielwert-Beleg, Rolle 4 Setup-Details).
"""

from __future__ import annotations

import json
import subprocess
import sys

import yaml

from repro_agent.dossier import REPO_ROOT
from repro_agent.llm import BillingBlocked
from repro_agent.loop import Orchestrator

CODE_COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
DIRTY = bool(subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip())
N_GEN = 3
STORE = REPO_ROOT / "runs" / "benchmark.jsonl"
CLAIMS = {   # echter Claim, Placebo-Claim
    "kipf_gcn_2017": ("gcn_cora_acc", "gcn_cora_acc_placebo"),
    "shchur_pitfalls_2018": ("shchur_gcn_cora_acc", "shchur_gcn_cora_acc_placebo"),
    "zeng_dlinear_2023": ("dlinear_etth1_96_mse", "dlinear_etth1_96_mse_placebo"),
    "wu_sgc_2019": ("sgc_cora_acc", "sgc_cora_acc_placebo"),
}


def main() -> None:
    papers = sys.argv[1:] or list(CLAIMS)
    yaml_targets = {c["id"]: c["target"] for p in yaml.safe_load((REPO_ROOT / "benchmark" / "papers.yaml").read_text())["papers"]
                    for c in p.get("claims", [])}
    rows = [json.loads(x) for x in STORE.read_text().splitlines()] if STORE.exists() else []
    for paper in papers:
        claim_ids = list(CLAIMS[paper])
        arms = {
            "A_ohne_tavily": Orchestrator(use_tavily=False, fresh=True, setup_details=False, claim_ids=claim_ids, tavily_mode="fixture"),
            "B_mit_tavily": Orchestrator(use_tavily=True, fresh=True, setup_details=True, claim_ids=claim_ids, tavily_mode="fixture"),
        }
        snapshots: dict = {}
        for orch in arms.values():
            orch._snapshots = snapshots
        for g in range(N_GEN):
            for arm, orch in arms.items():
                if any(r["paper"] == paper and r["arm"] == arm and r["generation"] == g for r in rows):
                    continue
                base = {"paper": paper, "arm": arm, "generation": g, "code_commit": CODE_COMMIT + ("+dirty" if DIRTY else "")}
                try:
                    d = orch.run(paper)
                except BillingBlocked:
                    raise
                except RuntimeError as exc:   # z. B. Kostenlimit, ungueltige Extraktion: als Ausfall festhalten, weiter
                    new = [{**base, "claim": cid, "origin": "placebo" if cid.endswith("_placebo") else "paper",
                            "urteil": "abgebrochen", "error": str(exc)[:300]} for cid in claim_ids]
                else:
                    new = []
                    for v in d.verdicts:
                        claim = next(c for c in d.claims if c.id == v.claim_id)
                        new.append({
                            **base, "claim": v.claim_id, "origin": claim.origin, "urteil": v.urteil, "mean": v.measured_mean,
                            "std": v.measured_std, "target": claim.target, "target_matches_yaml": abs(claim.target - yaml_targets[v.claim_id]) < 1e-9,
                            "gesamt": v.gesamtaussage, "methode": v.methode,
                            "ablation": d.ablations.get(v.claim_id, {}).get("verdict"), "cost_usd_run": round(orch.spent_usd, 4),
                            "run_dir": str(orch.last_run_dir.relative_to(REPO_ROOT)),
                        })
                for r in new:
                    rows.append(r)
                    with STORE.open("a") as f:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    print("ERGEBNIS", json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
