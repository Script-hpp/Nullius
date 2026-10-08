"""Live-Lauf der Einwand-Experimente (Teil 2 von benchmark/experiments/tavily_experiments_test.md).

Echte Token-Factory-, Sandbox- und Tavily-Aufrufe (TAVILY_MODE=live), Produktkonfiguration, Vollmodus, kein LLM-Cache.
  LLM_PROVIDER=tokenfactory TAVILY_MODE=live uv run python scripts/run_objection_live.py --paper kipf_gcn_2017 --claim gcn_cora_acc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repro_agent.loop import Orchestrator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    ap.add_argument("--claim", required=True)
    a = ap.parse_args()
    orch = Orchestrator(graph_repair=True, tavily_mode="live", llm_cache=False, objection_experiments=True, claim_ids=[a.claim])
    dossier = orch.run(a.paper)
    print("run dir:", orch.last_run_dir)
    for v in dossier.verdicts:
        print("verdict", v.claim_id, v.urteil, v.measured_mean, v.measured_std)
    for c in dossier.objection_checks:
        print(c["id"], c["type"], c["status"], "|", (c.get("note") or "")[:80], "|", c["quote"][:70])


if __name__ == "__main__":
    main()
