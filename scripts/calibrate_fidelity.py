"""Eicht den Methodentreue-Pruefer an von Hand eingeordneten Codes (benoetigt LLM-Aufrufe).

  LLM_PROVIDER=tokenfactory uv run python scripts/calibrate_fidelity.py

Vergleicht das Urteil (ja/teilweise/nein) mit benchmark/components/gold_labels.json und gibt je Code
die Baustein-Ergebnisse aus. Erst wenn der Pruefer hier stimmt, darf er Urteile beeinflussen.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from repro_agent.dossier import REPO_ROOT
from repro_agent.fidelity import COMPONENT_DIR, check_fidelity, load_components
from repro_agent.llm import LLMClient

CODES = REPO_ROOT / "tests" / "fixtures" / "agent_codes"


def main() -> None:
    gold = json.loads((COMPONENT_DIR / "gold_labels.json").read_text())
    only = sys.argv[1:]
    llm = LLMClient()
    names = [n for n in gold if not only or n in only]

    def run(name):
        comps = load_components(gold[name]["paper"])
        return name, check_fidelity(llm, comps, (CODES / f"{name}.py").read_text())

    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(run, names))
    hits = 0
    for name, rep in results:
        g = gold[name]
        ok = rep.overall == g["gold"]
        hits += ok
        print(f"{'OK ' if ok else 'ABW'} {name:32s} Pruefer: {rep.overall:9s} Gold: {g['gold']:9s}{' (schwerer Fall)' if g.get('hard') else ''}")
        for r in rep.results:
            print(f"      {r.id:34s} {r.status:16s} {r.note[:90]}")
    print(f"\n{hits}/{len(results)} stimmen mit dem Gold-Label ueberein; Kosten {llm.total_cost_usd:.4f} $")


if __name__ == "__main__":
    main()
