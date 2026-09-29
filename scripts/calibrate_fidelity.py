"""Eicht den Methodentreue-Pruefer an von Hand eingeordneten Codes (benoetigt LLM-Aufrufe).

  LLM_PROVIDER=tokenfactory uv run python scripts/calibrate_fidelity.py

Prueft jeden Code 3x (Mehrheit zaehlt, Regel in benchmark/experiments/calibration_fidelity.md) gegen benchmark/components/gold_labels.json und gibt je Code
die Baustein-Ergebnisse aus. Erst wenn der Pruefer hier stimmt, darf er Urteile beeinflussen.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from repro_agent.config import load_settings
from repro_agent.dossier import REPO_ROOT
from repro_agent.fidelity import COMPONENT_DIR, check_fidelity, load_components
from repro_agent.llm import LLMClient

CODES = REPO_ROOT / "tests" / "fixtures" / "agent_codes"


REPEATS = 3


def majority(verdicts: list[str]) -> str | None:
    top, n = Counter(verdicts).most_common(1)[0]
    return top if n >= 2 else None


def main() -> None:
    gold = json.loads((COMPONENT_DIR / "gold_labels.json").read_text())
    only = sys.argv[1:]
    settings = dataclasses.replace(load_settings(), llm_cache=False)   # Wiederholungen muessen echt sein
    llm = LLMClient(settings)
    names = [n for n in gold if not only or n in only]

    def run(name):
        comps = load_components(gold[name]["paper"])
        code = (CODES / f"{name}.py").read_text()
        return name, [check_fidelity(llm, comps, code) for _ in range(REPEATS)]

    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(run, names))
    hits = swaps = unstable = 0
    for name, reps in results:
        g = gold[name]
        votes = [r.overall for r in reps]
        maj = majority(votes)
        ok = maj == g["gold"]
        swap = {maj, g["gold"]} == {"ja", "nein"}
        hits += ok
        swaps += swap
        unstable += len(set(votes)) > 1
        print(f"{'OK ' if ok else 'ABW'} {name:32s} Mehrheit: {str(maj):9s} Gold: {g['gold']:9s} "
              f"Urteile: {votes}{' (schwerer Fall)' if g.get('hard') else ''}{' JA/NEIN VERTAUSCHT' if swap else ''}")
        for r in reps[0].results:
            print(f"      {r.id:34s} {r.status:16s} {r.note[:90]}")
    passed = hits >= 8 and swaps == 0
    print(f"\n{hits}/{len(results)} Mehrheitsurteile stimmen mit Gold; ja/nein vertauscht: {swaps}; "
          f"Codes mit uneinigen Urteilen: {unstable}/{len(results)}; Kosten {llm.total_cost_usd:.4f} $ netto")
    print(f"Eichregel (>= 8/10 und keine Vertauschung): {'ERFUELLT' if passed else 'NICHT ERFUELLT'}")


if __name__ == "__main__":
    main()
