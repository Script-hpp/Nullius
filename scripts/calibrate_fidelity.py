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

# v1 (verbraucht, Eichung gescheitert) und v2 (neue Codes). Standard: v3.
SETS = {
    "v1": (COMPONENT_DIR / "gold_labels.json", REPO_ROOT / "tests" / "fixtures" / "agent_codes"),
    "v2": (COMPONENT_DIR / "gold_labels_v2.json", REPO_ROOT / "tests" / "fixtures" / "agent_codes_v2"),
    "v3": (COMPONENT_DIR / "gold_labels_v3.json", REPO_ROOT / "tests" / "fixtures" / "agent_codes_v3"),
}
MIN_HITS = {"v1": 8, "v2": 8, "v3": 9}   # Regelschwelle je Version (v3: 9 von 11, siehe calibration_fidelity_v3.md)


REPEATS = 3


def majority(verdicts: list[str]) -> str | None:
    top, n = Counter(verdicts).most_common(1)[0]
    return top if n >= 2 else None


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    version = next((v for v in SETS if f"--{v}" in sys.argv), "v3")
    gold_path, codes_dir = SETS[version]
    gold = json.loads(gold_path.read_text())
    only = args
    settings = dataclasses.replace(load_settings(), llm_cache=False)   # Wiederholungen muessen echt sein
    llm = LLMClient(settings)
    names = [n for n in gold if not only or n in only]

    def run(name):
        comps = load_components(gold[name]["paper"])
        code = (codes_dir / f"{name}.py").read_text()
        return name, [check_fidelity(llm, comps, code) for _ in range(REPEATS)]

    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(run, names))
    # Regelmenge: Codes ohne dead_code. dead_code-Codes (Bau ohne Nutzung) werden nur berichtet.
    hits = swaps = unstable = rule_n = 0
    dead: list[str] = []
    for name, reps in results:
        g = gold[name]
        votes = [r.overall for r in reps]
        maj = majority(votes)
        ok = maj == g["gold"]
        swap = {maj, g["gold"]} == {"ja", "nein"}
        is_dead = bool(g.get("dead_code"))
        unstable += len(set(votes)) > 1
        if is_dead:
            dead.append(f"{name}: Mehrheit {maj}, Gold {g['gold']}")
        else:
            rule_n += 1
            hits += ok
            swaps += swap
        print(f"{'OK ' if ok else 'ABW'} {name:36s} Mehrheit: {str(maj):14s} Gold: {g['gold']:9s} "
              f"Urteile: {votes}{' (dead_code, nur berichtet)' if is_dead else ''}{' JA/NEIN VERTAUSCHT' if swap else ''}")
        for r in reps[0].results:
            print(f"      {r.id:34s} {r.status:16s} {r.note[:90]}")
    passed = hits >= MIN_HITS[version] and swaps == 0
    print(f"\nRegelmenge: {hits}/{rule_n} Mehrheitsurteile stimmen mit Gold; ja/nein vertauscht: {swaps}; "
          f"Codes mit uneinigen Urteilen (alle): {unstable}/{len(results)}; Kosten {llm.total_cost_usd:.4f} $ netto")
    for line in dead:
        print(f"  dead_code (nicht in der Regel): {line}")
    if version == "v1":
        print("v1 ist verbraucht (Regel v1: >= 8/10 ohne Vertauschung).")
    print(f"Eichregel ({version}: >= {MIN_HITS[version]} von {rule_n} und keine Vertauschung): {'ERFUELLT' if passed else 'NICHT ERFUELLT'}")


if __name__ == "__main__":
    main()
