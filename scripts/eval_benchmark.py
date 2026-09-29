"""Wertet runs/benchmark.jsonl nach benchmark/experiments/benchmark_protocol.md aus. Schreibt Markdown nach stdout.

  uv run python scripts/eval_benchmark.py > benchmark/experiments/results/benchmark_results.md
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict

from repro_agent.dossier import REPO_ROOT
from repro_agent.harness.verify import judge

# Referenzmessungen (results/*.txt) und Ziel/Toleranz je Claim, wie im Protokoll festgelegt
REF = {
    "gcn_cora_acc": (0.8136, 0.815), "gcn_cora_acc_placebo": (0.8136, 0.88),
    "sgc_cora_acc": (0.8081, 0.81), "sgc_cora_acc_placebo": (0.8081, 0.88),
    "dlinear_etth1_96_mse": (0.3809, 0.375), "dlinear_etth1_96_mse_placebo": (0.3809, 0.300),
    "shchur_gcn_cora_acc": (0.8107, 0.815), "shchur_gcn_cora_acc_placebo": (0.8107, 0.88),
}
TOL = 0.01
FAIL = ("nicht reproduzierbar (Setup)", "Agenten-Fehler vermutet", "abgebrochen")
TUNING = re.compile(r"grid|search|hyperopt|for\s+(wd|lr|weight_decay|hidden|dropout)\b|itertools\.product|best_val", re.I)
ARMS = ("A_ohne_tavily", "B_mit_tavily")


def main() -> None:
    rows = [json.loads(x) for x in (REPO_ROOT / "runs" / "benchmark.jsonl").read_text().splitlines()]
    for r in rows:
        r["ref"] = judge(*REF[r["claim"]], TOL)
        r["match"] = r["urteil"] == r["ref"]
        r["fail"] = r["urteil"] in FAIL
    out = [f"# Benchmark-Ergebnis ({len(rows)} Urteile aus {len(rows) // 2} Läufen)\n",
           "Protokoll: `../benchmark_protocol.md`. Rohdaten: `benchmark.jsonl`. Alle Urteile vorläufig (Achse 2 nicht geprüft).\n"]

    out.append("## Kernzahlen (Übereinstimmung mit dem Referenzurteil)\n")
    out.append("| | Arm | Übereinstimmungen | Ausfälle | „nah“ (nur echte Claims) | Urteilsverteilung |")
    out.append("|---|---|---|---|---|---|")
    for origin, label in (("paper", "echte Claims"), ("placebo", "Placebo-Claims")):
        for arm in ARMS:
            rs = [r for r in rows if r["origin"] == origin and r["arm"] == arm]
            near = sum(r["urteil"] in ("reproduziert", "teilweise") for r in rs) if origin == "paper" else None
            dist = ", ".join(f"{k} {v}" for k, v in Counter(r["urteil"] for r in rs).most_common())
            out.append(f"| {label} | {arm} | {sum(r['match'] for r in rs)} von {len(rs)} | {sum(r['fail'] for r in rs)} | "
                       f"{'–' if near is None else f'{near} von {len(rs)}'} | {dist} |")

    out.append("\n## Je Paper (Übereinstimmungen von 3 Generierungen; Mittelwerte der Messung)\n")
    out.append("| Paper | Claim | Referenzurteil | Arm A | Arm B | Mittel A (Gen. 0/1/2) | Mittel B (Gen. 0/1/2) |")
    out.append("|---|---|---|---|---|---|---|")
    by = defaultdict(list)
    for r in rows:
        by[(r["paper"], r["claim"])].append(r)
    for (paper, claim), rs in by.items():
        cells = []
        for arm in ARMS:
            a = sorted([r for r in rs if r["arm"] == arm], key=lambda r: r["generation"])
            cells.append((f"{sum(r['match'] for r in a)}/3", " / ".join("–" if r.get("mean") is None or r["mean"] != r["mean"] else f"{r['mean']:.4f}" for r in a)))
        out.append(f"| {paper} | {claim} | {rs[0]['ref']} | {cells[0][0]} | {cells[1][0]} | {cells[0][1]} | {cells[1][1]} |")

    out.append("\n## Graph-Ablation und Zielwert-Prüfung\n")
    for arm in ARMS:
        ab = Counter(r.get("ablation") for r in rows if r["arm"] == arm and r["paper"] != "zeng_dlinear_2023")
        out.append(f"- {arm}: Graph-Ablation {dict(ab)}")
    bad = [r for r in rows if r.get("target_matches_yaml") is False]
    out.append(f"- Belegter Zielwert weicht von `papers.yaml` ab: {len(bad)} Fälle" + (f" ({[(r['paper'], r['arm']) for r in bad]})" if bad else ""))

    out.append("\n## Aufwand: brauchen Placebo-Läufe auffällig mehr Versuche oder Tuning?\n")
    out.append("| Claim-Art | Arm | Versuche je Claim (Median) | Code-Versionen (Median) | Reparaturen gesamt | Code mit Tuning-Stichwort | Codelänge (Median, Zeilen) |")
    out.append("|---|---|---|---|---|---|---|")
    stats = defaultdict(list)
    for r in rows:
        if "run_dir" not in r:
            continue
        d = json.loads((REPO_ROOT / r["run_dir"] / "dossier.json").read_text())
        att = [a for a in d["attempts"] if a["claim_id"] == r["claim"]]
        versions = list(dict.fromkeys(a["code_hash"] for a in att))
        last = next(iter((REPO_ROOT / r["run_dir"]).glob(f"{r['claim']}_{versions[-1]}.py")), None) if versions else None
        code = last.read_text() if last else ""
        stats[(r["origin"], r["arm"])].append((len(att), len(versions), bool(TUNING.search(code)), len(code.splitlines())))
    for (origin, arm), v in sorted(stats.items()):
        out.append(f"| {origin} | {arm} | {statistics.median(x[0] for x in v)} | {statistics.median(x[1] for x in v)} | "
                   f"{sum(x[1] - 1 for x in v)} | {sum(x[2] for x in v)} von {len(v)} | {statistics.median(x[3] for x in v)} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
