"""Fehlerprotokoll eines Benchmark-Stores: jedes Urteil, das nicht "reproduziert" (echter Claim) bzw. nicht
"nicht reproduziert" (Placebo) ist, mit Ursache aus dossier.json/events.jsonl. Schreibt Markdown nach stdout.

  uv run python scripts/failure_log.py runs/benchmark_v3.jsonl > benchmark/experiments/results/failure_log_v3.md
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def events(run_dir: str, claim: str) -> list[dict]:
    p = REPO / run_dir / "events.jsonl"
    if not p.exists():
        return []
    return [e for e in map(json.loads, p.read_text().splitlines()) if e.get("claim_id") in (claim, None)]


def cause(r: dict, evs: list[dict], expl: str) -> str:
    u = r["urteil"]
    if u == "nicht reproduzierbar (Setup)":
        for key, label in (("Reparatur und Neuschreiben ohne Code-Block", "kein Code trotz Neuschreiben"),
                           ("Reparatur ohne Code-Block", "Reparatur ohne Code-Block"),
                           ("Timeout", "Timeout"),
                           ("Statische Pruefung", "statische Prüfung (verbotener Import o. ä.)"),
                           ("einzige Vorhersage-Klasse", "nur eine Vorhersage-Klasse"),
                           ("kein gueltiger Lauf nach", "Versuche erschöpft")):
            if key in expl:
                last = expl.split("letzter Fehler:")[-1].split(")")[0].strip()[:120] if "letzter Fehler" in expl else ""
                return f"Setup: {label}" + (f" ({last})" if last else "")
        return "Setup: sonstiges"
    if u == "Agenten-Fehler vermutet":
        m = r.get("mean")
        if r["paper"] != "zeng_dlinear_2023" and r["paper"] != "zeng_nlinear_2023" and m is not None and m < 0.3:
            return f"Agenten-Fehler: Genauigkeit {m:.3f} nahe Zufall (Vorhersage-Zuordnung oder kein Lernen)"
        return f"Agenten-Fehler: schlechter als triviale Baseline ({m:.4f})" if m is not None else "Agenten-Fehler"
    if u == "abgebrochen":
        return f"abgebrochen: {r.get('error', '')[:120]}"
    m, t = r.get("mean"), r.get("target")
    if m is None or m != m:
        return u
    side = "zu niedrig" if (m < t) == (r["paper"].startswith("zeng") is False) else "zu hoch"
    if r["paper"].startswith("zeng"):
        side = "MSE zu hoch" if m > t else "MSE zu niedrig"
    else:
        side = "Genauigkeit zu niedrig" if m < t else "Genauigkeit zu hoch"
    return f"Abweichung: {u}, {side} (gemessen {m:.4f}, Ziel {t})"


def main() -> None:
    store = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "runs" / "benchmark_v3.jsonl"
    rows = [json.loads(x) for x in store.read_text().splitlines()]
    out = [f"# Fehlerprotokoll `{store.name}` ({len(rows)} Urteile)\n",
           "Automatisch aus `dossier.json`/`events.jsonl` (`scripts/failure_log.py`). Erfasst: echte Claims, die nicht "
           "\"reproduziert\" sind, und Placebos, die nicht \"nicht reproduziert\" sind. Die Ursache ist eine Einordnung "
           "aus Erklärung und Ereignissen, kein Beweis.\n",
           "| Einheit | Arm | Gen | Art | Urteil | Ursache | Reparaturen (Fehler/Graph/Code) | Rolle 5 | Lauf |",
           "|---|---|---|---|---|---|---|---|---|"]
    counts: Counter = Counter()
    for r in rows:
        ok = (r["urteil"] == "reproduziert") if r["origin"] == "paper" else (r["urteil"] == "nicht reproduziert")
        if ok:
            continue
        d = REPO / r["run_dir"] / "dossier.json"
        expl = ""
        if d.exists():
            dj = json.loads(d.read_text())
            expl = next((v.get("explanation", "") for v in dj.get("verdicts", []) if v["claim_id"] == r["claim"]), "")
        evs = events(r["run_dir"], r["claim"])
        n = Counter(e["type"] for e in evs if e.get("claim_id") == r["claim"])
        c = cause(r, evs, expl)
        counts[c.split(":")[0] if not c.startswith("Abweichung") else f"Abweichung ({r['urteil']})"] += 1
        out.append(f"| {r['paper']} | {r['arm'][0]} | {r['generation']} | {r['origin']} | {r['urteil']} | {c} | "
                   f"{n['fault_repair']}/{n['graph_repair']}/{n['repair'] + n['rewrite']} | {n['error_search']} | `{r['run_dir']}` |")
    out.append("\n## Zählung nach Art\n")
    out += [f"- {k}: {v}" for k, v in counts.most_common()]
    print("\n".join(out))


if __name__ == "__main__":
    main()
