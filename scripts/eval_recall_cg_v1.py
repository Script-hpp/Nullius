"""Wertet runs/recall_cg_v1.jsonl nach benchmark/experiments/recall_cg_v1.md aus (Markdown nach stdout).

  uv run python scripts/eval_recall_cg_v1.py > benchmark/experiments/results/recall_cg_v1.md
Drei getrennte Zahlen, nie summiert: Entwicklungseinheiten X von 8 (In-Sample), neues Paper Y von 1, Placebos bestaetigt Z von 9 (Ziel 0).
Zaehlung Achse 1 gegen das Referenzurteil, Mehrheit je Einheit (Gen 0 und 1 gleich: Klasse steht; sonst Gen 2). Keine Signifikanzaussagen.
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import recall_cg_v1 as R  # noqa: E402


def _fmt(x) -> str:
    return "–" if x is None or x != x else f"{x:.4f}"


def label(urteil: str) -> str:
    return {"reproduziert": "rep", "teilweise": "teil", "nicht reproduziert": "nicht"}.get(urteil, f"Ausfall ({urteil})")


def evaluate(rows: list[dict], ref: dict, v4_note: bool = True) -> str:
    rows = [r for r in rows if r.get("config") == R.CONFIG]
    real = lambda u: sorted((r for r in rows if r["paper"] == u and r["origin"] == "paper"), key=lambda r: r["generation"])  # noqa: E731
    plc = lambda u: [r for r in rows if r["paper"] == u and r["origin"] == "placebo"]  # noqa: E731
    maj = {u: R.unit_majority(R.real_classes(rows, u)) for u in R.ALL_UNITS}
    hit = {u: R.matches(maj[u][0], ref[u]["ref"]) for u in R.ALL_UNITS}
    dev_hits, rl_hit = sum(hit[u] for u in R.DEV_UNITS), hit[R.HELDOUT]
    plc_rows = [r for u in R.ALL_UNITS for r in plc(u)]
    plc_conf = sum(r["urteil"] == "reproduziert" for r in plc_rows)
    g = R.gate(rows, ref)
    out = ["# recall cg v1: Ergebnis\n",
           "Vorab-Festlegung: `../recall_cg_v1.md`. Rohdaten: `runs/recall_cg_v1.jsonl` (nicht in Git). Modus: Nachbau aus Paper und Autorencode, Arm B (Fixtures), "
           "Produktkonfiguration plus code_grounded und io_helper. **method fidelity not checked** (Achse 2). Die Zahlen messen „Zielwert laut Harness erreicht“, "
           "nicht „Methode reproduziert“. Keine Signifikanzaussagen. Drei getrennte Zahlen, nie summiert; v4 (Arm B 2 von 8) bleibt der Vergleich.\n",
           "## Ergebnis (drei Zahlen)\n",
           f"- **Entwicklungseinheiten (In-Sample, „code-grounded rebuild“): {dev_hits} von {len(R.DEV_UNITS)}** stimmen mit dem Referenzurteil überein (v4: 2 von 8).",
           f"- **Neues Paper (RLinear, nicht beim Entwurf der Fixes betrachtet): {int(rl_hit)} von 1.**",
           f"- **Placebos bestätigt („reproduziert“): {plc_conf} von {len(plc_rows)}** (Ziel 0; v2, v3, v4 je 0 von 48).",
           f"- Stufen-Gate (nach Stufe 1): F = {g['F']}, U = {g['U']}, vollständig = {g['complete']}, weiter = {g['continue']} (Regel: F ≥ 2 und F + U ≥ 4).\n",
           "## Je Einheit (echter Claim; Klassen der Generationen 0/1/2)\n",
           "| Einheit | Referenzurteil (Mittel) | Klassen Gen 0 / 1 / 2 | Einheitsklasse | Treffer | Mittel je Generation | Placebo (Gen 0) |",
           "|---|---|---|---|---|---|---|"]
    for u in R.ALL_UNITS:
        rs = real(u)
        p = plc(u)
        out.append(f"| {u}{' (neu)' if u == R.HELDOUT else ''} | {ref[u]['ref']} ({ref[u]['reference_mean']}) | "
                   f"{' / '.join(label(r['urteil']) for r in rs) or '–'} | {maj[u][0]} | {'ja' if hit[u] else 'nein'} | "
                   f"{' / '.join(_fmt(r.get('mean')) for r in rs) or '–'} | "
                   f"{label(p[0]['urteil']) + ' (' + _fmt(p[0].get('mean')) + ', Ziel ' + str(p[0].get('target')) + ')' if p else '–'} |")
    out += ["\n## Sekundär: Rohzählung je Generation (post hoc beschreibend)\n"]
    real_rows = [r for u in R.ALL_UNITS for r in real(u)]
    per_gen_hits = sum(r["urteil"] == ref[r["paper"]]["ref"] for r in real_rows)
    near = sum(r["urteil"] in R.VALID[:2] for r in real_rows)
    out.append(f"- Einzelurteile = Referenzurteil: {per_gen_hits} von {len(real_rows)}; „nah“ (reproduziert oder teilweise): {near} von {len(real_rows)}.")
    out.append("- Urteilsverteilung: " + ", ".join(f"{k} {v}" for k, v in Counter(r["urteil"] for r in real_rows).most_common()))
    dist = [abs(r["mean"] - r["target"]) for r in real_rows if r.get("mean") is not None and r["mean"] == r["mean"]]
    out.append(f"- Median Abstand zum Zielwert (gültige Generationen): {statistics.median(dist):.4f} (n={len(dist)})" if dist else "- Median Abstand: –")
    fails = Counter(r["urteil"] for r in rows if R.klass(r["urteil"]) == R.AUSFALL)
    out.append(f"- Ausfälle nach Ursache (alle Zeilen): {dict(fails) or 'keine'}")
    abl = Counter(r.get("ablation") for r in real_rows if r["paper"] not in ("zeng_dlinear_2023", "zeng_nlinear_2023", R.HELDOUT))
    out.append(f"- Graph-Ablation (echte Claims, Graph-Einheiten): {dict(abl)}")
    bad = [(r["paper"], r["generation"]) for r in rows if r.get("target_matches_yaml") is False]
    out.append(f"- Belegter Zielwert weicht von `papers.yaml` ab: {len(bad)} Fälle {bad or ''}")
    out.append("\n## Placebos\n")
    for u in R.ALL_UNITS:
        for r in plc(u):
            out.append(f"- {u}: {r['urteil']} (Mittel {_fmt(r.get('mean'))}, Ziel {r.get('target')}, Referenzurteil Placebo {ref[u]['ref_placebo']})")
    out.append("\n## Kosten und Code-Stand\n")
    runs = {(r["paper"], r["generation"]): r["cost_usd_run"] for r in rows if "cost_usd_run" in r}
    net = sum(runs.values())
    out.append(f"- {len(runs)} Läufe, {net:.3f} $ netto, {net * (1 + R.VAT):.3f} $ brutto (Summe der Laufkosten; maßgeblich für die Kappe ist das Aufruf-Log, siehe Bericht).")
    stands = Counter(r.get("code_commit") for r in rows)
    out.append(f"- Code-Stand (`code_commit`): {dict(stands)}")
    return "\n".join(out)


def main() -> None:
    store = Path(sys.argv[1]) if len(sys.argv) > 1 else R.STORE
    print(evaluate(R.load_rows(store), R.reference_verdicts()))


if __name__ == "__main__":
    main()
