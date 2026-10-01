"""Wertet runs/benchmark_v2.jsonl nach benchmark/experiments/benchmark_v2.md aus. Schreibt Markdown nach stdout.

  uv run python scripts/eval_benchmark_v2.py > benchmark/experiments/results/benchmark_v2.md
Primaer: Mehrheit je Paper-Claim-Einheit ueber 3 Generierungen (mindestens 2 gleiche Klassen, Ausfall ist eine eigene Klasse), Uebereinstimmung
mit dem Referenzurteil; Placebos getrennt. Sekundaer: Rohzaehlung. Referenzen stehen als YAML-Block in benchmark_v2.md (`reference_verdicts`).
Keine Signifikanzaussagen. Alle Urteile vorlaeufig: Achse 2 (Methodentreue) nicht geprueft.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from repro_agent.dossier import REPO_ROOT
from repro_agent.harness.verify import judge

DOC = REPO_ROOT / "benchmark" / "experiments" / "benchmark_v2.md"
STORE = REPO_ROOT / "runs" / "benchmark_v2.jsonl"
ARMS = ("A_ohne_tavily", "B_mit_tavily")
VALID = ("reproduziert", "teilweise", "nicht reproduziert")
AUSFALL = "Ausfall"
NO_MAJORITY = "keine Mehrheit"
INCOMPLETE = "unvollstaendig"
N_GEN = 3
NO_GRAPH = ("zeng_dlinear_2023", "zeng_nlinear_2023")   # ETTh1: keine Graph-Ablation


def load_refs(doc: Path = DOC) -> dict:
    """YAML-Block `reference_verdicts` aus der Vorab-Festlegung."""
    for block in re.findall(r"```yaml\n(.*?)```", doc.read_text(), flags=re.S):
        data = yaml.safe_load(block)
        if isinstance(data, dict) and "reference_verdicts" in data:
            return data["reference_verdicts"]
    raise SystemExit(f"Kein YAML-Block `reference_verdicts` in {doc}")


def klass(urteil: str) -> str:
    return urteil if urteil in VALID else AUSFALL


def majority(urteile: list[str], n_gen: int = N_GEN) -> str:
    """Mehrheitsklasse ueber die Generierungen. Weniger als n_gen Urteile: unvollstaendig (zaehlt nie als Uebereinstimmung)."""
    if len(urteile) < n_gen:
        return INCOMPLETE
    top, n = Counter(klass(u) for u in urteile).most_common(1)[0]
    return top if n >= 2 else NO_MAJORITY


def reference_verdicts(refs: dict) -> dict[str, dict]:
    """Je Einheit Referenzurteil echt und Placebo aus reference_mean. Offener Referenzlauf (null) ist ein Fehler."""
    tol = refs["tolerance"]
    open_units = [u for u, v in refs["units"].items() if v["reference_mean"] is None]
    if open_units:
        raise SystemExit(f"Referenzlauf offen fuer {open_units}: reference_mean in benchmark_v2.md eintragen (vor den Agentenlaeufen)")
    return {u: {**v, "ref": judge(v["reference_mean"], v["target"], tol), "ref_placebo": judge(v["reference_mean"], v["placebo_target"], tol)}
            for u, v in refs["units"].items()}


def _fmt(x) -> str:
    return "–" if x is None or x != x else f"{x:.4f}"


def evaluate(rows: list[dict], refs: dict) -> str:
    ref = reference_verdicts(refs)
    by = defaultdict(list)   # (Einheit, Arm, echt|placebo) -> Zeilen nach Generierung
    for r in sorted(rows, key=lambda r: r["generation"]):
        if r["paper"] in ref:
            by[(r["paper"], r["arm"], r["origin"])].append(r)
    units = list(ref)
    out = [f"# Benchmark {next((c[:2] for c in (r.get('config', '') for r in rows) if c[:2] in ('v3', 'v4')), 'v2')} ({len(rows)} Urteile aus {len(rows) // 2} Läufen)\n",
           "Vorab-Festlegung: `../benchmark_v2.md`. Rohdaten: `runs/benchmark_v2.jsonl`. Alle Urteile vorläufig: **method fidelity not checked** (Achse 2). "
           "Die Zahlen messen „Zielwert laut Harness erreicht“, nicht „Methode reproduziert“. Keine Signifikanzaussagen.\n"]

    def maj(unit: str, arm: str, origin: str) -> str:
        return majority([r["urteil"] for r in by[(unit, arm, origin)]])

    out.append(f"## Primär: Mehrheit je Paper-Claim-Einheit ({len(units)} Einheiten, je 3 Generierungen)\n")
    out.append("| Arm | Echte Claims: stimmt mit Referenzurteil überein | Mehrheit Ausfall | keine Mehrheit | unvollständig | Placebos als „nicht reproduziert“ erkannt |")
    out.append("|---|---|---|---|---|---|")
    for arm in ARMS:
        m = [maj(u, arm, "paper") for u in units]
        agree = sum(mj == ref[u]["ref"] for mj, u in zip(m, units))
        pl = sum(maj(u, arm, "placebo") == "nicht reproduziert" for u in units)
        out.append(f"| {arm} | {agree} von {len(units)} | {m.count(AUSFALL)} | {m.count(NO_MAJORITY)} | {m.count(INCOMPLETE)} | {pl} von {len(units)} |")

    out.append("\n## Je Einheit (Urteile der Generierungen 0/1/2, Mehrheit, Mittelwerte der Messung)\n")
    out.append("| Einheit | Claim | Referenzurteil (Mittel, Herkunft) | Arm | Urteile | Mehrheit | Übereinstimmung | Mittel je Generierung | Placebo: Mehrheit |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for u in units:
        for arm in ARMS:
            rs = by[(u, arm, "paper")]
            mj = maj(u, arm, "paper")
            out.append(f"| {u} | {ref[u]['claim']} | {ref[u]['ref']} ({ref[u]['reference_mean']}, {ref[u]['origin']}) | {arm} | "
                       f"{' / '.join(klass_label(r['urteil']) for r in rs) or '–'} | {mj} | {'ja' if mj == ref[u]['ref'] else 'nein'} | "
                       f"{' / '.join(_fmt(r.get('mean')) for r in rs) or '–'} | {maj(u, arm, 'placebo')} |")

    out.append("\n## Sekundär: Rohzählung (einzelne Urteile)\n")
    out.append("| Claim-Art | Arm | Übereinstimmungen mit Referenzurteil | Ausfälle | „nah“ (nur echt) | Urteilsverteilung |")
    out.append("|---|---|---|---|---|---|")
    for origin, label in (("paper", "echt"), ("placebo", "Placebo")):
        for arm in ARMS:
            rs = [r for u in units for r in by[(u, arm, origin)]]
            want = lambda r: ref[r["paper"]]["ref" if origin == "paper" else "ref_placebo"]  # noqa: E731
            near = "–" if origin == "placebo" else f"{sum(r['urteil'] in VALID[:2] for r in rs)} von {len(rs)}"
            out.append(f"| {label} | {arm} | {sum(r['urteil'] == want(r) for r in rs)} von {len(rs)} | {sum(klass(r['urteil']) == AUSFALL for r in rs)} | {near} | "
                       f"{', '.join(f'{k} {v}' for k, v in Counter(r['urteil'] for r in rs).most_common())} |")
    dev = {arm: [abs(r["mean"] - r["target"]) for u in units for r in by[(u, arm, "paper")] if r.get("mean") == r.get("mean") and r.get("mean") is not None]
           for arm in ARMS}
    out.append("\nMedian der Abweichung vom Zielwert (gültige echte Generierungen, post hoc): "
               + ", ".join(f"{a} {statistics.median(v):.4f} (n={len(v)})" if v else f"{a} –" for a, v in dev.items()) + "\n")

    out.append("## Ausfälle nach Ursache\n")
    for arm in ARMS:
        c = Counter(r["urteil"] for r in rows if r["arm"] == arm and klass(r["urteil"]) == AUSFALL)
        out.append(f"- {arm}: {dict(c) or 'keine'}")

    out.append("\n## Graph-Ablation und Zielwert-Prüfung\n")
    for arm in ARMS:
        ab = Counter(r.get("ablation") for r in rows if r["arm"] == arm and r["paper"] not in NO_GRAPH and r["origin"] == "paper")
        out.append(f"- {arm}: Graph-Ablation (echte Claims, Graph-Tasks) {dict(ab)}")
    bad = [r for r in rows if r.get("target_matches_yaml") is False]
    out.append(f"- Belegter Zielwert weicht von `papers.yaml` ab: {len(bad)} Fälle" + (f" ({[(r['paper'], r['arm']) for r in bad]})" if bad else ""))

    out.append("\n## Kosten und Code-Stand\n")
    runs = {(r["paper"], r["arm"], r["generation"]): r for r in rows if "cost_usd_run" in r}
    for arm in ARMS:
        net = sum(r["cost_usd_run"] for (p, a, g), r in runs.items() if a == arm)
        out.append(f"- {arm}: {len([1 for k in runs if k[1] == arm])} Läufe, {net:.3f} $ netto, {net * 1.19:.3f} $ brutto")
    out.append("\n| Stand (`code_commit`, `config`) | Urteile | Übereinstimmungen echt | Ausfälle |")
    out.append("|---|---|---|---|")
    stands = defaultdict(list)
    for r in rows:
        if r["paper"] in ref:
            stands[f"{r.get('code_commit', 'ohne Feld')} / {r.get('config', 'ohne Feld')}"].append(r)
    for stand, rs in sorted(stands.items()):
        real = [r for r in rs if r["origin"] == "paper"]
        out.append(f"| {stand} | {len(rs)} | {sum(r['urteil'] == ref[r['paper']]['ref'] for r in real)} von {len(real)} | "
                   f"{sum(klass(r['urteil']) == AUSFALL for r in rs)} |")
    return "\n".join(out)


def klass_label(urteil: str) -> str:
    return {"reproduziert": "rep", "teilweise": "teil", "nicht reproduziert": "nicht"}.get(urteil, f"Ausfall ({urteil})")


def main() -> None:
    store = Path(sys.argv[1]) if len(sys.argv) > 1 else STORE
    rows = [json.loads(x) for x in store.read_text().splitlines()]
    print(evaluate(rows, load_refs()))


if __name__ == "__main__":
    main()
