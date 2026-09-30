"""Auswertung Benchmark v2: Mehrheitsregel, Referenzurteile, Tabellen. Synthetische Zeilen, keine Aufrufe."""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev, rb = _load("eval_benchmark_v2"), _load("run_benchmark")
REFS = {"tolerance": 0.01, "units": {
    "p1": {"claim": "c1", "placebo": "c1_pl", "target": 0.815, "placebo_target": 0.88, "reference_mean": 0.8136, "origin": "R"},
    "p2": {"claim": "c2", "placebo": "c2_pl", "target": 0.823, "placebo_target": 0.88, "reference_mean": 0.8047, "origin": "R"},
}}


def row(paper, arm, gen, urteil, origin="paper", mean=0.81, **kw):
    return {"paper": paper, "arm": arm, "generation": gen, "claim": paper + ("_pl" if origin == "placebo" else ""), "origin": origin,
            "urteil": urteil, "mean": mean, "target": 0.815, **kw}


def test_majority_rules():
    assert ev.majority(["reproduziert", "reproduziert", "teilweise"]) == "reproduziert"
    assert ev.majority(["reproduziert", "teilweise", "nicht reproduziert"]) == ev.NO_MAJORITY
    assert ev.majority(["abgebrochen", "Agenten-Fehler vermutet", "teilweise"]) == ev.AUSFALL   # Ausfall ist eine Klasse
    assert ev.majority(["reproduziert", "reproduziert"]) == ev.INCOMPLETE


def test_reference_verdicts_from_means():
    ref = ev.reference_verdicts(REFS)
    assert ref["p1"]["ref"] == "reproduziert" and ref["p2"]["ref"] == "teilweise"   # DGI-Fall: Abstand 0,0183
    assert ref["p1"]["ref_placebo"] == "nicht reproduziert"


def test_open_reference_aborts():
    bad = {"tolerance": 0.01, "units": {"p": {**REFS["units"]["p1"], "reference_mean": None}}}
    with pytest.raises(SystemExit, match="offen"):
        ev.reference_verdicts(bad)


def _rows():
    rows = []
    for g in range(3):
        rows += [row("p1", "A_ohne_tavily", g, ["teilweise", "teilweise", "reproduziert"][g]),
                 row("p1", "B_mit_tavily", g, "reproduziert"),
                 row("p2", "A_ohne_tavily", g, ["teilweise", "reproduziert", "nicht reproduziert"][g]),   # keine Mehrheit
                 row("p2", "B_mit_tavily", g, ["teilweise", "teilweise", "abgebrochen"][g])]
        for p in ("p1", "p2"):
            for arm in ev.ARMS:
                rows.append(row(p, arm, g, "nicht reproduziert" if not (p == "p2" and arm == ev.ARMS[0] and g == 0) else "teilweise", "placebo"))
    return rows


def test_evaluate_primary_counts_and_tables():
    md = ev.evaluate(_rows(), REFS)
    a = [l for l in md.splitlines() if l.startswith("| A_ohne_tavily |") and "von 2" in l][0].split("|")
    b = [l for l in md.splitlines() if l.startswith("| B_mit_tavily |") and "von 2" in l][0].split("|")
    # Arm A: p1 Mehrheit teilweise (Ref reproduziert: nein), p2 keine Mehrheit -> 0 von 2; Placebos: p1 3x, p2 Mehrheit nicht (2 von 3) -> 2 von 2
    assert a[2].strip() == "0 von 2" and a[4].strip() == "1" and a[6].strip() == "2 von 2"
    # Arm B: p1 reproduziert (ja), p2 teilweise Mehrheit (Ref teilweise: ja) -> 2 von 2
    assert b[2].strip() == "2 von 2" and b[3].strip() == "0"
    assert "method fidelity not checked" in md and "Keine Signifikanzaussagen" in md
    assert "Ausfall (abgebrochen)" in md


def test_incomplete_unit_is_not_counted():
    rows = [row("p1", "A_ohne_tavily", 0, "reproduziert"), row("p1", "A_ohne_tavily", 1, "reproduziert")]
    md = ev.evaluate(rows, REFS)
    a = [l for l in md.splitlines() if l.startswith("| A_ohne_tavily |") and "von 2" in l][0].split("|")
    assert a[2].strip() == "0 von 2" and a[5].strip() == "2"   # beide Einheiten unvollstaendig


def test_reference_block_in_doc_matches_run_benchmark_and_papers_yaml():
    refs = ev.load_refs()
    assert set(refs["units"]) == set(rb.CLAIMS_V2)
    papers = {p["id"]: p for p in yaml.safe_load((ROOT / "benchmark" / "papers.yaml").read_text())["papers"]}
    for unit, v in refs["units"].items():
        claims = {c["id"]: c for c in papers[unit]["claims"]}
        assert (v["claim"], v["placebo"]) == rb.CLAIMS_V2[unit]
        assert claims[v["claim"]]["target"] == v["target"] and claims[v["placebo"]]["target"] == v["placebo_target"]
