"""export_benchmark_json: neue summary-Felder (Placebo-Zaehlung, einzelne 'reproduziert'-Urteile, Median der Abweichung) mit kleinem kuenstlichen Store."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_benchmark_json as ex  # noqa: E402

A, B = "A_ohne_tavily", "B_mit_tavily"
REFS = {"tolerance": 0.01, "units": {
    "u1": {"claim": "c1", "placebo": "c1_p", "target": 0.80, "placebo_target": 0.9, "reference_mean": 0.80, "origin": "R"},   # Referenz: reproduziert
    "u2": {"claim": "c2", "placebo": "c2_p", "target": 0.70, "placebo_target": 0.9, "reference_mean": 0.72, "origin": "R"},   # Referenz: teilweise
}}
PAPERS = {u: {"title": u, "claims": [{"id": f"c{u[1]}", "metric": "accuracy"}]} for u in ("u1", "u2")}


def row(unit, arm, gen, claim, urteil, mean, target):
    return {"paper": unit, "arm": arm, "generation": gen, "claim": claim, "origin": "placebo" if claim.endswith("_p") else "paper",
            "urteil": urteil, "mean": mean, "target": target, "run_dir": f"runs/{unit}/20260101-00000{gen}", "code_commit": "abc1234"}


def store(with_means=True):
    m = (lambda x: x) if with_means else (lambda x: None)
    rows = []
    for unit, claim, tgt, arms in (
        ("u1", "c1", 0.80, {A: (["reproduziert", "reproduziert", "teilweise"], [0.801, 0.802, 0.82]), B: (["reproduziert"] * 3, [0.81, 0.79, 0.81])}),
        # u2: Referenz "teilweise", aber Einzelurteile "reproduziert" (A: 2, B: 0)
        ("u2", "c2", 0.70, {A: (["reproduziert", "reproduziert", "nicht reproduziert"], [0.70, 0.70, 0.75]), B: (["teilweise"] * 3, [0.73, 0.73, 0.67])}),
    ):
        for arm, (verdicts, means) in arms.items():
            for g in range(3):
                rows.append(row(unit, arm, g, claim, verdicts[g], m(means[g]), tgt))
                plac = "reproduziert" if (unit, arm, g) == ("u2", A, 0) else "nicht reproduziert"   # ein Placebo faellt durch
                rows.append(row(unit, arm, g, claim + "_p", plac, m(0.5), 0.9))
    return rows


def test_new_summary_fields():
    s = ex.version("vX", store(), REFS, PAPERS)["summary"]
    assert s["placebo_total"] == 12 and s["placebo_reproduced"] == 1 and s["placebo_passed"] == 1
    assert s["reproduced_real"] == 2 + 3 + 2                        # u1 A 2, u1 B 3, u2 A 2
    assert s["reproduced_real_agree"] == 5                          # u2: Referenz teilweise, zaehlt in reproduced_real, nicht in agree
    assert s["median_distance_n_A"] == 6 and s["median_distance_n_B"] == 6
    assert s["median_distance_A"] == pytest.approx(0.0015) and s["median_distance_B"] == pytest.approx(0.02)   # Median ueber echte Claims, Placebos nicht
    assert s["n_units"] == 2 and s["agree_A"] == 1                  # alte Felder unveraendert (u1 A: Mehrheit reproduziert; u2 A: reproduziert != teilweise)


def test_distance_fields_omitted_without_means():
    s = ex.version("vX", store(with_means=False), REFS, PAPERS)["summary"]
    assert "median_distance_A" not in s and "median_distance_B" not in s and "median_distance_n_A" not in s
    assert s["reproduced_real"] == 7 and s["placebo_total"] == 12


def test_distances_reuse_eval_function():
    assert ex.ev.distances([{"mean": 0.5, "target": 0.4}, {"mean": None, "target": 0.4}, {"mean": float("nan"), "target": 0.4}]) == [pytest.approx(0.1)]
