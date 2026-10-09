"""recall cg v1: Laufplan, Stufen-Gate, Mehrheit, Reservierung, Auswertung (scripts/recall_cg_v1.py, scripts/eval_recall_cg_v1.py). Ohne LLM, ohne Sandbox."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import recall_cg_v1 as R  # noqa: E402
from eval_recall_cg_v1 import evaluate  # noqa: E402

REF = {u: {"ref": "reproduziert", "ref_placebo": "nicht reproduziert", "reference_mean": 0.8, "target": 0.8, "placebo_target": 0.88} for u in R.ALL_UNITS}
REF["velickovic_dgi_2019"]["ref"] = "teilweise"


def row(unit, gen, urteil, origin="paper", **kw):
    real, placebo = R.CLAIMS[unit]
    return {"paper": unit, "generation": gen, "config": R.CONFIG, "origin": origin, "claim": real if origin == "paper" else placebo,
            "urteil": urteil, "mean": 0.8, "target": 0.8, "cost_usd_run": 0.2, **kw}


def test_units_and_claims_cover_all_nine():
    assert len(R.DEV_UNITS) == 8 and R.HELDOUT not in R.DEV_UNITS and set(R.CLAIMS) == set(R.ALL_UNITS)
    assert all(p.endswith("_placebo") for _, p in R.CLAIMS.values())


def test_claims_exist_in_papers_yaml():
    import yaml
    papers = {p["id"]: p for p in yaml.safe_load((R.ROOT / "benchmark" / "papers.yaml").read_text())["papers"]}
    for unit, (real, placebo) in R.CLAIMS.items():
        ids = {c["id"]: c for c in papers[unit]["claims"]}
        assert real in ids and placebo in ids and ids[placebo]["origin"] == "placebo"


def test_stage1_has_16_jobs_placebo_only_in_generation_0():
    jobs = R.stage1_jobs()
    assert len(jobs) == 16
    assert all(placebo == (gen == 0) for _, gen, placebo in jobs) and {g for _, g, _ in jobs} == {0, 1}
    assert R.claim_ids(("kipf_gcn_2017", 0, True)) == ["gcn_cora_acc", "gcn_cora_acc_placebo"]
    assert R.claim_ids(("kipf_gcn_2017", 1, False)) == ["gcn_cora_acc"]


def test_majority_rules():
    m = R.unit_majority
    assert m({0: "reproduziert", 1: "reproduziert"}) == ("reproduziert", False)
    assert m({0: "reproduziert"}) == (R.INCOMPLETE, False)
    assert m({0: "reproduziert", 1: "teilweise"}) == (R.INCOMPLETE, True)
    assert m({0: "reproduziert", 1: "teilweise", 2: "reproduziert"}) == ("reproduziert", False)
    assert m({0: "reproduziert", 1: "teilweise", 2: "nicht reproduziert"}) == (R.NO_MAJORITY, False)
    assert m({0: R.AUSFALL, 1: R.AUSFALL}) == (R.AUSFALL, False)
    assert not R.matches(R.AUSFALL, "reproduziert") and not R.matches(R.INCOMPLETE, "reproduziert")


def test_failure_classes_are_one_class():
    assert R.klass("nicht reproduzierbar (Setup)") == R.klass("abgebrochen") == R.klass("Agenten-Fehler vermutet") == R.AUSFALL


def _stage1(rows_by_unit):
    rows = []
    for u, (a, b) in rows_by_unit.items():
        rows += [row(u, 0, a), row(u, 1, b)]
    return rows


def test_gate_thresholds():
    # F = 2 (both generations hit), U = 2 (one hit): F + U = 4 -> continue
    hits = {u: ("nicht reproduziert", "nicht reproduziert") for u in R.DEV_UNITS}
    hits["kipf_gcn_2017"] = ("reproduziert", "reproduziert")
    hits["wu_sgc_2019"] = ("reproduziert", "reproduziert")
    hits["zeng_nlinear_2023"] = ("reproduziert", "teilweise")
    hits["zeng_dlinear_2023"] = ("nicht reproduziert", "reproduziert")
    g = R.gate(_stage1(hits), REF)
    assert (g["F"], g["U"], g["complete"], g["continue"]) == (2, 2, True, True)
    # F = 1: stop even with F + U = 5
    hits["wu_sgc_2019"] = ("reproduziert", "teilweise")
    hits["shchur_gat_2018"] = ("reproduziert", "teilweise")
    g = R.gate(_stage1(hits), REF)
    assert (g["F"], g["U"], g["continue"]) == (1, 4, False)
    # F = 2 but F + U = 3: stop
    hits = {u: ("nicht reproduziert", "nicht reproduziert") for u in R.DEV_UNITS}
    hits["kipf_gcn_2017"] = ("reproduziert", "reproduziert")
    hits["wu_sgc_2019"] = ("reproduziert", "reproduziert")
    hits["zeng_nlinear_2023"] = ("reproduziert", "teilweise")
    g = R.gate(_stage1(hits), REF)
    assert (g["F"], g["U"], g["continue"]) == (2, 1, False)


def test_gate_dgi_reference_is_teilweise():
    hits = {u: ("nicht reproduziert", "nicht reproduziert") for u in R.DEV_UNITS}
    hits["velickovic_dgi_2019"] = ("teilweise", "teilweise")
    assert R.gate(_stage1(hits), REF)["F"] == 1


def test_gate_incomplete_never_continues():
    rows = _stage1({u: ("reproduziert", "reproduziert") for u in R.DEV_UNITS[:-1]})
    g = R.gate(rows, REF)
    assert g["F"] == 6 and not g["complete"] and not g["continue"]


def test_stage2_order_and_adaptive_generation_2():
    rows = _stage1({u: ("reproduziert", "reproduziert") for u in R.DEV_UNITS})
    rows = [r for r in rows if r["paper"] != "kipf_gcn_2017"] + [row("kipf_gcn_2017", 0, "reproduziert"), row("kipf_gcn_2017", 1, "teilweise")]
    jobs = R.stage2_jobs(rows)
    assert jobs[:2] == [(R.HELDOUT, 0, True), (R.HELDOUT, 1, False)]       # held-out first
    assert jobs[2:] == [("kipf_gcn_2017", 2, False)]                       # only the differing dev unit
    rows += [row(R.HELDOUT, 0, "reproduziert"), row(R.HELDOUT, 1, "teilweise")]
    assert R.stage2_jobs(rows)[-1] == (R.HELDOUT, 2, False)                # held-out gen 2 last
    assert R.done(rows, ("kipf_gcn_2017", 1, False)) and not R.done(rows, ("kipf_gcn_2017", 2, False))


def test_reservation_blocks_start_near_the_stop():
    job_p, job_r = ("kipf_gcn_2017", 0, True), ("kipf_gcn_2017", 1, False)
    assert R.can_start(0.0, [], job_p)
    assert R.can_start(R.STOP_GROSS_USD - R.reserve(job_r), [], job_r)
    assert not R.can_start(R.STOP_GROSS_USD - R.reserve(job_r) + 0.01, [], job_r)
    assert not R.can_start(7.0, [job_p, job_p, job_r], job_r)               # 7.0 + 0.6 + 0.6 + 0.4 + 0.4 > 8.75


def test_call_level_stop_matches_the_costs_module():
    from repro_agent.costs import ABORT_FRACTION, budget_status
    assert R.ABORT_FRACTION == ABORT_FRACTION
    assert budget_status(R.STOP_GROSS_USD - 0.001, R.BUDGET_TOTAL_FOR_STOP) == "warn"
    assert budget_status(R.STOP_GROSS_USD + 0.001, R.BUDGET_TOTAL_FOR_STOP) == "abort"
    assert R.STOP_GROSS_USD < R.CAP_GROSS_USD
    env = R.worker_env("2026-10-09T01:00:00Z")
    assert env["LLM_PROVIDER"] == "tokenfactory" and env["LLM_CACHE"] == "off" and env["SANDBOX_BACKEND"] == "nebius" and env["TAVILY_MODE"] == "fixture"


def test_reference_block_in_preregistration_gives_expected_verdicts():
    if not R.DOC.exists():
        pytest.skip("Vorab-Festlegung noch nicht geschrieben")
    ref = R.reference_verdicts()
    assert set(ref) == set(R.ALL_UNITS)
    assert ref["velickovic_dgi_2019"]["ref"] == "teilweise"
    assert all(ref[u]["ref"] == "reproduziert" for u in R.ALL_UNITS if u != "velickovic_dgi_2019")
    assert all(v["ref_placebo"] == "nicht reproduziert" for v in ref.values())


def test_evaluate_reports_three_separate_numbers():
    rows = []
    for u in R.DEV_UNITS:
        rows += [row(u, 0, "reproduziert" if u == "kipf_gcn_2017" else "nicht reproduziert"), row(u, 1, "reproduziert" if u == "kipf_gcn_2017" else "nicht reproduziert"),
                 row(u, 0, "nicht reproduziert", origin="placebo")]
    rows += [row(R.HELDOUT, 0, "reproduziert"), row(R.HELDOUT, 1, "reproduziert"), row(R.HELDOUT, 0, "nicht reproduziert", origin="placebo")]
    text = evaluate(rows, REF)
    assert "Entwicklungseinheiten (In-Sample, „code-grounded rebuild“): 1 von 8" in text
    assert "Neues Paper (RLinear, nicht beim Entwurf der Fixes betrachtet): 1 von 1" in text
    assert "Placebos bestätigt („reproduziert“): 0 von 9" in text
    assert "method fidelity not checked" in text


def test_evaluate_counts_a_confirmed_placebo():
    rows = [row("kipf_gcn_2017", 0, "reproduziert", origin="placebo")]
    assert "Placebos bestätigt („reproduziert“): 1 von 1" in evaluate(rows, REF)
