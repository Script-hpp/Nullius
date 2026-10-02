"""Benchmark-Skript: Laufliste, Budget, Fixture-Abdeckung, Dry-Run ohne Aufrufe. Keine echten Aufrufe."""

import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_benchmark", ROOT / "scripts" / "run_benchmark.py")
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)

PAPERS = {p["id"]: p for p in yaml.safe_load((ROOT / "benchmark" / "papers.yaml").read_text())["papers"]}


def test_v1_claims_unchanged():
    assert list(rb.CLAIMS) == ["kipf_gcn_2017", "shchur_pitfalls_2018", "zeng_dlinear_2023", "wu_sgc_2019"]


def test_v2_claims_exist_in_papers_yaml_with_placebo_and_extract():
    assert set(rb.CLAIMS) < set(rb.CLAIMS_V2) and len(rb.CLAIMS_V2) == 8
    for paper, (real, placebo) in rb.CLAIMS_V2.items():
        ids = {c["id"]: c for c in PAPERS[paper]["claims"]}
        assert ids[placebo]["origin"] == "placebo" and ids[placebo]["target"] != ids[real]["target"]
        assert "extract" in ids[real] and "origin" not in ids[real]


def test_v2_targets_and_tolerances_as_preregistered():
    c = {c["id"]: c for p in PAPERS.values() for c in p.get("claims", [])}
    assert (c["shchur_gat_cora_acc"]["target"], c["nlinear_etth1_96_mse"]["target"]) == (0.818, 0.374)
    assert (c["gat_cora_acc_placebo"]["target"], c["shchur_gat_cora_acc_placebo"]["target"], c["nlinear_etth1_96_mse_placebo"]["target"]) == (0.88, 0.88, 0.3)
    assert all(x["tolerance"] == 0.01 for x in c.values())


def test_pending_runs_skips_existing_rows_and_keeps_order():
    rows = [{"paper": "a", "arm": "A_ohne_tavily", "generation": 0}]
    runs = rb.pending_runs(["a", "b"], rows, 2)
    assert runs[0] == ("a", "B_mit_tavily", 0) and len(runs) == 7
    assert ("a", "A_ohne_tavily", 0) not in runs


def test_gross_spent_counts_each_run_once_with_vat():
    rows = [{"run_dir": "r1", "cost_usd_run": 1.0}, {"run_dir": "r1", "cost_usd_run": 1.0}, {"run_dir": "r2", "cost_usd_run": 0.5}]
    assert abs(rb.gross_spent(rows, 0.19) - 1.5 * 1.19) < 1e-9


def test_fixture_status_reports_missing_readme_and_text(tmp_path):
    (tmp_path / "repos").mkdir()
    paper = {"arxiv_url": "https://arxiv.org/abs/1234.5678"}
    assert "fehlt" in rb.fixture_status(paper, tmp_path)[0]
    (tmp_path / "1234.5678.json").write_text(json.dumps({"text": "Code is available at https://github.com/x/y ."}))
    assert "x_y_readme.json" in rb.fixture_status(paper, tmp_path)[0]
    (tmp_path / "repos" / "x_y_readme.json").write_text("{}")
    assert rb.fixture_status(paper, tmp_path) == []


def test_dry_run_makes_no_calls_and_reports_cost(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("Dry-Run darf keinen Orchestrator bauen")

    monkeypatch.setattr(rb, "Orchestrator", boom)
    out = rb.dry_run(True, list(rb.CLAIMS_V2), [], PAPERS, rb.CLAIMS_V2)
    assert "48 offene Laeufe" in out and "brutto" in out and "Budget-Grenze: 6.00" in out and "Keine Aufrufe ausgefuehrt" in out
    assert "shchur_gat_2018" in out and "zeng_nlinear_2023" in out


def test_v2_main_builds_arms_with_graph_repair_and_stops_at_budget(monkeypatch, tmp_path, capsys):
    made = []

    class Fake:
        def __init__(self, **kw):
            made.append(kw)

        def run(self, paper):
            raise AssertionError("Budget-Grenze erreicht, kein Lauf erwartet")

    store = tmp_path / "benchmark_v2.jsonl"
    store.write_text(json.dumps({"paper": "x", "arm": "A_ohne_tavily", "generation": 0, "run_dir": "r", "cost_usd_run": 6.0}) + "\n")
    monkeypatch.setattr(rb, "Orchestrator", Fake)
    monkeypatch.setattr(rb, "STORE_V2", store)
    monkeypatch.setattr(rb.sys, "argv", ["run_benchmark.py", "--v2", "kipf_gcn_2017"])
    rb.main()
    assert [k["graph_repair"] for k in made] == [True, True]
    assert [k["use_tavily"] for k in made] == [False, True] and all(k["tavily_mode"] == "fixture" for k in made)
    assert "Budget-Grenze v2 erreicht" in capsys.readouterr().out


def test_v1_main_builds_arms_without_graph_repair(monkeypatch, tmp_path):
    made = []

    class Fake:
        def __init__(self, **kw):
            made.append(kw)

    store = tmp_path / "benchmark.jsonl"
    rows = [{"paper": "kipf_gcn_2017", "arm": a, "generation": g} for a in rb.ARMS for g in range(3)]
    store.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(rb, "Orchestrator", Fake)
    monkeypatch.setattr(rb, "STORE", store)
    monkeypatch.setattr(rb.sys, "argv", ["run_benchmark.py", "kipf_gcn_2017"])
    rb.main()
    assert len(made) == 2 and all("graph_repair" not in k for k in made)


def test_load_rows_skips_broken_lines(tmp_path):
    f = tmp_path / "s.jsonl"
    f.write_text('{"a": 1}\n\n{"b": 2}\n{"half": ')
    assert rb.load_rows(f) == [{"a": 1}, {"b": 2}] and rb.load_rows(tmp_path / "missing.jsonl") == []


class ParallelOrch:
    """Attrappe: Aufruf n schreibt, als haette ein paralleler Prozess gearbeitet, Zeilen in den Store (Hook `foreign`)."""
    calls, foreign = [], None

    def __init__(self, **kw):
        self.kw, self._snapshots = kw, None

    def run(self, paper):
        ParallelOrch.calls.append((paper, self.kw["use_tavily"]))
        if ParallelOrch.foreign:
            ParallelOrch.foreign(len(ParallelOrch.calls))
        return type("D", (), {"verdicts": [], "claims": [], "ablations": {}})()


def run_main(monkeypatch, tmp_path, foreign):
    import sys

    store = tmp_path / "v2.jsonl"
    monkeypatch.setattr(rb, "STORE_V2", store)
    monkeypatch.setattr(rb, "Orchestrator", ParallelOrch)
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--v2", "kipf_gcn_2017"])
    ParallelOrch.calls, ParallelOrch.foreign = [], lambda n: foreign(store, n)
    rb.main()
    return store


def row(arm, gen, cost=0.1):
    return {"paper": "kipf_gcn_2017", "arm": arm, "generation": gen, "run_dir": f"runs/x/{arm}{gen}", "cost_usd_run": cost}


def test_main_skips_runs_finished_by_a_parallel_process(monkeypatch, tmp_path):
    def foreign(store, n):
        if n == 1:   # waehrend Lauf 1 (A, Gen 0) beendet ein anderer Prozess B Gen 0 und A Gen 1
            with store.open("a") as f:
                f.write(json.dumps(row("B_mit_tavily", 0)) + "\n" + json.dumps(row("A_ohne_tavily", 1)) + "\n")

    run_main(monkeypatch, tmp_path, foreign)
    # Reihenfolge je Generierung: A, B. Gen 0: A laeuft, B uebersprungen; Gen 1: A uebersprungen, B laeuft; Gen 2: A, B laufen
    assert [use for _, use in ParallelOrch.calls] == [False, True, False, True]


def test_main_stops_when_parallel_costs_reach_the_budget(monkeypatch, tmp_path, capsys):
    def foreign(store, n):
        with store.open("a") as f:   # ein anderer Prozess hat das Budget aufgebraucht
            f.write(json.dumps(row("B_mit_tavily", 0, cost=100.0)) + "\n")

    run_main(monkeypatch, tmp_path, foreign)
    assert len(ParallelOrch.calls) == 1 and "Budget-Grenze v2 erreicht" in capsys.readouterr().out
