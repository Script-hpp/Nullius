"""Export fuer die statische UI: catalog.json und index.json (Tmp-Verzeichnis, keine echten Daten veraendert)."""

import importlib.util
import json
import shutil
from pathlib import Path

import yaml

from repro_agent import catalog as cat
from test_catalog import cand, dossier_with, entry

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_site_data", ROOT / "scripts" / "export_site_data.py")
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


def make_catalog(tmp_path):
    a = entry(cand())
    cat.record_result(a, cat.CHECKED, dossier=dossier_with("nicht reproduziert", (0.78, 0.79)), run_dir=ROOT / "runs" / "disc_2511_08094" / "20260930-200050",
                      cost_gross_usd=0.0246)
    b = entry(cand("2001.00002"))
    cat.record_result(b, cat.NOT_RUNNABLE, reason="Cora ohne matched_task")
    f = tmp_path / "catalog.yaml"
    cat.save_catalog([a, b], f)
    return f


def test_catalog_only_writes_catalog_json_and_nothing_else(tmp_path):
    out = tmp_path / "data"
    exp.main(["--catalog-only"], out=out, catalog_yaml=make_catalog(tmp_path))
    assert [p.name for p in out.iterdir()] == ["catalog.json"]
    a, b = json.loads((out / "catalog.json").read_text())
    assert a["id"] == "disc_2511_08094" and a["task"] == "Cora-LCC" and a["status"] == "geprueft"
    assert a["claim"]["method"] == "SL-GCN" and a["claim"]["target"] == 82.92 and "82.92" in a["claim"]["quote"]
    assert a["protocol"] == "passt" and a["column_check"] == "ok" and a["github_url"] and a["source_url"]
    assert a["verdict"]["urteil"] == "nicht reproduziert" and a["verdict"]["run_dir"] == "runs/disc_2511_08094/20260930-200050"
    assert a["run_id"] == "disc_2511_08094--20260930-200050" and a["jury"]["ready"] is True and a["cost_gross_usd"] == 0.0246
    assert b["status"] == "nicht_laufbar" and b["reason"] == "Cora ohne matched_task" and b["verdict"] is None and b["run_id"] is None
    assert "imported_from" not in a and "community" not in a                     # nur Felder, die die UI braucht


def test_index_titles_and_gesamtaussage_for_disc_runs(tmp_path):
    out = tmp_path / "data"
    run = out / "disc_2511_08094" / "20260930-200050"
    run.mkdir(parents=True)
    shutil.copy(ROOT / "web/public/data/disc_2511_08094/20260930-200050/dossier.json", run / "dossier.json")
    (out / "index.json").write_text(json.dumps([
        {"id": "disc_2511_08094--20260930-200050", "paper_id": "disc_2511_08094", "title": "disc_2511_08094", "timestamp": "20260930-200050"},
        {"id": "kipf--1", "paper_id": "kipf_gcn_2017", "title": "Kipf", "timestamp": "1"}]))
    exp.main([], out=out, catalog_yaml=make_catalog(tmp_path))
    disc, kipf = json.loads((out / "index.json").read_text())
    assert disc["title"] == "Paper 2511.08094" and disc["catalog_id"] == "disc_2511_08094"
    assert disc["gesamtaussage"] in ("vorlaeufig", "nicht bewertbar (Methode nicht umgesetzt)")
    assert kipf == {"id": "kipf--1", "paper_id": "kipf_gcn_2017", "title": "Kipf", "timestamp": "1"}   # Benchmark unveraendert


def test_missing_catalog_gives_empty_json(tmp_path):
    out = tmp_path / "data"
    exp.main(["--catalog-only"], out=out, catalog_yaml=tmp_path / "none.yaml")
    assert json.loads((out / "catalog.json").read_text()) == []


def full(claims):
    """Complete claim dicts (Dossier.load, used by the badge, needs every field)."""
    return [{"text": "t", "metric": "accuracy", "target": 0.8, "tolerance": 0.01, "source": "s", **c} for c in claims]


def run_dir(tmp_path, claims, plans):
    claims = full(claims)
    run = tmp_path / "runs" / "p1" / "20261001-000000"
    run.mkdir(parents=True)
    (run / "dossier.json").write_text(json.dumps({"paper_id": "p1", "claims": claims, "attempts": [], "verdicts": []}))
    for name, text in plans.items():
        (run / name).write_text(text)
    return run


def test_export_copies_the_plan_of_the_first_paper_claim_not_a_placebos(tmp_path):
    from repro_agent import site_export as se
    claims = [{"id": "c_placebo", "origin": "placebo"}, {"id": "c", "origin": "paper"}, {"id": "c_obj", "origin": "einwand"}]
    run = run_dir(tmp_path, claims, {"c_plan.md": "PAPER PLAN", "c_placebo_plan.md": "PLACEBO PLAN", "c_obj_plan.md": "OBJECTION PLAN"})
    for _ in range(3):   # the glob order of the file system must not matter
        out = tmp_path / "out"
        se.export_run(run, out, {})
        assert (out / "p1" / "20261001-000000" / "plan.md").read_text() == "PAPER PLAN"


def test_export_without_the_claims_plan_copies_nothing_when_several_plans_exist(tmp_path):
    from repro_agent import site_export as se
    run = run_dir(tmp_path, [{"id": "c", "origin": "paper"}], {"x_plan.md": "X", "y_plan.md": "Y"})
    se.export_run(run, tmp_path / "out", {})
    assert not (tmp_path / "out" / "p1" / "20261001-000000" / "plan.md").exists()
    run2 = run_dir(tmp_path / "two", [{"id": "c", "origin": "paper"}], {"only_plan.md": "ONLY"})   # a single plan is taken as before
    se.export_run(run2, tmp_path / "out2", {})
    assert (tmp_path / "out2" / "p1" / "20261001-000000" / "plan.md").read_text() == "ONLY"


def test_catalog_entries_get_head_from_exported_dossier_summary(tmp_path):
    out = tmp_path / "data"
    run = out / "disc_2511_08094" / "20260930-200050"
    run.mkdir(parents=True)
    summary = {"text": "T", "authors": ["A B"], "authors_source": "abs", "year": 2025, "institutions": [{"name": "Uni X", "quote": "Uni X"}],
               "institutions_source": "pdf", "source": "abs", "generated_by": "m", "extra": "dropped"}
    (run / "dossier.json").write_text(json.dumps({"summary": summary}))
    exp.main(["--catalog-only"], out=out, catalog_yaml=make_catalog(tmp_path))
    a, b = json.loads((out / "catalog.json").read_text())
    assert a["head"] == {k: v for k, v in summary.items() if k != "extra"}
    assert "head" not in b                                                       # no run, no head, nothing invented


def test_catalog_entry_without_summary_has_no_head(tmp_path):
    out = tmp_path / "data"
    run = out / "disc_2511_08094" / "20260930-200050"
    run.mkdir(parents=True)
    (run / "dossier.json").write_text(json.dumps({"summary": {"institutions": []}}))
    exp.main(["--catalog-only"], out=out, catalog_yaml=make_catalog(tmp_path))
    assert all("head" not in e for e in json.loads((out / "catalog.json").read_text()))
