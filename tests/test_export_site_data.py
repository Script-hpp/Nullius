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
