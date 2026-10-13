"""A persistent data volume must receive corrected definitions without losing evidence."""
import json

from repro_agent import catalog, site_export


def test_redeploy_corrects_fits_setup_preserves_outcomes_and_live_only_rows(tmp_path):
    out = tmp_path / "data"
    out.mkdir()
    old = {"id": "disc_2307_03756", "title": "FITS", "claim": {"target": .372, "input_length": 720},
           "status": "geprueft", "verdict": {"measured_mean": .374}, "run_id": "fits--timestamp",
           "cost_gross_usd": .12, "checked_at": "2026-10-13", "head": {"text": "saved summary"}}
    live = {"id": "live-candidate", "status": "geprueft", "verdict": {"measured_mean": .38}}
    (out / "catalog.json").write_text(json.dumps([old, live]))
    index = b'[{"id":"finished-run"}]'
    (out / "index.json").write_bytes(index)
    site_export.refresh_catalog_metadata(catalog.CATALOG_YAML, out)
    rows = json.loads((out / "catalog.json").read_text())
    fits = next(row for row in rows if row["id"] == old["id"])
    assert fits["claim"]["input_length"] == 360 and fits["claim"]["target"] == .372
    for key in ("status", "verdict", "run_id", "cost_gross_usd", "checked_at", "head"):
        assert fits[key] == old[key]
    assert live in rows and (out / "index.json").read_bytes() == index
    before = (out / "catalog.json").read_bytes()
    site_export.refresh_catalog_metadata(catalog.CATALOG_YAML, out)
    assert (out / "catalog.json").read_bytes() == before


def test_new_export_contains_current_catalog_definitions(tmp_path):
    site_export.refresh_catalog_metadata(catalog.CATALOG_YAML, tmp_path)
    rows = json.loads((tmp_path / "catalog.json").read_text())
    assert len(rows) == len(catalog.load_catalog())
