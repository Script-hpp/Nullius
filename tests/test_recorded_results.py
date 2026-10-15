"""Recorded catalog results (TSMixer, SparseTSF, BANGS) are shown with date and note, but never offered for a new check."""
import json

import pytest

from repro_agent import api, catalog as cat, live_catalog, site_export
from test_api_repro import env, FakeOrch, start  # noqa: F401 (pytest fixture)
from test_catalog import cand

RECORDED = ("disc_2303_06053", "disc_2405_00946", "disc_2410_09348_d8991277dd33d43f")


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(live_catalog, "PATH", tmp_path / "logs" / "live_candidates.json")
    monkeypatch.setattr(live_catalog, "PREFLIGHT_DIR", tmp_path / "logs" / "preflight")


def by_id():
    return {e["id"]: e for e in cat.load_catalog(cat.CATALOG_YAML)}


def test_recorded_entries_are_locked_dated_and_explained():
    entries = by_id()
    for cid in RECORDED:
        entry = entries[cid]
        assert entry["status"] == cat.CHECKED and entry["verdict"]["run_dir"]
        assert cat.jury_ready(entry)[0] is False
        assert live_catalog.curated_offer(entry)["runnable"] is False
        assert entry["recorded"]["date"] and len(entry["recorded"]["note"]) > 40
    assert "1 mixer block" in entries[RECORDED[0]]["recorded"]["note"] and "128" in entries[RECORDED[0]]["recorded"]["note"]
    assert "linear variant" in entries[RECORDED[1]]["recorded"]["note"] and "720" in entries[RECORDED[1]]["recorded"]["note"]
    assert "80 minutes" in entries[RECORDED[2]]["jury_override"]["reason"]
    assert entries[RECORDED[2]]["verdict"]["urteil"] == "teilweise"
    # historical numbers stay as recorded
    assert entries[RECORDED[0]]["verdict"]["measured_mean"] == 0.362416
    assert entries[RECORDED[1]]["verdict"]["measured_mean"] == 0.365314


def test_jury_set_is_exactly_the_three_featured_checks():
    ready = {cid for cid, e in by_id().items() if cat.jury_ready(e)[0]}
    assert ready == {"disc_1609_02907", "disc_2307_03756", "disc_2501_01087"}


def test_catalog_json_carries_recorded_and_redeploy_applies_it(tmp_path):
    rows = {r["id"]: r for r in site_export.catalog_json(site_export.catalog_entries(cat.CATALOG_YAML))}
    assert all(rows[cid]["recorded"]["date"] for cid in RECORDED)
    assert rows["disc_2307_03756"]["recorded"] is None
    (tmp_path / "catalog.json").write_text(json.dumps([{"id": RECORDED[0], "jury": None}]))
    site_export.refresh_catalog_metadata(cat.CATALOG_YAML, tmp_path)
    stored = {r["id"]: r for r in json.loads((tmp_path / "catalog.json").read_text())}
    assert stored[RECORDED[0]]["recorded"]["note"].startswith("Target reached")


def test_saved_live_find_does_not_duplicate_or_unlock_a_curated_entry(env, tmp_path):
    event = live_catalog.save_candidate(cand())
    live = live_catalog.find_entry(event["catalog_id"])
    live["readiness"] = {"ready": True, "stage": "ready", "reason": None}
    curated = {**live, "status": cat.CHECKED, "recorded": {"date": "2026-10-15", "note": "x" * 50},
               "jury_override": {"ready": False, "reason": "Takes about 80 minutes on CPU."}}
    entries = cat.load_catalog(env.cat) + [curated]
    cat.save_catalog(entries, env.cat)
    assert live_catalog.offer(live)["runnable"] is True   # alone, the saved find would get a start button
    eligible = env.client.get("/api/reproduce/eligibility").json()
    assert eligible[curated["id"]] == {"runnable": False, "reason": "Takes about 80 minutes on CPU.", "warnings": []}
    response = start(env, catalog_id=curated["id"])
    assert response.status_code == 400 and "80 minutes" in response.json()["detail"]
    assert FakeOrch.instances == []
    site_export.write_catalog(env.cat, tmp_path / "out", [live])
    rows = json.loads((tmp_path / "out" / "catalog.json").read_text())
    assert [r["id"] for r in rows].count(curated["id"]) == 1
