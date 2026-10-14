"""Referenz-Papers im Katalog (origin reference): Laden, Adapter, Zitate gegen die Fixtures, Import, Jury-Filter, Benchmark-Kennzahl. Kein Netz, kein Geld."""

import json
from pathlib import Path

import pytest

from repro_agent import catalog as cat
from repro_agent import site_export as se
from repro_agent.discovery import _quote_in_text
from test_catalog import cand, entry

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {   # arxiv_id -> (Task, Seeds, Fixture-Text)
    "1609.02907": ("Cora", 10, "1609.02907.json"),
    "1811.05868": ("Cora-LCC", 30, "1811.05868.json"),
}


def refs_of(entries):
    # first entry per arXiv id wins: the Kipf CiteSeer entry (case-citeseer) shares the id with the Kipf Cora entry
    out = {}
    for e in entries:
        if cat.origin_of(e) == cat.REFERENCE:
            out.setdefault(e["arxiv_id"], e)
    return out


def reference_entries():
    return refs_of(cat.load_catalog())


def test_origin_defaults_to_discovery():
    assert cat.origin_of(entry(cand())) == cat.DISCOVERY
    assert cat.origin_of({"id": "x", "origin": None}) == cat.DISCOVERY
    assert cat.origin_of({"id": "x", "origin": "reference"}) == cat.REFERENCE


def test_committed_catalog_has_both_references():
    refs = reference_entries()
    assert set(refs) == set(REFERENCES)
    for e in refs.values():
        # kein erfundenes Urteil: entweder ungeprueft ohne Urteil, oder geprueft mit Urteil aus einem echten Lauf
        if e["status"] == cat.UNCHECKED:
            assert "verdict" not in e and "jury" not in e
        else:
            assert e["status"] == cat.CHECKED and e["verdict"].get("run_dir")


@pytest.mark.parametrize("arxiv_id", REFERENCES)
def test_build_paper_accepts_reference(arxiv_id):
    task, seeds, _ = REFERENCES[arxiv_id]
    paper, claims = cat.build_paper(reference_entries()[arxiv_id])
    c = claims[0]
    assert paper["dataset"] == task and paper["seeds"] == seeds
    assert c.target == pytest.approx(0.815) and c.tolerance == 0.01 and c.metric == "accuracy"
    assert c.source.startswith("reference (benchmark papers.yaml)")
    assert not any("Protokoll" in w or "Tabellenspalte" in w for w in paper["warnings"])   # belegt, keine Warnung


@pytest.mark.parametrize("arxiv_id", REFERENCES)
def test_quotes_stand_in_the_stored_paper_text(arxiv_id):
    e = reference_entries()[arxiv_id]
    text = json.loads((ROOT / "benchmark" / "fixtures" / REFERENCES[arxiv_id][2]).read_text())["text"]
    claim = e["candidate"]["claim"]
    assert _quote_in_text(claim["quote"], text)
    assert _quote_in_text(claim["protocol_quote"], text)
    assert e["candidate"]["protocol"] == "passt" and claim["column_check"] == "ok"


def test_import_never_overwrites_or_duplicates_a_reference(tmp_path):
    entries = cat.load_catalog()
    before = json.dumps(reference_entries(), sort_keys=True)
    n = len(entries)
    clash = cand("1609.02907", "Cora", title="Other title", claim={"method": "GCN", "quote": "x 1", "metric": "accuracy", "target": 99})
    other = cand("1811.05868", "Cora")
    assert cat.import_candidates(entries, [(clash, "t"), (other, "t")]) == 0
    assert len(entries) == n and json.dumps(refs_of(entries), sort_keys=True) == before


def test_unchecked_reference_is_not_jury_ready():
    for e in reference_entries().values():
        fresh = {k: v for k, v in e.items() if k not in ("verdict", "jury", "jury_override")} | {"status": cat.UNCHECKED}
        assert cat.jury_ready(fresh) == (False, "Status ungeprueft")


# ---- Benchmark-Kennzahl im Export --------------------------------------------------
def bench(verdicts_b=("reproduziert", "reproduziert", "teilweise"), majority_b="reproduziert"):
    def unit(paper, claim, ref, b_verdicts, b_major):
        return {"paper": paper, "claim": claim, "reference_verdict": ref, "arms": {"A": {"majority": "teilweise", "verdicts": ["teilweise"] * 3},
                                                                                  "B": {"majority": b_major, "verdicts": list(b_verdicts)}}}
    return {"versions": [{"id": "v4", "units": [unit("p1", "c1", "reproduziert", verdicts_b, majority_b),
                                                 unit("p2", "c2", "reproduziert", ["teilweise"] * 3, "teilweise")]}]}


def ref_entry(version="v4", paper="p1", claim="c1"):
    return {"id": "disc_x", "benchmark_unit": {"version": version, "paper": paper, "claim": claim}}


def test_reference_benchmark_is_computed_from_verdicts():
    assert se.reference_benchmark(ref_entry(), bench()) == {
        "version": "v4", "arm": "with Tavily", "reached": 2, "runs": 3, "units_matched": 1, "units": 2}


@pytest.mark.parametrize("e,b", [(ref_entry(version="v9"), bench()), (ref_entry(paper="nope"), bench()), ({"id": "d"}, bench()), (ref_entry(), None),
                                 (ref_entry(), {"versions": []})])
def test_reference_benchmark_missing_source_gives_none(e, b):
    assert se.reference_benchmark(e, b) is None


def test_committed_catalog_json_matches_benchmark_json():
    data = {e["id"]: e for e in json.loads((ROOT / "web" / "public" / "data" / "catalog.json").read_text())}
    for arxiv_id in REFERENCES:
        e = data[cat.catalog_id(arxiv_id)]
        assert e["origin"] == "reference" and e["benchmark"]["version"] == "v4" and e["benchmark"]["units"] == 8
    assert data["disc_1609_02907"]["benchmark"]["reached"] == 3 and data["disc_1811_05868"]["benchmark"]["reached"] == 2
    # Other entries carry no benchmark line; a second claim of a reference paper (Kipf on CiteSeer) keeps origin "reference".
    others = [e for e in data.values() if e["id"] not in {cat.catalog_id(a) for a in REFERENCES}]
    assert all(e["benchmark"] is None for e in others)
    assert all(e["origin"] == "discovery" or e.get("arxiv_id") in REFERENCES for e in others)
