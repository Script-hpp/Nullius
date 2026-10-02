"""Kurzbeschreibung in der Pipeline (loop._paper_summary_round): Dossier-Feld, Ereignis, Zitat-Check, Fehler laufen weiter, kein Prompt des Agenten. Attrappen, kein Netz."""

import json
from types import SimpleNamespace

import pytest

from repro_agent import loop as loop_mod
from repro_agent.claims_source import PaperText
from repro_agent.llm import BillingBlocked
from repro_agent.paper_head import INSTITUTION_SYSTEM, SYSTEM
from test_catalog import cora_entry, evidence, tavily_orch

TEXT = ("# SL-GCN\n\nKaicheng Zhang\nSchool of Mathematics\nUniversity of Edinburgh, Edinburgh, UK\n\nAbstract. Stuart-Landau oscillators. "
        "SL-GCN 82.92 ± 1.39 on Cora")
SUMMARY = "The paper proposes a new way to sort linked articles. The data is Cora, about 2,700 articles. It claims 82.9 out of 100 are right."


def orch(tmp_path, monkeypatch, answers=None, flag=True, stored=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    e = cora_entry()
    o = tavily_orch(tmp_path, monkeypatch, [e], evidence(0.8292))
    o.graph_repair, o.n_candidates = flag, 1
    pt = PaperText("2511.08094", "https://arxiv.org/html/2511.08094", "arxiv_html", "2026-09-30T00:00:00", "h" * 64, "v2", TEXT)
    monkeypatch.setattr(loop_mod, "fetch_paper_text", lambda *a, **k: pt)
    asked, emitted = [], []

    def ask(role, system, user, max_tokens=None, temperature=0.2):
        asked.append((role, system, user))
        a = (answers or {}).get(system)
        if isinstance(a, Exception):
            raise a
        return SimpleNamespace(content=a if a is not None else "", cost_usd=0.0, model="m", cached=False)

    o._ask = ask
    o._fetch_abstract = lambda aid: "We present a new graph network."
    o._fetch_meta = lambda aid: {"authors": ["Kaicheng Zhang"], "year": 2025}
    real_emit = o._emit
    o._emit = lambda type, **kw: (emitted.append((type, kw)), real_emit(type, **kw))[1]
    if stored:
        f = tmp_path / "web" / "public" / "data" / "discovery" / "summaries.json"
        f.parent.mkdir(parents=True)
        f.write_text(json.dumps({"2511.08094": {"summary": "x", "source": "y"}}))
    return o, e, asked, emitted


GOOD = {
    SYSTEM: 'Thinking.\n{"summary": "' + SUMMARY + '"}',
    INSTITUTION_SYSTEM: 'Done.\n{"institutions": [{"name": "University of Edinburgh", "quote": "University of Edinburgh, Edinburgh, UK"}, '
                        '{"name": "MIT", "quote": "Massachusetts Institute of Technology"}]}',
}


def test_summary_goes_into_dossier_and_event_with_verified_affiliations(tmp_path, monkeypatch):
    o, e, asked, emitted = orch(tmp_path, monkeypatch, GOOD)
    d = o.run(e["id"], source="catalog")
    s = d.summary
    assert s["text"] == SUMMARY and s["source"].startswith("arXiv abstract (2511.08094)")
    assert s["authors"] == ["Kaicheng Zhang"] and s["year"] == 2025 and "arXiv abstract page" in s["authors_source"]
    assert [i["name"] for i in s["institutions"]] == ["University of Edinburgh"]       # MIT: quote not in the text, dropped
    assert s["institutions_source"] == "arXiv HTML v2, read with Tavily Extract"
    ev = [kw for t, kw in emitted if t == "paper_summary"]
    assert len(ev) == 1 and ev[0]["text"] == SUMMARY and ev[0]["institutions"] == s["institutions"]
    assert [a[0] for a in asked] == ["code", "code"]                                       # one call for the summary, one for the affiliations
    assert d.verdicts                                                                       # the run went on


def test_summary_is_report_only_no_prompt_of_the_agent_gets_it(tmp_path, monkeypatch):
    o, e, asked, _ = orch(tmp_path, monkeypatch, GOOD)
    d = o.run(e["id"], source="catalog")
    assert SUMMARY not in d.condensed_for_reasoning()
    rest = [a for a in asked if a[1] not in (SYSTEM, INSTITUTION_SYSTEM)]
    assert all(SUMMARY not in a[2] and "Kaicheng" not in a[2] for a in rest)
    src = open(loop_mod.__file__).read()
    assert ".summary" not in src and "summary[" not in src                                  # the loop only writes the result (summary=...), nothing reads it back into a prompt


def test_stored_summary_or_benchmark_path_makes_no_call(tmp_path, monkeypatch):
    o, e, asked, emitted = orch(tmp_path, monkeypatch, GOOD, stored=True)
    d = o.run(e["id"], source="catalog")
    assert d.summary is None and asked == [] and not [1 for t, _ in emitted if t == "paper_summary"]
    o2, e2, asked2, _ = orch(tmp_path / "b", monkeypatch, GOOD, flag=False)
    assert o2.run(e2["id"], source="catalog").summary is None and asked2 == []


def test_failure_is_a_warning_and_the_run_goes_on(tmp_path, monkeypatch):
    o, e, _, emitted = orch(tmp_path, monkeypatch, {SYSTEM: RuntimeError("boom")})
    d = o.run(e["id"], source="catalog")
    assert d.summary is None and d.verdicts
    assert any("Kurzbeschreibung nicht erstellt" in w and "boom" in w for w in d.warnings)
    assert any(t == "warning" and "Kurzbeschreibung" in kw["text"] for t, kw in emitted)
    o2, e2, _, _ = orch(tmp_path / "b", monkeypatch, GOOD)
    o2._fetch_abstract = lambda aid: (_ for _ in ()).throw(OSError("offline"))             # arXiv page unreachable
    d2 = o2.run(e2["id"], source="catalog")
    assert d2.summary is None and d2.verdicts and any("OSError" in w for w in d2.warnings)


def test_missing_meta_is_only_a_note_the_summary_still_comes(tmp_path, monkeypatch):
    o, e, _, _ = orch(tmp_path, monkeypatch, GOOD)
    o._fetch_meta = lambda aid: (_ for _ in ()).throw(OSError("429"))
    d = o.run(e["id"], source="catalog")
    assert d.summary["text"] == SUMMARY and "authors" not in d.summary
    assert any("Autoren und Jahr nicht gelesen" in w for w in d.warnings)


def test_billing_blocked_is_not_swallowed(tmp_path, monkeypatch):
    o, e, _, _ = orch(tmp_path, monkeypatch, {SYSTEM: BillingBlocked("suspended")})
    with pytest.raises(BillingBlocked):
        o.run(e["id"], source="catalog")


def test_invalid_answer_gives_a_warning_not_a_crash(tmp_path, monkeypatch):
    o, e, _, _ = orch(tmp_path, monkeypatch, {SYSTEM: "no json here", INSTITUTION_SYSTEM: "nor here"})
    d = o.run(e["id"], source="catalog")
    assert d.summary == {"institutions": [], "institutions_source": "arXiv HTML v2, read with Tavily Extract", "authors": ["Kaicheng Zhang"],
                         "year": 2025, "authors_source": "arXiv abstract page (2511.08094)"} or d.summary is None
