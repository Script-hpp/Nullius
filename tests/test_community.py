"""Community-Check nach dem Urteil: nur Bericht, Zitatpruefung, Cache, Kontingent. Attrappen, kein Netz, keine echten Tavily-Aufrufe."""

import dataclasses
import inspect
import json
import re
import threading
import time
from types import SimpleNamespace

import pytest

from repro_agent import community
from repro_agent.dossier import Dossier, Verdict
from repro_agent.llm import BillingBlocked
from repro_agent.loop import Orchestrator
from repro_agent.report import render_markdown

TITLE, ARXIV = "GraphNet: Better Node Classification", "2501.00001"
ISSUE = "The reported test accuracy uses drop_last=True in the test loader, so the last batch is dropped."
REPORT = "We could not reproduce the 83.0 accuracy with the released code."
PAGE_ISSUE = f"GraphNet: Better Node Classification (arXiv 2501.00001) github issue. {ISSUE} Thanks."
PAGE_REPORT = f"Review of GraphNet: Better Node Classification. {REPORT}"
PAGE_OTHER_PAPER = f"Some other paper entirely. {ISSUE}"
ERRATUM = "Erratum: Table 2 of GraphNet: Better Node Classification was recomputed; the correct value is 81.9."


class FakeTools:
    def __init__(self, search_pages=None, crawl_pages=None, read_pages=None, explain_pages=None, repo_pages=None):
        self.calls = []
        self.search_pages = search_pages if search_pages is not None else [("https://github.com/a/b/issues/1", PAGE_ISSUE), ("https://example.org/r", PAGE_REPORT)]
        self.crawl_pages, self.read, self.explain_pages, self.repo_pages = crawl_pages or [], read_pages or [], explain_pages or [], repo_pages or []

    @staticmethod
    def _res(pages):
        return {"results": [{"url": u, "raw_content": t} for u, t in pages]}

    def search_community(self, title, limit=5):
        self.calls.append(("search", title))
        return self._res(self.search_pages)

    def search_repo_issues(self, slug, limit=5):
        self.calls.append(("repo_search", slug))
        return self._res(self.repo_pages)

    def crawl_issues(self, repo_url, limit=3):
        self.calls.append(("crawl", repo_url, limit))
        return self._res(self.crawl_pages)

    def read_pages(self, urls, limit=3):
        self.calls.append(("extract", tuple(urls)))
        return self._res(self.read)

    def clarify_discrepancy(self, title, max_results=3):
        self.calls.append(("explain", title))
        return self._res(self.explain_pages)


class FakeLLM:
    """Gibt fuer jeden Chunk die Aussagen zurueck, deren Zitat (Teilsatz) im Chunk steht, plus eine erfundene Aussage."""

    def __init__(self, quotes, fabricate=True):
        self.quotes, self.fabricate, self.n = quotes, fabricate, 0

    def complete(self, role, messages, **kw):
        self.n += 1
        user = messages[-1]["content"]
        explain = "explain such a difference" in user
        out = [{"topic": "t", ("explanation" if explain else "statement"): "s", "quote": q} for q in self.quotes if q in user]
        if self.fabricate:
            out.append({"topic": "t", ("explanation" if explain else "statement"): "made up", "quote": "this sentence is nowhere in the page"})
        return SimpleNamespace(content="thinking...\n" + json.dumps(out))


def orch(tools, llm, flag=True, tavily=True, mode="live", cache=None, monkeypatch=None):
    o = Orchestrator.__new__(Orchestrator)
    o.graph_repair, o.use_tavily, o.tavily_mode = flag, tavily, mode
    o._tavily_calls, o._repo_files, o.llm = 0, None, llm
    o._lock = threading.Lock()
    o.events_seen, o.warnings = [], []
    o._emit = lambda type, **kw: o.events_seen.append((type, kw))
    o._warn = lambda text: o.warnings.append(text)
    o._tavily_getter = lambda: (lambda: tools)
    return o


@pytest.fixture(autouse=True)
def cache_file(monkeypatch, tmp_path):
    monkeypatch.setattr(community, "CACHE_PATH", tmp_path / "community_cache.json")
    return tmp_path / "community_cache.json"


def paper(community_hits=None):
    return {"title": TITLE, "dataset": "Cora", "arxiv_url": f"https://arxiv.org/abs/{ARXIV}", "community": community_hits or []}


def dossier(urteil="reproduziert", repo=None):
    v = Verdict("c", urteil, 0.8, 0.01, "x")
    return Dossier(paper_id="p", verdicts=[v], repo=repo)


def types(o, name):
    return [kw for t, kw in o.events_seen if t == name]


def test_findings_need_quote_in_text_and_page_must_name_the_paper():
    tools = FakeTools(search_pages=[("https://github.com/a/b/issues/1", PAGE_ISSUE), ("https://example.org/r", PAGE_REPORT),
                                    ("https://example.org/other", PAGE_OTHER_PAPER)])
    o, d = orch(tools, FakeLLM([ISSUE, REPORT])), dossier()
    o._community_round(paper(), d)
    f = d.community["findings"]
    assert {x["url"] for x in f} == {"https://github.com/a/b/issues/1", "https://example.org/r"}      # fremde Seite nie gelesen
    assert all(x["quote"] in (PAGE_ISSUE + PAGE_REPORT) for x in f)
    assert d.community["dropped"] == 2                                                                  # je Seite die erfundene Aussage
    assert {x["url"]: x["kind"] for x in f} == {"https://github.com/a/b/issues/1": "protocol note", "https://example.org/r": "reproduction report"}
    assert d.community["mode"] == "live" and len(d.community["pages"]) == 2 and d.community["pages"][0]["sha256"]
    assert [s["kind"] for s in d.sources] == ["community", "community"]


def test_classify_is_deterministic_keywords():
    assert community.classify("x", "the test set was reused", True) == "protocol note"
    assert community.classify("authors could not reproduce the number", "q", False) == "reproduction report"
    assert community.classify("nice paper", "q", False) == "other"


def test_no_llm_classification_calls_beyond_extraction():
    tools = FakeTools()
    llm = FakeLLM([ISSUE])
    orch(tools, llm)._community_round(paper(), dossier())
    assert llm.n == 2                                                                                  # ein extract-Aufruf je Seite, sonst keiner


def test_repo_issues_by_search_crawl_only_as_fallback():
    repo = {"url": "https://github.com/a/b"}
    # Search ohne Issue-Treffer des Repos, Repo-Suche liefert eine Issue-Seite: kein Crawl
    tools = FakeTools(search_pages=[("https://example.org/r", PAGE_REPORT)], repo_pages=[("https://github.com/a/b/issues/7", "Issue about reproduce accuracy " + ISSUE)])
    d = dossier(repo=repo)
    orch(tools, FakeLLM([REPORT, ISSUE]))._community_round(paper(), d)
    assert [c[0] for c in tools.calls] == ["search", "repo_search"] and tools.calls[1][1] == "a/b"
    assert "https://github.com/a/b/issues/7" in [p["url"] for p in d.community["pages"]]            # Repo-Seite braucht den Titel nicht
    # auch die Repo-Suche leer: Crawl als Rueckfall, limit 3
    community.CACHE_PATH.unlink(missing_ok=True)
    tools2 = FakeTools(search_pages=[("https://example.org/r", PAGE_REPORT)])
    orch(tools2, FakeLLM([REPORT]))._community_round(paper(), dossier(repo=repo))
    assert [c[0] for c in tools2.calls] == ["search", "repo_search", "crawl"] and tools2.calls[2][2] == community.MAX_CRAWL == 3
    # Search lieferte schon eine Issue-Seite dieses Repos: weder Repo-Suche noch Crawl
    community.CACHE_PATH.unlink(missing_ok=True)
    tools3 = FakeTools()
    orch(tools3, FakeLLM([ISSUE]))._community_round(paper(), dossier(repo=repo))
    assert [c[0] for c in tools3.calls] == ["search"]
    # kein Repo: nie Repo-Suche oder Crawl
    community.CACHE_PATH.unlink(missing_ok=True)
    tools4 = FakeTools(search_pages=[("https://example.org/r", PAGE_REPORT)])
    orch(tools4, FakeLLM([REPORT]))._community_round(paper(), dossier())
    assert [c[0] for c in tools4.calls] == ["search"]


FIXTURE = json.loads((__import__("pathlib").Path(__file__).parent / "fixtures" / "community_kipf_raw.json").read_text())


def test_real_kipf_raw_data_citation_pages_dropped_repo_issues_kept():
    """Rohdaten des ersten echten Aufrufs (Kipf GCN, tkipf/gcn): Zitierseiten und die eigene arXiv-Seite fallen raus, Repo-Issues bleiben."""
    search = FIXTURE["search"]["results"]
    crawl = FIXTURE["crawl"]["results"]
    kept = community.drop_noise({"results": search}, "1609.02907")["results"]
    urls = [r["url"] for r in kept]
    assert not any("scirp.org" in u or "semanticscholar" in u or "arxiv.org/abs/1609.02907" in u for u in urls)
    assert "https://ojs.aaai.org/index.php/AAAI/article/view/11604/11463" in urls and "https://tkipf.github.io/graph-convolutional-networks" in urls
    title = "Semi-Supervised Classification with Graph Convolutional Networks"
    kws = community.keywords(title, "1609.02907")
    # alter Filter: die kurzen Repo-Issues nennen den Titel nicht und fielen weg
    from repro_agent.objections import docs_from_collect
    assert not [d for d in docs_from_collect({"crawl": {"results": crawl}}, kws) if "github.com/tkipf/gcn/issues" in d.name]
    kept_issues = docs_from_collect({"crawl": {"results": crawl}}, kws, ("https://github.com/tkipf/gcn/",))
    assert {d.name for d in kept_issues} == {r["url"] for r in crawl}
    # Pipeline: Search (Rohdaten), Repo-Suche leer, Crawl (Rohdaten) -> Repo-Issues werden gelesen, Zitierseiten nicht
    tools = FakeTools(search_pages=[(r["url"], r["raw_content"]) for r in search], crawl_pages=[(r["url"], r["raw_content"]) for r in crawl])
    quote = "I'm trying the train the model in my own conell data, where the training is getting completed without any error."
    o, d = orch(tools, FakeLLM([quote])), dossier(repo={"url": "https://github.com/tkipf/gcn"})
    o._community_round({**paper(), "title": title, "arxiv_url": "https://arxiv.org/abs/1609.02907"}, d)
    read = {p["url"] for p in d.community["pages"]}
    assert {r["url"] for r in crawl} <= read
    assert not any("scirp" in u or "semanticscholar" in u or u.endswith("abs/1609.02907") for u in read)
    assert [c[0] for c in tools.calls] == ["search", "repo_search", "crawl"]


def test_noise_filter_handles_versions_and_pdf():
    res = {"results": [{"url": u} for u in ("https://arxiv.org/abs/2501.00001v2", "https://arxiv.org/pdf/2501.00001", "https://arxiv.org/abs/2501.99999",
                                            "https://www.researchgate.net/x", "https://github.com/a/b/issues/1")]}
    assert [r["url"] for r in community.drop_noise(res, "2501.00001")["results"]] == ["https://arxiv.org/abs/2501.99999", "https://github.com/a/b/issues/1"]


def test_discovery_hits_replace_the_search(cache_file):
    tools = FakeTools(read_pages=[("https://github.com/a/b/issues/1", PAGE_ISSUE)])
    hits = [{"url": "https://github.com/a/b/issues/1", "title": "t", "snippet": "s"}]
    d = dossier()
    o = orch(tools, FakeLLM([ISSUE]))
    o._community_round(paper(hits), d)
    assert [c[0] for c in tools.calls] == ["extract"] and d.community["reused_discovery"] is True       # kein Search
    assert d.community["queries"][0]["kind"] == "discovery_hits"


def test_cache_per_arxiv_id_seven_days(cache_file, monkeypatch):
    tools = FakeTools()
    o1, d1 = orch(tools, FakeLLM([ISSUE])), dossier()
    o1._community_round(paper(), d1)
    n_calls = len(tools.calls)
    assert json.loads(cache_file.read_text())[ARXIV]["community"]["findings"]
    o2, d2 = orch(tools, FakeLLM([ISSUE])), dossier()
    o2._community_round(paper(), d2)
    assert len(tools.calls) == n_calls and d2.community["mode"] == "cache"                              # keine Tavily-Aufrufe
    assert d2.community["findings"] == d1.community["findings"] and types(o2, "community")[0]["mode"] == "cache"
    assert [s["mode"] for s in d2.sources] == ["cache", "cache"]
    # abgelaufen: neuer Abruf
    cache = json.loads(cache_file.read_text())
    cache[ARXIV]["ts"] -= community.CACHE_TTL_S + 10
    cache_file.write_text(json.dumps(cache))
    o3, d3 = orch(tools, FakeLLM([ISSUE])), dossier()
    o3._community_round(paper(), d3)
    assert len(tools.calls) > n_calls and d3.community["mode"] == "live"


def test_explanations_only_for_deviating_verdicts_and_marked_unverified():
    for urteil, expected in (("reproduziert", 0), ("nicht reproduzierbar (Setup)", 0), ("Agenten-Fehler vermutet", 0), ("teilweise", 1), ("nicht reproduziert", 1)):
        community.CACHE_PATH.unlink(missing_ok=True)
        tools = FakeTools(explain_pages=[("https://example.org/errata", ERRATUM)])
        o, d = orch(tools, FakeLLM([ERRATUM[:60]])), dossier(urteil)
        o._community_round(paper(), d)
        assert len([c for c in tools.calls if c[0] == "explain"]) == expected, urteil
        assert bool(d.explanations) == bool(expected)
        if expected:
            items = d.explanations["c"]["items"]
            assert items and all(i["status"] == "unverified hypothesis" and i["quote"] in ERRATUM for i in items)
            assert d.explanations["c"]["dropped"] >= 1                                                  # erfundene Erklaerung verworfen


def test_explanations_cached_and_not_refetched(cache_file):
    tools = FakeTools(explain_pages=[("https://example.org/errata", ERRATUM)])
    orch(tools, FakeLLM([ERRATUM[:60]]))._community_round(paper(), dossier("teilweise"))
    n = len(tools.calls)
    d = dossier("nicht reproduziert")
    o = orch(tools, FakeLLM([ERRATUM[:60]]))
    o._community_round(paper(), d)
    assert len(tools.calls) == n and d.explanations["c"]["items"]


def test_verdicts_unchanged_and_run_hooks_only_after_verdicts():
    d = dossier("teilweise")
    before = dataclasses.asdict(d.verdicts[0])
    tools = FakeTools(explain_pages=[("https://example.org/errata", ERRATUM)])
    orch(tools, FakeLLM([ISSUE, ERRATUM[:60]]))._community_round(paper(), d)
    assert dataclasses.asdict(d.verdicts[0]) == before and d.verdicts[0].gesamtaussage == Verdict(**before).gesamtaussage
    # Einbindung: genau ein Aufruf in run(), nach der Claim-Schleife; sonst kommt `community` im Loop nirgends vor
    run_src = inspect.getsource(Orchestrator.run)
    assert run_src.index("self.run_claim(") < run_src.index("self._community_round(") < run_src.index("dossier.save(")
    loop_src = inspect.getsource(inspect.getmodule(Orchestrator))
    outside = loop_src.replace(inspect.getsource(Orchestrator._community_round), "")
    assert "community." not in outside and "possible_explanations" not in outside
    for name in ("plan", "write_code", "repair_code", "_run_seed", "run_claim", "_verdict", "_seed0_quality_loop"):
        assert "community" not in inspect.getsource(getattr(Orchestrator, name))


@pytest.mark.parametrize("kw", [{"flag": False}, {"tavily": False}, {"mode": "fixture"}])
def test_no_calls_outside_the_live_product_path(kw):
    tools = FakeTools()
    o, d = orch(tools, FakeLLM([ISSUE]), **kw), dossier("teilweise")
    o._community_round(paper(), d)
    assert tools.calls == [] and d.community is None and d.explanations == {} and o.events_seen == []


def test_tavily_failure_only_warns_and_billing_passes_through():
    class Boom(FakeTools):
        def search_community(self, *a, **k):
            raise RuntimeError("rate limit")

    o, d = orch(Boom(), FakeLLM([ISSUE])), dossier("teilweise")
    o._community_round(paper(), d)
    assert d.community is None and d.verdicts[0].urteil == "teilweise" and "Community-Check fehlgeschlagen" in o.warnings[0]

    class Billing(FakeTools):
        def search_community(self, *a, **k):
            raise BillingBlocked("blocked")

    with pytest.raises(BillingBlocked):
        orch(Billing(), FakeLLM([ISSUE]))._community_round(paper(), dossier())


def test_paid_hits_survive_a_failing_explanation_step(cache_file):
    class Boom(FakeTools):
        def clarify_discrepancy(self, *a, **k):
            raise RuntimeError("down")

    o = orch(Boom(), FakeLLM([ISSUE]))
    o._community_round(paper(), dossier("teilweise"))
    assert json.loads(cache_file.read_text())[ARXIV]["community"]["findings"]                          # Funde im Cache trotz Fehler


def test_zeugnis_sections_and_empty_message():
    d = dossier("teilweise")
    o = orch(FakeTools(explain_pages=[("https://example.org/errata", ERRATUM)]), FakeLLM([ISSUE, ERRATUM[:60]]))
    o._community_round(paper(), d)
    md = render_markdown(d)
    assert "## What others found" in md and "[protocol note]" in md and "nicht Teil des Urteils" in md and "Possible explanations" in md
    d2 = dossier()
    d2.community = {"findings": [], "dropped": 0, "pages": [], "mode": "live", "queried_at": "t"}
    assert "No public discussion found." in render_markdown(d2)
    assert "What others found" not in render_markdown(dossier())                                       # ohne Check kein Abschnitt


def test_dossier_roundtrip_with_defaults(tmp_path):
    d = dossier()
    d.community = {"findings": [], "dropped": 0, "pages": [], "queries": [], "reused_discovery": False, "mode": "live"}
    p = d.save("20260101-000000")
    assert Dossier.load(p).community == d.community
    old = json.loads(p.read_text())
    old.pop("community"); old.pop("explanations")
    p.write_text(json.dumps(old))
    assert Dossier.load(p).community is None and Dossier.load(p).explanations == {}
    import shutil
    shutil.rmtree(p.parent.parent)
