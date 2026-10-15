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
ISSUE = "In GraphNet: Better Node Classification the reported test accuracy uses drop_last=True in the test loader, so the last batch is dropped."
REPORT = "We could not reproduce the 83.0 accuracy of GraphNet: Better Node Classification with the released code."
PAGE_ISSUE = f"GraphNet: Better Node Classification (arXiv 2501.00001) github issue. {ISSUE} Thanks."
PAGE_REPORT = f"Review of GraphNet: Better Node Classification. {REPORT}"
PAGE_OTHER_PAPER = "Some other paper entirely about something else. It mentions drop_last=True once."
ERRATUM = "Erratum: Table 2 of GraphNet: Better Node Classification was recomputed; the correct value is 81.9."


class FakeTools:
    def __init__(self, search_pages=None, crawl_pages=None, read_pages=None, explain_pages=None, repo_pages=None, method_pages=None, critique_pages=None):
        self.calls = []
        self.search_pages = search_pages if search_pages is not None else [("https://github.com/a/b/issues/1", PAGE_ISSUE), ("https://example.org/r", PAGE_REPORT)]
        self.crawl_pages, self.read, self.explain_pages, self.repo_pages = crawl_pages or [], read_pages or [], explain_pages or [], repo_pages or []
        self.method_pages = method_pages or []
        self.critique_pages = critique_pages or []
        self.critique_limit = None

    @staticmethod
    def _res(pages):
        return {"results": [{"url": u, "raw_content": t} for u, t in pages]}

    def search_community(self, title, limit=5):
        self.calls.append(("search", title))
        return self._res(self.search_pages)

    def search_method_critique(self, method, dataset, limit=5):
        self.calls.append(("method_search", method, dataset))
        return self._res(self.method_pages)

    def search_critique(self, method, dataset, limit=5):
        self.calls.append(("critique", method, dataset))
        self.critique_limit = limit
        return {**self._res(self.critique_pages), "queries": ["q1", "q2", "q3", "q4"]}

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
    assert [c[0] for c in tools.calls] == ["search", "repo_search", "extract"] and tools.calls[1][1] == "a/b"   # extract: Issue-Text nachladen
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
    assert [c[0] for c in tools3.calls] == ["search", "extract"]   # Issue-Text nachladen, sonst nichts
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
    run_src = inspect.getsource(Orchestrator._execute)
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


# ---- zweite Runde (Kipf-Aufruf mit b4518d0): schwache Seiten, zu allgemeine Funde, Methoden-Query ------------------------------------
KIPF = "Semi-Supervised Classification with Graph Convolutional Networks"
TOO_GOOD = "results appear too good"
CRITIQUE = ("Pitfalls of Graph Neural Network Evaluation. We re-evaluate GCN on Cora with random splits and find that GCN reaches 81.5 +- 1.3, "
            "and that the ranking of models changes with the split. Different splits lead to a completely different ranking of models.")


def synthetic_kipf_pages():
    """Synthetisch nach der Beschreibung des echten Aufrufs: Sammel-Repo (70k), zwei Gists (152k, 276k), Repo-Issues."""
    filler = lambda n: ("Paper notes. " + KIPF + ". " + TOO_GOOD + ". ") + "x" * n
    return [("https://github.com/nplan-io/ml_paper_club", filler(70_000)),
            ("https://gist.github.com/u/aaa", filler(152_000)), ("https://gist.github.com/u/bbb", filler(276_000))]


def test_big_pages_and_collections_are_not_read():
    items = [{"url": u, "raw_content": t} for u, t in synthetic_kipf_pages()]
    items += [{"url": "https://github.com/awesome-gnn/list", "raw_content": KIPF + " reading list"},
              {"url": "https://example.org/reading-list", "raw_content": KIPF},
              {"url": "https://github.com/tkipf/gcn/issues/5", "raw_content": "short issue " + "y" * 100},
              {"url": "https://openreview.net/forum?id=1", "raw_content": f"Review of {KIPF}: " + TOO_GOOD}]
    docs = community.select_docs(items, KIPF, "1609.02907", ("https://github.com/tkipf/gcn/",))
    assert [d.name for d in docs] == ["https://github.com/tkipf/gcn/issues/5", "https://openreview.net/forum?id=1"]
    assert community.is_collection("https://gist.github.com/u/aaa") and community.is_collection("https://github.com/nplan-io/ml_paper_club")
    assert not community.is_collection("https://github.com/tkipf/gcn/issues/5")


class QuoteLLM:
    """Liefert fuer jede Seite die vorgegebenen (Aussage, Zitat)-Paare, deren Zitat im Chunk steht."""

    def __init__(self, pairs):
        self.pairs = pairs

    def complete(self, role, messages, **kw):
        user = messages[-1]["content"]
        out = [{"topic": "t", "statement": st, "quote": q} for st, q in self.pairs if q in user]
        return SimpleNamespace(content=json.dumps(out))


def kipf_paper(with_method=True):
    claims = [{"id": "c", "extract": {"method": "GCN", "dataset": "Cora"}}] if with_method else []
    return {"title": KIPF, "dataset": "Cora", "arxiv_url": "https://arxiv.org/abs/1609.02907", "claims": claims, "community": []}


def kipf_paper_of(method):
    return {"title": "Some Unrelated Title", "dataset": "Cora", "arxiv_url": "https://arxiv.org/abs/2501.00009",
            "claims": [{"id": "c", "extract": {"method": method, "dataset": "Cora"}}], "community": []}


def test_finding_must_name_method_or_title_but_repo_pages_are_exempt():
    page = f"Review of {KIPF}. Some say: {TOO_GOOD}. Another review: GCN accuracy on Cora is lower with random splits."
    issue = "Issue in the repo: the numbers differ on my machine, please explain the preprocessing step."
    tools = FakeTools(search_pages=[("https://openreview.net/forum?id=1", page), ("https://github.com/tkipf/gcn/issues/9", issue)])
    llm = QuoteLLM([("Results look too good.", TOO_GOOD), ("GCN is worse with random splits.", "GCN accuracy on Cora is lower with random splits"),
                    ("Preprocessing differs.", "the numbers differ on my machine")])
    d = dossier(repo={"url": "https://github.com/tkipf/gcn"})
    o = orch(tools, llm)
    o._community_round(kipf_paper(), d)
    got = {f["quote"] for f in d.community["findings"]}
    assert got == {"GCN accuracy on Cora is lower with random splits", "the numbers differ on my machine"}   # TOO_GOOD nennt nichts: verworfen
    assert d.community["dropped"] >= 1


def test_method_query_reads_critique_papers_without_the_title():
    tools = FakeTools(search_pages=[], method_pages=[("https://arxiv.org/abs/1811.05868", CRITIQUE), ("https://arxiv.org/abs/1609.02907", "own page " + KIPF)])
    llm = QuoteLLM([("Ranking depends on the split.", "GCN reaches 81.5 +- 1.3, and that the ranking of models changes with the split")])
    d, o = dossier(), orch(tools, llm)
    o._community_round(kipf_paper(), d)
    assert ("method_search", "GCN", "Cora") in tools.calls and [c[0] for c in tools.calls][:2] == ["search", "method_search"]
    assert [f["url"] for f in d.community["findings"]] == ["https://arxiv.org/abs/1811.05868"]      # Kritik-Paper, nicht die eigene Seite
    f = d.community["findings"][0]
    assert f["kind"] == "other" and f["label"] == "Note"                                                  # Label-Feld nie leer
    assert {q["kind"] for q in d.community["queries"]} == {"search", "method_search", "critique_search"}


def test_method_query_needs_method_and_dataset_in_the_page():
    tools = FakeTools(search_pages=[], method_pages=[("https://arxiv.org/abs/9999.00001", "A paper about GCN on Citeseer only, the ranking of models changes.")])
    d = dossier()
    orch(tools, QuoteLLM([("x", "the ranking of models changes")]))._community_round(kipf_paper(), d)
    assert d.community["findings"] == [] and d.community["pages"] == []                                    # Methode da, Datensatz fehlt


def test_labels_for_every_kind():
    for kind, label in community.KIND_LABEL.items():
        assert label and kind in (community.KIND_PROTOCOL, community.KIND_REPORT, community.KIND_OTHER)
    d = dossier()
    orch(FakeTools(), FakeLLM([ISSUE, REPORT]))._community_round(paper(), d)
    assert all(f["label"] == community.KIND_LABEL[f["kind"]] for f in d.community["findings"]) and d.community["findings"]


def test_old_cache_entries_are_ignored(cache_file):
    cache_file.write_text(json.dumps({ARXIV: {"ts": time.time(), "community": {"findings": [{"x": 1}]}}}))
    assert community.cache_get(ARXIV) is None
    community.cache_put(ARXIV, {"ts": time.time(), "v": community.CACHE_VERSION, "community": {"findings": []}})
    assert community.cache_get(ARXIV) is not None


def test_tavily_queries_of_the_new_methods():
    from repro_agent.tavily_tools import TavilyTools

    seen = []
    t = TavilyTools.__new__(TavilyTools)
    t.on_call = None
    t._client = SimpleNamespace(search=lambda **kw: seen.append(kw) or {"results": []})
    t.search_community(KIPF, 5)
    t.search_method_critique("GCN", "Cora", 5)
    t.search_repo_issues("tkipf/gcn", 5)
    a, b, c = seen
    assert a["include_domains"] == ["github.com", "openreview.net", "arxiv.org", "reddit.com"] and "scirp.org" in a["exclude_domains"] and "could not reproduce" in a["query"]
    assert b["query"] == 'GCN Cora reproducibility OR re-evaluation OR pitfalls OR "could not reproduce"' and b["include_domains"] == ["arxiv.org", "openreview.net"] and b["max_results"] == 5
    assert c["query"] == "gcn reproduce results accuracy" and c["include_domains"] == ["github.com/tkipf/gcn"] and c["max_results"] == 5


def test_same_page_with_query_string_or_fragment_is_read_once():
    base = "https://github.com/tkipf/gcn/issues/14"
    items = [{"url": base, "raw_content": "issue text A"}, {"url": base + "?timeline_page=1", "raw_content": "issue text B"},
             {"url": base + "#issuecomment-1", "raw_content": "issue text C"}, {"url": base + "/", "raw_content": "issue text D"},
             {"url": "https://github.com/tkipf/gcn/issues/15", "raw_content": "other issue"}]
    docs = community.select_docs(items, KIPF, "1609.02907", ("https://github.com/tkipf/gcn/",))
    assert [d.name for d in docs] == [base, "https://github.com/tkipf/gcn/issues/15"] and docs[0].text == "issue text A"


def test_hits_and_drop_reasons_are_logged():
    page = f"Review of {KIPF}. Some say: {TOO_GOOD}. Another review: GCN accuracy on Cora is lower with random splits."
    tools = FakeTools(search_pages=[("https://openreview.net/forum?id=1", page)])
    llm = QuoteLLM([("Results look too good.", TOO_GOOD), ("GCN is worse with random splits.", "GCN accuracy on Cora is lower with random splits")])
    d = dossier()
    o = orch(tools, llm)
    o._community_round(kipf_paper(), d)
    c = d.community
    assert [h["url"] for h in c["hits"]] == ["https://openreview.net/forum?id=1"] and c["not_read"] == []
    assert [x["reason"] for x in c["dropped_items"]] == ["names_neither_method_title_nor_arxiv_id"]
    assert c["dropped_items"][0]["quote"] == TOO_GOOD[:300] and len(c["dropped_items"]) == c["dropped"]
    ev = types(o, "community")[0]
    assert ev["hits"] == c["hits"] and ev["dropped_items"] == c["dropped_items"]


def test_quote_not_in_page_reason():
    tools = FakeTools(search_pages=[("https://github.com/a/b/issues/1", PAGE_ISSUE)])
    o, d = orch(tools, FakeLLM([ISSUE, REPORT])), dossier()
    o._community_round(paper(), d)
    assert {x["reason"] for x in d.community["dropped_items"]} <= {"quote_not_in_page", "names_neither_method_title_nor_arxiv_id"}
    assert len(d.community["dropped_items"]) == d.community["dropped"]


# ---- generische Kritik-Vorlagen, Foren, Issue-Texte (Pivot 2026-10-08) -----------------------------------------------------

def test_critique_templates_are_generic_and_search_forums():
    from repro_agent import tavily_tools as tt

    seen = []
    t = tt.TavilyTools.__new__(tt.TavilyTools)
    t.on_call = None
    t._client = SimpleNamespace(search=lambda **kw: seen.append(kw) or {"results": [{"url": "https://x/" + kw["query"].replace(" ", "_")}, {"url": "https://x/dup"}]})
    res = t.search_critique("GCN", "Cora", 5)
    assert [k["query"] for k in seen] == ["GCN Cora reproduce results differ", "GCN bug evaluation", "GCN baseline re-evaluation fixed",
                                           "we re-run GCN on Cora and compare with the reported numbers"]
    assert all(k["max_results"] == 5 and "arxiv.org" in k["include_domains"] and "openreview.net" in k["include_domains"]
               and "github.com" in k["include_domains"] for k in seen)
    assert [("discuss.dgl.ai" in k["include_domains"]) for k in seen] == [True, True, True, False]
    assert len(res["results"]) == 5 and res["queries"] == [k["query"] for k in seen]   # Duplikat-URL einmal
    # keine Paper-Eigenheiten in den Vorlagen
    assert not any(w in " ".join(q for q, _ in tt.CRITIQUE_TEMPLATES).lower() for w in ("drop_last", "dlinear", "etth1", "github.com/"))


def test_critique_hits_are_read_with_method_and_dataset_but_paper_and_quote_checks_stay():
    page = "We re-run MyNet on Cora and find the reported accuracy drops by two points. Details follow."
    tools = FakeTools(search_pages=[("https://example.org/r", PAGE_REPORT)], critique_pages=[("https://arxiv.org/abs/2601.00002", page)])
    quote = "We re-run MyNet on Cora and find the reported accuracy drops by two points."
    d = dossier()
    orch(tools, QuoteLLM([("Re-run drops accuracy.", quote), ("Made up.", "nowhere in the page")]))._community_round(kipf_paper_of("MyNet"), d)
    assert ("critique", "MyNet", "Cora") in tools.calls
    assert [f["url"] for f in d.community["findings"]] == ["https://arxiv.org/abs/2601.00002"]
    assert d.community["calls"].count("search") == 1 + 1 + 4   # title, method, 4 critique templates
    # Seite ohne Methode+Datensatz im Text wird auch als Treffer der Kritik-Vorlagen nicht gelesen
    tools2 = FakeTools(search_pages=[], critique_pages=[("https://arxiv.org/abs/2601.00003", "Totally unrelated text about transformers.")])
    d2 = dossier()
    community.CACHE_PATH.unlink(missing_ok=True)
    orch(tools2, QuoteLLM([]))._community_round(kipf_paper_of("MyNet"), d2)
    assert d2.community["pages"] == []


def test_short_issue_hits_get_their_body_by_extract_limited():
    repo = {"url": "https://github.com/a/b"}
    heads = [(f"https://github.com/a/b/issues/{i}", f"Issue {i} title") for i in range(1, 8)]
    body = "Issue 2 title. Description: " + ISSUE
    tools = FakeTools(search_pages=heads, read_pages=[("https://github.com/a/b/issues/2", body)])
    d = dossier(repo=repo)
    orch(tools, FakeLLM([ISSUE]))._community_round(paper(), d)
    ext = [c for c in tools.calls if c[0] == "extract"]
    assert len(ext) == 1 and len(ext[0][1]) == community.MAX_BODIES == 4 and ext[0][1][0].endswith("/issues/1")
    assert [f["url"] for f in d.community["findings"]] == ["https://github.com/a/b/issues/2"]   # Zitat steht nur im Text
    assert {"kind": "issue_bodies", "urls": list(ext[0][1])} in d.community["queries"]


def test_issue_list_page_alone_triggers_the_crawl_for_bodies():
    repo = {"url": "https://github.com/a/b"}
    tools = FakeTools(search_pages=[("https://github.com/a/b/issues", "Issues\n#1 Something\n#2 Other")],
                      crawl_pages=[("https://github.com/a/b/issues/9", "Issue nine " + ISSUE)])
    d = dossier(repo=repo)
    orch(tools, FakeLLM([ISSUE]))._community_round(paper(), d)
    assert ("crawl", "https://github.com/a/b", 3) in tools.calls
    assert [f["url"] for f in d.community["findings"]] == ["https://github.com/a/b/issues/9"]


def test_extract_failure_keeps_the_paid_hits():
    class Boom(FakeTools):
        def read_pages(self, urls, limit=3):
            raise RuntimeError("down")
    repo = {"url": "https://github.com/a/b"}
    tools = Boom(search_pages=[("https://github.com/a/b/issues/1", PAGE_ISSUE[:100] + ISSUE)])
    d = dossier(repo=repo)
    orch(tools, FakeLLM([ISSUE]))._community_round(paper(), d)
    assert d.community is not None and [f["url"] for f in d.community["findings"]] == ["https://github.com/a/b/issues/1"]


# ---- Addendum B: Seitenebene fuer Setup-Themen -----------------------------------------------------------------------------

DL = "the test dataloader uses drop_last=True, which may exclude a significant portion of test data"


def _page_rule(page_text, pairs):
    tools = FakeTools(search_pages=[], critique_pages=[("https://example.org/fits", page_text)])   # Seite kommt als Kritik-Treffer
    d = dossier()
    o = orch(tools, QuoteLLM(pairs))
    community.CACHE_PATH.unlink(missing_ok=True)
    o._community_round(kipf_paper(), d)
    return d.community


def test_setup_finding_passes_when_page_names_method_but_statement_does_not():
    c = _page_rule(f"README of another model. Compared with GCN on Cora. Note: {DL}.", [("Test loader drops data.", DL)])
    assert [f["quote"] for f in c["findings"]] == [DL] and c["findings"][0]["pass_reason"] == "page_names_method"
    assert [x["outcome"] for x in c["candidates"]] == ["kept_page_names_method"] and c["dropped_items"] == []


def test_setup_finding_dropped_when_page_does_not_name_method(monkeypatch):
    # select_docs laesst solche Seiten normalerweise nicht durch; hier erzwungen, damit die Regel allein geprueft wird
    monkeypatch.setattr(community, "select_docs", lambda items, *a, **kw: [community.Doc(i["url"], i["raw_content"]) for i in items])
    c = _page_rule(f"README of another model about Cora arXiv 2501.00001. Note: {DL}.", [("Test loader drops data.", DL)])
    assert c["findings"] == [] and c["dropped_items"][0]["reason"] == "names_neither_method_title_nor_arxiv_id"
    assert c["candidates"][0]["outcome"] == "dropped_names_neither_method_title_nor_arxiv_id"


def test_non_setup_finding_keeps_statement_level_rule_even_if_page_names_method():
    c = _page_rule(f"Mentions GCN on Cora once. {TOO_GOOD}.", [("Results look too good.", TOO_GOOD)])
    assert c["findings"] == [] and c["candidates"][0]["outcome"] == "dropped_names_neither_method_title_nor_arxiv_id"


def test_quote_check_stays_strict_and_candidates_store_raw_output():
    c = _page_rule(f"Compared with GCN on Cora. {DL}.", [("Test loader drops data.", DL)])
    assert c["findings"][0]["pass_reason"] == "page_names_method"

    class Bad:
        def complete(self, role, messages, **kw):
            return SimpleNamespace(content=json.dumps([{"topic": "t", "statement": "drop_last used", "quote": "invented quote GCN"}]))
    tools = FakeTools(search_pages=[], critique_pages=[("https://example.org/fits", "GCN on Cora. Nothing else here at all.")])
    d = dossier()
    community.CACHE_PATH.unlink(missing_ok=True)
    orch(tools, Bad())._community_round(kipf_paper(), d)
    assert d.community["findings"] == [] and d.community["candidates"][0]["outcome"] == "dropped_quote_not_in_page"
    assert d.community["candidates"][0]["quote_in_page"] is False


# Nachtrag C: Seitenreihenfolge vor dem Schnitt auf MAX_CHUNKS. Die FITS-Seite stand im Lauf von Nachtrag A an Stelle 12 von 22 gelesenen
# (0-basiert) und fiel unbemerkt hinter docs[:12]. Text: woertlicher Auszug aus dem FITS-README (github.com/VEWOXIC/FITS, "Important Update:
# 2023-12-25", frei abgerufen am 2026-10-08, nicht ueber Tavily); die Fuellseiten sind synthetisch.
FITS_README = (
    "## Important Update: 2023-12-25\n\nWe've identified a significant bug in our code, originally found in Informer. This issue has "
    "implications for a broad spectrum of research on time series forecasting, including but not limited to:\n\n"
    "- PatchTST (ICLR 2023)\n- DLinear (AAAI 2022 reported version)\n- FEDformer (ICML 2022)\n\n"
    "### Description of the Bug:\n\nThe bug stems from an incorrect implementation in the data loader. Specifically, "
    "the test dataloader uses `drop_last=True`, which may exclude a significant portion of test data, particularly with large batch sizes, "
    "leading to unfair model comparisons.\n\nThe discovered bug predominantly impacts results on smaller datasets like ETTh1 and ETTh2.\n"
)
FITS_QUOTE = "the test dataloader uses `drop_last=True`, which may exclude a significant portion of test data"


def dlinear_paper():
    return {"title": "Are Transformers Effective for Time Series Forecasting?", "dataset": "ETTh1", "arxiv_url": "https://arxiv.org/abs/2205.13504",
            "claims": [{"id": "c", "extract": {"method": "DLinear", "dataset": "ETTh1"}}], "community": []}


def late_fits_scenario():
    fillers = [(f"https://arxiv.org/html/2601.{i:05d}", f"We compare DLinear on ETTh1 with our model {i}. We tune the hyperparameter grid, "
                f"report the test set MSE over 3 seeds and the validation set curve. " + "z" * 3000) for i in range(12)]
    other = ("https://arxiv.org/html/2604.99999", "A KAN model on Weather and ETTm1. We use a unified protocol. Test set results over seeds. "
             "Our dataloader sets drop_last=True for training only; leakage is avoided.")   # nennt weder DLinear noch ETTh1: wird nicht gelesen
    return fillers + [other, ("https://github.com/VEWOXIC/FITS", FITS_README)]


def test_rank_docs_puts_rare_protocol_keywords_first_and_keeps_ties_in_order():
    common = "tune the hyperparameter on the validation set and report the test set accuracy"
    docs = [community.Doc(f"u{i}", common + " " * i) for i in range(4)] + [community.Doc("rare", common + " drop_last leakage")]
    assert [d.name for d in community.rank_docs(docs)] == ["rare", "u0", "u1", "u2", "u3"]
    assert [d.name for d in community.rank_docs(docs[:1])] == ["u0"] and community.rank_docs([]) == []


def test_late_page_with_rare_protocol_keyword_reaches_the_extractor_and_passes_page_rule(monkeypatch):
    def run():
        tools = FakeTools(search_pages=[], critique_pages=late_fits_scenario())
        d = dossier()
        community.CACHE_PATH.unlink(missing_ok=True)
        orch(tools, QuoteLLM([("Test loader drops data.", FITS_QUOTE)]))._community_round(dlinear_paper(), d)
        return tools, d.community
    tools, c = run()
    assert tools.critique_limit == community.CRITIQUE_RESULTS >= 8
    assert [f["url"] for f in c["findings"]] == ["https://github.com/VEWOXIC/FITS"] and c["findings"][0]["pass_reason"] == "page_names_method"
    assert c["findings"][0]["kind"] == community.KIND_PROTOCOL
    assert "https://arxiv.org/html/2604.99999" not in {p["url"] for p in c["pages"]}      # Seite ohne Methode/Datensatz: nicht gelesen
    assert len(c["pages"]) == 13 and c["not_extracted"] and "https://github.com/VEWOXIC/FITS" not in c["not_extracted"]
    # alte Reihenfolge (Eingang der Suchergebnisse): die Seite faellt hinter den Schnitt, nichts wird extrahiert
    monkeypatch.setattr(community, "rank_docs", lambda docs: docs)
    _, old = run()
    assert old["findings"] == [] and old["not_extracted"] == ["https://github.com/VEWOXIC/FITS"]


def test_generic_pages_without_rare_keywords_do_not_make_findings():
    fillers = [(u, t) for u, t in late_fits_scenario()[:12]]
    tools = FakeTools(search_pages=[], critique_pages=fillers)
    d = dossier()
    community.CACHE_PATH.unlink(missing_ok=True)
    orch(tools, QuoteLLM([("Test loader drops data.", FITS_QUOTE)]))._community_round(dlinear_paper(), d)
    assert d.community["findings"] == [] and d.community["candidates"] == []


def test_fits_style_finding_is_accepted_as_protocol_change_by_the_fixed_validation():
    from repro_agent.objection_experiments import validate
    finding = {"statement": "The test dataloader drops part of the test data.", "quote": FITS_QUOTE}
    out = validate({"type": "protocol_change", "reason": "r"}, finding, "mse", {}, ("drop_last",))
    assert out["type"] == "protocol_change" and out["param"] == "drop_last"
    generic = {"statement": "Baselines are quoted from original publications.", "quote": "Nonlinear baselines are quoted from their original publications"}
    assert validate({"type": "protocol_change"}, generic, "mse", {}, ("drop_last",))["type"] == "not_measurable"
