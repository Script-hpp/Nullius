"""Tavily-Credit-Zaehler: genau ein Log-Eintrag je Aufruf, Credits laut Tabelle. Kein Netz, Log in tmp_path."""

import json

import pytest

from repro_agent import tavily_usage as tu
from repro_agent.tavily_tools import TavilyTools


class FakeClient:
    def __init__(self):
        self.fail = False

    def search(self, **kw):
        if self.fail:
            raise RuntimeError("boom")
        return {"results": []}

    def extract(self, **kw):
        return {"results": []}

    def crawl(self, **kw):
        return {"results": []}


@pytest.fixture
def log(tmp_path, monkeypatch):
    p = tmp_path / "tavily_usage.jsonl"
    monkeypatch.setattr(tu, "TAVILY_LOG_PATH", p)
    return p


def entries(p):
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def tools(client=None):
    t = TavilyTools.__new__(TavilyTools)        # kein echter Client, kein Key
    t._client = tu.CountedClient(client or FakeClient())
    t._research_calls = 0
    t.on_call = None
    return t


@pytest.mark.parametrize("endpoint,kw,credits", [
    ("search", {}, 1), ("search", {"search_depth": "advanced"}, 2),
    ("extract", {"urls": ["a"], "extract_depth": "advanced"}, 2), ("extract", {"urls": ["a"] * 6}, 2),
    ("extract", {"urls": ["a"] * 5, "extract_depth": "advanced"}, 2),
    ("crawl", {"url": "u", "limit": 5, "instructions": "x"}, 3), ("crawl", {"url": "u", "limit": 10}, 3),
])
def test_credit_table(endpoint, kw, credits):
    assert tu.estimate_credits(endpoint, kw) == credits


def test_each_call_writes_exactly_one_entry(log):
    t = tools()
    t.find_material("paper")
    t.read_claims("https://arxiv.org/html/x")
    t.crawl_issues("https://github.com/o/r", limit=3)
    t.cite_sources(["u1", "u2"], "paper")                     # extract + research-search = zwei Aufrufe
    es = entries(log)
    assert [e["endpoint"] for e in es] == ["search", "extract", "crawl", "extract", "search"]
    assert [e["credits"] for e in es] == [1, 2, 3, 2, 2]
    assert tu.used_credits() == 10


def test_failed_call_is_counted_with_error(log):
    c = FakeClient()
    c.fail = True
    with pytest.raises(RuntimeError):
        tools(c).find_material("p")
    (e,) = entries(log)
    assert e["endpoint"] == "search" and e["credits"] == 1 and "boom" in e["error"]


def test_used_credits_since(log):
    log.write_text(json.dumps({"ts": 1.0, "endpoint": "search", "credits": 5}) + "\n"
                   + json.dumps({"ts": 5e9, "endpoint": "search", "credits": 7}) + "\n")
    assert tu.used_credits() == 12 and tu.used_credits("2100-01-01T00:00Z") == 7


def test_transient_connection_error_is_retried(log, monkeypatch):
    import requests
    monkeypatch.setattr(tu, "_sleep", lambda s: None)

    class Flaky(FakeClient):
        n = 0

        def search(self, **kw):
            Flaky.n += 1
            if Flaky.n < 3:
                raise requests.exceptions.ConnectionError("Remote end closed connection without response")
            return {"results": [1]}

    assert tu.CountedClient(Flaky()).search(query="x") == {"results": [1]}
    e = entries(log)
    assert len(e) == 3 and "error" in e[0] and "error" not in e[2]


def test_connection_error_gives_up_after_three_and_other_errors_not_retried(log, monkeypatch):
    import requests
    monkeypatch.setattr(tu, "_sleep", lambda s: None)

    class Down(FakeClient):
        def search(self, **kw):
            raise requests.exceptions.ConnectionError("down")

    with pytest.raises(requests.exceptions.ConnectionError):
        tu.CountedClient(Down()).search(query="x")
    assert len(entries(log)) == 3
    c = FakeClient()
    c.fail = True
    with pytest.raises(RuntimeError):
        tu.CountedClient(c).search(query="x")
    assert len(entries(log)) == 4
