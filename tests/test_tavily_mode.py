import json
from types import SimpleNamespace

import pytest

from repro_agent.claims_source import fetch_paper_text
from repro_agent.loop import Orchestrator
from repro_agent.repo_source import fetch_readme

TEXT = "x" * 3000


@pytest.fixture(autouse=True)
def no_arxiv_network(monkeypatch, tmp_path):
    monkeypatch.setattr("repro_agent.discovery._fetch_arxiv_source", lambda *a: "")
    monkeypatch.setattr("repro_agent.claims_source.SOURCE_CACHE_DIR", tmp_path / "paper_sources")
    monkeypatch.setattr("repro_agent.claims_source.SOURCE_RESERVE_DIR", tmp_path / "source_reserves")


class FakeTavily:
    def __init__(self):
        self.calls = 0

    def read_claims(self, url):
        self.calls += 1
        return {"results": [{"raw_content": TEXT}]}

    def get_setup_details(self, url):
        self.calls += 1
        return {"results": [{"raw_content": "README live"}]}


def _fixture(tmp_path, arxiv_id="1234.5678"):
    fx = {"arxiv_id": arxiv_id, "url": "u", "source_kind": "pdf", "retrieved_at": "t", "sha256": "h", "arxiv_version": None, "text": "FIXTURE TEXT"}
    (tmp_path / f"{arxiv_id}.json").write_text(json.dumps(fx))


def test_paper_text_fixture_mode_uses_fixture_without_call(tmp_path, monkeypatch):
    monkeypatch.setattr("repro_agent.claims_source.latest_arxiv_version", lambda i: None)
    _fixture(tmp_path)
    tv = FakeTavily()
    assert fetch_paper_text("1234.5678", lambda: tv, tmp_path, mode="fixture").text == "FIXTURE TEXT" and tv.calls == 0


def test_paper_text_live_mode_calls_tavily_even_with_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr("repro_agent.claims_source.latest_arxiv_version", lambda i: None)
    _fixture(tmp_path)
    tv = FakeTavily()
    p = fetch_paper_text("1234.5678", lambda: tv, tmp_path, mode="live")
    assert p.text == TEXT and tv.calls >= 1
    assert json.loads((tmp_path / "1234.5678.json").read_text())["text"] == "FIXTURE TEXT"    # live schreibt nie in fixture_dir


def test_paper_text_fixture_miss_creates_fixture_live_does_not(tmp_path, monkeypatch):
    monkeypatch.setattr("repro_agent.claims_source.latest_arxiv_version", lambda i: None)
    tv = FakeTavily()
    live_dir, fx_dir = tmp_path / "live", tmp_path / "fx"
    assert fetch_paper_text("1234.5678", lambda: tv, live_dir, mode="live").text == TEXT
    assert not live_dir.exists()
    assert fetch_paper_text("1234.5678", lambda: tv, fx_dir, mode="fixture").text == TEXT
    assert json.loads((fx_dir / "1234.5678.json").read_text())["text"] == TEXT
    calls = tv.calls
    assert fetch_paper_text("1234.5678", lambda: tv, fx_dir, mode="fixture").text == TEXT and tv.calls == calls


def test_live_mode_writes_nothing_in_repo_functions(tmp_path):
    from types import SimpleNamespace

    from repro_agent.claims_source import PaperText, TargetEvidence, extract_target_cached
    from repro_agent import claims_source, repo_source

    tv = SimpleNamespace(
        get_setup_details=lambda url: {"results": [{"raw_content": "README live"}]},
        find_material=lambda title, domains: {"results": [{"url": "https://github.com/a/b", "content": "1234.5678"}]},
    )
    url = "https://github.com/a/b"
    live, fx = tmp_path / "live", tmp_path / "fx"
    assert repo_source.fetch_readme(url, lambda: tv, live, mode="live") == "README live"
    assert repo_source.search_repo_url("T", "1234.5678", lambda: tv, live, mode="live") == url
    ev = TargetEvidence(value=0.8, quote="q", table="t", url="u", source_kind="k", retrieved_at="r", sha256="h", arxiv_version=None)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(claims_source, "extract_target", lambda *a, **k: ev)
    try:
        paper = PaperText("1234.5678", "u", "k", "r", "h" * 64, None, "text")
        assert extract_target_cached(None, paper, "m", "d", "acc", live, mode="live") is ev
        assert not live.exists()
        extract_target_cached(None, paper, "m", "d", "acc", fx)       # fixture mode: legt an
        assert len(list(fx.glob("*_target_*.json"))) == 1
    finally:
        monkey.undo()
    assert repo_source.fetch_readme(url, lambda: tv, fx, mode="fixture") == "README live"
    assert (fx / "a_b_readme.json").exists()
    assert repo_source.search_repo_url("T", "1234.5678", lambda: tv, fx, mode="fixture") == url
    assert (fx / "search_1234.5678.json").exists()


def test_readme_modes(tmp_path):
    url = "https://github.com/a/b"
    from repro_agent.repo_source import _repo_key

    (tmp_path / f"{_repo_key(url)}_readme.json").write_text(json.dumps({"url": url, "text": "README fixture"}))
    tv = FakeTavily()
    assert fetch_readme(url, lambda: tv, tmp_path, mode="fixture") == "README fixture" and tv.calls == 0
    assert fetch_readme(url, lambda: tv, tmp_path, mode="live") == "README live" and tv.calls == 1


def _make(monkeypatch, **kw):
    from repro_agent import loop
    from repro_agent.config import LLMConfig, Settings

    st = Settings(llm=LLMConfig("tokenfactory", "http://x", "k", "m", "m", "m"), sandbox_backend="local", tavily_api_key="",
                  max_cost_per_run_usd=0.5, max_attempts_per_claim=3, llm_cache=False)
    monkeypatch.setattr(loop, "load_settings", lambda: st)
    monkeypatch.setattr(loop, "LLMClient", lambda settings: SimpleNamespace(total_cost_usd=0.0))
    monkeypatch.setattr(loop, "get_sandbox", lambda backend: object())
    return Orchestrator(**kw)


def test_default_mode_is_live_and_env_or_arg_selects_fixture(monkeypatch):
    monkeypatch.delenv("TAVILY_MODE", raising=False)
    assert _make(monkeypatch).tavily_mode == "live"                        # Standard: live
    monkeypatch.setenv("TAVILY_MODE", "fixture")
    assert _make(monkeypatch).tavily_mode == "fixture"
    assert _make(monkeypatch, tavily_mode="live").tavily_mode == "live"     # ausdruecklicher Wert gewinnt
    monkeypatch.setenv("TAVILY_MODE", "kaputt")
    with pytest.raises(ValueError):
        _make(monkeypatch)


def test_record_source_marks_live_or_fixture_with_hash_and_time():
    from repro_agent.events import EventLog

    o = Orchestrator.__new__(Orchestrator)
    o._sources, o.events = [], EventLog(None)
    o._record_source("paper", "https://arxiv.org/pdf/x", "abc", live=True)
    o._record_source("readme", "https://github.com/a/b", "def", live=False, retrieved_at="2026-01-01T00:00:00+00:00", sha256="H")
    assert [s["mode"] for s in o._sources] == ["live", "fixture"]
    assert o._sources[0]["sha256"] and o._sources[0]["retrieved_at"] and o._sources[0]["n_chars"] == 3
    assert o._sources[1]["sha256"] == "H"
