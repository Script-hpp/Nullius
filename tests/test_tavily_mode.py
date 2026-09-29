import json
from types import SimpleNamespace

import pytest

from repro_agent.claims_source import fetch_paper_text
from repro_agent.loop import Orchestrator
from repro_agent.repo_source import fetch_readme

TEXT = "x" * 3000


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
    assert json.loads((tmp_path / "1234.5678.json").read_text())["text"] == TEXT    # Fixture wird aktualisiert


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
