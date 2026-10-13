"""Source outages never remove identity checks or require paid alternate extraction."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import MethodType

import httpx
import pytest

from repro_agent import catalog, claims_source, discovery, loop
from repro_agent.claims_source import ExtractionInvalid, PaperText, TargetVerificationBlocked
from test_discovery_target_handoff import target_gate
from test_selected_target_table import TABLES

RESERVES = Path(__file__).resolve().parents[1] / "benchmark/source_reserves"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(claims_source, "SOURCE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(claims_source, "SOURCE_RESERVE_DIR", tmp_path / "reserves")
    monkeypatch.setattr(claims_source, "latest_arxiv_version", lambda _: None)


def saved(version="v3", text=TABLES):
    return PaperText("2307.03756", f"https://arxiv.org/src/2307.03756{version}", "latex", "2026-10-13",
                     hashlib.sha256(text.encode()).hexdigest(), version, text)


def write_cache(paper, name=None):
    directory = claims_source.SOURCE_CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name or f"{paper.arxiv_id}{paper.arxiv_version}.json")
    path.write_text(json.dumps(asdict(paper)))
    return path


def test_runtime_cache_persists_fresh_source_and_never_writes_live_fixtures(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, "_fetch_arxiv_source", lambda _: discovery.ArxivSourceText(TABLES, "https://arxiv.org/src/2307.03756v3"))
    path = tmp_path / "fixtures"
    first = claims_source.fetch_paper_text("2307.03756", lambda: pytest.fail("No Tavily"), path, "live", True, "v3")
    assert not first.from_cache and first.url == "https://arxiv.org/src/2307.03756v3"
    assert not path.exists()
    monkeypatch.setattr(discovery, "_fetch_arxiv_source", lambda _: "")
    second = claims_source.fetch_paper_text("2307.03756", lambda: pytest.fail("No Tavily"), path, "live", True, "v3")
    assert second.from_cache and second.sha256 == first.sha256 and second.text == first.text


@pytest.mark.parametrize("change", ["wrong_id", "wrong_version", "wrong_kind", "wrong_hash", "changed_hash", "broken_json", "wrong_text_type"])
def test_corrupt_or_wrong_identity_cache_cannot_authorize_the_selected_claim(tmp_path, monkeypatch, change):
    paper = saved()
    expected = paper.sha256
    data = asdict(paper)
    if change == "wrong_id": data["arxiv_id"] = "2307.00000"
    if change == "wrong_version": data["arxiv_version"] = "v2"
    if change == "wrong_kind": data["source_kind"] = "pdf"
    if change == "wrong_hash": data["sha256"] = "0" * 64
    if change == "changed_hash":
        data["text"] = TABLES + "changed text"
        data["sha256"] = hashlib.sha256(data["text"].encode()).hexdigest()
    if change == "wrong_text_type": data["text"] = [0] * 3000
    path = write_cache(paper)
    path.write_text("broken" if change == "broken_json" else json.dumps(data))
    monkeypatch.setattr(discovery, "_fetch_arxiv_source", lambda _: "")
    with pytest.raises(ExtractionInvalid, match="temporarily unavailable"):
        claims_source.fetch_paper_text("2307.03756", lambda: pytest.fail("No paid fallback"), tmp_path, "live",
                                      True, "v3", expected)


def test_cache_for_another_requested_version_is_rejected(tmp_path, monkeypatch):
    write_cache(saved("v3"))
    monkeypatch.setattr(discovery, "_fetch_arxiv_source", lambda _: "")
    with pytest.raises(ExtractionInvalid):
        claims_source.fetch_paper_text("2307.03756", lambda: pytest.fail("No paid fallback"), tmp_path, "live", True, "v4")


def test_unknown_current_version_uses_highest_explicit_version_and_discloses_unknown_state(tmp_path, monkeypatch):
    for version in ("v1", "v3", "v10"):
        write_cache(saved(version))
    # Malformed names and mislabelled contents must not derail version selection.
    write_cache(saved("v1"), "2307.03756v99.json")
    write_cache(saved("v3"), "2307.03756vgarbage.json")
    monkeypatch.setattr(discovery, "_fetch_arxiv_source", lambda _: "")
    result = claims_source.fetch_paper_text("2307.03756", lambda: pytest.fail("No Tavily"), tmp_path, "live", True)
    assert result.arxiv_version == "v10" and result.from_cache and result.current_version_unknown


def test_fresh_changed_source_wins_over_a_valid_cached_target_and_is_still_blocked(monkeypatch, tmp_path):
    write_cache(saved())
    monkeypatch.setattr(discovery, "_fetch_arxiv_source", lambda _: TABLES.replace(r"FITS & \textbf{0.372}", r"FITS & \textbf{0.373}"))
    config, claims = catalog.build_paper(catalog.find_entry(catalog.load_catalog(), "disc_2307_03756"))
    orchestrator, _, _ = target_gate(monkeypatch, tmp_path, saved())
    monkeypatch.setattr(loop, "fetch_paper_text", claims_source.fetch_paper_text)
    with pytest.raises(TargetVerificationBlocked, match="source may have changed"):
        orchestrator.resolve_targets(config, claims)
    assert not orchestrator.paper_text.from_cache


def test_production_fits_can_verify_its_pinned_source_during_a_complete_arxiv_outage(monkeypatch, tmp_path):
    monkeypatch.setattr(claims_source, "SOURCE_RESERVE_DIR", RESERVES)
    monkeypatch.setattr(claims_source, "latest_arxiv_version", lambda _: pytest.fail("Pinned version needs no API lookup"))
    monkeypatch.setattr(discovery, "_fetch_arxiv_source", lambda _: discovery.ArxivSourceText("", "https://arxiv.org/src/2307.03756v3", "HTTP 403 at arxiv.org"))
    config, claims = catalog.build_paper(catalog.find_entry(catalog.load_catalog(), "disc_2307_03756"))
    orchestrator, events, _ = target_gate(monkeypatch, tmp_path, saved())
    orchestrator._sources = []
    orchestrator._record_source = MethodType(loop.Orchestrator._record_source, orchestrator)
    orchestrator._warn = lambda text: events.append({"type": "warning", "text": text})
    monkeypatch.setattr(loop, "fetch_paper_text", claims_source.fetch_paper_text)
    orchestrator.resolve_targets(config, claims)
    assert claims[0].target == .372 and claims[0].target_evidence["table"] == "tab:FITSett"
    assert orchestrator.paper_text.from_cache and orchestrator.paper_text.arxiv_version == "v3"
    assert orchestrator._sources[0]["from_cache"] is True
    assert any(e["type"] == "warning" and "saved LaTeX" in e["text"] and "403" in e["text"] for e in events)


def test_official_second_host_preserves_actual_source_url_and_rejects_html_error_pages(monkeypatch):
    urls = []
    def get(url, **kwargs):
        urls.append(url)
        body = b"<html>blocked " + b"\\begin{table}" * 200 if "export.arxiv.org" in url else TABLES.encode()
        return httpx.Response(200, request=httpx.Request("GET", url), content=body)
    monkeypatch.setattr(httpx, "get", get)
    text = discovery._fetch_arxiv_source("2307.03756v3")
    assert len(urls) == 2 and text.url == "https://arxiv.org/src/2307.03756v3"
    assert "tab:FITSett" in text


def test_download_failure_class_reaches_the_stopped_event(monkeypatch, tmp_path):
    monkeypatch.setattr(discovery, "_fetch_arxiv_source", lambda _: discovery.ArxivSourceText("", "https://arxiv.org/src/2307.03756v3", "ConnectError at arxiv.org"))
    config, claims = catalog.build_paper(catalog.find_entry(catalog.load_catalog(), "disc_2307_03756"))
    orchestrator, events, _ = target_gate(monkeypatch, tmp_path, saved())
    monkeypatch.setattr(loop, "fetch_paper_text", claims_source.fetch_paper_text)
    with pytest.raises(TargetVerificationBlocked, match="ConnectError"):
        orchestrator.resolve_targets(config, claims)
    assert any(e["type"] == "source_unavailable" and "ConnectError" in e["reason"] for e in events)
