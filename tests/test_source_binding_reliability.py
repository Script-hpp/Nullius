"""Budget-free production source failures: identity, drift, retry, source format."""
import gzip

import httpx
import pytest

from repro_agent import catalog, claims_source, discovery
from repro_agent.claims_source import ExtractionInvalid, TargetVerificationBlocked
from test_selected_target_table import TABLES, paper
from test_discovery_target_handoff import target_gate


def test_strict_latex_claim_does_not_spend_tavily_credits_on_alternate_format(tmp_path, monkeypatch):
    monkeypatch.setattr(claims_source, "latest_arxiv_version", lambda *a: None)
    monkeypatch.setattr(discovery, "_fetch_arxiv_source", lambda *a: "")
    with pytest.raises(ExtractionInvalid, match="temporarily unavailable"):
        claims_source.fetch_paper_text("2307.03756", lambda: pytest.fail("No paid fallback"),
                                      tmp_path, "live", require_latex=True)


def test_source_url_and_version_identify_the_actual_download_without_constructing_tavily(tmp_path, monkeypatch):
    monkeypatch.setattr(claims_source, "latest_arxiv_version", lambda *a: "v3")
    ids = []
    monkeypatch.setattr(discovery, "_fetch_arxiv_source", lambda pid: ids.append(pid) or TABLES)
    evidence = claims_source.fetch_paper_text("2307.03756", lambda: pytest.fail("No Tavily needed"),
                                             tmp_path, "live", require_latex=True)
    assert ids == ["2307.03756v3"]
    assert evidence.url.endswith("2307.03756v3") and evidence.arxiv_version == "v3"


def test_missing_fits_quote_cannot_fall_back_to_another_model_table(monkeypatch, tmp_path):
    config, claims = catalog.build_paper(catalog.find_entry(catalog.load_catalog(), "disc_2307_03756"))
    source = paper(TABLES.replace(r"FITS & \textbf{0.372}", r"FITS & \textbf{0.373}"))
    orchestrator, _, _ = target_gate(monkeypatch, tmp_path, source)
    with pytest.raises(TargetVerificationBlocked, match="source may have changed"):
        orchestrator.resolve_targets(config, claims)


def test_second_location_outside_a_standard_table_is_not_ignored():
    duplicated = TABLES + r"\begin{wraptable}{r}{4cm}\begin{tabular}{cc}FITS & \textbf{0.372}\end{tabular}\end{wraptable}"
    with pytest.raises(ExtractionInvalid, match="multiple source locations"):
        claims_source.verify_selected_target(paper(duplicated), "FITS", "ETTh1", "MSE for prediction length 96",
                                             .372, r"FITS & \textbf{0.372}", False)


def test_explicit_standard_deviation_is_not_mistaken_for_another_metric():
    text = TABLES.replace(r"FITS & \textbf{0.372}", r"FITS & \textbf{0.372}$\pm$0.002")
    evidence = claims_source.verify_selected_target(paper(text), "FITS", "ETTh1", "MSE for prediction length 96",
                                                    .372, r"FITS & \textbf{0.372}", False)
    assert evidence.value == .372


def test_two_metrics_need_explicit_aligned_metric_headers():
    text = r"""\begin{table}\caption{MSE and MAE results}\label{mixed}
\begin{tabular}{c|cc}Dataset & \multicolumn{2}{c}{ETTh1}\\
Horizon & 96 & 96\\ Metric & MSE & MAE\\
FrequencyNet & 0.372 & 0.399\\\end{tabular}\end{table}"""
    for metric, value in [("MSE", .372), ("MAE", .399)]:
        evidence = claims_source.verify_selected_target(paper(text), "FrequencyNet", "ETTh1",
                                                        f"{metric} for prediction length 96", value,
                                                        "FrequencyNet & 0.372", False)
        assert evidence.value == value


def test_arxiv_download_retries_transient_failure_once(monkeypatch):
    calls = []
    def get(url, **kwargs):
        calls.append(kwargs["timeout"])
        if len(calls) == 1:
            raise httpx.ReadTimeout("temporary transport failure")
        return httpx.Response(200, request=httpx.Request("GET", url), content=gzip.compress(TABLES.encode()))
    monkeypatch.setattr(httpx, "get", get)
    assert "tab:FITSett" in discovery._fetch_arxiv_source("2307.03756v3")
    assert calls == [30.0, 30.0]


@pytest.mark.parametrize("status,count", [(404, 1), (500, 2)])
def test_arxiv_retry_is_bounded_and_does_not_retry_missing_papers(monkeypatch, status, count):
    calls = []
    monkeypatch.setattr(httpx, "get", lambda url, **k: calls.append(url) or
                        httpx.Response(status, request=httpx.Request("GET", url)))
    assert discovery._fetch_arxiv_source("2307.03756") == ""
    assert len(calls) == count
