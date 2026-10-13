"""A catalog claim identifies a source result, not any result for the same method.

FITS publishes two ETTh1/96 MSE results in different tables. No model/network calls.
"""
import hashlib
from pathlib import Path

import pytest

from repro_agent import catalog
from repro_agent.claims_source import PaperText, TargetVerificationBlocked, _norm
from repro_agent.table_check import check_columns
from test_discovery_target_handoff import target_gate


TABLES = (Path(__file__).parent / "fixtures/fits_results_tables.tex").read_text()


def paper(text=TABLES):
    return PaperText("2307.03756", "https://export.arxiv.org/e-print/2307.03756v3",
                     "latex", "2026-10-13T15:14:43Z", hashlib.sha256(text.encode()).hexdigest(), "v3", text)


def test_real_fits_catalog_target_survives_other_valid_appendix_result(monkeypatch, tmp_path):
    entry = catalog.find_entry(catalog.load_catalog(), "disc_2307_03756")
    config, claims = catalog.build_paper(entry)
    source = paper()
    orchestrator, events, _ = target_gate(monkeypatch, tmp_path, source)
    orchestrator.resolve_targets(config, claims)
    evidence = claims[0].target_evidence
    assert claims[0].target == .372
    assert evidence["value"] == .372 and evidence["column_check"] == "ok"
    assert evidence["table"] == "tab:FITSett"
    assert evidence["quote"] in source.text
    assert evidence["sha256"] == source.sha256
    assert any(e["type"] == "target_resolved" for e in events)
    assert not any(e["type"] == "llm_call" for e in events)


@pytest.mark.parametrize("value,label", [(.372, "tab:FITSett"), (.368, "tab:app_ett")])
def test_either_explicit_source_claim_can_be_verified_without_substituting_the_other(value, label):
    from repro_agent.claims_source import verify_selected_target
    evidence = verify_selected_target(paper(), "FITS", "ETTh1", "MSE for prediction length 96",
                                      value, rf"FITS & \textbf{{{value}}}", False)
    assert evidence and evidence.value == value and evidence.table == label


def test_transposed_table_checks_method_dataset_horizon_and_caption_metric():
    result = check_columns(_norm(TABLES), _norm(r"FITS & \textbf{0.372}"), .372,
                           "FITS", "ETTh1", "MSE for prediction length 96", False)
    assert result.status == "ok" and result.horizon == 96
    assert result.source_row and _norm(r"FITS & \textbf{0.372}") in result.source_row


@pytest.mark.parametrize("method,dataset,metric,value,quote,text", [
    ("FITS", "ETTh2", "MSE for prediction length 96", .372, r"FITS & \textbf{0.372}", TABLES),
    ("FITS", "ETTh1", "MAE for prediction length 96", .372, r"FITS & \textbf{0.372}", TABLES),
    ("FITS", "ETTh1", "MSE for prediction length 192", .372, r"FITS & \textbf{0.372}", TABLES),
    ("PatchTST", "ETTh1", "MSE for prediction length 96", .372, r"FITS & \textbf{0.372}", TABLES),
    ("FITS", "ETTh1", "MSE for prediction length 96", .404, r"FITS & \textbf{0.372} & \textbf{0.404}", TABLES),
    ("FITS", "ETTh1", "MSE for prediction length 96", .372, r"FITS & \textbf{0.372}", TABLES + TABLES),
    ("FITS", "ETTh1", "MSE for prediction length 96", .372, r"FITS & \textbf{0.372}",
     TABLES.replace(r"FITS & \textbf{0.372}", r"FITS & \textbf{0.372} & 0.001")),
])
def test_selected_quote_does_not_bypass_other_identity_checks(method, dataset, metric, value, quote, text):
    from repro_agent.claims_source import ExtractionInvalid, verify_selected_target
    with pytest.raises(ExtractionInvalid):
        verify_selected_target(paper(text), method, dataset, metric, value, quote, False)


def test_binding_is_general_for_other_method_names_and_mae_tables():
    from repro_agent.claims_source import verify_selected_target
    text = TABLES.replace("FITS", "FrequencyNet").replace("in MSE", "in MAE")
    evidence = verify_selected_target(paper(text), "FrequencyNet", "ETTm2",
                                      "MAE for prediction length 96", .162,
                                      r"FrequencyNet & \textbf{0.372}", False)
    assert evidence and evidence.value == .162


def test_real_sparsetsf_multirow_method_target_is_bound_to_its_own_main_table():
    from repro_agent.claims_source import verify_selected_target
    text = (Path(__file__).parent / "fixtures/sparsetsf_main_results.tex").read_text()
    entry = catalog.find_entry(catalog.load_catalog(), "disc_2405_00946")
    evidence = verify_selected_target(paper(text), "SparseTSF", "ETTh1", "MSE for prediction length 96",
                                      .359, entry["candidate"]["claim"]["quote"], False)
    assert evidence and evidence.table == "main_reslut" and evidence.value == .359


def test_real_segrnn_nested_header_tables_do_not_truncate_the_source_result():
    from repro_agent.claims_source import ExtractionInvalid, verify_selected_target
    text = (Path(__file__).parent / "fixtures/segrnn_main_results.tex").read_text()
    entry = catalog.find_entry(catalog.load_catalog(), "disc_2308_11200")
    quote = entry["candidate"]["claim"]["quote"]
    evidence = verify_selected_target(paper(text), "SegRNN", "ETTh1", "MSE for prediction length 96",
                                      .341, quote, False)
    assert evidence and evidence.table == "main_result" and evidence.value == .341
    assert evidence.quote in text
    with pytest.raises(ExtractionInvalid):
        verify_selected_target(paper(text), "SegRNN", "ETTh2", "MSE for prediction length 96",
                               .341, quote, False)
    with pytest.raises(ExtractionInvalid):
        verify_selected_target(paper(text), "SegRNN", "ETTh1", "MAE for prediction length 96",
                               .341, quote, False)


def test_incomplete_nested_tabular_does_not_produce_a_column_proof():
    from repro_agent.table_check import latex_tabulars
    assert latex_tabulars(r"\begin{tabular}{cc} \begin{tabular}{c} name\end{tabular}") == []


def test_caption_after_tabular_and_nested_multiline_formatting_are_supported():
    from repro_agent.claims_source import verify_selected_target
    import re
    text = TABLES[TABLES.index(r"\begin{table*}", TABLES.index(r"\end{table*}") + 1):]
    caption = re.search(r"\\caption\{[^\n]*\}\n", text).group()
    text = text.replace(caption, "").replace(r"\end{table*}",
                                             caption.replace("in MSE", "in \\textbf{\nMSE\n}") + r"\end{table*}")
    evidence = verify_selected_target(paper(text), "FITS", "ETTh1", "MSE for prediction length 96",
                                      .372, r"FITS & \textbf{0.372}", False)
    assert evidence and evidence.value == .372


@pytest.mark.parametrize("caption", ["MSE and MAE results", "Forecasting results", "MSE results}\\caption{MAE results"])
def test_ambiguous_or_missing_table_metric_does_not_pass(caption):
    from repro_agent.claims_source import ExtractionInvalid, verify_selected_target
    import re
    text = re.sub(r"\\caption\{[^\n]*\}\n", lambda _: r"\caption{" + caption + "}\n", TABLES)
    with pytest.raises(ExtractionInvalid):
        verify_selected_target(paper(text), "FITS", "ETTh1", "MSE for prediction length 96",
                               .372, r"FITS & \textbf{0.372}", False)


def test_changed_fresh_source_still_blocks_selected_catalog_claim(monkeypatch, tmp_path):
    entry = catalog.find_entry(catalog.load_catalog(), "disc_2307_03756")
    config, claims = catalog.build_paper(entry)
    # The quote continues to occur elsewhere in the row; ETTh1/96 itself changed.
    text = TABLES.replace(r"FITS & \textbf{0.372} & \textbf{0.404}",
                         r"FITS & \textbf{0.373} & \textbf{0.372}")
    config["discovery_quote"] = r"\textbf{0.372}"
    orchestrator, _, _ = target_gate(monkeypatch, tmp_path, paper(text))
    with pytest.raises(TargetVerificationBlocked):
        orchestrator.resolve_targets(config, claims)
    assert claims[0].target == .372


def test_live_discovery_transposed_claim_reaches_same_catalog_target_gate(monkeypatch, tmp_path):
    from repro_agent import discovery, live_catalog
    from test_discovery_reliability import Model, Search
    monkeypatch.setenv("TAVILY_MODE", "live")
    row = r"FITS & \textbf{0.372}"
    model = Model({"method": "FITS", "metric": "mse", "target": .372, "horizon": 96,
                   "quote": row, "protocol": "unklar"})
    agent = discovery.DiscoveryAgent(llm=model, tavily=Search(), cache_path=str(tmp_path / "cache.json"))
    monkeypatch.setattr(agent, "_find_papers_tavily", lambda dataset: [{
        "arxiv_id": "2307.03756", "title": "FITS: Modeling Time Series", "authors": [],
        "abstract": "We introduce FITS, a model for multivariate time series forecasting on ETTh1."}])
    monkeypatch.setattr(discovery, "_fetch_arxiv_source", lambda *a: TABLES)
    monkeypatch.setattr(agent, "_find_repo_and_context", lambda *a, **k: (
        "https://github.com/VEWOXIC/FITS", "Public code", []))
    monkeypatch.setattr(live_catalog, "PATH", tmp_path / "candidates.json")
    events = []
    candidates = agent.discover(["ETTh1"], on_event=events.append)
    assert len(candidates) == 1, events
    event = live_catalog.save_candidate(agent._candidate_event(candidates[0], False))
    assert event["claim"]["column_check"] == "ok" and event["runnable"]
    assert event["claim"]["source_evidence"]["sha256"] == hashlib.sha256(TABLES.encode()).hexdigest()
    assert event["claim"]["source_evidence"]["table"] == "tab:FITSett"
    config, claims = catalog.build_paper(live_catalog.find_entry(event["catalog_id"]))
    orchestrator, _, _ = target_gate(monkeypatch, tmp_path, paper())
    orchestrator.resolve_targets(config, claims)
    assert claims[0].target_evidence["table"] == "tab:FITSett"
