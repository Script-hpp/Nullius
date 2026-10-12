"""Full automatic discovery with source evidence and deliberately bad model selections.

Only search/provider/repo network boundaries are mocked; real source parsing,
claim validation, persistence and reproduction eligibility all run.
"""
import json
from pathlib import Path

import pytest

from repro_agent import discovery, live_catalog
from repro_agent.catalog import build_paper
from repro_agent.claims_source import _norm
from repro_agent.table_check import check_columns

FIX = Path(__file__).parent / "fixtures"
PAPERS = json.loads((FIX / "discovery_reliability_papers.json").read_text())
SOURCES = {
    "2606.04135": (FIX / "saraf_main_results.tex").read_text(),
    "2501.01087": (FIX / "glinear_main_results.tex").read_text(),
    "2505.20716": (FIX / "data_embeddings_main_results.tex").read_text(),
}


class Model:
    def __init__(self, response):
        self.response, self.calls = response, 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        return type("Response", (), {"content": json.dumps(self.response)})()


class Search:
    on_call = None


def agent_for(monkeypatch, tmp_path, model, papers):
    agent = discovery.DiscoveryAgent(llm=model, tavily=Search(), cache_path=str(tmp_path / "cache.json"))
    monkeypatch.setattr(agent, "_find_papers_tavily", lambda dataset: papers)
    monkeypatch.setattr(discovery, "_fetch_arxiv_source", lambda paper: SOURCES[paper])
    monkeypatch.setattr(agent, "_find_repo_and_context", lambda *args, **kwargs: (
        "https://github.com/ShiqiaoZhou/SARAF" if args[3] == "2606.04135" else "https://github.com/t-rizvi/GLinear",
        "Mocked repository availability; claim checks use the original source table.", []))
    monkeypatch.setattr(live_catalog, "PATH", tmp_path / "candidates.json")
    return agent


@pytest.mark.parametrize("model_claim", [
    {},
    {"method": "SARAF", "target": .415, "metric": "mse", "horizon": 96, "quote": "averaged result"},
    {"method": "SARAF", "target": .366, "metric": "mse", "horizon": 96, "quote": "RAFT baseline"},
    {"method": "RAFT", "target": .366, "metric": "mse", "horizon": 192, "quote": "wrong model and horizon"},
])
def test_incident_papers_yield_one_verified_runnable_claim_despite_bad_model_numbers(monkeypatch, tmp_path, model_claim):
    model = Model(model_claim)
    # Time-VLM appeared in the real failed run; it must consume no extraction.
    agent = agent_for(monkeypatch, tmp_path, model, [PAPERS[1], PAPERS[2], PAPERS[0]])
    events = []

    def record(event):
        events.append(live_catalog.save_candidate(event) if event["type"] == "candidate" else event)

    result = agent.discover(["ETTh1"], on_event=record)
    assert len(result) == 1 and result[0].arxiv_id == "2606.04135"
    event = next(e for e in events if e["type"] == "candidate")
    assert event["claim"]["target"] == .371 and event["runnable"]
    assert event["claim"]["column_check"] == "ok"
    assert sum(e["type"] == "extracting" for e in events) == 2
    assert any(e["type"] == "reject" and e["stage"] == "feasibility" for e in events)
    stored = live_catalog.find_entry(event["catalog_id"])
    paper, (claim,) = build_paper(stored)
    assert paper["input_length"] == 720 and claim.target == .371
    assert paper["claims"][0]["require_verified_target"]
    assert .366 not in [e["claim"]["target"] for e in live_catalog.candidate_events()]
    if not model_claim:
        # Two empty calls for the comparison paper; only one for SARAF, whose
        # source already supplies its number. No increased model retry budget.
        assert model.calls == 3


def test_empty_numeric_extraction_still_selects_glinear_and_retains_unknown_setup_warning(monkeypatch, tmp_path):
    model = Model({})
    agent = agent_for(monkeypatch, tmp_path, model, [PAPERS[3]])
    result = agent.discover(["ETTh1"])
    assert len(result) == 1
    c = result[0]
    assert c.claim["method"] == "GLinear" and c.claim["target"] == .382
    assert c.protocol == "unklar"  # never invent a protocol assessment
    event = live_catalog.save_candidate(agent._candidate_event(c, False))
    assert event["runnable"] and event["run_warnings"]
    check = check_columns(_norm(SOURCES[c.arxiv_id]), _norm(c.claim["quote"]), .382,
                          "GLinear", "ETTh1", "MSE for prediction length 96", False)
    assert check.status == "ok"


def test_source_table_never_overrides_an_explicit_feasibility_rejection(monkeypatch, tmp_path):
    model = Model({"infeasible": "requires external weights"})
    agent = agent_for(monkeypatch, tmp_path, model, [PAPERS[0]])
    events = []
    assert not agent.discover(["ETTh1"], on_event=events.append)
    assert any(e["type"] == "reject" and e["stage"] == "feasibility" for e in events)


def test_data_embedding_comparison_is_not_a_new_times2d_method(monkeypatch, tmp_path):
    model = Model({"method": "Times2D", "target": .378, "metric": "mse", "horizon": 96,
                   "quote": "placeholder"})
    source = SOURCES["2505.20716"]
    # Its cited-baseline sentence is from the paper's public source, not a model output.
    SOURCES["2505.20716"] = source + r" Times2D \cite{nematirad2025times2d} proposes a multi-block decomposition."
    try:
        agent = agent_for(monkeypatch, tmp_path, model, [PAPERS[2]])
        # Use an actual row quoted by the previous deployed extraction.
        row = next(row for row in discovery.re.split(r"\\\\", source) if "0.378" in row and "96" in row)
        model.response["quote"] = row.strip()
        events = []
        assert not agent.discover(["ETTh1"], on_event=events.append)
        assert any(e["type"] == "reject" and "cited prior work" in e["reason"] for e in events)
    finally:
        SOURCES["2505.20716"] = source
