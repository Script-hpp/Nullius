"""Discovery -> actual target gate with independent fresh source evidence, no paid calls."""
import hashlib
import json
from dataclasses import asdict

import pytest

from repro_agent import catalog, loop
from repro_agent.claims_source import (
    ExtractionInvalid, PaperText, TargetVerificationBlocked,
    extract_target, extract_target_cached, verify_extraction,
)
from repro_agent.loop import Orchestrator
from test_discovery_reliability import Model, PAPERS, SOURCES, agent_for


class NoModel:
    def complete(self, *args, **kwargs):
        raise AssertionError("A verified source table must not need another numerical model call")


def source(paper_id, text=None):
    text = text or SOURCES[paper_id]
    # GLinear names its proposed model in the actual abstract, not an '(ours)' column.
    abstract = next(p["abstract"] for p in PAPERS if p["arxiv_id"] == paper_id)
    text = r"\begin{abstract}" + abstract + r"\end{abstract}" + "\n" + text
    return PaperText(paper_id, f"https://export.arxiv.org/e-print/{paper_id}", "latex",
                     "2026-10-13T00:00:00Z", hashlib.sha256(text.encode()).hexdigest(), "v1", text)


def target_gate(monkeypatch, tmp_path, fresh):
    o = Orchestrator.__new__(Orchestrator)
    o.paper_text, o.tavily_mode, o._tavily_calls, o._run_dir = None, "live", 0, tmp_path
    o.llm = NoModel()
    events, fetches = [], []
    o._emit = lambda event_type, **data: events.append({"type": event_type, **data})
    o._record_source = lambda *args: None

    def fetch(paper_id, *args, **kwargs):
        fetches.append(paper_id)
        return fresh

    monkeypatch.setattr(loop, "fetch_paper_text", fetch)
    return o, events, fetches


@pytest.mark.parametrize("index,target,length", [(0, .371, 720), (3, .382, 336)])
def test_actual_discovery_claim_passes_the_second_target_gate_with_verbatim_evidence(monkeypatch, tmp_path, index, target, length):
    agent = agent_for(monkeypatch, tmp_path, Model({}), [PAPERS[index]])
    (candidate,) = agent.discover(["ETTh1"])
    entry = catalog.entry_from_candidate(agent._candidate_event(candidate, False), "test discovery")
    paper, claims = catalog.build_paper(entry)
    fresh = source(candidate.arxiv_id)
    o, events, fetches = target_gate(monkeypatch, tmp_path, fresh)
    o.resolve_targets(paper, claims)
    assert fetches == [candidate.arxiv_id]
    assert claims[0].target == target and paper["input_length"] == length
    assert claims[0].target_evidence["column_check"] == "ok"
    assert claims[0].target_evidence["quote"] in fresh.text
    assert claims[0].target_evidence["sha256"] == fresh.sha256
    assert any(e["type"] == "target_check" and e["agrees"] for e in events)
    assert any(e["type"] == "target_resolved" for e in events)
    assert not any(e["type"] == "target_blocked" for e in events)


@pytest.mark.parametrize("configured,changed", [(.366, False), (.371, True)])
def test_wrong_old_claim_and_changed_fresh_source_still_stop_before_experiment(monkeypatch, tmp_path, configured, changed):
    agent = agent_for(monkeypatch, tmp_path, Model({}), [PAPERS[0]])
    (candidate,) = agent.discover(["ETTh1"])
    candidate.claim["target"] = configured
    paper, claims = catalog.build_paper(catalog.entry_from_candidate(agent._candidate_event(candidate, False), "test"))
    fresh = source(candidate.arxiv_id, SOURCES[candidate.arxiv_id].replace("0.371", "0.372") if changed else None)
    o, events, _ = target_gate(monkeypatch, tmp_path, fresh)
    with pytest.raises(TargetVerificationBlocked, match="discovery selected|Selected result table"):
        o.resolve_targets(paper, claims)
    assert claims[0].target == configured
    assert any(e["type"] == "target_blocked" for e in events)
    assert not any(e["type"] == "target_resolved" for e in events)


def test_synthesized_live_quote_stays_invalid_but_real_source_target_is_independently_verified(tmp_path):
    paper = source("2606.04135")
    invalid = r"ETTh1 & 96 & \underline{0.371} & \underline{0.399}"
    with pytest.raises(ExtractionInvalid, match="Textausschnitt nicht im Quelltext"):
        verify_extraction({"value": .371, "quote": invalid}, paper.text, False)
    ev = extract_target(NoModel(), paper, "SARAF", "ETTh1", "MSE for prediction length 96", False)
    assert ev.value == .371 and ev.quote in paper.text and ev.quote != invalid
    ev_mae = extract_target(NoModel(), paper, "SARAF", "ETTh1", "MAE for prediction length 96", False)
    assert ev_mae.value == .399 and ev_mae.column_check == "ok"

    # A bad derived cache cannot force another model call or silently change history.
    raw = f"{paper.sha256}|SARAF|ETTh1|MSE for prediction length 96|raw"
    key = hashlib.sha256(raw.encode()).hexdigest()[:16]
    path = tmp_path / f"{paper.arxiv_id}_target_{key}.json"
    cached = {**asdict(ev), "quote": invalid}
    path.write_text(json.dumps(cached))
    checked = extract_target_cached(NoModel(), paper, "SARAF", "ETTh1", "MSE for prediction length 96",
                                    fixture_dir=tmp_path, percent=False, mode="live")
    assert checked.value == .371 and checked.column_check == "ok" and not checked.from_store
    assert json.loads(path.read_text()) == cached
