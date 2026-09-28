import json
from pathlib import Path

from repro_agent.claims_source import PaperText, extract_target_cached

FIXTURE = Path(__file__).resolve().parents[1] / "benchmark" / "fixtures" / "1609.02907.json"


def test_cached_target_needs_no_llm_and_reverifies():
    paper = PaperText(**json.loads(FIXTURE.read_text()))
    ev = extract_target_cached(None, paper, "GCN (this paper)", "Cora", "accuracy")
    assert abs(ev.value - 0.815) < 1e-9
