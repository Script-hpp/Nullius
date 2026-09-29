import json
from pathlib import Path

import pytest

from repro_agent.claims_source import ExtractionInvalid, PaperText, verify_extraction

FIXTURE = Path(__file__).resolve().parents[1] / "benchmark" / "fixtures" / "1609.02907.json"
TEXT = json.loads(FIXTURE.read_text())["text"]  # echte Tavily-Antwort, keine neuen Credits

GOOD = {"value": 81.5, "quote": "GCN (this paper) 70 .3 (7s) 81 .5 (4s) 79 .0 (38s)"}


def test_real_fixture_value_verified():
    assert verify_extraction(GOOD, TEXT) == pytest.approx(0.815)


def test_quote_not_in_text_rejected():
    with pytest.raises(ExtractionInvalid, match="nicht im Quelltext"):
        verify_extraction({"value": 81.5, "quote": "GCN (this paper) 99 .9 81 .5"}, TEXT)


def test_number_not_in_quote_rejected():
    with pytest.raises(ExtractionInvalid, match="steht nicht im Textausschnitt"):
        verify_extraction({**GOOD, "value": 82.5}, TEXT)


def test_missing_fields_rejected():
    with pytest.raises(ExtractionInvalid):
        verify_extraction({"value": None, "quote": ""}, TEXT)


def test_fixture_is_valid_papertext():
    PaperText(**json.loads(FIXTURE.read_text()))


def test_windows_keep_the_result_table_when_the_dataset_name_fills_the_paper():
    import json
    from pathlib import Path

    from repro_agent.claims_source import _windows

    fix = Path(__file__).resolve().parents[1] / "benchmark" / "fixtures"
    gat = json.loads((fix / "1710.10903.json").read_text())["text"]
    assert "GAT (ours) 83.0" not in _windows(gat, "Cora")[:14000]          # altes Verhalten: Tabelle abgeschnitten
    assert "GAT (ours) 83.0" in _windows(gat, "Cora", prefer="GAT")        # mit Methodenname bleibt sie drin
    for fixture, kw, method, needle in [("1609.02907", "Cora", "GCN (this paper)", "81 .5"), ("1811.05868", "CORA", "GCN", "81 .5"),
                                        ("1902.07153", "Cora", "SGC", "81 .0")]:
        text = json.loads((fix / f"{fixture}.json").read_text())["text"]
        assert needle in _windows(text, kw, prefer=method)                  # bisherige Papers weiter abgedeckt
