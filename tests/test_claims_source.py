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
