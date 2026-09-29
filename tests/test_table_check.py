import json
from pathlib import Path

import pytest

from repro_agent.claims_source import ExtractionInvalid, PaperText, _column_status, _norm
from repro_agent.table_check import check_columns

FIX = Path(__file__).resolve().parents[1] / "benchmark" / "fixtures"
DL = json.loads((FIX / "2205.13504.json").read_text())["text"]
GCN = json.loads((FIX / "1609.02907.json").read_text())["text"]
ROW = "ETTh1  96 0.80% 0.375 0.397 0.374 0.394 0.375 0.399 0.376 0.419 0.449 0.459 0.865 0.713 0.664 0.612 0.878 0.740 1.295 0.713"


def chk(text, quote, value, method, dataset, metric, percent):
    return check_columns(_norm(text), _norm(quote), value, method, dataset, metric, percent)


@pytest.mark.parametrize("method,metric,value,status", [
    ("DLinear", "MSE for prediction length 96", 0.375, "ok"),
    ("DLinear", "MAE for prediction length 96", 0.399, "ok"),      # drittes Paar
    ("Repeat", "MSE for prediction length 96", 1.295, "ok"),       # letztes Paar
    ("Repeat", "MAE for prediction length 96", 0.713, "ok"),
    ("NLinear", "MSE for prediction length 96", 0.374, "ok"),
    ("DLinear", "MAE for prediction length 96", 0.397, "mismatch"),   # Linear-MAE gewaehlt
    ("DLinear", "MSE for prediction length 96", 0.376, "mismatch"),   # FEDformer-MSE gewaehlt
    ("Repeat", "MSE for prediction length 96", 0.713, "mismatch"),    # MAE statt MSE
])
def test_dlinear_columns(method, metric, value, status):
    assert chk(DL, ROW, value, method, "ETTh1", metric, False).status == status


def test_gcn_columns():
    row = "GCN (this paper) 70 .3 (7s) 81 .5 (4s) 79 .0 (38s) 66 .0 (48s)"
    assert chk(GCN, row, 0.815, "GCN (this paper)", "Cora", "accuracy", True).status == "ok"
    assert chk(GCN, row, 0.703, "GCN (this paper)", "Cora", "accuracy", True).status == "mismatch"  # Citeseer
    assert chk(GCN, row, 0.79, "GCN (this paper)", "Pubmed", "accuracy", True).status == "ok"


def test_unknown_header_is_not_checkable():
    assert chk("nothing here", "a 1.0 2.0", 1.0, "M", "D", "MSE", False).status == "nicht pruefbar"


def test_extraction_with_wrong_column_is_invalid():
    with pytest.raises(ExtractionInvalid, match="Spaltenpruefung"):
        _column_status(DL, ROW, 0.397, "DLinear", "ETTh1", "MAE for prediction length 96", False)
    assert _column_status(DL, ROW, 0.399, "DLinear", "ETTh1", "MAE for prediction length 96", False) == (None, "ok")
