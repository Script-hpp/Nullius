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


SH = json.loads((FIX / "1811.05868.json").read_text())["text"]


def test_shchur_table1_headers_without_method_word():
    gcn = "GCN 81 .5 ± 1.3 71.9 ± 1.9 77 .8 ± 2.9 62.2 ± 0.6"
    mlp = "MLP 58 .2 ± 2.1 59 .1 ± 2.3 70 .0 ± 2.1 36 .8 ± 1.0"
    assert chk(SH, gcn, 0.815, "GCN", "CORA", "accuracy", True).status == "ok"
    assert chk(SH, gcn, 0.719, "GCN", "CORA", "accuracy", True).status == "mismatch"      # CiteSeer-Spalte
    assert chk(SH, gcn, 0.622, "GCN", "CORA Full", "accuracy", True).status == "ok"        # 'CORA Full' ist eine Spalte
    assert chk(SH, mlp, 0.582, "MLP", "CORA", "accuracy", True).status == "ok"
    assert chk(SH, mlp, 0.591, "MLP", "CORA", "accuracy", True).status == "mismatch"


def test_shchur_row_not_in_text_is_not_checkable():
    assert chk(SH, "GCN 1.0 2.0 3.0 4.0", 0.01, "GCN", "CORA", "accuracy", True).status == "nicht pruefbar"


SGC = json.loads((FIX / "1902.07153.json").read_text())["text"]


def test_partial_quote_is_extended_to_full_row():
    # Das Modell zitierte nur "SGC 81.0 ± 0.0"; die Zeile hat drei Spalten (Cora Citeseer Pubmed)
    assert chk(SGC, "SGC 81 .0 ± 0.0", 0.81, "SGC", "Cora", "accuracy", True).status == "ok"
    assert chk(SGC, "SGC 81 .0 ± 0.0", 0.719, "SGC", "Cora", "accuracy", True).status == "mismatch"
    assert chk(SGC, "SGC 81 .0 ± 0.0", 0.719, "SGC", "Citeseer", "accuracy", True).status == "ok"


DGI = json.loads((FIX / "1809.10341.json").read_text())["text"]
GAT = json.loads((FIX / "1710.10903.json").read_text())["text"]


def test_percent_signs_after_std_do_not_break_the_row():
    dgi = "X, A DGI (ours) 82.3 ± 0.6% 71.8 ± 0.7% 76.8 ± 0.6%"
    assert chk(DGI, dgi, 0.823, "DGI (ours)", "Cora", "accuracy", True).status == "ok"
    assert chk(DGI, dgi, 0.718, "DGI (ours)", "Cora", "accuracy", True).status == "mismatch"
    assert chk(DGI, dgi, 0.718, "DGI (ours)", "Citeseer", "accuracy", True).status == "ok"
    gat = "GAT (ours) 83.0 ± 0.7% 72.5 ± 0.7% 79.0 ± 0.3%"
    assert chk(GAT, gat, 0.83, "GAT", "Cora", "accuracy", True).status == "ok"
