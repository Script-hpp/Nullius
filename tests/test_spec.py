import pytest

from repro_agent.spec import (
    parse_number,
    _number_in_quote,
    other_model_selected,
    resolve,
    SpecItem,
    verify_entries
)
from repro_agent.setup_details import Doc

def test_parse_number():
    assert parse_number("lr", "0.005") == 0.005
    assert parse_number("lr", "5e-3") == 0.005
    assert parse_number("epochs", "100") == 100
    assert parse_number("epochs", "100.5") is None  # integer required
    assert parse_number("lr", "abc") is None

def test_number_in_quote():
    assert _number_in_quote("0.005", "learning rate is 0.005")
    assert _number_in_quote("0.005", "lr = 5e-3")
    assert _number_in_quote("100", "epochs: 100")
    assert not _number_in_quote("100", "epochs: 200")

def test_other_model_selected():
    assert other_model_selected("model_name = 'DLinear'", "NLinear") == "DLinear"
    assert other_model_selected("model_name='NLinear'", "NLinear") == ""
    assert other_model_selected("args.model == 'str'", "NLinear") == ""

def test_verify_entries():
    docs = [Doc("paper", "We use DLinear with lr 0.005.")]
    raw = [
        {"key": "lr", "value": "0.005", "quote": "lr 0.005", "source": "paper"}
    ]
    kept, dropped = verify_entries(raw, docs, "DLinear", "forecast")
    assert len(kept) == 1
    assert kept[0].number == 0.005
    assert len(dropped) == 0

    # Wrong quote
    raw2 = [
        {"key": "lr", "value": "0.005", "quote": "lr 0.01", "source": "paper"}
    ]
    kept, dropped = verify_entries(raw2, docs, "DLinear", "forecast")
    assert len(kept) == 0
    assert len(dropped) == 1
    assert "Zitat nicht im Quelltext" in dropped[0]["reason"]

def test_resolve():
    # Ranks: start script > config > paper > code > README > argparse default
    # source_rank parses names like "run.sh" -> "Startskript"
    i1 = SpecItem("lr", "0.01", "quote", "run.sh", "Startskript", 0.01)
    i2 = SpecItem("lr", "0.005", "quote", "paper", "Paper", 0.005)
    
    items, missing = resolve([i1, i2], "forecast")
    # i1 should win
    assert len(items) > 0
    lr_item = next(i for i in items if i.key == "lr")
    assert lr_item.value == "0.01"
    assert lr_item.enforce is True
    assert len(lr_item.alternatives) == 1
    assert lr_item.alternatives[0]["value"] == "0.005"

    # Tie should not be enforced
    i3 = SpecItem("epochs", "10", "quote", "paper", "Paper", 10)
    i4 = SpecItem("epochs", "20", "quote", "paper", "Paper", 20)
    items, missing = resolve([i3, i4], "forecast")
    ep_item = next(i for i in items if i.key == "epochs")
    assert not ep_item.enforce
    assert ep_item.note.startswith("gleichrangige Quellen widersprechen sich")
