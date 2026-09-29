import pytest

from repro_agent.claims_source import _norm, _norm_v1
from repro_agent.setup_details import Doc, numbers_in_quote, quote_in_text, verify_raw


def q_in(quote, text, **kw):
    return quote_in_text(quote, _norm(text), **kw)


def test_pdf_typography_is_smoothed():
    assert q_in("a two‑layer GCN for semi‑supervised classification", "a two-layer GCN for semi-supervised classification")   # nicht trennender Bindestrich
    assert q_in("the ﬁrst conﬂict", "the first conflict")                                                                                # Ligaturen
    assert q_in("conducted transductively.", "conducted transductively .")                                                                   # Leerraum vor Punkt
    assert q_in("learning rate 0.2", "learning rate 0 .2".replace("0 .2", "0.2"))
    assert q_in("Dropout (p = 0.6)", "Dropout ( p = 0.6 )")
    assert not _norm_v1("conducted transductively.") == _norm_v1("conducted transductively .")     # alte Pruefung scheiterte hier


def test_numbers_and_identifiers_stay_exact():
    assert not q_in("learning rate 0.05", "learning rate 0.5")
    assert not q_in("kernel_size=5", "kernel_size=25")
    assert not q_in("lr=0.001", "lr=0.01")
    assert not q_in("weight decay 5e-4", "weight decay 5e-5")


def test_ellipsis_at_border_and_inside():
    text = "we train SGC for 100 epochs using Adam with learning rate 0.2. Then we tune the weight decay on the validation set."
    assert q_in("...we train SGC for 100 epochs using Adam with learning rate 0.2.", text)
    assert q_in("we train SGC for 100 epochs ... tune the weight decay on the validation set", text)
    assert not q_in("we tune the weight decay on the validation set ... we train SGC for 100 epochs", text)      # falsche Reihenfolge
    assert not q_in("we train SGC ... epochs", text)                                                               # Stuecke zu kurz
    assert not q_in("we train SGC for 100 epochs ... invented sentence that is not in the text at all", text)


def test_number_in_value_must_be_in_quote_as_whole_number():
    assert numbers_in_quote("0.005", "initial learning rate of 0.005 for all")
    assert not numbers_in_quote("0.005", "initial learning rate of 0.05 for all")
    assert not numbers_in_quote("336", "--seq_len $seq_len")
    assert numbers_in_quote("patience-based early stopping", "early stopping with patience")     # kein Zahlenwert


def test_verify_raw_reasons():
    docs = [Doc("paper", "We use a learning rate of 0.005. Dropout p = 0.6 is applied.")]
    raw = [
        {"doc": "paper", "item": {"topic": "lr", "value": "0.005", "quote": "learning rate of 0.005"}},
        {"doc": "paper", "item": {"topic": "lr", "value": "0.01", "quote": "learning rate of 0.005"}},          # Zahl nicht im Zitat
        {"doc": "paper", "item": {"topic": "x", "value": "1", "quote": "text that is not in the paper 1"}},
        {"doc": "paper", "item": None},
        {"doc": "paper", "item": {"topic": "", "value": "1", "quote": "Dropout p = 0.6"}},
    ]
    kept, dropped = verify_raw(raw, docs)
    assert [d.value for d in kept] == ["0.005"]
    assert [d["reason"] for d in dropped] == ["Zahl des Werts nicht im Zitat", "Zitat nicht im Quelltext", "kein Objekt", "Feld fehlt"]
