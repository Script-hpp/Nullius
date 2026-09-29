import pytest

from repro_agent.fidelity import Component, ComponentResult, FidelityReport, aggregate, apply_coverage
from repro_agent.harness.coverage_check import Coverage, locate_quote, quote_coverage

CODE = """import torch
def used(x):
    y = x + 1
    return y

def never_called(x):
    z = x * 2
    return z

for i in range(2):
    used(i)
"""
# ausfuehrbare Zeilen: 1,2,3(used-Rumpf),4,6(def),7,8,10,11 ; nie gelaufen: 7,8
COV = Coverage(executed=frozenset({1, 2, 3, 4, 6, 10, 11}), missing=frozenset({7, 8}))


def test_locate_quote_ignores_whitespace_prefix_and_blank_lines():
    assert locate_quote(CODE, "y = x+1\n  return   y") == (3, 4)
    assert locate_quote(CODE, "   7| z = x * 2\n   8| return z") == (7, 8)
    assert locate_quote(CODE, "x * 2") == (7, 7)          # Teil einer Zeile
    assert locate_quote(CODE, "not in code") is None


def test_quote_coverage_states():
    assert quote_coverage(CODE, "y = x + 1\nreturn y", COV) == "ausgefuehrt"
    assert quote_coverage(CODE, "z = x * 2\nreturn z", COV) == "nicht ausgefuehrt"
    assert quote_coverage(CODE, "return y\n\ndef never_called(x):\n    z = x * 2", COV) == "teilweise ausgefuehrt"
    assert quote_coverage(CODE, "nirgends", COV) == "nicht pruefbar"


def test_from_coverage_json():
    data = {"files": {"/work/train.py": {"executed_lines": [1, 2], "missing_lines": [7]}}}
    cov = Coverage.from_coverage_json(data)
    assert cov.executed == {1, 2} and cov.missing == {7}
    assert Coverage.from_coverage_json({"files": {}}) is None


COMPS = [Component("a", "A", "d", "q", True), Component("b", "B", "d", "q", True)]


def rep(*pairs):
    results = [ComponentResult(i, "umgesetzt", q, "") for i, q in pairs]
    return FidelityReport(aggregate(COMPS, results), results)


def test_ja_with_dead_quote_is_downgraded_to_teilweise_or_nein():
    r = rep(("a", "y = x + 1"), ("b", "z = x * 2"))
    assert r.overall == "ja"
    new, notes = apply_coverage(COMPS, r, CODE, COV)
    assert new.overall == "teilweise" and "b:" in notes[0]
    new, _ = apply_coverage(COMPS, rep(("a", "z = x * 2"), ("b", "return z")), CODE, COV)
    assert new.overall == "nein"


def test_coverage_never_upgrades():
    for overall_in in ("teilweise", "nein"):
        results = [ComponentResult("a", "umgesetzt" if overall_in == "teilweise" else "nicht umgesetzt", "y = x + 1", ""),
                   ComponentResult("b", "nicht umgesetzt", None, "")]
        r = FidelityReport(aggregate(COMPS, results), results)
        assert r.overall == overall_in
        new, _ = apply_coverage(COMPS, r, CODE, COV)
        assert new is r and new.overall == overall_in          # unveraendert


def test_no_coverage_keeps_verdict_with_note():
    r = rep(("a", "z = x * 2"), ("b", "return z"))
    new, notes = apply_coverage(COMPS, r, CODE, None)
    assert new.overall == "ja" and "nicht verfuegbar" in notes[0]


def test_faithful_quote_not_downgraded_and_unfindable_quote_unchanged():
    r = rep(("a", "y = x + 1"), ("b", "return y"))
    assert apply_coverage(COMPS, r, CODE, COV)[0].overall == "ja"
    r = rep(("a", "y = x + 1"), ("b", "gibt es nicht"))
    assert apply_coverage(COMPS, r, CODE, COV)[0].overall == "ja"   # nicht pruefbar: keine Herabstufung
