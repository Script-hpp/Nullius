"""Zaehl- und Entscheidungsregel des code_grounded-Pilots (scripts/review/cg_pilot.py). Ohne LLM, ohne Docker."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "review"))

import cg_pilot as cp  # noqa: E402


def _results(table: dict[tuple[str, str], list[str]]) -> list[cp.Result]:
    out = []
    for (unit, arm), urteile in table.items():
        for t, u in zip(cp.TEMPS, urteile):
            out.append(cp.Result(unit, arm, t, u, 0.8, 0.01, 5, 5))
    return out


def _full(heute_fail: int, cg_fail: int, kipf_h: int, kipf_c: int) -> list[cp.Result]:
    R, N = "reproduziert", "nicht reproduziert"

    def split(total, units):   # verteilt Treffer auf Einheiten (je max. 2)
        d = {}
        for u in units:
            k = min(2, total)
            total -= k
            d[u] = [R] * k + [N] * (2 - k)
        return d

    t = {}
    for arm, tot, ktot in (("heute", heute_fail, kipf_h), ("cg", cg_fail, kipf_c)):
        for u, v in split(tot, cp.FAIL_UNITS).items():
            t[(u, arm)] = v
        for u, v in split(ktot, cp.NONREG_UNITS).items():
            t[(u, arm)] = v
    return _results(t)


def test_classify_needs_enough_valid_seeds():
    f = lambda m: "reproduziert"
    assert cp.classify([], 5, f) == ("nicht reproduzierbar (Setup)", None, None)
    assert cp.classify([0.8, 0.8], 10, f)[0] == "nicht reproduzierbar (Setup)"      # 2 von 10 gueltig
    u, mean, std = cp.classify([0.8, 0.82], 4, f)
    assert u == "reproduziert" and mean == pytest.approx(0.81) and std > 0


@pytest.mark.parametrize("h,c,kh,kc,decision", [
    (0, 3, 2, 2, "bestanden"),
    (1, 3, 2, 2, "bestanden"),          # 3 >= 3 und 3 >= 1 + 2
    (2, 4, 2, 1, "bestanden"),          # Kipf cg = heute - 1 erlaubt
    (2, 4, 2, 0, "nicht bestanden"),    # Kipf cg < heute - 1
    (0, 2, 2, 2, "nicht bestanden"),    # unter 3 Treffer
    (2, 3, 2, 2, "nicht bestanden"),    # nur +1 gegenueber heute
    (4, 4, 2, 2, "nicht bestanden"),
])
def test_decide_pass_rule(h, c, kh, kc, decision):
    d = cp.decide(_full(h, c, kh, kc))
    assert d["decision"] == decision, d


def test_decide_incomplete_is_never_a_pass():
    res = _full(0, 6, 2, 2)
    res[0].urteil = None                                  # eine Probe nicht gelaufen
    assert cp.decide(res)["decision"] == "unvollstaendig"
    assert cp.decide(res[:-1])["decision"] == "unvollstaendig"


def test_reference_verdicts_and_units_registered():
    assert set(cp.FAIL_UNITS) | set(cp.NONREG_UNITS) == set(cp.UNITS)
    assert all(v[1] == "reproduziert" for v in cp.UNITS.values())
    assert (len(cp.UNITS), len(cp.ARMS), len(cp.TEMPS)) == (4, 2, 2) and cp.CALL_CAP == 45


def test_call_counter_hard_cap(tmp_path):
    class Resp:
        status_code, text = 200, "ok"

    class Client:
        def post(self, *a, **k):
            return Resp()

    cp.MIN_INTERVAL_S, c = 0.0, Client()
    counter = cp.CallCounter(tmp_path / "calls.jsonl", cap=3)
    counter.wrap(c)
    for _ in range(3):
        c.post("/x")
    assert counter.count() == 3
    with pytest.raises(cp.QuotaStop):
        c.post("/x")
    assert counter.count() == 3                           # die vierte Anfrage ging nie hinaus


def test_call_counter_stops_on_daily_limit(tmp_path):
    class Resp:
        status_code, text = 429, '{"error": "Rate limit exceeded: free-models-per-day"}'

    class Client:
        def post(self, *a, **k):
            return Resp()

    cp.MIN_INTERVAL_S, c = 0.0, Client()
    cp.CallCounter(tmp_path / "c.jsonl").wrap(c)
    with pytest.raises(cp.QuotaStop):
        c.post("/x")


def test_spec_json_roundtrip_keeps_excerpt_and_enforcement():
    from repro_agent.spec import Spec, SpecItem

    spec = Spec(model="NLinear", family="forecast", excerpt="=== a ===\nx", dropped_defaults=["lr"],
                items=[SpecItem("lr", "0.005", "q", "repo:a.sh@abc", "start script", 0.005, enforce=True)])
    back = cp._spec_from_json(cp._spec_to_json(spec))
    assert back == spec and back.enforced()[0].number == 0.005
