from repro_agent.fidelity import Component, ComponentResult, aggregate, _last_json


def comps():
    return [Component("decomp", "Zerlegung", "d", "q1", True), Component("two", "zwei Zweige", "d", "q2", True),
            Component("misc", "klein", "d", "q3", False)]


def res(*statuses):
    return [ComponentResult(i, s, None, "") for i, s in zip(("decomp", "two", "misc"), statuses)]


def test_aggregate():
    assert aggregate(comps(), res("umgesetzt", "umgesetzt", "nicht gefunden")) == "ja"   # nur tragende zaehlen
    assert aggregate(comps(), res("umgesetzt", "nicht umgesetzt", "umgesetzt")) == "teilweise"
    assert aggregate(comps(), res("nicht gefunden", "nicht umgesetzt", "umgesetzt")) == "nein"
    assert aggregate(comps(), res("teilweise", "umgesetzt", "umgesetzt")) == "teilweise"   # Wert fehlt


def test_last_json_ignores_thinking_brackets():
    assert _last_json('think [x] more\n[{"id": "a"}]', "[") == [{"id": "a"}]


# ---- technischer Ausfall ist nie ein Urteil; Zitat-Normalisierung ----
import json as _json
from types import SimpleNamespace

from repro_agent.fidelity import check_fidelity, value_in_quote


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, role, messages, temperature=0.2, max_tokens=None, timeout_s=None):
        self.calls.append((role, temperature, max_tokens))
        content, finish = self.replies.pop(0)
        return SimpleNamespace(content=content, finish_reason=finish)


CODE = "class M:\n    def __init__(self):\n        self.k = 25\n        self.avg = AvgPool1d(kernel_size=25)\n"
COMP = [Component("k", "Kernel", "d", "q", True, value="5"), Component("pool", "Pool", "d", "q", True)]


def reply(*items):
    return ("thinking...\n" + _json.dumps(list(items)), "stop")


def item(i, quote, status="implemented"):
    return {"id": i, "status": status, "code_quote": quote, "note": "n"}


def test_value_is_whole_token():
    assert not value_in_quote("5", "kernel_size=25")
    assert value_in_quote("25", "kernel_size=25")
    assert value_in_quote("0.5", "p = 0.5)") and not value_in_quote("0.5", "p = 10.55")


def test_quote_normalisation_ignores_whitespace_prefix_and_typography_but_not_identifiers():
    llm = FakeLLM([reply(item("k", "  3| self.k  =  25"), item("pool", "self.avg = AvgPool1d(kernel_size=25)"))])
    rep = check_fidelity(llm, [Component("k", "Kernel", "d", "q", True, value="25"), COMP[1]], CODE)
    assert rep.overall == "ja"
    llm = FakeLLM([reply(item("k", "self.k = 25"), item("pool", "self.avg = avgpool1d(kernel_size=25)"))])
    assert check_fidelity(llm, [Component("k", "Kernel", "d", "q", True, value="25"), COMP[1]], CODE) \
        .results[1].status == "nicht gefunden"       # Gross-/Kleinschreibung zaehlt


def test_value_5_does_not_match_kernel_25():
    llm = FakeLLM([reply(item("k", "self.k = 25"), item("pool", "self.avg = AvgPool1d(kernel_size=25)"))])
    rep = check_fidelity(llm, COMP, CODE)          # Wert "5" verlangt
    assert rep.results[0].status == "teilweise"


def test_unreadable_answer_is_not_a_verdict():
    llm = FakeLLM([("no json at all", "stop"), ("still none [x]", "length")])
    rep = check_fidelity(llm, COMP, CODE)
    assert rep.overall == "nicht geprueft"
    assert all(r.status == "nicht geprueft" for r in rep.results)
    assert "technischer Ausfall" in rep.results[0].note
    assert llm.calls[1][2] > llm.calls[0][2] and llm.calls[1][1] != llm.calls[0][1]   # Grenze hoch, anderer Cache-Schluessel


def test_retry_once_recovers():
    llm = FakeLLM([("cut off [{\"id\": \"k\"", "length"), reply(item("k", "self.k = 25"), item("pool", "AvgPool1d(kernel_size=25)"))])
    assert check_fidelity(llm, [Component("k", "K", "d", "q", True, value="25"), COMP[1]], CODE).overall == "ja"
    assert len(llm.calls) == 2


def test_missing_component_answer_is_not_checked_not_nein():
    llm = FakeLLM([reply(item("k", "self.k = 25", "not_implemented"))])   # 'pool' fehlt
    rep = check_fidelity(llm, COMP, CODE)
    assert rep.results[1].status == "nicht geprueft" and rep.overall == "nicht geprueft"


# ---- Standardargument wird beim Aufruf ueberschrieben ----
from repro_agent.fidelity import default_overrides

BASE = "class Model:\n    def __init__(self, seq, pred, kernel_size=25):\n        self.k = kernel_size\n"


def test_default_used_or_same_value_is_fine():
    assert default_overrides(BASE + "m = Model(1, 2)\n", "25", "kernel_size=25") == []
    assert default_overrides(BASE + "m = Model(1, 2, 25)\nn = Model(1, 2, kernel_size=25)\n", "25", "kernel_size=25") == []


def test_positional_and_keyword_override_detected():
    pos = default_overrides(BASE + "m = Model(1, 2, 5)\n", "25", "kernel_size=25")
    kw = default_overrides(BASE + "m = Model(1, 2, kernel_size=49)\n", "25", "kernel_size=25")
    assert pos and "5" in pos[0] and kw and "49" in kw[0]


def test_variable_argument_is_undecidable_and_unrelated_param_ignored():
    assert default_overrides(BASE + "k = 5\nm = Model(1, 2, k)\n", "25", "kernel_size=25") == []
    assert default_overrides(BASE + "m = Model(1, 2, 5)\n", "25", "self.other = 1") == []   # Parameter nicht im Zitat


def test_checker_marks_overridden_default_as_partial():
    code = BASE + "m = Model(1, 2, 5)\n"
    llm = FakeLLM([reply(item("k", "def __init__(self, seq, pred, kernel_size=25):"))])
    rep = check_fidelity(llm, [Component("k", "Kernel", "d", "q", True, value="25")], code)
    assert rep.results[0].status == "teilweise" and "ueberschreibt" in rep.results[0].note
