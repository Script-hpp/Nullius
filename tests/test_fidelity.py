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
