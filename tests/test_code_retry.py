from types import SimpleNamespace

import pytest

from repro_agent.loop import Orchestrator


def _orch(replies):
    o = Orchestrator.__new__(Orchestrator)
    temps = []

    def ask(role, system, user, max_tokens=None, temperature=0.2):
        temps.append(temperature)
        return SimpleNamespace(content=replies.pop(0))

    o._ask = ask
    return o, temps


def test_retries_with_higher_temperature_until_code_block():
    o, temps = _orch(["no code here", "```python\nprint(1)\n```"])
    assert o._code_from_llm("s", "u").strip() == "print(1)"
    assert temps[0] < temps[1]           # sonst gaebe der Cache dieselbe Antwort


def test_gives_up_after_three_replies_without_code():
    o, _ = _orch(["x", "y", "z"])
    with pytest.raises(ValueError):
        o._code_from_llm("s", "u")
