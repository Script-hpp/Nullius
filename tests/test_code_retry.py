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


def _orch_full(replies):
    """Wie _orch, protokolliert zusaetzlich max_tokens und Prompt; Antworten als (Inhalt, finish_reason)."""
    o = Orchestrator.__new__(Orchestrator)
    calls = []

    def ask(role, system, user, max_tokens=None, temperature=0.2):
        calls.append({"max_tokens": max_tokens, "user": user})
        content, finish = replies.pop(0)
        return SimpleNamespace(content=content, finish_reason=finish)

    o._ask = ask
    return o, calls


def test_truncated_reply_is_retried_with_more_tokens_and_code_only_note():
    # Abgeschnitten: halber Code-Block ohne schliessenden Zaun, finish_reason length
    o, calls = _orch_full([("```python\nprint(", "length"), ("```python\nprint(1)\n```", "stop")])
    assert o._code_from_llm("s", "u").strip() == "print(1)"
    assert calls[0]["max_tokens"] is None and "IMPORTANT" not in calls[0]["user"]
    assert calls[1]["max_tokens"] > 8192 and "IMPORTANT" in calls[1]["user"]


def test_truncated_reply_is_not_used_even_if_a_block_matches():
    # Ein abgeschlossener Teil-Block in einer abgeschnittenen Antwort ist kein vollstaendiges Skript
    o, _ = _orch_full([("```python\nx = 1\n```\nand then", "length"), ("```python\nprint(2)\n```", "stop")])
    assert o._code_from_llm("s", "u").strip() == "print(2)"


def test_all_replies_truncated_raises():
    o, calls = _orch_full([("```python\na", "length")] * 3)
    with pytest.raises(ValueError, match="abgeschnitten"):
        o._code_from_llm("s", "u")
    assert len(calls) == 3


def test_code_temperature_is_capped_and_retry_prompts_stay_distinct():
    # FITS-Live-Lauf 35758c32: Kandidat 2 lief bei 0.8/1.1/1.4 und lieferte abgebrochenen Code und Wortsalat
    o = Orchestrator.__new__(Orchestrator)
    calls = []

    def ask(role, system, user, max_tokens=None, temperature=0.2):
        calls.append((temperature, user))
        return SimpleNamespace(content="no code", finish_reason="stop")

    o._ask = ask
    o._emit = lambda *a, **k: None
    with pytest.raises(ValueError):
        o._code_from_llm("s", "u", temperature=0.7)
    assert all(t <= 0.7 for t, _ in calls)
    assert len({(t, u) for t, u in calls}) == 3   # sonst lieferte der Cache dreimal dieselbe Antwort
