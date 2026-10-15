"""Author launch scripts derive settings with shell arithmetic (SST: seq_len=$(($label_len*2))). 2026-10-15 the readiness
check stopped SST with 'unresolved model parameters: seq_len' because only literal bindings were read."""
from pathlib import Path

import pytest

from repro_agent.experiment_settings import shell_arithmetic, shell_calls

SST = (Path(__file__).parent / "fixtures" / "sst_etth1.sh").read_text()


def test_sst_launch_resolves_its_derived_input_length():
    calls = shell_calls(SST, {"pred_len": 96, "data": "ETTh1"})
    assert len(calls) == 1
    _, expanded, bindings = calls[0]
    assert bindings["seq_len"] == "672" and bindings["label_len"] == "336" and bindings["pred_len"] == "96"
    assert "--seq_len 672" in expanded and "--pred_len 96" in expanded


@pytest.mark.parametrize("value,bindings,expected", [
    ("$(($label_len*2))", {"label_len": "336"}, "672"),
    ("$(( (x+1) / 2 ))", {"x": "6"}, "3"),
    ("$(( -7 / 2 ))", {}, "-3"),          # bash truncates toward zero
    ("$(( -7 % 2 ))", {}, "-1"),          # and keeps the dividend's sign
    ("$((${a}+b))", {"a": "1", "b": "2"}, "3"),
])
def test_bash_integer_arithmetic(value, bindings, expected):
    assert shell_arithmetic(value, bindings) == expected


@pytest.mark.parametrize("value,bindings", [
    ("$((y*2))", {}),                      # unbound
    ("$((x*2))", {"x": "0.5"}),            # not an integer binding
    ("$((x**2))", {"x": "3"}),             # unsupported operator
    ("$((1/0))", {}),
    ("$(cat f)", {}),                      # command substitution is never evaluated
    ("$((x))", {"x": "$(rm -rf /)"}),
])
def test_anything_else_stays_unresolved(value, bindings):
    assert shell_arithmetic(value, bindings) is None


def test_derived_value_inside_a_conditional_is_never_bound():
    text = 'label_len=336\nif [ "$x" = 1 ]; then\n  seq_len=$(($label_len*2))\nfi\npython run.py --seq_len $seq_len --pred_len 96\n'
    (_, expanded, bindings), = shell_calls(text, {"pred_len": 96})
    assert "seq_len" not in bindings and "--seq_len $seq_len" in expanded
