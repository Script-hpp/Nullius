import pytest

from repro_agent.spec import Spec, SpecItem
from repro_agent.spec_check import config_check, lines_in_code, repair_message

def test_config_check_overrides():
    code = """import torch
CONFIG = {"lr": 0.01, "epochs": 100}
print(CONFIG["lr"])
"""
    spec = Spec("DLinear", "forecast", items=[
        SpecItem("lr", "0.005", "quote", "paper", "Paper", 0.005, enforce=True),
        SpecItem("epochs", "100", "quote", "paper", "Paper", 100, enforce=True)
    ])
    
    new_code, report = config_check(code, spec)
    assert report["has_config"]
    assert len(report["overrides"]) == 1
    assert report["overrides"][0]["key"] == "lr"
    assert report["overrides"][0]["old"] == 0.01
    assert report["overrides"][0]["new"] == 0.005
    
    assert "CONFIG['lr'] = 0.005  # spec override (Paper)" in new_code

def test_config_check_missing_config():
    code = """import torch\nlr = 0.01\n"""
    spec = Spec("DLinear", "forecast", items=[
        SpecItem("lr", "0.005", "quote", "paper", "Paper", 0.005, enforce=True)
    ])
    
    new_code, report = config_check(code, spec)
    assert not report["has_config"]
    assert "lr" in report["unused"]

def test_config_check_unused():
    code = """import torch
CONFIG = {"lr": 0.005, "epochs": 100}
print(CONFIG["lr"])
# epochs is not used!
"""
    spec = Spec("DLinear", "forecast", items=[
        SpecItem("lr", "0.005", "quote", "paper", "Paper", 0.005, enforce=True),
        SpecItem("epochs", "100", "quote", "paper", "Paper", 100, enforce=True)
    ])
    
    new_code, report = config_check(code, spec)
    assert "epochs" in report["unused"]
    assert "lr" not in report["unused"]

def test_config_check_conflicts():
    code = """import torch
CONFIG = {"lr": 0.005}
optim = torch.optim.Adam(model.parameters(), lr=0.01)
"""
    spec = Spec("DLinear", "forecast", items=[
        SpecItem("lr", "0.005", "quote", "paper", "Paper", 0.005, enforce=True)
    ])
    
    new_code, report = config_check(code, spec)
    assert len(report["conflicts"]) == 1
    assert report["conflicts"][0]["key"] == "lr"
    assert report["conflicts"][0]["literal"] == 0.01

def test_lines_in_code():
    code = "def model():\n    x = x + 1\n    return x"
    assert lines_in_code("x = x + 1", code)
    assert lines_in_code("def model():\n...\nreturn x", code)
    assert not lines_in_code("x = x + 2", code)

def test_repair_message():
    spec = Spec("DLinear", "forecast", items=[
        SpecItem("lr", "0.005", "quote with info", "paper", "Paper", 0.005, enforce=True)
    ])
    fails = [{"key": "lr", "kind": "literal", "literal": 0.01, "line": 5}]
    msg = repair_message(fails, spec)
    assert "Line 5 hard-codes lr = 0.01" in msg
    assert "quote with info" in msg
