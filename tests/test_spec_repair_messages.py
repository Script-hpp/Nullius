"""BANGS live-path run 2026-10-15: every candidate stayed 'invalid' because (a) heads=8 from the GAT branch's launch default
was required to be read by a GCN backbone, and (b) the static message only said 'contradicts ... epochs', so repairs moved
the value to CONFIG["max_epochs"] and back."""
from repro_agent import spec_check
from repro_agent.spec import Spec, SpecItem


def spec():
    return Spec(model="BANGS", family="graph", items=[
        SpecItem("epochs", "2000", "--epochs 2000", "repo:main_node.py", "argparse default", 2000, enforce=True),
        SpecItem("heads", "8", "--nb_heads 8", "repo:main_node.py", "argparse default", 8, enforce=True)])


def test_heads_need_not_be_read_by_a_model_without_attention():
    _, report = spec_check.config_check('CONFIG = {"epochs": 2000, "heads": 8}\nfor e in range(CONFIG["epochs"]): pass\n', spec())
    assert report["unused"] == [] and spec_check.score(spec(), report, []) == (2, 2)


def test_heads_are_still_enforced_when_the_code_sets_them():
    code, report = spec_check.config_check('CONFIG = {"epochs": 2000, "heads": 4}\nn = CONFIG["heads"]\nfor e in range(CONFIG["epochs"]): pass\n', spec())
    assert report["overrides"] == [{"key": "heads", "old": 4, "new": 8}] and "CONFIG['heads'] = 8" in code


def test_messages_name_the_finding_and_the_fix():
    _, report = spec_check.config_check('CONFIG = {"epochs": 2000, "heads": 8}\nCONFIG["max_epochs"] = 40\n'
                                        'for e in range(CONFIG["max_epochs"]): pass\n', spec())
    lines = spec_check.failure_lines(spec_check.failures(report, []), spec())
    assert len(lines) == 1 and 'CONFIG["epochs"] is never read' in lines[0] and "another key" in lines[0]
    value = spec_check.failure_lines([{"key": "epochs", "kind": "value", "old": 40, "new": 2000}], spec())[0]
    assert "is 40 in the CONFIG block" in value and "requires 2000" in value


def test_reading_a_key_the_config_never_defines_is_found_before_the_sandbox():
    code = ('CONFIG = {"epochs": 2000, "heads": 8}\nfor e in range(CONFIG["epochs"]): pass\n'
            'def build():\n    return CONFIG["padding_patch"]\n')
    _, report = spec_check.config_check(code, spec())
    assert report["missing"] == ["padding_patch"]
    (line,) = spec_check.failure_lines(spec_check.failures(report, []), spec())
    assert 'CONFIG["padding_patch"]' in line and "no such key" in line
    added = code + 'CONFIG["padding_patch"] = "end"\n'
    assert spec_check.config_check(added, spec())[1]["missing"] == []
    dynamic = code + 'CONFIG.update(extra)\n'
    assert spec_check.config_check(dynamic, spec())[1]["missing"] == []


def test_candidate_left_unfixed_by_the_spec_rounds_is_not_repaired_again(monkeypatch):
    from repro_agent.loop import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    o.independent_selection = True
    o.spec = spec()
    monkeypatch.setattr(o, "_active_spec", lambda: o.spec)
    assert o._static_spec_blockers('CONFIG = {"epochs": 2000, "heads": 8}\nn = 40\n') == ["epochs"]
    assert o._static_spec_blockers('CONFIG = {"epochs": 2000, "heads": 8}\nfor e in range(CONFIG["epochs"]): pass\n') == []
    o.independent_selection = False
    assert o._static_spec_blockers('CONFIG = {"epochs": 2000}\n') == []
