"""code_grounded (docs/recall_analysis_3.md Hebel 3): roher Repo-Ausschnitt in Plan-, Code- und Reparatur-Prompt, argparse-Defaults
weg, wo ein Startskript das Flag setzt, nur Zahlenschluessel im Steckbrief. Flag aus = Prompts byte-identisch. Attrappen, kein Netz."""

import json

import pytest

from repro_agent.dossier import Claim
from repro_agent.loop import Orchestrator
from repro_agent.setup_details import Doc, SetupDetailV2
from repro_agent.spec import (
    CG_INTRO, Spec, SpecItem, as_dict, build_spec, drop_run_script_defaults, format_code_grounded, keys_for, resolve, verify_entries)
from repro_agent.tasks import cora_task, etth1_task

EXCERPT = (
    "=== scripts/etth1.sh ===\npython run_longExp.py --model NLinear --seq_len 336 --learning_rate 0.005\n\n"
    "=== run_longExp.py ===\n"
    "parser.add_argument('--seq_len', type=int, default=96, help='input')\n"
    "parser.add_argument('--learning_rate', type=float, default=0.0001, help='lr')\n"
    "parser.add_argument('--patience', type=int, default=3, help='stop')\n\n"
    "=== utils/tools.py (function body) ===\ndef adjust_learning_rate(optimizer, epoch, args):\n    lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}\n"
)
CLAIM = Claim("c", "NLinear erreicht MSE 0,374 auf ETTh1", "mse", 0.374, 0.01, "T", "paper")


def _spec(excerpt: str = EXCERPT) -> Spec:
    items = [SpecItem("lr", "0.005", "--learning_rate 0.005", "scripts/etth1.sh", "start script", 0.005, enforce=True)]
    return Spec(model="NLinear", family="forecast", items=items, excerpt=excerpt)


def _orch(code_grounded, with_spec: bool, details=None) -> Orchestrator:
    o = Orchestrator.__new__(Orchestrator)
    o.details, o.graph_repair = details or [], False
    if code_grounded is not None:
        o.code_grounded = code_grounded
        o.spec_mode = code_grounded
    if with_spec:
        o.spec, o.spec_mode = _spec(), True if code_grounded else getattr(o, "spec_mode", False)
    return o


def _prompts(o: Orchestrator, task) -> dict:
    seen = {}
    o._ask = lambda role, system, user, **kw: seen.setdefault("plan", user) and type("R", (), {"content": "plan", "finish_reason": "stop"})()
    o._code_from_llm = lambda system, user, **kw: user
    o.plan({"title": "T"}, CLAIM, task)
    return {"plan": seen["plan"], "write": o.write_code("PLAN", task), "repair": o.repair_code("PLAN", task, "print(1)", "boom")}


@pytest.mark.parametrize("task", [cora_task(), etth1_task()])
def test_flag_off_prompts_byte_identical(task):
    legacy = _prompts(_orch(None, False), task)         # Objekt ohne Attribut code_grounded (Stand vor dem Flag)
    off = _prompts(_orch(False, False), task)           # Flag explizit aus
    off_with_spec = _prompts(_orch(False, True), task)  # Flag aus, Steckbrief samt Ausschnitt vorhanden: nie roher Code im Prompt
    assert legacy == off
    assert all("Authors' code" not in p and "run_longExp" not in p for p in off_with_spec.values())


def test_flag_off_keys_for_and_as_dict_unchanged():
    assert keys_for("forecast") == keys_for("forecast", False)
    assert keys_for("forecast", True)[1] == []                                  # nur Zahlenschluessel
    d = as_dict(Spec(model="M", family="graph"))
    assert "excerpt" not in d and "dropped_defaults" not in d and "excerpt_chars" not in d


def test_flag_on_puts_raw_excerpt_in_plan_code_and_repair():
    o = _orch(True, True)
    p = _prompts(o, etth1_task())
    for k in ("plan", "write", "repair"):
        assert EXCERPT in p[k]
        assert CG_INTRO.format(model="NLinear") in p[k]
        assert "- lr = 0.005" in p[k]                                          # S2-Zahl steht im Steckbrief
    assert "CONFIG = {" in p["write"] and "CONFIG = {" in p["repair"]
    assert "Sources give different values" not in p["plan"]


def test_flag_on_without_items_still_shows_excerpt():
    o = _orch(True, False)
    o.spec = Spec(model="NLinear", family="forecast", excerpt=EXCERPT)
    assert EXCERPT in o._details_prompt() and o._conflicts_prompt() == ""
    assert o._spec_note() == ""                                                # nichts erzwungen, kein CONFIG-Zwang


def test_paper_details_kept_but_repo_details_replaced():
    det = [SetupDetailV2("dropout", "0.2", "dropout 0.2", "paper", source_rank="paper"),
           SetupDetailV2("lr", "0.0001", "learning_rate default=0.0001", "repo:run_longExp.py@abc1234", source_rank="argparse default")]
    text = format_code_grounded(_spec(), det)
    head = text.split("Authors' code for")[0]
    assert "dropout: 0.2" in head and "default=0.0001" not in head


def test_drop_defaults_only_for_flags_set_by_script():
    out, flags = drop_run_script_defaults(EXCERPT)
    assert flags == ["learning_rate", "seq_len"]
    assert "default=96" not in out and "default=0.0001" not in out
    assert "default=3" in out                                                  # --patience setzt das Skript nicht
    assert "--seq_len 336 --learning_rate 0.005" in out                        # Skript selbst bleibt
    assert drop_run_script_defaults("=== a.py ===\nparser.add_argument('--x', default=1)\n") == (
        "=== a.py ===\nparser.add_argument('--x', default=1)\n", [])


def test_drop_defaults_with_real_doc_names():
    ex = ("=== repo:scripts/etth1.sh@0c11366 ===\npython run.py --seq_len 336\n\n"
          "=== repo:run_longExp.py@0c11366 ===\nparser.add_argument('--seq_len', type=int, default=96)\n")
    out, flags = drop_run_script_defaults(ex)
    assert flags == ["seq_len"] and "default=96" not in out and "--seq_len 336" in out


def test_drop_defaults_multiline_and_string_default():
    ex = ("=== run.sh ===\npython main.py --model_name GAT --dropout 0.6\n\n=== main.py ===\n"
          "ap.add_argument('--model_name',\n    type=str, default='DLinear',\n    help='m')\nap.add_argument('--dropout', type=float, default=0.5)\n")
    out, flags = drop_run_script_defaults(ex)
    assert flags == ["dropout", "model_name"] and "default=" not in out and "help='m'" in out


def test_build_spec_numeric_only_prompt_has_no_text_keys_and_keeps_raw_excerpt():
    docs = [Doc("paper", "NLinear on ETTh1."),
            Doc("repo:scripts/etth1.sh@abc1234", "python run_longExp.py --model NLinear --seq_len 336 --learning_rate 0.005 --batch_size 32 --e 1 --f 2"),
            Doc("repo:run_longExp.py@abc1234", "parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')\n# NLinear etth1\nloss.backward()\n"),
            Doc("repo:models/NLinear.py@abc1234", "class Model:\n    def forward(self, x):\n        seq_last = x[:,-1:,:].detach()\n        return x - seq_last\n")]
    seen = {}

    def ask(system, user):
        seen["user"] = user
        return json.dumps([{"key": "lr", "value": "0.005", "quote": "--learning_rate 0.005", "source": "repo:scripts/etth1.sh@abc1234"},
                           {"key": "model_core", "value": "x", "quote": "return x - seq_last", "source": "repo:models/NLinear.py@abc1234"}])

    spec = build_spec(ask, [], docs, "NLinear erreicht MSE 0,374 auf ETTh1", "ETTh1", numeric_only=True)
    assert "- model_core:" not in seen["user"] and "- lr:" in seen["user"]
    assert spec.excerpt and "seq_last = x[:,-1:,:].detach()" in spec.excerpt
    assert "default=96" not in spec.excerpt and "seq_len" in spec.dropped_defaults
    # F1/F4: Startskript-Flags (--batch_size 32, --seq_len 336) werden zusaetzlich deterministisch gelesen
    assert [i.key for i in spec.items] == ["lr", "batch_size", "seq_len"] and any(d["reason"] == "unbekannter Schluessel" for d in spec.dropped)
    assert all(i.enforce for i in spec.items) and [i.origin for i in spec.items] == ["model", "raw", "raw"]
    flag_off = build_spec(ask, [], docs, "NLinear erreicht MSE 0,374 auf ETTh1", "ETTh1")
    assert flag_off.excerpt == "" and "model_core" in {i.key for i in flag_off.items}


def test_orchestrator_flag_implies_spec_mode_and_setup_v2(monkeypatch):
    import repro_agent.loop as loop_mod
    monkeypatch.setattr(loop_mod, "get_sandbox", lambda name: None)
    from test_fault_repair import settings
    monkeypatch.setattr(loop_mod, 'load_settings', lambda: settings(False))
    o = Orchestrator(code_grounded=True)
    assert o.code_grounded and o.spec_mode and o.setup_v2
    o = Orchestrator()
    assert not o.code_grounded and not o.spec_mode and not o.setup_v2
