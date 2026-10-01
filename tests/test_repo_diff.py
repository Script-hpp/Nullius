"""A2: Abgleich mit Repo-Dateien, Zitatpruefung, Uebernahmeregeln. Attrappen, kein Netz, kein Geld."""

import json
from types import SimpleNamespace

import pytest

from repro_agent.dossier import Claim, Dossier
from repro_agent.repo_diff import Deviation, build_prompt, repair_message, truncate_files, verify_deviations
from repro_agent.tasks import cora_task
from test_candidates import C0, C1, ValSandbox
from test_loop_fakes import make_orch

REPO = {"args.py": "parser.add_argument('--dropout', default=0.5)\n", "model.py": "x = F.dropout(x, 0.5)\n"}
DEV = {"aspect": "dropout", "file": "args.py", "repo_quote": "parser.add_argument('--dropout', default=0.5)",
       "our_quote": "print('c0')", "difference": "no dropout"}


def test_truncate_is_deterministic_and_reports_cuts():
    files = {"a": "x" * 10, "b": "y" * 10, "c": "z" * 10}
    shown, cuts = truncate_files(files, 15)
    assert list(shown) == ["a", "b"] and shown["b"] == "y" * 5
    assert cuts == [{"file": "b", "kept_chars": 5, "of_chars": 10}, {"file": "c", "kept_chars": 0, "of_chars": 10}]
    assert truncate_files(files, 100) == (files, [])


def test_verify_drops_unverified_quotes():
    bad_file = {**DEV, "file": "nope.py"}
    bad_repo = {**DEV, "repo_quote": "dropout = 0.9"}
    bad_ours = {**DEV, "our_quote": "never in our script"}
    missing = {k: v for k, v in DEV.items() if k != "difference"}
    kept, dropped = verify_deviations([DEV, DEV, bad_file, bad_repo, bad_ours, missing, "x"], C0, REPO)
    assert kept == [Deviation(**DEV)]                                        # Duplikat entfernt
    assert [d["reason"] for d in dropped] == [
        "Datei nicht unter den gezeigten Repo-Dateien", "Repo-Zitat nicht in der Datei", "Zitat nicht in unserem Code",
        "Feld fehlt", "kein Objekt"]
    other_file = {**DEV, "file": "model.py"}                                 # Zitat steht in einer anderen Datei
    assert verify_deviations([other_file], C0, REPO)[0] == []


def make(val_by_code, repairs, repo=REPO, flag=True, ask_content=None, test_by_code=None, same_ablation=False):
    class SB(ValSandbox):
        def run(self, branch, cmd, timeout_s):
            code = self.code_of.get(branch)
            if same_ablation and "cora_eval.py" in cmd and branch in self.ablated and code == C1:   # nur C1 ignoriert den Graph
                from repro_agent.sandbox.base import RunResult
                self.cmds.append((branch, cmd))
                return RunResult(0, json.dumps({"status": "ok", "metrics": {"accuracy": self.test_by_code[C1]},
                                                "baselines": {"majority": {"accuracy": 0.13}, "no_graph": {"accuracy": 0.2}}}), "", False)
            return super().run(branch, cmd, timeout_s)

    sb = SB(test_by_code or {C0: 0.80, C1: 0.84}, val_by_code)
    o, calls = make_orch(sb, seeds=3, repairs=repairs, write=C0)
    o.graph_repair, o.n_candidates, o._repo_files = flag, 1, dict(repo)
    asks = []

    def ask(role, system, user, **kw):
        asks.append((role, user, kw))
        return SimpleNamespace(content=json.dumps([DEV]) if ask_content is None else ask_content)

    o._ask = ask
    events = []
    o._emit = lambda type, **kw: events.append((type, kw))
    claim = Claim("c", "GCN 81,5 %", "accuracy", 0.815, 0.01, "T", "paper")
    d = Dossier(paper_id="p", claims=[claim])
    return o, calls, asks, events, claim, d, sb


def run(o, claim, d, tmp_path):
    o.run_claim({"seeds": 3, "dataset": "Cora"}, claim, cora_task(), "snap", d, tmp_path)


def test_adopted_when_val_equal_or_better_and_ablation_passes(tmp_path):
    o, calls, asks, events, claim, d, sb = make({C0: 0.7312, C1: 0.7312}, [C1])   # Val gleich = uebernehmen
    run(o, claim, d, tmp_path)
    (ev,) = [kw for t, kw in events if t == "repo_diff"]
    assert ev["result"] == "adopted" and ev["truncation"] == [] and ev["deviations"][0]["file"] == "args.py"
    assert len(asks) == 1 and asks[0][0] == "extract" and asks[0][2]["temperature"] == 0.0
    assert calls["repair"] == [repair_message([Deviation(**DEV)])]
    assert {a.code_hash for a in d.attempts if a.seed > 0} == {d.attempts[-1].code_hash} and d.verdicts[0].measured_mean == pytest.approx(0.84)
    assert d.repo_diff["c"]["result"] == "adopted" and "ersetzt" in d.verdicts[0].explanation
    assert sum(1 for t, kw in events if t == "ablation" and kw.get("runde") == "repo_diff") == 1


@pytest.mark.parametrize("val_new,reason", [(0.60, "Validierung schlechter"), (None, "Validierungswert fehlt")])
def test_kept_when_val_worse_or_missing(tmp_path, val_new, reason):
    vals = {C0: 0.70} | ({C1: val_new} if val_new is not None else {})
    o, calls, asks, events, claim, d, sb = make(vals, [C1])
    run(o, claim, d, tmp_path)
    (ev,) = [kw for t, kw in events if t == "repo_diff"]
    assert ev["result"] == "kept" and reason in ev["reason"]
    assert d.verdicts[0].measured_mean == pytest.approx(0.80)               # alter Code, Messungen des neuen zaehlen nicht
    assert {a.code_hash for a in d.attempts if a.seed > 0} == {d.attempts[0].code_hash}


def test_kept_when_new_code_ignores_graph(tmp_path):
    o, calls, asks, events, claim, d, sb = make({C0: 0.70, C1: 0.80}, [C1], same_ablation=True)
    run(o, claim, d, tmp_path)
    (ev,) = [kw for t, kw in events if t == "repo_diff"]
    assert ev["result"] == "kept" and "Graph" in ev["reason"]


def test_kept_when_new_code_fails_fault_check(tmp_path):
    o, calls, asks, events, claim, d, sb = make({C0: 0.70, C1: 0.10}, [C1], test_by_code={C0: 0.80, C1: 0.84})
    run(o, claim, d, tmp_path)
    (ev,) = [kw for t, kw in events if t == "repo_diff"]
    assert ev["result"] == "kept"                                            # Val unter Mehrheitsklasse (0.3)


def test_prompts_carry_no_val_test_or_target_numbers(tmp_path):
    o, calls, asks, events, claim, d, sb = make({C0: 0.7312, C1: 0.7312}, [C1])
    run(o, claim, d, tmp_path)
    for text in [asks[0][1], calls["repair"][0]]:
        for needle in ("0.7312", "7312", "0.815", "81.5", "81,5", "0.84", "0.80"):
            assert needle not in text
    assert "GCN" not in asks[0][1]                                           # kein Claim-Text


def test_no_call_without_repo_files_or_without_flag(tmp_path):
    for kw in ({"repo": {}}, {"flag": False}):                               # Arm A (keine Repo-Dateien) / alter Pfad
        o, calls, asks, events, claim, d, sb = make({C0: 0.7}, [], **kw)
        if not kw.get("flag", True):
            o.write_code = lambda plan, task: C0
        run(o, claim, d, tmp_path)
        assert asks == [] and not [t for t, _ in events if t == "repo_diff"] and d.repo_diff == {}


def test_no_deviation_means_no_repair(tmp_path):
    o, calls, asks, events, claim, d, sb = make({C0: 0.7}, [], ask_content="[]")
    run(o, claim, d, tmp_path)
    assert len(asks) == 1 and calls["repair"] == [] and d.repo_diff["c"]["result"] == "no_deviation"


def test_prompt_lists_aspects_and_files():
    p = build_prompt("code", REPO, "Cora")
    assert "=== FILE args.py ===" in p and "stopping criterion" in p and "SCRIPT:\ncode" in p


def test_failed_llm_call_keeps_code_and_finishes_run(tmp_path):
    """v4-Abbruch: Super liefert leere Antwort (Denken zehrt Token-Limit auf), Client wirft RuntimeError nach 5 Versuchen.
    Der optionale Abgleich darf den Lauf nicht beenden; Urteil mit altem Code, Ereignis result=error."""
    o, calls, asks, events, claim, d, sb = make({C0: 0.7}, [C1])

    def failing(role, system, user, **kw):
        asks.append((role, user, kw))
        raise RuntimeError("LLM-Aufruf fehlgeschlagen nach 5 Versuchen: Antwort ohne Inhalt")

    o._ask = failing
    run(o, claim, d, tmp_path)
    assert calls["repair"] == [] and d.repo_diff["c"]["result"] == "error" and len(d.verdicts) == 1
    assert d.verdicts[0].measured_mean == pytest.approx(0.80) and asks[0][2]["max_tokens"] == 16000


def test_cost_limit_still_propagates(tmp_path):
    from repro_agent.loop import CostLimitExceeded
    o, calls, asks, events, claim, d, sb = make({C0: 0.7}, [C1])

    def over(role, system, user, **kw):
        raise CostLimitExceeded("limit")

    o._ask = over
    with pytest.raises(CostLimitExceeded):
        run(o, claim, d, tmp_path)
