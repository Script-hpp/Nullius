"""Einwand-Experimente (Rolle 3, Stufe 2): Klassifikation, feste Regeln, Grenzen, Ereignisse. Attrappen, kein Netz, kein Geld."""

import dataclasses
import inspect
import json
import math
import threading
from types import SimpleNamespace

import pytest

from repro_agent import objection_experiments as oe
from repro_agent.dossier import Attempt, Claim, Dossier, Verdict
from repro_agent.loop import Orchestrator
from repro_agent.report import render_markdown
from repro_agent.tasks import TASKS, cora_task, etth1_task

CLAIM = Claim(id="c", text="DLinear MSE 0.375", metric="mse", target=0.375, tolerance=0.01, source="t", origin="paper")
MAIN_CODE = "import numpy\nWINDOW = 96\n"
MAIN_HASH = oe._hash(MAIN_CODE)

Q_WINDOW = "we used an input window of 336 instead of 96 and got a much better result"
Q_SEED = "the result changes a lot with the random seed, we saw large run-to-run variation"
Q_NUMBER = "i run etth1 and find mse = 0.426, not in paper 0.386"
Q_UNFAIR = "the comparison is unfair because the baselines were not tuned"
Q_SPLIT = "with a different train/test split of 70/30 the number drops"


def finding(i, quote):
    return {"topic": f"t{i}", "statement": f"s{i}: {quote}", "quote": quote, "url": f"https://github.com/a/b/issues/{i}", "kind": "other"}


class FakeLLM:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def complete(self, role, messages, **kw):
        user = messages[-1]["content"]
        self.calls.append(role)
        for quote, ans in self.answers.items():
            if f"QUOTE: {quote}" in user:
                return SimpleNamespace(content="thinking...\n" + json.dumps(ans))
        return SimpleNamespace(content="no idea")


WINDOW = {"type": "parameter_change", "reason": "r", "param": "input_window", "to_value": "336"}
SEEDV = {"type": "seed_variance", "reason": "r"}
NUMBER = {"type": "reported_number", "reason": "r", "reported_value": 0.426, "reported_metric": "mse", "same_setup": "unknown"}
NOT = {"type": "not_measurable", "reason": "fairness critique"}
SPLIT = {"type": "parameter_change", "reason": "r", "param": "data split", "to_value": "70/30"}


class FakeOrch:
    """Minimaler Orchestrator: Sandbox-Messungen aus einer Funktion (Code, Seed) -> Wert oder None."""

    def __init__(self, llm, measure, main_values=None, budget=10.0, verdict_mean=0.376, task=None):
        self.llm, self.measure = llm, measure
        self.events, self.warnings, self.branches = [], [], 0
        self.settings = SimpleNamespace(max_cost_per_run_usd=budget, max_attempts_per_claim=3)
        self.spent_usd = self.spent_gross_usd = 0.0
        self._lock = threading.Lock()
        self.code_requests, self.repairs = [], []
        self.new_code = "import numpy\nWINDOW = 336\n"
        task = task or etth1_task()
        self._claim_ctx = {"c": {"code": MAIN_CODE, "plan": "plan", "task": task, "snapshot": "snap", "n_seeds": 5}}
        main = main_values or {0: 0.376, 1: 0.378, 2: 0.374, 3: 0.377, 4: 0.375}
        self.dossier = Dossier(paper_id="p", claims=[CLAIM], verdicts=[Verdict("c", "reproduziert", verdict_mean, 0.002, "x")],
                               community={"findings": []})
        self.dossier.attempts = [Attempt("c", f"b{s}", "snap", MAIN_HASH, s, "stabil", v, None) for s, v in main.items()]

    def _emit(self, type, **kw):
        self.events.append((type, kw))

    def _warn(self, text):
        self.warnings.append(text)

    def _attempt(self, snapshot, code, seed, task, metric, **kw):
        with self._lock:
            self.branches += 1
            name = f"ob{self.branches}"
        v = self.measure(code, seed)
        return name, ({"status": "ok", "measured": v} if v is not None else {"status": "fehler", "detail": "boom"})

    def _details_prompt(self):
        return "details"

    def _product_notes(self, task):
        return ""

    def _code_from_llm(self, system, user, temperature=0.2):
        self.code_requests.append(user)
        return self.new_code

    def repair_code(self, plan, task, code, error):
        self.repairs.append(error)
        return self.new_code

    def of(self, name):
        return [kw for t, kw in self.events if t == name]


def run(orch, quotes_answers, quick=False):
    findings = [finding(i, q) for i, (q, _) in enumerate(quotes_answers, 1)]
    orch.dossier.community = {"findings": findings}
    orch.llm.answers = {q: a for q, a in quotes_answers}
    oe.run_round(orch, orch.dossier, {"title": "T", "dataset": "ETTh1"}, quick=quick)
    return {c["id"]: c for c in orch.dossier.objection_checks}


def fresh(measure, **kw):
    return FakeOrch(FakeLLM({}), measure, **kw)


def seeded(code, seed):
    return {336: 0.37}.get(0) if False else (0.376 + 0.001 * seed if code == MAIN_CODE else 0.30 + 0.001 * seed)


# ---- Klassifikation: feste Pruefung nach dem LLM --------------------------------------------------------------------------

def test_the_four_types_with_valid_claims():
    o = fresh(seeded)
    r = run(o, [(Q_WINDOW, WINDOW), (Q_SEED, SEEDV), (Q_NUMBER, NUMBER), (Q_UNFAIR, NOT)])
    assert [r[k]["type"] for k in ("obj1", "obj2", "obj3", "obj4")] == ["parameter_change", "seed_variance", "reported_number", "not_measurable"]
    assert r["obj4"]["status"] == "nicht pruefbar"
    assert all(r[k]["quote"] and r[k]["url"] for k in r)   # woertliches Zitat und URL bleiben


@pytest.mark.parametrize("quote,ans,why", [
    (Q_WINDOW, {**WINDOW, "to_value": "512"}, "value is not written in the quote"),
    (Q_WINDOW, {**WINDOW, "param": "dropout_of_the_moon"}, "not changeable in this task"),
    (Q_SPLIT, SPLIT, "data and splits"),
    (Q_UNFAIR, SEEDV, "no statement about seeds"),
    (Q_NUMBER, {**NUMBER, "reported_value": 0.5}, "reported number is not written"),
    (Q_NUMBER, {**NUMBER, "reported_metric": "accuracy"}, "not the claim metric"),
    (Q_NUMBER, {"type": "magic"}, "unknown type"),
])
def test_llm_claims_not_covered_by_quote_or_task_are_demoted(quote, ans, why):
    r = run(fresh(seeded), [(quote, ans)])["obj1"]
    assert r["type"] == "not_measurable" and r["status"] == "nicht pruefbar" and why in r["note"]


def test_unparseable_answer_is_not_measurable():
    o = fresh(seeded)
    o.dossier.community = {"findings": [finding(1, Q_WINDOW)]}
    oe.run_round(o, o.dossier, {"title": "T", "dataset": "ETTh1"})
    assert o.dossier.objection_checks[0]["type"] == "not_measurable" and "missing" in o.dossier.objection_checks[0]["note"]


def test_tasks_declare_what_can_change_and_never_data():
    for name in ("Cora", "ETTh1", "Cora-LCC"):
        keys = [k for k, _ in TASKS[name]().tunable]
        assert keys and not any("split" in k or "label" in k or "test" in k for k in keys)
    assert "input_window" in dict(etth1_task().tunable) and "dropout" in dict(cora_task().tunable)


# ---- Regeln an den Grenzen ----------------------------------------------------------------------------------------------

TOL = 0.01


def test_seed_variance_rule_boundaries():
    main = [0.375, 0.376, 0.374, 0.375, 0.375]
    s, m, _ = oe.rule_seed_variance(main, [0.375, 0.376, 0.374], TOL, 3)
    assert s == "widerlegt" and m["std_pool"] < TOL
    assert oe.rule_seed_variance(main, [0.375, 0.376, 0.42], TOL, 3)[0] == "bestaetigt"      # Streuung des Pools >= Toleranz
    assert oe.rule_seed_variance(main, [0.39, 0.391, 0.389], TOL, 3)[0] == "bestaetigt"      # Mittel der neuen Seeds verschoben
    assert oe.rule_seed_variance(main, [0.385, 0.385], TOL, 5)[0] == "bestaetigt"            # |Verschiebung| = 0.010..: > tol
    assert oe.rule_seed_variance(main, [0.375], TOL, 3)[0] == "offen"                        # weniger als 2 gueltige Seeds
    assert oe.rule_seed_variance(main, [0.375, 0.375], TOL, 5)[0] == "offen"                 # unter ceil(K/2) = 3
    assert oe.rule_seed_variance([], [0.375, 0.375, 0.375], TOL, 3)[0] == "offen"


def test_parameter_change_rule_boundaries():
    main = {0: 0.375, 1: 0.376, 2: 0.374}
    assert oe.rule_parameter_change({0: 0.380, 1: 0.381, 2: 0.379}, main, TOL)[0] == "widerlegt"       # |delta| 0.005 <= tol
    assert oe.rule_parameter_change({0: 0.405, 1: 0.406, 2: 0.404}, main, TOL)[0] == "bestaetigt"      # 0.03 > max(tol, 2 SE)
    s, m, note = oe.rule_parameter_change({0: 0.30, 1: 0.45, 2: 0.43}, main, TOL)                       # |delta| > tol, aber im Rauschen
    assert abs(m["delta"]) > TOL and s == "offen" and "2*SE" in note
    assert oe.rule_parameter_change({0: 0.2}, main, TOL)[0] == "offen"
    assert oe.rule_parameter_change({5: 0.2, 6: 0.3}, main, TOL)[0] == "offen"                           # keine gepaarten Seeds
    s, m, _ = oe.rule_parameter_change({1: 0.405, 2: 0.404}, main, TOL)                                  # nur gleiche Seeds werden gepaart
    assert m["seeds"] == [1, 2] and s == "bestaetigt"


def test_reported_number_rule():
    t, r = 0.375, 0.426
    assert oe.rule_reported(r, 0.427, t, TOL, "yes")[0] == "bestaetigt" and oe.rule_reported(r, 0.427, t, TOL, "yes")[1]["supports"] == "reported"
    assert oe.rule_reported(r, 0.376, t, TOL, "yes")[0] == "widerlegt"
    assert oe.rule_reported(r, 0.376, t, TOL, "unknown")[0] == "offen"                                   # Setup nicht belegt: nie bestaetigt/widerlegt
    assert oe.rule_reported(r, 0.376, t, TOL, "no")[0] == "offen"
    assert oe.rule_reported(r, 0.40, t, TOL, "yes")[1]["supports"] == "neither"
    assert oe.rule_reported(0.380, 0.378, t, TOL, "yes") [1]["supports"] == "both" and oe.rule_reported(0.380, 0.378, t, TOL, "yes")[0] == "offen"
    assert oe.rule_reported(r, float("nan"), t, TOL, "yes")[0] == "offen"


# ---- Lauf: Branches, Messung, Ereignisse ---------------------------------------------------------------------------------

def test_reported_number_needs_no_branch_and_issue_117_is_offen():
    o = fresh(seeded)
    r = run(o, [(Q_NUMBER, NUMBER)])["obj1"]
    assert r["status"] == "offen" and r["measured"]["supports"] == "paper" and o.branches == 0 and "does not state the same setup" in r["note"]
    r = run(fresh(seeded), [(Q_NUMBER, {**NUMBER, "same_setup": "yes"})])["obj1"]
    assert r["status"] == "widerlegt" and r["measured"]["mean"] == pytest.approx(0.376)


def test_seed_variance_runs_extra_seeds_with_the_main_code():
    seen = []
    o = fresh(lambda code, seed: seen.append((code, seed)) or 0.375 + 0.0001 * seed)
    r = run(o, [(Q_SEED, SEEDV)])["obj1"]
    assert sorted(s for _, s in seen) == [5, 6, 7, 8, 9] and {c for c, _ in seen} == {MAIN_CODE}
    assert r["status"] == "widerlegt" and r["main_code_hash"] == MAIN_HASH and len(r["attempts"]) == 5
    assert all(a["branch_id"].startswith("ob") and a["parent_snapshot"] == "snap" for a in r["attempts"])
    assert r["measured"]["n_extra"] == 5 and r["measured"]["n_main"] == 5


def test_seed_variance_confirmed_when_new_seeds_scatter():
    o = fresh(lambda code, seed: 0.375 + (0.05 if seed % 2 else -0.05))
    assert run(o, [(Q_SEED, SEEDV)])["obj1"]["status"] == "bestaetigt"


def test_parameter_change_new_code_paired_seeds_and_target_note():
    o = fresh(seeded)
    r = run(o, [(Q_WINDOW, WINDOW)])["obj1"]
    assert "set input_window to 336" in o.code_requests[0] and MAIN_CODE in o.code_requests[0] and Q_WINDOW in o.code_requests[0]
    assert r["code_hash"] == oe._hash(o.new_code) != r["main_code_hash"]
    assert sorted(a["seed"] for a in r["attempts"]) == [0, 1, 2]
    assert r["status"] == "bestaetigt" and r["measured"]["delta"] < -0.07 and r["measured"]["within_target_tolerance"] is False


def test_parameter_change_widerlegt_when_effect_inside_tolerance():
    o = fresh(lambda code, seed: 0.376 + 0.001 * seed + (0.001 if code != MAIN_CODE else 0))
    assert run(o, [(Q_WINDOW, WINDOW)])["obj1"]["status"] == "widerlegt"


def test_parameter_change_repairs_seed0_and_goes_offen_without_valid_seeds():
    calls = {"n": 0}

    def flaky(code, seed):
        if code == MAIN_CODE:
            return 0.376
        calls["n"] += 1
        return None if calls["n"] < 2 else 0.38 + 0.001 * seed

    o = fresh(flaky)
    r = run(o, [(Q_WINDOW, WINDOW)])["obj1"]
    assert len(o.repairs) == 1 and r["status"] in ("widerlegt", "bestaetigt")
    o = fresh(lambda code, seed: None)
    r = run(o, [(Q_WINDOW, WINDOW)])["obj1"]
    assert r["status"] == "offen" and len(o.repairs) == 2 and "fewer than 2" in r["note"]


def test_new_code_must_contain_the_named_value():
    o = fresh(seeded)
    o.new_code = "import numpy\nWINDOW = 96\n"
    r = run(o, [(Q_WINDOW, WINDOW)])["obj1"]
    assert r["status"] == "offen" and "does not contain the value 336" in r["note"] and o.branches == 0


def test_limits_priority_and_duplicates():
    q2 = "we used an input window of 336 here too, and it was far better"
    q3 = "we used an input window of 720 with a different result entirely"
    o = fresh(seeded)
    r = run(o, [(Q_SEED, SEEDV), (Q_WINDOW, WINDOW), (q2, WINDOW), (q3, {**WINDOW, "to_value": "720"})])
    assert r["obj2"]["status"] == "bestaetigt"                                           # parameter_change zuerst
    assert r["obj3"]["note"] == "same experiment as obj2" and r["obj3"]["status"] == r["obj2"]["status"] and r["obj3"]["attempts"] == []
    assert r["obj4"]["status"] == "bestaetigt" and r["obj1"]["status"] == "offen" and "limit of 3" in r["obj1"]["note"] or r["obj1"]["status"] != "offen"
    started = [kw["id"] for kw in o.of("objection_experiment") if kw["stage"] == "start"]
    assert len(started) <= oe.MAX_EXPERIMENTS and started[0] == "obj2"


def test_quick_mode_two_experiments_and_three_extra_seeds():
    o = fresh(seeded)
    q3 = "we used an input window of 720 with a different result entirely"
    r = run(o, [(Q_SEED, SEEDV), (Q_WINDOW, WINDOW), (q3, {**WINDOW, "to_value": "720"})], quick=True)
    assert sum(1 for kw in o.of("objection_experiment") if kw["stage"] == "start") == 2
    assert r["obj1"]["status"] == "offen" and "limit of 2" in r["obj1"]["note"]
    o = fresh(lambda code, seed: 0.375)
    r = run(o, [(Q_SEED, SEEDV)], quick=True)["obj1"]
    assert sorted(a["seed"] for a in r["attempts"]) == [5, 6, 7] and o.of("objection_round_end")[0]["quick"] is True


def test_run_budget_stops_experiments():
    o = fresh(seeded, budget=0.0)
    r = run(o, [(Q_WINDOW, WINDOW)])["obj1"]
    assert r["status"] == "offen" and r["note"] == "run budget reached" and o.branches == 0


def test_experiment_failure_is_a_warning_not_a_crash():
    def boom(code, seed):
        raise RuntimeError("sandbox down")

    o = fresh(boom)
    r = run(o, [(Q_SEED, SEEDV)])["obj1"]
    assert r["status"] == "offen" and "experiment failed" in r["note"] and o.warnings


def test_events_in_order_with_added_time():
    o = fresh(seeded)
    run(o, [(Q_SEED, SEEDV), (Q_NUMBER, NUMBER)])
    names = [t for t, _ in o.events]
    assert names[:2] == ["objection_classified"] * 2
    assert names.index("objection_experiment") > names.index("objection_classified")
    assert names.index("objection_checked") > max(i for i, n in enumerate(names) if n == "objection_experiment")
    assert names[-1] == "objection_round_end"
    end = o.of("objection_round_end")[0]
    assert end["n_experiments"] == 1 and end["seconds"] >= 0 and end["n_findings"] == 2
    checked = {kw["id"]: kw for kw in o.of("objection_checked")}
    assert checked["obj1"]["status"] == "widerlegt" and checked["obj1"]["quote"] == Q_SEED and checked["obj1"]["url"]
    assert all(json.dumps(kw) for _, kw in o.events)   # JSON-faehig


def test_verdicts_and_attempts_untouched():
    o = fresh(seeded)
    before = (dataclasses.asdict(o.dossier.verdicts[0]), [dataclasses.asdict(a) for a in o.dossier.attempts])
    run(o, [(Q_SEED, SEEDV), (Q_WINDOW, WINDOW)])
    assert (dataclasses.asdict(o.dossier.verdicts[0]), [dataclasses.asdict(a) for a in o.dossier.attempts]) == before


def test_no_valid_main_claim_skips_the_round():
    o = fresh(seeded)
    o.dossier.verdicts = [Verdict("c", "nicht reproduzierbar (Setup)", float("nan"), float("nan"), "x")]
    run(o, [(Q_SEED, SEEDV)])
    assert o.dossier.objection_checks == [] and o.of("objection_round_end")[0]["skipped"]


# ---- Einbindung in den Orchestrator --------------------------------------------------------------------------------------

def bare(flag=True, tavily=True, mode="live", switch=True):
    o = Orchestrator.__new__(Orchestrator)
    o.graph_repair, o.use_tavily, o.tavily_mode, o.objection_experiments = flag, tavily, mode, switch
    o.events_seen, o.warnings = [], []
    o._warn = lambda text: o.warnings.append(text)
    return o


@pytest.mark.parametrize("kw", [{"flag": False}, {"tavily": False}, {"mode": "fixture"}, {"switch": False}])
def test_round_runs_only_in_the_live_product_path_with_the_switch(kw, monkeypatch):
    called = []
    monkeypatch.setattr(oe, "run_round", lambda *a, **k: called.append(1))
    bare(**kw)._objection_round({}, Dossier(paper_id="p"))
    assert called == []
    bare()._objection_round({}, Dossier(paper_id="p"))
    assert called == [1]


def test_round_failure_warns_and_billing_passes(monkeypatch):
    from repro_agent.llm import BillingBlocked
    monkeypatch.setattr(oe, "run_round", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    o = bare()
    o._objection_round({}, Dossier(paper_id="p"))
    assert "Einwand-Experimente fehlgeschlagen" in o.warnings[0]
    monkeypatch.setattr(oe, "run_round", lambda *a, **k: (_ for _ in ()).throw(BillingBlocked("b")))
    with pytest.raises(BillingBlocked):
        bare()._objection_round({}, Dossier(paper_id="p"))


def test_switch_defaults_off_and_run_order():
    sig = inspect.signature(Orchestrator.__init__)
    assert sig.parameters["objection_experiments"].default is False and sig.parameters["objection_quick"].default is False
    src = inspect.getsource(Orchestrator.run)
    assert src.index("self._community_round(") < src.index("self._objection_round(") < src.index("dossier.save(")
    for name in ("plan", "write_code", "repair_code", "_run_seed", "_verdict"):
        assert "objection" not in inspect.getsource(getattr(Orchestrator, name))


def test_api_live_path_turns_it_on_in_quick_mode(monkeypatch):
    from repro_agent import api
    seen = {}
    monkeypatch.setattr("repro_agent.loop.Orchestrator", lambda **kw: seen.update(kw) or SimpleNamespace())
    api.make_orchestrator(True)
    assert seen["objection_experiments"] is True and seen["objection_quick"] is True
    api.make_orchestrator(False)
    assert seen["objection_quick"] is False


# ---- Dossier und Zeugnis -------------------------------------------------------------------------------------------------

def test_dossier_roundtrip_and_old_dossiers(tmp_path, monkeypatch):
    monkeypatch.setattr("repro_agent.dossier.REPO_ROOT", tmp_path)
    o = fresh(seeded)
    run(o, [(Q_SEED, SEEDV)])
    path = o.dossier.save("t1")
    assert Dossier.load(path).objection_checks == o.dossier.objection_checks
    data = json.loads(path.read_text())
    data.pop("objection_checks")
    path.write_text(json.dumps(data))
    assert Dossier.load(path).objection_checks == []


def test_zeugnis_section_quote_url_measurement_status():
    o = fresh(seeded)
    run(o, [(Q_WINDOW, WINDOW), (Q_NUMBER, NUMBER), (Q_SPLIT, SPLIT)])
    text = render_markdown(o.dossier)
    assert "## Objections checked" in text
    assert f'"{Q_WINDOW}" (https://github.com/a/b/issues/1) -> input_window = 336: mean' in text and "-> bestaetigt" in text
    assert "reported number 0.426; our measurement 0.3760 supports the paper's value" in text and "-> offen" in text
    assert "not changeable in this task (the task fixes data and splits)" in text and "-> nicht pruefbar" in text
    assert "## Objections checked" not in render_markdown(Dossier(paper_id="p"))


# ---- protocol_change (drop_last): Neubewertung derselben Vorhersagen, kein Lauf --------------------------------------------

Q_DROP = "the test loader uses drop_last=True, so the last incomplete batch of the test set is dropped (batch_size 32 in the script)"
DROP = {"type": "protocol_change", "reason": "r"}


def droplast_metrics(shift):
    """metrics_droplast eines Seeds: je Batchgroesse verschiebt sich die MSE um `shift` (Fenster wie bei 2785 Testfenstern)."""
    return {str(b): {"mse": 0.376 + shift * (b / 256), "mae": 0.4, "windows": 2785 - d, "dropped": d} for b, d in ((32, 1), (64, 33), (128, 97), (256, 225))}


def with_droplast(o, shift, seeds=(0, 1, 2, 3, 4)):
    for a in o.dossier.attempts:
        a.metrics_droplast = droplast_metrics(shift) if a.seed in seeds else None
    return o


def test_protocol_change_is_validated_against_quote_task_and_metric():
    r = run(with_droplast(fresh(seeded), 0.0), [(Q_DROP, DROP)])["obj1"]
    assert r["type"] == "protocol_change" and r["param"] == "drop_last" and r["to_value"] == "32"
    r = run(fresh(seeded), [(Q_UNFAIR, DROP)])["obj1"]
    assert r["type"] == "not_measurable" and "no statement about a dropped last test batch" in r["note"]
    cora = fresh(seeded, task=cora_task())
    r = run(cora, [(Q_DROP, DROP)])["obj1"]
    assert r["type"] == "not_measurable" and "no protocol re-evaluation" in r["note"]
    r = oe.validate(DROP, finding(1, Q_DROP), "accuracy", {}, ("drop_last",))
    assert r["type"] == "not_measurable" and "mse and mae" in r["note"]


def test_protocol_change_needs_no_branch_no_code_and_keeps_main_untouched():
    o = with_droplast(fresh(seeded), 0.02)
    before = [dataclasses.asdict(a) for a in o.dossier.attempts]
    r = run(o, [(Q_DROP, {**DROP})])["obj1"]
    assert o.branches == 0 and o.code_requests == [] and r["attempts"] == []
    assert r["status"] == "bestaetigt" and r["measured"]["seeds"] == [0, 1, 2, 3, 4] and r["measured"]["main_branches"] == ["b0", "b1", "b2", "b3", "b4"]
    assert r["measured"]["sizes"]["32"]["delta"] == pytest.approx(0.02 * 32 / 256) and r["measured"]["restricted_to"] == "32"
    assert [dataclasses.asdict(a) for a in o.dossier.attempts] == before and o.dossier.verdicts[0].urteil == "reproduziert"
    assert o.of("objection_round_end")[0]["n_experiments"] == 0


def test_protocol_change_rule_boundaries():
    seeds = lambda shift: [(0.376, droplast_metrics(shift)) for _ in range(3)]
    thr = TOL * oe.PROTOCOL_EFFECT                      # 0.0025
    st, m, _ = oe.rule_protocol_droplast(seeds(thr), "mse", TOL, 0.375)       # b=256: shift = thr exactly -> effect >= thr
    assert st == "bestaetigt" and m["max_effect"] == pytest.approx(thr)
    st, m, _ = oe.rule_protocol_droplast(seeds(thr * 0.99), "mse", TOL, 0.375)
    assert st == "widerlegt"
    st, _, _ = oe.rule_protocol_droplast(seeds(thr), "mse", TOL, 0.375, batch="32")   # nur b=32: Effekt thr/8 -> widerlegt
    assert st == "widerlegt"
    assert oe.rule_protocol_droplast(seeds(0.1)[:1], "mse", TOL, 0.375)[0] == "offen"
    assert oe.rule_protocol_droplast(seeds(0.1), "mse", TOL, 0.375, batch="48")[0] == "offen"
    assert oe.rule_protocol_droplast([], "mse", TOL, 0.375)[0] == "offen"


def test_protocol_change_offen_without_stored_droplast_metrics():
    r = run(fresh(seeded), [(Q_DROP, DROP)])["obj1"]
    assert r["status"] == "offen" and "fewer than 2 valid seeds" in r["note"]


def test_protocol_change_not_counted_against_experiment_limit_and_zeugnis_line():
    o = with_droplast(fresh(seeded), 0.02)
    r = run(o, [(Q_WINDOW, WINDOW), (Q_SEED, SEEDV), (Q_DROP, DROP), (Q_NUMBER, NUMBER)])
    assert r["obj3"]["status"] == "bestaetigt" and r["obj3"]["type"] == "protocol_change"
    assert o.of("objection_round_end")[0]["n_experiments"] == 2
    from repro_agent.report import describe_check
    assert "drop_last=True loader keeps them" in describe_check(r["obj3"])


def test_etth1_declares_drop_last_and_others_do_not():
    assert etth1_task().protocols == ("drop_last",) and cora_task().protocols == ()
