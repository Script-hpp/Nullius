"""Katalog-Pipeline: Adapter, Import, Status, Loop mit Attrappen, Kuratierungsskript. Kein Netz, kein Geld."""

import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from repro_agent import catalog as cat
from repro_agent import loop as loop_mod
from repro_agent.claims_source import ExtractionInvalid, TargetVerificationBlocked, PaperText, TargetEvidence
from repro_agent.dossier import Claim, Dossier, Verdict
from repro_agent.events import EventLog
from repro_agent.llm import BillingBlocked
from repro_agent.report import render_markdown
from test_loop_fakes import FakeSandbox, make_orch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("curate_catalog", ROOT / "scripts" / "curate_catalog.py")
curate_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curate_mod)


def cand(arxiv_id="2511.08094", dataset="Cora", **over):
    base = {
        "type": "candidate", "arxiv_id": arxiv_id, "title": f"Paper {arxiv_id}", "dataset": dataset,
        "claim": {"method": "SL-GCN", "quote": "SL-GCN 82.92 ± 1.39", "metric": "accuracy", "target": 82.92,
                  "protocol": "passt", "matched_task": "Cora-LCC", "column_check": "ok"},
        "protocol": "passt", "github_url": "https://github.com/x/y", "source_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "found_by": "tavily", "reason": "Code found",
    }
    base.update(over)
    return base


def etth1(**claim_over):
    claim = {"method": "MoHETS", "quote": "| ETTh1 | 96 | 0.350 | 0.383 |", "metric": "mse", "target": 0.35,
             "horizon": 96, "protocol": "unklar", "matched_task": None, "column_check": "unchecked"}
    claim.update(claim_over)
    return cand("2601.21866", "ETTh1", claim=claim, protocol="unklar", github_url=None)


def entry(c, source="test"):
    return cat.entry_from_candidate(c, source)


# ---- Adapter -------------------------------------------------------------------
def test_adapter_converts_shchur_candidate():
    paper, claims = cat.build_paper(entry(cand()))
    c = claims[0]
    assert paper["id"] == "disc_2511_08094" and paper["dataset"] == "Cora-LCC"          # matched_task, nicht Discovery-Datensatz
    assert paper["arxiv_url"] == "https://arxiv.org/abs/2511.08094"
    assert c.target == pytest.approx(0.8292) and c.tolerance == 0.01 and c.metric == "accuracy"
    assert c.source == "discovery: https://arxiv.org/pdf/2511.08094, Zitat: SL-GCN 82.92 ± 1.39"
    assert c.origin == "paper" and paper["seeds"] == cat.TASK_SEEDS["Cora-LCC"]
    assert paper["claims"][0]["extract"] == {"method": "SL-GCN", "dataset": "Cora", "metric": "accuracy", "percent": True}
    assert any("Streuung +-1.39" in w and "bleibt fest" in w for w in paper["warnings"])   # Info, Toleranz unveraendert
    assert not any("nicht belegt" in w for w in paper["warnings"])


def test_adapter_keeps_fraction_targets_and_mse_unchanged():
    p, (c,) = cat.build_paper(entry(cand(claim={**cand()["claim"], "target": 0.83})))
    assert c.target == 0.83
    p, (c,) = cat.build_paper(entry(etth1()))
    assert c.target == 0.35 and c.metric == "mse" and c.tolerance == 0.01 and p["dataset"] == "ETTh1"
    assert p["claims"][0]["extract"]["percent"] is False and "prediction length 96" in p["claims"][0]["extract"]["metric"]


def test_seeds_per_entry():
    e = entry(cand())
    e["seeds"] = 7
    assert cat.build_paper(e)[0]["seeds"] == 7


def test_e1_etth1_unclear_protocol_runs_with_warning():
    p, _ = cat.build_paper(entry(etth1()))
    assert any("Protokoll im Paper nicht belegt" in w and "vorlaeufig" in w for w in p["warnings"])


def test_e2_unchecked_column_runs_with_warning():
    p, _ = cat.build_paper(entry(etth1()))
    assert any("Tabellenspalte nicht geprueft" in w for w in p["warnings"])


def test_cora_without_matched_task_is_not_runnable():
    c = cand(protocol="unklar", claim={**cand()["claim"], "matched_task": None, "protocol": "unklar"})
    with pytest.raises(cat.NotRunnable, match="matched_task"):
        cat.build_paper(entry(c))


@pytest.mark.parametrize("horizon", [192, None, "avg"])
def test_etth1_needs_horizon_96(horizon):
    with pytest.raises(cat.NotRunnable, match="Horizont"):
        cat.build_paper(entry(etth1(horizon=horizon)))


def test_unsupported_metric_and_dataset_not_runnable():
    with pytest.raises(cat.NotRunnable, match="Metrik"):
        cat.build_paper(entry(etth1(metric="accuracy")))
    with pytest.raises(cat.NotRunnable, match="Datensatz"):
        cat.build_paper(entry(cand(dataset="Citeseer")))


def test_input_length_passed_as_detail_with_discovery_quote():
    quote = "MoHETS uses input sequence length 336. ETTh1 96 0.350."
    p, _ = cat.build_paper(entry(etth1(input_length=336, quote=quote)))
    (d,) = cat.discovery_details(p, quote)
    assert (d.topic, d.value, d.source) == ("input length", "336", "discovery") and d.quote == p["input_length_quote"]
    assert not any("Eingabelaenge im Paper nicht belegt" in w for w in p["warnings"])
    p2, _ = cat.build_paper(entry(etth1()))
    assert cat.discovery_details(p2) == [] and any("Eingabelaenge" in w for w in p2["warnings"])


def test_order_e5():
    a = entry(cand("2001.00001"))                                                     # passt + ok + Code
    b = entry(cand("2001.00002", github_url=None, feasibility="no_repo"))             # passt, kein Code -> ans Ende
    c = entry(cand("2001.00003", claim={**cand()["claim"], "column_check": "unchecked"}))   # Spalte ungeprueft
    d = entry(etth1())                                                                # unklar
    assert [cat.order_key(e)[0] for e in (a, b, c, d)] == [1, 4, 3, 3]
    chosen = curate_mod.select([d, c, b, a], None)   # ohne Code standardmaessig uebersprungen
    assert [e["arxiv_id"] for e in chosen] == ["2001.00001", "2001.00003", "2601.21866"]
    chosen = curate_mod.select([d, c, b, a], None, include_no_repo=True)
    assert [e["arxiv_id"] for e in chosen] == ["2001.00001", "2001.00003", "2601.21866", "2001.00002"]
    assert [e["arxiv_id"] for e in curate_mod.select([b], b["arxiv_id"])] == ["2001.00002"]   # --only ignoriert die Markierung


def test_select_only_matches_arxiv_id_and_entry_id():
    a, b = entry(cand("2001.00001")), entry(cand("2001.00002"))
    assert [e["id"] for e in curate_mod.select([a, b], "2001.00002")] == [b["id"]]
    assert [e["id"] for e in curate_mod.select([a, b], a["id"])] == [a["id"]]
    assert curate_mod.select([a, b], "nope") == []


# ---- Import und Datei ------------------------------------------------------------
def test_import_is_idempotent_and_keeps_status(tmp_path):
    entries = []
    assert cat.import_candidates(entries, [(cand(), "events:r.jsonl"), (etth1(), "cache:c.json")]) == 2
    entries[0]["status"], entries[0]["verdict"] = cat.CHECKED, {"urteil": "reproduziert"}
    assert cat.import_candidates(entries, [(cand(), "cache:other.json")]) == 0
    assert entries[0]["status"] == cat.CHECKED and entries[0]["verdict"] == {"urteil": "reproduziert"}
    assert entries[0]["imported_from"] == "events:r.jsonl"             # Quelle pro Eintrag
    f = tmp_path / "catalog.yaml"
    cat.save_catalog(entries, f)
    assert cat.load_catalog(f) == yaml.safe_load(f.read_text())["candidates"] == entries


def test_read_cache_and_events(tmp_path):
    cache = tmp_path / "c.json"
    ok = {**cand(), "status": "ungeprueft", "cache_version": 2}
    cache.write_text(json.dumps({
        "a": ok, "b": {**ok, "cache_version": 3, "arxiv_id": "2001.1"}, "c": {**ok, "cache_version": 1},
        "d": {**ok, "status": "rejected"}, "e": {"status": "rejected", "reason": "x", "cache_version": 3}}))
    assert [c["arxiv_id"] for c, src in cat.read_cache_candidates(cache)] == ["2511.08094", "2001.1"]
    assert cat.read_cache_candidates(cache)[0][1] == "cache:c.json"
    ev = tmp_path / "r.jsonl"
    ev.write_text(json.dumps({"type": "searching"}) + "\n" + json.dumps(cand()) + "\n\n")
    got = cat.read_event_candidates(ev)
    assert len(got) == 1 and got[0][1] == "events:r.jsonl"


def test_committed_replay_yields_sl_gcn_pilot():
    (c, _), = [x for x in cat.read_event_candidates(ROOT / "web/public/data/discovery/replay.jsonl")
               if x[0]["arxiv_id"] == "2511.08094"]
    paper, (claim,) = cat.build_paper(entry(c))
    assert paper["dataset"] == "Cora-LCC" and claim.target == pytest.approx(0.8292)


def test_committed_catalog_file_loads():
    assert isinstance(cat.load_catalog(), list)


def dossier_with(urteil="reproduziert", measured=(0.83, 0.84)):
    claim = Claim("disc_x_accuracy", "t", "accuracy", 0.8292, 0.01, "s", "paper")
    d = Dossier(paper_id="disc_x", claims=[claim])
    d.attempts = [SimpleNamespace(claim_id=claim.id, seed=i, measured=m) for i, m in enumerate(measured)]
    mean = sum(measured) / len(measured) if measured else float("nan")
    d.verdicts = [Verdict(claim.id, urteil, mean, 0.005, "x")]
    return d


def test_record_result_writes_verdict():
    e = entry(cand())
    cat.record_result(e, cat.CHECKED, dossier=dossier_with(), run_dir=ROOT / "runs" / "disc_x" / "t", cost_gross_usd=0.12345)
    v = e["verdict"]
    assert e["status"] == "geprueft" and e["cost_gross_usd"] == 0.1235
    assert v["urteil"] == "reproduziert" and v["measured_mean"] == pytest.approx(0.835) and v["seeds"] == 2
    assert v["gesamtaussage"] == "vorlaeufig" and v["run_dir"] == "runs/disc_x/t" and v["target"] == 0.8292
    cat.record_result(e, cat.FAILED, reason="boom")
    assert "verdict" not in e and e["reason"] == "boom"
    cat.record_result(e, cat.CHECKED, dossier=dossier_with("nicht reproduzierbar (Setup)", ()))
    assert e["verdict"]["measured_mean"] is None and e["verdict"]["seeds"] == 0     # NaN -> null, YAML-tauglich
    yaml.safe_dump(e)


# ---- Loop mit Attrappen --------------------------------------------------------------
def tavily_orch(tmp_path, monkeypatch, catalog_entries, evidence, seeds=1):
    sb = FakeSandbox()
    o, calls = make_orch(sb, seeds=seeds)
    o.use_tavily = o.use_setup_details = True
    o.paper_only, o.claim_ids, o.tavily_mode = False, None, "fixture"
    o._tavily_calls, o._sources, o.paper_text, o.last_run_dir, o._run_dir = 0, [], None, None, None
    o.events = EventLog(None)
    cf = tmp_path / "catalog.yaml"
    cat.save_catalog(catalog_entries, cf)
    monkeypatch.setattr(loop_mod, "load_catalog_claims", lambda pid: cat.load_catalog_claims(pid, cf))
    monkeypatch.setattr(loop_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr("repro_agent.dossier.REPO_ROOT", tmp_path)
    text = PaperText("2511.08094", "https://arxiv.org/html/2511.08094", "arxiv_html", "2026-09-30T00:00:00", "h" * 64, "v1",
                     "SL-GCN 82.92 ± 1.39 on Cora")
    monkeypatch.setattr(loop_mod, "fetch_paper_text", lambda *a, **k: text)
    monkeypatch.setattr(loop_mod, "find_repo_url", lambda t: None)

    def fake_extract(*a, **k):
        if isinstance(evidence, Exception):
            raise evidence
        return evidence
    monkeypatch.setattr(loop_mod, "extract_target_cached", fake_extract)
    monkeypatch.setattr(loop_mod, "extract_setup_details", lambda llm, docs, text, ds: ([], 0))
    return o


def evidence(value):
    return TargetEvidence(value=value, quote="SL-GCN 82.92", table="Table 2", url="https://arxiv.org/html/2511.08094",
                          source_kind="arxiv_html", retrieved_at="t", sha256="h" * 64, arxiv_version="v1", column_check="ok")


def cora_entry(**claim_over):
    return entry(cand(claim={**cand()["claim"], "matched_task": "Cora", **claim_over}))


def test_run_from_catalog_rolle2_agrees(tmp_path, monkeypatch):
    e = cora_entry()
    o = tavily_orch(tmp_path, monkeypatch, [e], evidence(0.8292))
    d = o.run(e["id"], source="catalog")
    c = d.claims[0]
    assert c.target == pytest.approx(0.8292) and c.target_evidence["quote"] == "SL-GCN 82.92"   # Rolle 2 belegt den Wert
    assert [w for w in d.warnings if "weicht ab" in w] == []
    assert d.verdicts[0].urteil in ("reproduziert", "teilweise", "nicht reproduziert") 
    assert d.verdicts[0].methode in ("nicht geprueft", "nein")   # nie "ja": Pruefer nicht geeicht; Fake-Ablation liefert "Graph nicht genutzt"
    assert d.verdicts[0].gesamtaussage in ("vorlaeufig", "nicht bewertbar (Methode nicht umgesetzt)")
    assert (tmp_path / "runs" / e["id"]).exists()


def test_run_from_catalog_rolle2_disagreement_stops_before_experiments(tmp_path, monkeypatch):
    e = cora_entry()
    o = tavily_orch(tmp_path, monkeypatch, [e], evidence(0.8115))
    monkeypatch.setattr(o, "run_claim", lambda *a: pytest.fail("No experiment for a mismatched target"))
    with pytest.raises(TargetVerificationBlocked, match="discovery selected.*paper check found"):
        o.run(e["id"], source="catalog")
    assert cat.build_paper(e)[1][0].target == pytest.approx(0.8292)


def test_run_from_catalog_rolle2_failure_stops_before_experiments(tmp_path, monkeypatch):
    e = cora_entry()
    o = tavily_orch(tmp_path, monkeypatch, [e], ExtractionInvalid("Zahl nicht im Text"))
    monkeypatch.setattr(o, "run_claim", lambda *a: pytest.fail("No experiment for an unverified target"))
    with pytest.raises(TargetVerificationBlocked, match="target could not be verified"):
        o.run(e["id"], source="catalog")


def test_catalog_stops_when_value_matches_but_column_is_unverified(tmp_path, monkeypatch):
    e = cora_entry()
    ev = evidence(.8292)
    ev.column_check = "nicht pruefbar"
    o = tavily_orch(tmp_path, monkeypatch, [e], ev)
    monkeypatch.setattr(o, "run_claim", lambda *a: pytest.fail("No experiment for an ambiguous column"))
    with pytest.raises(TargetVerificationBlocked, match="column could not be established"):
        o.run(e["id"], source="catalog")


def test_benchmark_rolle2_failure_still_aborts(tmp_path, monkeypatch):
    claim = Claim("c", "t", "accuracy", 0.815, 0.01, "s", "paper")
    o = tavily_orch(tmp_path, monkeypatch, [], ExtractionInvalid("x"))
    with pytest.raises(ExtractionInvalid):
        o.resolve_targets({"arxiv_url": "https://arxiv.org/abs/1", "claims": [{"id": "c", "extract": {
            "method": "GCN", "dataset": "Cora", "metric": "accuracy"}}]}, [claim])


def test_input_length_reaches_the_plan(tmp_path, monkeypatch):
    quote = "MoHETS uses input sequence length 336. ETTh1 96 0.350."
    e = entry(etth1(input_length=336, quote=quote))
    o = tavily_orch(tmp_path, monkeypatch, [e], evidence(0.35))
    monkeypatch.setattr(loop_mod, "fetch_paper_text", lambda *a, **k:
                        PaperText("2601.21866", "u", "arxiv_html", "t", "h", "v1", quote))
    monkeypatch.setattr(loop_mod, "TASKS", {"ETTh1": lambda: SimpleNamespace(ablation_cmd=None, image="i", setup_cmds=[], selection_contract=False, selection_info=lambda: None)})
    monkeypatch.setattr(o, "run_claim", lambda *a, **k: None, raising=False)
    d = o.run(e["id"], source="catalog")
    assert [(x["topic"], x["value"], x["source"]) for x in d.setup_details] == [("input length", "336", "discovery")]
    assert any("Protokoll im Paper nicht belegt" in w for w in d.warnings)


@pytest.mark.parametrize("quote", ["ETTh1 96 192 336 720 0.350", "Look-back window 360. ETTh1 0.350",
                                 "Prediction length 720; MSE 0.350"])
def test_catalog_input_length_hint_is_not_passed_as_a_verified_setup_fact(quote):
    p, _ = cat.build_paper(entry(etth1(input_length=720, quote=quote)))
    assert cat.discovery_details(p, quote) == []
    assert any("Eingabelaenge im Paper nicht belegt" in w for w in p["warnings"])


def test_setup_length_quote_must_still_occur_in_the_fresh_source():
    quote = "Input length is 360."
    p, _ = cat.build_paper(entry(etth1(input_length=360, input_length_quote=quote)))
    assert cat.discovery_details(p, "Input length is 720.") == []
    assert cat.discovery_details(p) == []
    assert cat.discovery_details(p, quote)[0].value == "360"


@pytest.mark.parametrize("quote,length,source", [
    ("Input Sequence Length for all experiments is 336", 336, "Input Sequence Length for all experiments is 336"),
    ("We set the input length L=512 as suggested in Nie et al.", 512, r"We set the input length $L = 512$ as suggested in~\citet{YN23}"),
    ("input length of 720", 720, "input length of 720"),
    ("look-back of 720", 720, "look-back of 720"),
])
def test_setup_length_can_be_proved_by_protocol_or_formatted_source(quote, length, source):
    p, _ = cat.build_paper(entry(etth1(input_length=length, protocol_quote=quote)))
    assert cat.discovery_details(p, source)[0].value == str(length)


@pytest.mark.parametrize("paper_id", ["disc_2609_30894", "disc_2603_13702"])
def test_audited_unsupported_catalog_claims_are_not_startable(paper_id):
    e = cat.find_entry(cat.load_catalog(), paper_id)
    with pytest.raises(cat.NotRunnable, match="frozen|backbone"):
        cat.build_paper(e)


def test_run_rejects_unknown_source(tmp_path, monkeypatch):
    o = tavily_orch(tmp_path, monkeypatch, [], None)
    with pytest.raises(ValueError):
        o.run("x", source="nope")


# ---- Kuratierungsskript -----------------------------------------------------------------
def test_dry_run_reads_only_and_makes_no_calls(tmp_path, capsys):
    ev = tmp_path / "r.jsonl"
    ev.write_text(json.dumps(cand()) + "\n")
    catalog_file = tmp_path / "catalog.yaml"
    rc = curate_mod.main(["--dry-run", "--import-events", str(ev), "--only", "2511.08094", "--seeds", "10",
                          "--catalog", str(catalog_file), "--cache", str(tmp_path / "none.json")],
                         orchestrator_factory=lambda: pytest.fail("Orchestrator im Dry-Run"))
    out = capsys.readouterr().out
    assert rc == 0 and not catalog_file.exists()                       # nichts geschrieben
    assert '"seeds": 10' in out and '"task": "Cora-LCC"' in out and "DRY-RUN" in out


def test_dry_run_shows_not_runnable_reason(tmp_path, capsys):
    ev = tmp_path / "r.jsonl"
    ev.write_text(json.dumps(etth1(horizon=192)) + "\n")
    curate_mod.main(["--dry-run", "--import-events", str(ev), "--catalog", str(tmp_path / "c.yaml"), "--cache", str(tmp_path / "n.json")])
    assert '"runnable": false' in capsys.readouterr().out


class FakeOrch:
    """Ersetzt Orchestrator: gibt Dossiers oder wirft Fehler, zaehlt Kosten."""

    def __init__(self, plan, cost=0.2):
        self.plan, self.cost, self.ran, self.spent_gross_usd, self.last_run_dir = list(plan), cost, [], 0.0, None

    def run(self, paper_id, source="benchmark"):
        assert source == "catalog"
        self.ran.append(paper_id)
        step = self.plan.pop(0)
        self.spent_gross_usd = self.cost
        if isinstance(step, Exception):
            raise step
        return dossier_with(step)


def run_curate(tmp_path, entries, orch, max_usd=1.0, only=None):
    f = tmp_path / "catalog.yaml"
    cat.save_catalog(entries, f)
    chosen = curate_mod.select(entries, only)
    rc = curate_mod.curate(entries, chosen, f, max_usd, None, False, orchestrator_factory=lambda: orch)
    return rc, cat.load_catalog(f)


def test_curate_writes_status_and_verdict(tmp_path):
    entries = [entry(cand("2001.00001")), entry(etth1(horizon=192))]
    rc, saved = run_curate(tmp_path, entries, FakeOrch(["teilweise"]))
    by = {e["arxiv_id"]: e for e in saved}
    assert rc == 0
    assert by["2001.00001"]["status"] == "geprueft" and by["2001.00001"]["verdict"]["urteil"] == "teilweise"
    assert by["2601.21866"]["status"] == "nicht_laufbar" and "Horizont 192" in by["2601.21866"]["reason"]


def test_curate_crash_marks_failed_and_continues(tmp_path):
    entries = [entry(cand("2001.00001")), entry(cand("2001.00002"))]
    orch = FakeOrch([RuntimeError("Sandbox weg"), "reproduziert"])
    rc, saved = run_curate(tmp_path, entries, orch)
    assert rc == 0 and orch.ran == ["disc_2001_00001", "disc_2001_00002"]
    assert saved[0]["status"] == "lauf_fehlgeschlagen" and "Sandbox weg" in saved[0]["reason"]
    assert saved[1]["status"] == "geprueft"


@pytest.mark.parametrize("exc", [BillingBlocked("gesperrt"), __import__("repro_agent.costs", fromlist=["x"]).BudgetExceeded("leer")])
def test_curate_aborts_on_billing_or_budget(tmp_path, exc):
    entries = [entry(cand("2001.00001")), entry(cand("2001.00002"))]
    orch = FakeOrch([exc, "reproduziert"])
    rc, saved = run_curate(tmp_path, entries, orch)
    assert rc == 2 and orch.ran == ["disc_2001_00001"]                 # zweiter Kandidat nicht gestartet
    assert saved[0]["status"] == "ungeprueft" and "Abbruch" in saved[0]["reason"]   # wiederholbar


def test_curate_max_usd_stops_before_next_candidate(tmp_path):
    entries = [entry(cand(f"2001.0000{i}")) for i in range(1, 4)]
    orch = FakeOrch(["reproduziert"] * 3, cost=0.3)
    rc, saved = run_curate(tmp_path, entries, orch, max_usd=0.5)
    assert rc == 2 and len(orch.ran) == 2                              # 0.3 < 0.5: zweiter laeuft, 0.6 >= 0.5: Stopp
    assert [e["status"] for e in saved] == ["geprueft", "geprueft", "ungeprueft"]


def test_only_reruns_any_status_default_skips_done(tmp_path):
    a, b = entry(cand("2001.00001")), entry(cand("2001.00002"))
    a["status"] = cat.CHECKED
    assert [e["arxiv_id"] for e in curate_mod.select([a, b], None)] == ["2001.00002"]
    assert [e["arxiv_id"] for e in curate_mod.select([a, b], "2001.00001")] == ["2001.00001"]


def test_dossier_warnings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("repro_agent.dossier.REPO_ROOT", tmp_path)
    path = Dossier(paper_id="p", warnings=["w1"]).save("t")
    assert Dossier.load(path).warnings == ["w1"]
    (tmp_path / "old.json").write_text(json.dumps({"paper_id": "p", "claims": [], "attempts": [], "verdicts": []}))
    assert Dossier.load(tmp_path / "old.json").warnings == []          # alte Laeufe ohne Feld


def test_details_reset_at_run_start(tmp_path, monkeypatch):
    e = cora_entry()
    o = tavily_orch(tmp_path, monkeypatch, [e], evidence(0.8292))
    o.use_setup_details = False                                         # Kandidat ohne Rolle 4
    o.details = [SimpleNamespace(topic="lr", value="0.01", quote="q", source="paper")]   # Rest des vorigen Kandidaten
    d = o.run(e["id"], source="catalog")
    assert d.setup_details == []


def test_rolle2_agreement_tolerates_rounding(tmp_path, monkeypatch):
    claim = Claim("c", "t", "accuracy", 82.92 / 100, 0.01, "s", "paper")
    o = tavily_orch(tmp_path, monkeypatch, [], evidence(0.8292))
    events = []
    o._emit = lambda type, **kw: events.append((type, kw))
    paper = {"arxiv_url": "https://arxiv.org/abs/1", "claims": [{"id": "c", "keep_target": True, "extract": {
        "method": "SL-GCN", "dataset": "Cora", "metric": "accuracy"}}]}
    o.resolve_targets(paper, [claim])
    assert dict(events)["target_check"]["agrees"] is True and not getattr(o, "_warnings", [])
    claim2 = Claim("c", "t", "accuracy", 0.8292, 0.01, "s", "paper")
    o.resolve_targets(paper, [claim2])   # 0.8292 gegen 0.8292000000000001 o. ae.
    o2 = tavily_orch(tmp_path, monkeypatch, [], evidence(0.8293))
    o2.resolve_targets(paper, [Claim("c", "t", "accuracy", 0.8292, 0.01, "s", "paper")])
    assert any("weicht ab" in w for w in o2._warnings)                  # 1e-4 Abstand bleibt Abweichung


def test_unproven_protocol_caps_gesamtaussage_in_verdict():
    d = dossier_with("reproduziert")
    d.verdicts[0].methode = "ja"                                        # Pruefer hypothetisch geeicht
    assert d.verdicts[0].gesamtaussage == "Paper reproduziert"
    unclear = entry(etth1())
    cat.record_result(unclear, cat.CHECKED, dossier=d)
    assert unclear["verdict"]["gesamtaussage"] == "vorlaeufig"
    proven = entry(cand())
    cat.record_result(proven, cat.CHECKED, dossier=d)
    assert proven["verdict"]["gesamtaussage"] == "Paper reproduziert"   # Protokoll belegt: keine Deckelung
    d2 = dossier_with("nicht reproduzierbar (Setup)", ())
    cat.record_result(unclear, cat.CHECKED, dossier=d2)
    assert unclear["verdict"]["gesamtaussage"] == "nicht reproduzierbar (Setup)"   # strengere Aussagen bleiben
