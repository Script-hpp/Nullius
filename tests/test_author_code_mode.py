"""Modus "Nachbau aus Paper und Autorencode" als Produktstandard (2026-10-09): Produktpfade setzen code_grounded + io_helper, Benchmark- und
Experiment-Skripte behalten ihre vorab festgelegte Konfiguration, Rueckfall auf reinen Paper-Nachbau wird nie falsch etikettiert. Attrappen, kein Netz."""

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from repro_agent import api
from repro_agent import catalog as cat
from repro_agent import loop as loop_mod
from repro_agent import site_export
from repro_agent.dossier import Dossier
from repro_agent.loop import AUTHOR_CODE_KWARGS, MODE_AUTHOR_CODE, MODE_PAPER, Orchestrator, product_kwargs
from repro_agent.report import mode_line, render_markdown
from repro_agent.repo_source import RepoDump
from repro_agent.spec import Spec, SpecItem
from test_tavily_mode import _make
from test_catalog import cora_entry, curate_mod, dossier_with, evidence, tavily_orch

ROOT = Path(__file__).resolve().parents[1]


# ---- Produktpfade setzen cg, Schalter schaltet ab ---------------------------------------------------------------------------
def test_product_kwargs_default_is_author_code_and_switch_turns_it_off():
    on, off = product_kwargs(), product_kwargs(author_code=False)
    assert on["code_grounded"] is True and on["io_helper"] is True and on["graph_repair"] and on["setup_v2"]
    assert "code_grounded" not in off and "io_helper" not in off and off == {"graph_repair": True, "setup_v2": True, "repo_verify": True}   # = Produktkonfiguration von v4
    assert AUTHOR_CODE_KWARGS == {"code_grounded": True, "io_helper": True}


def test_orchestrator_default_is_unchanged(monkeypatch):
    o = _make(monkeypatch)
    assert not o.code_grounded and not o.io_helper and not o.spec_mode and o.mode == MODE_PAPER   # kein stilles Umstellen fuer Benchmark/Experimente


def test_api_make_orchestrator_defaults_to_author_code(monkeypatch):
    got = {}
    monkeypatch.setattr(loop_mod, "Orchestrator", lambda **kw: got.update(kw) or SimpleNamespace())
    api.make_orchestrator(True)
    assert got["code_grounded"] is True and got["io_helper"] is True and got["llm_cache"] is False and got["seeds"] == 3
    got.clear()
    api.make_orchestrator(False, author_code=False)
    assert "code_grounded" not in got and "io_helper" not in got and got["graph_repair"] and got["seeds"] is None


def test_cli_default_author_code_and_flag(monkeypatch):
    got = {}

    class Fake:
        def __init__(self, **kw):
            got.update(kw)

        def run(self, *a, **k):
            pass

    monkeypatch.setattr(loop_mod, "Orchestrator", Fake)
    monkeypatch.setattr(sys, "argv", ["loop", "kipf_gcn_2017"])
    loop_mod.main()
    assert got["code_grounded"] and got["io_helper"] and got["graph_repair"]
    got.clear()
    monkeypatch.setattr(sys, "argv", ["loop", "kipf_gcn_2017", "--no-author-code"])
    loop_mod.main()
    assert "code_grounded" not in got and "graph_repair" not in got   # exakt die bisherige CLI-Konfiguration


def test_catalog_reproduction_script_default_and_flag(monkeypatch):
    got = []
    monkeypatch.setattr(loop_mod, "Orchestrator", lambda **kw: got.append(kw) or SimpleNamespace())
    curate_mod.curate([], [], ROOT / "nonexistent.yaml", 1.0, None, False)
    curate_mod.curate([], [], ROOT / "nonexistent.yaml", 1.0, None, False, author_code=False)
    assert got[0]["code_grounded"] and got[0]["io_helper"] and got[0]["graph_repair"]
    assert "code_grounded" not in got[1] and got[1]["graph_repair"]


def test_benchmark_and_experiment_scripts_keep_their_preregistered_configuration():
    """Skripte mit vorab festgelegter Konfiguration bauen den Orchestrator selbst und nutzen weder product_kwargs noch AUTHOR_CODE_KWARGS."""
    allowed = {"curate_catalog.py"}   # Katalog-Reproduktion = Produktpfad
    offenders = []
    for path in (ROOT / "scripts").rglob("*.py"):
        text = path.read_text()
        if path.name not in allowed and re.search(r"\b(product_kwargs|AUTHOR_CODE_KWARGS)\b", text):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
    bench = (ROOT / "scripts" / "run_benchmark.py").read_text()
    assert "code_grounded" not in bench and "io_helper" not in bench
    # die cg-Experimente legen den Modus weiter ausdruecklich fest
    assert "code_grounded=True, io_helper=True" in (ROOT / "scripts" / "recall_cg_v1.py").read_text()


# ---- Laufzeit: Modus im Dossier, Rueckfall ohne Autorencode ------------------------------------------------------------------
def cg_orch(tmp_path, monkeypatch):
    e = cora_entry()
    o = tavily_orch(tmp_path, monkeypatch, [e], evidence(0.8292))
    o._base_modes = {"code_grounded": True, "io_helper": True, "spec_mode": False}
    o.code_grounded = o.io_helper = o.spec_mode = True
    o.events = loop_mod.EventLog(None)
    seen = []
    real = o._emit
    o._emit = lambda type, **kw: (seen.append((type, kw)), real(type, **kw))[1]
    return o, e, seen


def test_no_repository_falls_back_to_paper_rebuild_and_is_labelled_so(tmp_path, monkeypatch):
    o, e, seen = cg_orch(tmp_path, monkeypatch)   # tavily_orch: find_repo_url -> None
    d = o.run(e["id"], source="catalog")
    assert d.mode == MODE_PAPER and d.repo is None
    assert not o.code_grounded and not o.io_helper and o.spec is None
    (ev,) = [kw for t, kw in seen if t == "mode"]
    assert ev["mode"] == MODE_PAPER and ev["requested"] == MODE_AUTHOR_CODE and "no repository" in ev["fallback_reason"]
    assert [kw["mode_requested"] for t, kw in seen if t == "run_start"] == [MODE_AUTHOR_CODE]
    assert any(w.startswith("Autorencode nicht verfügbar") for w in d.warnings)
    md = (o.last_run_dir / "zeugnis.md").read_text()
    assert "Rebuild from paper:" in md and "Rebuild from paper and authors' code" not in md


def test_repository_with_code_excerpt_runs_as_author_code(tmp_path, monkeypatch):
    o, e, seen = cg_orch(tmp_path, monkeypatch)
    monkeypatch.setattr(loop_mod, "find_repo_url", lambda t: "https://github.com/x/y")
    monkeypatch.setattr(loop_mod, "fetch_readme", lambda *a, **k: "")
    monkeypatch.setattr(loop_mod, "read_repo", lambda *a, **k: RepoDump(url="https://github.com/x/y", commit="abc1234567",
                                                                       files={"models/gcn.py": "class GCN: pass"}, n_tracked=1))
    spec = Spec(model="GCN", family="graph", items=[SpecItem("lr", "0.01", "lr 0.01", "paper", "paper text", 0.01, enforce=True)],
                excerpt="=== repo:models/gcn.py@abc1234 ===\nclass GCN: pass\n", excerpt_docs=["repo:models/gcn.py@abc1234"])
    monkeypatch.setattr(loop_mod, "build_spec", lambda *a, **k: spec)
    d = o.run(e["id"], source="catalog")
    assert d.mode == MODE_AUTHOR_CODE and d.repo["url"] == "https://github.com/x/y"
    assert o.code_grounded and o.io_helper
    (ev,) = [kw for t, kw in seen if t == "mode"]
    assert ev["mode"] == MODE_AUTHOR_CODE and ev["fallback_reason"] is None and ev["excerpt_docs"] == ["repo:models/gcn.py@abc1234"]
    assert not any(w.startswith("Autorencode nicht verfügbar") for w in d.warnings)
    assert "Rebuild from paper and authors' code" in (o.last_run_dir / "zeugnis.md").read_text()


def test_repo_without_model_code_falls_back(tmp_path, monkeypatch):
    o, e, seen = cg_orch(tmp_path, monkeypatch)
    monkeypatch.setattr(loop_mod, "find_repo_url", lambda t: "https://github.com/x/y")
    monkeypatch.setattr(loop_mod, "fetch_readme", lambda *a, **k: "")
    monkeypatch.setattr(loop_mod, "read_repo", lambda *a, **k: RepoDump(url="https://github.com/x/y", commit="abc1234567", files={}, n_tracked=0))
    monkeypatch.setattr(loop_mod, "build_spec", lambda *a, **k: Spec(model="GCN", family="graph", items=[], excerpt=""))
    d = o.run(e["id"], source="catalog")
    assert d.mode == MODE_PAPER and d.repo is not None   # Repo wurde gelesen, aber ohne Modellcode: kein Autorencode im Prompt
    (ev,) = [kw for t, kw in seen if t == "mode"]
    assert "no model code" in ev["fallback_reason"]


def test_fallback_does_not_leak_into_next_run_and_flag_off_stays_paper(tmp_path, monkeypatch):
    o, e, seen = cg_orch(tmp_path, monkeypatch)
    o.run(e["id"], source="catalog")
    assert not o.code_grounded
    o._set_author_code(True)   # naechster Lauf beginnt wieder im angeforderten Modus
    assert o.code_grounded and o.io_helper and o.spec_mode
    (tmp_path / "b").mkdir()
    off = tavily_orch(tmp_path / "b", monkeypatch, [e], evidence(0.8292))
    d = off.run(e["id"], source="catalog")
    assert d.mode == MODE_PAPER and d.repo is None


# ---- Bericht, Export, Katalog ------------------------------------------------------------------------------------------------
def test_report_mode_line_names_both_modes():
    assert "Rebuild from paper and authors' code" in mode_line("nachbau_autorencode") and "never executed" in mode_line("nachbau_autorencode")
    assert "harness" in mode_line("nachbau_autorencode")
    assert mode_line("nachbau").startswith("Modus: nachbau. Rebuild from paper:")
    d = dossier_with()
    d.mode = "nachbau_autorencode"
    assert "nachbau_autorencode" in render_markdown(d)


def test_dossier_roundtrip_keeps_mode(tmp_path, monkeypatch):
    monkeypatch.setattr("repro_agent.dossier.REPO_ROOT", tmp_path)
    d = Dossier(paper_id="p", mode="nachbau_autorencode")
    path = d.save("20261009-000000")
    assert Dossier.load(path).mode == "nachbau_autorencode"


def test_verdict_dict_and_export_ref_carry_the_mode(tmp_path):
    d = dossier_with()
    d.mode = "nachbau_autorencode"
    assert cat.verdict_dict(cora_entry(), d)["mode"] == "nachbau_autorencode"
    run = tmp_path / "runs" / "p" / "20261009-000000"
    run.mkdir(parents=True)
    import dataclasses, json
    for mode, want in (("nachbau_autorencode", "nachbau_autorencode"), (None, "nachbau")):
        dd = dataclasses.asdict(d)
        dd.pop("mode")
        if mode:
            dd["mode"] = mode
        dd["attempts"] = []
        (run / "dossier.json").write_text(json.dumps(dd, default=str))
        ref = site_export.export_run(run, tmp_path / "out", {})
        assert ref["mode"] == want
