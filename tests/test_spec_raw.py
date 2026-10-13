"""F1 (Startskript schlaegt Paper) und F4 (Konfigurationswerte aus Rohdateien, Quellen-Normalisierung, Rueckfall-Text) im Steckbrief."""

import json

import pytest

from repro_agent.setup_details import Doc, SetupDetail
from repro_agent.spec import (
    Spec, SpecItem, build_spec, excerpt_docs, format_spec, normalize_source, paper_numbers, raw_items, rebuild_spec, resolve, verify_entries)

START = "import time\nfrom models import GAT\n\ndataset = 'cora'\nbatch_size = 1\nnb_epochs = 100000\npatience = 100\nlr = 0.005  # learning rate\nl2_coef = 0.0005  # weight decay\nhid_units = [8]\nn_heads = [8, 1]\n"
PAPER = "We utilize a batch size of 2 graphs during training. Learning rate 0.005 and L2 0.0005."
S_NAMES = ["repo:execute_cora.py@abc1234"]


def test_raw_items_read_assignments_and_skip_lists():
    docs = [Doc("repo:execute_cora.py@abc1234", START)]
    got = {i.key: (i.number, i.rank, i.quote) for i in raw_items(docs, S_NAMES, "GAT", "graph")}
    assert got["batch_size"] == (1, "start script", "batch_size = 1")
    assert got["epochs"][0] == 100000 and got["patience"][0] == 100
    assert got["lr"][0] == 0.005 and got["weight_decay"][0] == 0.0005
    assert "hidden" not in got and "heads" not in got   # Listen sind keine skalaren Werte


def test_raw_items_only_unindented_lines_and_only_listed_docs():
    docs = [Doc("repo:execute_cora.py@abc1234", "def f():\n    batch_size = 7\nbatch_size == 3\n"),
            Doc("repo:execute_other.py@abc1234", "batch_size = 9\n"),
            Doc("repo:models/gat.py@abc1234", "lr = 0.1\n")]
    assert raw_items(docs, S_NAMES + ["repo:models/gat.py@abc1234"], "GAT", "graph") == []   # eingerueckt/Vergleich; Modelldatei ist weder Startskript noch Konfiguration


def test_raw_items_shell_flags_ambiguous_runs_are_skipped():
    sh = "python run.py --model A --seq_len 336 --learning_rate 0.005 --batch_size 32 \\\n  --e 1\npython run.py --model B --seq_len 96\n"
    got = {i.key: i.number for i in raw_items([Doc("repo:scripts/x.sh@a", sh)], ["repo:scripts/x.sh@a"], "A", "forecast")}
    assert got == {"lr": 0.005, "batch_size": 32, "seq_len": 336}   # E09: der Aufruf fuer Modell B verdraengt den Wert von A nicht mehr
    both = "python run.py --seq_len 336 --learning_rate 0.005\npython run.py --seq_len 96 --learning_rate 0.005\n"
    got = {i.key: i.number for i in raw_items([Doc("repo:scripts/x.sh@a", both)], ["repo:scripts/x.sh@a"], "A", "forecast")}
    assert got == {"lr": 0.005}   # zwei passende Aufrufe mit verschiedenen seq_len: wirklich mehrdeutig -> nicht gelesen


def test_yaml_config_values_including_float_tag():
    cfg = "model_name: \"GAT\"\nnum_layers: 2\nhidden_size: 8\nlearning_rate: 0.01\nweight_decay: !!float 5e-4\ninput_dropout_prob: 0.6\n"
    got = {i.key: (i.number, i.rank) for i in raw_items([Doc("repo:config/reference/gat.conf.yaml@a", cfg)], ["repo:config/reference/gat.conf.yaml@a"], "GAT", "graph")}
    assert got == {"layers": (2, "config"), "hidden": (8, "config"), "lr": (0.01, "config"), "weight_decay": (0.0005, "config")}   # input_dropout_prob ist kein 'dropout'


def _build(extract: list, docs: list[Doc], details=None, model="GAT erreicht 83 % auf Cora"):
    return build_spec(lambda s, u: json.dumps(extract), details or [], docs, model, "Cora", numeric_only=True)


def test_f1_start_script_beats_paper():
    """V-GAT: das Paper nennt batch size 2, das Startskript setzt 1. Ohne den Extraktor-Eintrag aus dem Skript gewinnt jetzt trotzdem das Skript."""
    docs = [Doc("paper", PAPER), Doc("repo:execute_cora.py@abc1234", START)]
    ext = [{"key": "batch_size", "value": "2", "quote": "We utilize a batch size of 2 graphs during training.", "source": "paper"}]
    spec = _build(ext, docs)
    bs = spec.get("batch_size")
    assert bs.number == 1 and bs.enforce and bs.rank == "start script" and bs.origin == "raw"
    assert bs.alternatives[0]["value"] == "2" and bs.alternatives[0]["source"] == "paper"


def test_f1_no_conflict_changes_nothing_else():
    docs = [Doc("paper", PAPER), Doc("repo:execute_cora.py@abc1234", START)]
    ext = [{"key": "lr", "value": "0.005", "quote": "Learning rate 0.005", "source": "paper"}]
    spec = _build(ext, docs)
    assert spec.get("lr").number == 0.005 and spec.get("lr").enforce and spec.get("lr").rank == "start script"


def test_numeric_only_off_keeps_old_behaviour():
    docs = [Doc("paper", PAPER), Doc("repo:execute_cora.py@abc1234", START)]
    ext = [{"key": "batch_size", "value": "2", "quote": "We utilize a batch size of 2 graphs during training.", "source": "paper"}]
    spec = build_spec(lambda s, u: json.dumps(ext), [], docs, "GAT erreicht 83 % auf Cora", "Cora")
    assert spec.get("batch_size").number == 2 and [i.key for i in spec.items] == ["batch_size"]


def test_normalize_source_maps_missing_commit_uniquely():
    names = ["paper", "repo:config/optimized/gat.conf.yaml@1e72912", "repo:a/x.py@1", "repo:b/x.py@1"]
    assert normalize_source("repo:config/optimized/gat.conf.yaml", names) == "repo:config/optimized/gat.conf.yaml@1e72912"
    assert normalize_source("config/optimized/gat.conf.yaml", names) == "repo:config/optimized/gat.conf.yaml@1e72912"
    assert normalize_source("repo:x.py", names) == "repo:x.py"                          # mehrdeutig/unbekannt: unveraendert
    assert normalize_source("paper", names) == "paper"


def test_verify_entries_f4a_only_with_flag():
    docs = [Doc("repo:config/optimized/gat.conf.yaml@1e72912", "num_layers: 2\n")]
    raw = [{"key": "layers", "value": 2, "quote": "num_layers: 2", "source": "repo:config/optimized/gat.conf.yaml"}]
    kept, dropped = verify_entries(raw, docs, "GAT", "graph", True)
    assert not kept and dropped[0]["reason"] == "Quelle unbekannt"
    kept, dropped = verify_entries(raw, docs, "GAT", "graph", True, normalize=True)
    assert kept[0].source == "repo:config/optimized/gat.conf.yaml@1e72912" and not dropped


REF = "model_name: \"GAT\"\nnum_layers: 2\nhidden_size: 8\nlearning_rate: 0.01\nweight_decay: !!float 5e-4\n"
OPT = "model_name: \"GAT\"\nnum_layers: 2\nhidden_size: 8\nlearning_rate: 0.01\nweight_decay: 0.01\n"
TRAIN = "num_epochs: 100000\nlearning_rate: 0.01\nweight_decay: !!float 5e-4\n"
CFG_DOCS = [Doc("repo:config/reference/gat.conf.yaml@c1", REF), Doc("repo:config/optimized/gat.conf.yaml@c1", OPT), Doc("repo:config/train.conf.yaml@c1", TRAIN)]
CFG_NAMES = [d.name for d in CFG_DOCS]


def test_f4_model_config_beats_general_and_paper_number_breaks_tie():
    items = raw_items(CFG_DOCS, CFG_NAMES, "GAT", "graph")
    paper = [SetupDetail("hyper", "x", "GAT 64 0.01 0.6/0.3 0.01 92 K", "paper")]
    res, missing = resolve(items, "graph", True, corroborate=paper_numbers(paper, []))
    wd = next(i for i in res if i.key == "weight_decay")
    assert wd.number == 0.01 and wd.enforce and wd.source.endswith("optimized/gat.conf.yaml@c1") and "Paper" in wd.note
    assert next(i for i in res if i.key == "epochs").number == 100000   # nur die allgemeine Konfiguration nennt epochs
    assert "dropout" in missing and "batch_size" in missing


def test_f4_unresolved_tie_is_not_enforced_and_prompt_says_so():
    items = raw_items(CFG_DOCS, CFG_NAMES, "GAT", "graph")
    res, missing = resolve(items, "graph", True, corroborate=set())
    wd = next(i for i in res if i.key == "weight_decay")
    assert not wd.enforce and {t["value"] for t in wd.tie} == {"5e-4", "0.01"}
    spec = Spec("GAT", "graph", res, missing=missing, excerpt="=== x ===\n...")
    text = format_spec(spec)
    assert "weight_decay: sources of equal rank disagree" in text and "weight_decay = " not in text
    assert "Not extracted into this spec (if the authors' code excerpt below defines one of them" in text
    assert "Not stated in any source" not in text


def test_f4b_text_unchanged_without_excerpt():
    spec = Spec("GAT", "graph", [SpecItem("lr", "0.01", "lr 0.01", "paper", "paper", 0.01, enforce=True)], missing=["dropout"])
    assert "Not stated in any source (choose a standard value" in format_spec(spec)


def test_rebuild_spec_replays_saved_spec_without_model_call():
    """G0-Steckbrief von Shchur-GAT: Zitate aus Tabelle verworfen, Config-Quelle ohne @commit verworfen. Die Wiederholung braucht keinen Extraktor."""
    excerpt = "\n\n".join(f"=== {d.name} ===\n{d.text}" for d in CFG_DOCS)
    old = {"model": "GAT", "family": "graph", "excerpt": excerpt, "excerpt_docs": CFG_NAMES, "dropped_defaults": [],
           "missing": ["lr", "weight_decay", "layers"], "items": [
               {"key": "epochs", "value": "100000", "quote": "Maximum number of epochs: 100000.", "source": "paper", "rank": "paper", "number": 100000,
                "enforce": True, "alternatives": [], "note": ""}],
           "dropped": [{"item": {"key": "layers", "value": 2, "quote": "num_layers: 2", "source": "repo:config/optimized/gat.conf.yaml"}, "reason": "Quelle unbekannt"},
                       {"item": {"key": "lr", "value": 0.01, "quote": "table row", "source": "paper"}, "reason": "Zitat nicht im Quelltext"}]}
    details = [SetupDetail("Hidden size for GAT", "64", "GAT 64 0.01 0.6/0.3 0.01 92 K", "paper")]
    spec = rebuild_spec(old, details)
    got = {i.key: (i.number, i.enforce, i.rank) for i in spec.items}
    assert got["weight_decay"] == (0.01, True, "config") and got["layers"] == (2, True, "config") and got["lr"] == (0.01, True, "config")
    assert got["epochs"] == (100000, True, "config")   # Konfiguration schlaegt das Paper (Rangregel)
    assert any(d["reason"] == "Zitat nicht im Quelltext" for d in spec.dropped) and not any(d["reason"] == "Quelle unbekannt" for d in spec.dropped)
    assert spec.excerpt == excerpt and "layers" not in spec.missing


def test_excerpt_docs_merges_function_body_names():
    ex = "=== repo:a.py@1 ===\nx\n\n=== repo:b.py@1 (function body) ===\ny\n\n=== repo:b.py@1 (function body) ===\nz\n"
    assert {d.name: " ".join(d.text.split()) for d in excerpt_docs(ex)} == {"repo:a.py@1": "x", "repo:b.py@1": "y z"}
