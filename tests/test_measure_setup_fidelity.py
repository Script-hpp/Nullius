"""v5 Schritt A, Messwerkzeug: Auswertung, Muster, Belegbarkeit, Dry-Run. Attrappen, kein LLM, kein Netz."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("measure_setup_fidelity", ROOT / "scripts" / "measure_setup_fidelity.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

SOLL = m.load_soll()


def det(topic, value, quote="q", source="paper"):
    return {"topic": topic, "value": value, "quote": quote, "source": source}


def test_soll_file_is_complete_and_patterns_compile():
    assert set(SOLL["units"]) == set(SOLL["soll"]) == set(SOLL["patterns"]) and len(SOLL["units"]) == 8
    for u, points in SOLL["soll"].items():
        assert set(points) == set(SOLL["patterns"][u]), u                       # je Soll-Punkt genau ein Muster
        for p, spec_ in SOLL["patterns"][u].items():
            for key in ("topic", "value", "wrong"):
                if spec_.get(key):
                    re.compile(spec_[key])
            assert spec_["topic"] and spec_["value"], (u, p)
    assert sum(len(v) for v in SOLL["soll"].values()) == 50                      # Zaehlung der Datei (Vorab-Festlegung nennt 48)
    for u, claim in ((u, v["claim"]) for u, v in SOLL["units"].items()):
        assert m.claim_for(u, claim)["id"] == claim                              # Claim existiert in papers.yaml


def test_score_point_states():
    p = SOLL["patterns"]["kipf_gcn_2017"]["lr"]
    assert m.score_point([], p)[0] == "fehlt"
    assert m.score_point([det("learning rate", "0.01")], p)[0] == "korrekt"
    assert m.score_point([det("learning rate", "0.0001")], p)[0] == "falsch"
    assert m.score_point([det("learning rate", "see config")], p)[0] == "unklar"        # Thema getroffen, Wert unklar
    assert m.score_point([det("dropout", "0.5")], p)[0] == "fehlt"                       # anderes Thema
    st, info = m.score_point([det("learning rate", "0.0001"), det("lr", "0.01")], p)
    assert st == "korrekt" and info["conflict"] is True                                   # richtiger Wert gewinnt, Konflikt vermerkt


def test_feature_norm_row_vs_standardised():
    p = SOLL["patterns"]["wu_sgc_2019"]["feature_norm"]
    assert m.score_point([det("feature normalization", "row-normalized (sums to one)")], p)[0] == "korrekt"
    assert m.score_point([det("feature normalization", "zero mean and unit standard deviation")], p)[0] == "falsch"
    q = SOLL["patterns"]["shchur_pitfalls_2018"]["feature_norm"]
    assert m.score_point([det("feature normalization", "features are not normalised")], q)[0] == "korrekt"
    assert m.score_point([det("feature normalization", "row-normalized")], q)[0] == "falsch"   # Shchur: NICHT normalisiert


def test_dgi_applies_to_and_shchur_hidden():
    p = SOLL["patterns"]["velickovic_dgi_2019"]["applies_to"]
    assert m.score_point([det("setting", "transductive (Cora)")], p)[0] == "korrekt"
    assert m.score_point([det("setting", "inductive (Reddit)")], p)[0] == "falsch"
    h = SOLL["patterns"]["shchur_gat_2018"]["hidden"]
    assert m.score_point([det("hidden size", "8 attention heads, 64 in total")], h)[0] == "korrekt"
    assert m.score_point([det("hidden size", "64 per head")], h)[0] == "falsch"


def hits(unit, line):
    return [(h["id"], h["status"]) for h in m.misleading_hits(line, SOLL["misleading"][unit], SOLL["misleading_unless"])]


def test_sgc_standardisation_and_markers():
    line = '- feature normalization: zero mean and unit standard deviation  (repo-readme: "normalize features to zero mean")'
    assert hits("wu_sgc_2019", line) == [("sgc_standardisation", "hit")]
    assert hits("wu_sgc_2019", line + "  [optional advice]") == []                  # markiert = nicht als gueltig gefuehrt
    assert hits("wu_sgc_2019", "- learning rate: 0.2 (paper)") == []


def test_dlinear_rule_only_hits_valid_96_or_1e4():
    unless = "  [repository default (the paper may tune this per dataset)]"
    # Fehlalarme der Basis-Messung: richtige Zeilen duerfen nicht treffen
    assert hits("zeng_dlinear_2023", '- input_length: 336  (paper: "we re-port L=336 for LTSF-Linear and L=96 for Transformers by default.")') == []
    assert hits("zeng_dlinear_2023", '- model_id: ETTh1_336_96  (repo:etth1.sh@c: "--model_id ETTh1_336_96")') == []
    assert hits("zeng_dlinear_2023", '- pred_len: 96  (repo:etth1.sh@c: "--pred_len 96")') == []
    assert hits("zeng_dlinear_2023", '- seq_len: 336  (repo:etth1.sh@c: "--seq_len 336")') == []
    # echte Treffer: 96 bzw. 1e-4 als gueltige Angabe
    assert hits("zeng_dlinear_2023", '- seq_len: 96  (repo:run_longExp.py@c: "--seq_len 96")') == [("dlinear_argparse_default", "hit")]
    assert hits("zeng_dlinear_2023", '- input length: 96  (paper: "L=96")') == [("dlinear_argparse_default", "hit")]
    assert hits("zeng_dlinear_2023", '- learning_rate: 0.0001  (repo:run_longExp.py@c: "default=0.0001")') == [("dlinear_argparse_default", "hit")]
    assert hits("zeng_dlinear_2023", '- learning rate: 1e-4  (paper: "lr 1e-4")') == [("dlinear_argparse_default", "hit")]
    # nicht als gueltig gefuehrt: Default/optional markiert, Transformer/Abbildung, Konfliktzeile
    assert hits("zeng_dlinear_2023", '- seq_len: default=96  (repo:run_longExp.py@c: "default=96")' + unless) == []
    assert hits("zeng_dlinear_2023", '- input length: 96 for Transformer baselines  (paper: "L=96 for Transformers")') == []
    assert hits("zeng_dlinear_2023", '- input length: 96  (paper: "Figure 5 visualization uses L=96")') == []
    assert hits("zeng_dlinear_2023", '- seq_len: paper says 96 (paper: "L=96"); repository says 336 (repo:etth1.sh@c: "seq_len=336")') == []


def test_shchur_gat_rule_statuses():
    q = '(paper: "GAT 64 0.01 0.6/0.3 0.01 92 K")'
    assert hits("shchur_gat_2018", f"- hidden size: 64 per attention head  {q}") == [("shchur_gat_64_per_head", "hit")]
    assert hits("shchur_gat_2018", '- hidden_size: 64  (repo:config/optimized/gcn.conf.yaml@c: "hidden_size: 64")') == [("shchur_gat_64_per_head", "hit")]
    assert hits("shchur_gat_2018", '- learning_rate: 0.01  (repo:config/optimized/gcn.conf.yaml@c: "learning_rate: 0.01")') == [("shchur_gat_64_per_head", "hit")]
    # mehrdeutig: 64 ohne Kopfzahl; zaehlt nicht als Treffer, wird aber gelistet
    assert hits("shchur_gat_2018", f"- Hyperparameter - hidden size: 64  {q}") == [("shchur_gat_64_ambiguous", "ambiguous")]
    conflict = f'- hidden size: paper says 64 {q}; repository says 16 (repo:config/reference/gcn.conf.yaml@c: "hidden_size: 16")'
    assert hits("shchur_gat_2018", conflict) == [("shchur_gat_64_ambiguous", "ambiguous")]
    assert hits("shchur_gat_2018", "- hidden size: 8 attention heads x 8 = 64 in total  (paper)") == []
    assert hits("shchur_gat_2018", f"- learning rate: 0.01  {q}") == []


def test_dgi_rule_only_hits_mp_layers_sampling_minibatch():
    assert hits("velickovic_dgi_2019", '- features: F = 512 features, PReLU  (paper: "We compute F = 512 features")') == []   # F=512 gilt auch fuer Cora
    assert hits("velickovic_dgi_2019", "- learning rate: 0.001 (paper)") == []
    assert hits("velickovic_dgi_2019", "- Model architecture - MP layers: three MP layers, each computing 512 features  (paper)") == [("dgi_inductive", "hit")]
    assert hits("velickovic_dgi_2019", "- Subsampling: sample 10, 10 and 25 neighbors at the first, second and third level  (paper)") == [("dgi_inductive", "hit")]
    assert hits("velickovic_dgi_2019", "- Minibatch size: minibatches of 256 nodes throughout training  (paper)") == [("dgi_inductive", "hit")]
    assert hits("velickovic_dgi_2019", "- Minibatch size: minibatches of 256 nodes  (paper)  [optional advice]") == []


def test_score_output_names_the_rule(tmp_path):
    plan = "- seq_len: 96  (repo:run_longExp.py@c: \"--seq_len 96\")\n- hidden size: 64  (paper: \"GAT 64\")"
    write_runs(tmp_path, "r", {"zeng_dlinear_2023": [([], plan)] * 1, "shchur_gat_2018": [([], plan)]})
    text = m.format_score("r", m.evaluate(SOLL, m.load_label("r", tmp_path)))
    assert "Regel: Eingabelänge 96 als gültige Angabe" in text and "[dlinear_argparse_default]" in text
    assert "Mehrdeutige Zeilen des Plan-Textes" in text and "mehrdeutig: shchur_gat_2018" in text


def write_runs(root, label, per_unit):
    d = root / label
    d.mkdir(parents=True)
    for u, reps in per_unit.items():
        for i, (details, plan) in enumerate(reps):
            (d / f"{u}_{i}.json").write_text(json.dumps({"unit": u, "rep": i, "details": details, "plan_text": plan}))


def test_evaluate_goals_and_doubts(tmp_path):
    good = [det("feature normalization", "row-normalized"), det("learning rate", "0.01")]
    bad = [det("feature normalization", "standardised"), det("learning rate", "see config")]
    write_runs(tmp_path, "basis", {"kipf_gcn_2017": [(good, ""), (good, ""), (bad, "")]})
    ev = m.evaluate(SOLL, m.load_label("basis", tmp_path))
    pts = ev["units"]["kipf_gcn_2017"]["points"]
    assert pts["feature_norm"] == ["korrekt", "korrekt", "falsch"] and pts["lr"] == ["korrekt", "korrekt", "unklar"]
    assert pts["dropout"] == ["fehlt"] * 3
    assert ev["goals"]["ziel1"]["units_ok"] == 1 and not ev["goals"]["ziel1"]["erreicht"]       # 2 von 3 Wdh, aber nur 1 Einheit
    assert [d["point"] for d in ev["doubts"]] == ["lr"] and ev["totals"]["korrekt"] == 4
    assert not ev["goals"]["ziel2"]["erreicht_ohne_basis"]
    assert not ev["goals"]["ziel3"]["erreicht"]                                                   # nicht alle vier Einheiten gemessen


def test_goal3_needs_all_four_units_and_no_hits(tmp_path):
    clean = {u: [([], "- learning rate: 0.01 (paper)")] * 3 for u in SOLL["misleading"]}
    write_runs(tmp_path, "v1", clean)
    assert m.evaluate(SOLL, m.load_label("v1", tmp_path))["goals"]["ziel3"]["erreicht"]
    dirty = dict(clean, wu_sgc_2019=[([], "- feature normalization: zero mean and unit standard deviation (readme)")] * 3)
    write_runs(tmp_path, "v2", dirty)
    g = m.evaluate(SOLL, m.load_label("v2", tmp_path))["goals"]["ziel3"]
    assert not g["erreicht"] and g["treffer"] == 3


def test_score_text_compares_with_base(tmp_path):
    good = [det("learning rate", "0.01")]
    write_runs(tmp_path, "a", {"kipf_gcn_2017": [(good, "")] * 3})
    write_runs(tmp_path, "b", {"kipf_gcn_2017": [([], "")] * 3})
    text = m.format_score("a", m.evaluate(SOLL, m.load_label("a", tmp_path)), m.evaluate(SOLL, m.load_label("b", tmp_path)), "b")
    assert "Ziel 1" in text and "Ziel 2" in text and "Ziel 3" in text and "Basis b 0.0%" in text and "Zweifelsfaelle" in text


def make_fixtures(root, unit_text="learning rate of 0.01 was used with dropout 0.5"):
    (root / "repos").mkdir(parents=True)
    (root / "1609.02907.json").write_text(json.dumps({"text": unit_text + " code: https://github.com/tkipf/gcn"}))
    (root / "repos" / "tkipf_gcn_readme.json").write_text(json.dumps({"text": "readme"}))
    (root / "repos" / "tkipf_gcn.json").write_text(json.dumps({"commit": "abcdef123", "files": {"train.py": "epochs = 200  # epochs to train"}}))


def test_check_soll_finds_value_near_keyword(tmp_path):
    make_fixtures(tmp_path)
    rows = {(r["unit"], r["point"]): r for r in m.check_soll(SOLL, ["kipf_gcn_2017"], tmp_path)}
    assert rows[("kipf_gcn_2017", "lr")]["where"] == "paper" and "0.01" in rows[("kipf_gcn_2017", "lr")]["snippet"]
    assert rows[("kipf_gcn_2017", "epochs")]["where"] == "repo:train.py"
    assert rows[("kipf_gcn_2017", "hidden")]["where"] is None                     # nichts gestrichen, nur "nicht gefunden"
    text = m.format_check(list(rows.values()))
    assert "belegbar in paper" in text and "nicht gefunden" in text and "kipf_gcn_2017.hidden" in text


def test_dry_run_lists_missing_fixtures(tmp_path):
    make_fixtures(tmp_path)
    (tmp_path / "repos" / "tkipf_gcn.json").unlink()
    out = m.dry_run(SOLL, ["kipf_gcn_2017"], tmp_path)
    assert "kipf_gcn_2017" in out and "Klon noetig: melden" in out and "geschaetzte LLM-Aufrufe" in out
    docs, missing = m.fixture_sources("kipf_gcn_2017", tmp_path)
    assert "paper" in docs and m.estimated_calls(docs) >= 2 and len(missing) == 1


def test_forbidden_guard_raises():
    with pytest.raises(RuntimeError, match="verboten"):
        m.Forbidden().find_material("x")


def test_no_soll_values_in_src_prompts():
    """Kein Wert und keine Methode der Soll-Liste in Prompts oder Code unter src/ (Vorab-Festlegung)."""
    src = "\n".join(p.read_text() for p in (ROOT / "src").rglob("*.py") if "harness" not in p.parts)
    assert "v5_step_a_soll" not in src and "measure_setup_fidelity" not in src
