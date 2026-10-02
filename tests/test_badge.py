"""Abzeichen je Lauf (badge.py): Texte nur aus dem Dossier, Farbe nach Urteil, Gesamtaussage, Escaping, Export schreibt die Datei."""

import xml.etree.ElementTree as ET
from dataclasses import asdict
import json

from repro_agent.badge import COLORS, badge_lines, render_badge, write_badge
from repro_agent.dossier import Claim, Dossier, Verdict
from repro_agent import site_export as se


def make(urteil="reproduziert", mean=0.8213, metric="accuracy", target=0.815, methode="nicht geprueft", text="GCN 81,5 %"):
    return Dossier(paper_id="p", claims=[Claim("c", text, metric, target, 0.01, "Tabelle", "paper")],
                   verdicts=[Verdict("c", urteil, mean, 0.002, "x", methode=methode)])


def test_reproduced_text_numbers_and_notes():
    t = badge_lines(make(), "20260929-130133")
    assert t["verdict"] == "reproduced with our setup" and t["tone"] == "green"
    assert t["detail"] == "measured 82.1% · claimed 81.5% · method not checked" and t["date"] == "2026-09-29"


def test_mse_uses_three_decimals_and_every_number_is_labelled():
    t = badge_lines(make("nicht reproduziert", 0.5, "mse", 0.375), "20260929-130133")
    assert t["detail"].startswith("measured 0.500 · claimed 0.375") and t["tone"] == "red"
    assert t["verdict"] == "not reproduced with our setup"


def test_tone_per_verdict():
    tones = {u: badge_lines(make(u), "20260101-000000")["tone"] for u in
             ("reproduziert", "teilweise", "nicht reproduziert", "nicht reproduzierbar (Setup)", "Agenten-Fehler vermutet")}
    assert tones == {"reproduziert": "green", "teilweise": "amber", "nicht reproduziert": "red", "nicht reproduzierbar (Setup)": "gray", "Agenten-Fehler vermutet": "violet"}


def test_no_measurement_shows_only_the_claim():
    t = badge_lines(make("nicht reproduzierbar (Setup)", float("nan")), "20260101-000000")
    assert t["detail"].startswith("claimed 81.5%") and "measured" not in t["detail"]


def test_overall_statement_when_method_was_judged():
    t = badge_lines(make(methode="nein"), "20260101-000000")        # Zielwert erreicht mit anderem Modell
    assert "overall: target reached with a different model" in t["detail"] and "method not checked" not in t["detail"]
    t = badge_lines(make(methode="ja"), "20260101-000000")
    assert "overall: paper reproduced" in t["detail"]


def test_catalog_not_assessable_hides_the_measurement():
    t = badge_lines(make("nicht reproduziert", 0.55, methode="nein"), "20260930-000000", catalog=True)   # nicht bewertbar
    assert t["verdict"] == "not assessable" and "measured" not in t["detail"] and t["tone"] == "gray"


def test_no_verdict_gives_no_badge(tmp_path):
    d = Dossier(paper_id="p", claims=[Claim("c", "t", "accuracy", 0.8, 0.01, "s", "paper")])
    assert badge_lines(d, "20260101-000000") is None
    (tmp_path / "badge.svg").write_text("old")
    assert write_badge(tmp_path, d, "20260101-000000") is None and not (tmp_path / "badge.svg").exists()


def test_svg_is_valid_and_text_is_escaped():
    lines = badge_lines(make(), "20260929-130133")
    lines["detail"] = "measured 1 <2> & \"3\""
    svg = render_badge(lines)
    root = ET.fromstring(svg)                                  # gueltiges XML trotz Sonderzeichen
    assert root.tag.endswith("svg") and "&lt;2&gt; &amp;" in svg and COLORS["green"] in svg


def test_export_run_writes_badge_next_to_dossier(tmp_path):
    run = tmp_path / "runs" / "kipf_gcn_2017" / "20260929-130133"
    run.mkdir(parents=True)
    (run / "dossier.json").write_text(json.dumps(asdict(make())))
    out = tmp_path / "data"
    se.export_run(run, out, {})
    svg = (out / "kipf_gcn_2017" / "20260929-130133" / "badge.svg").read_text()
    assert "reproduced with our setup" in svg and "measured 82.1%" in svg


def test_with_our_setup_only_for_measured_verdicts():
    assert badge_lines(make("teilweise"), "20260101-000000")["verdict"] == "partially reproduced with our setup"
    assert badge_lines(make("Agenten-Fehler vermutet"), "20260101-000000")["verdict"] == "agent error suspected"
    assert badge_lines(make("nicht reproduzierbar (Setup)", float("nan")), "20260101-000000")["verdict"] == "setup not reproducible"


def test_badge_uses_the_paper_claim_never_placebo_or_objection():
    d = make()
    d.claims = [Claim("p", "Placebo", "accuracy", 0.95, 0.01, "s", "placebo"), Claim("o", "Einwand", "accuracy", 0.5, 0.01, "s", "einwand"), d.claims[0]]
    d.verdicts = [Verdict("p", "nicht reproduziert", 0.82, 0.0, "x"), Verdict("o", "teilweise", 0.82, 0.0, "x"), d.verdicts[0]]
    t = badge_lines(d, "20260101-000000")
    assert t["verdict"] == "reproduced with our setup" and "claimed 81.5%" in t["detail"]
    d.claims[2] = Claim("c", "t", "accuracy", 0.815, 0.01, "s", "einwand")   # kein Paper-Claim: Fallback erstes Urteil
    assert badge_lines(d, "20260101-000000")["verdict"] == "not reproduced with our setup"
