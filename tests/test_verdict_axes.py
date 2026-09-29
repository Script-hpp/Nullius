import pytest

from repro_agent.dossier import BENCHMARK_AUSSAGEN, Verdict, gesamtaussage


@pytest.mark.parametrize("urteil,methode,erwartet", [
    ("reproduziert", "ja", "Paper reproduziert"),
    ("reproduziert", "nein", "Zielwert erreicht mit anderem Modell"),
    ("reproduziert", "teilweise", "Zielwert erreicht mit anderem Modell"),
    ("nicht reproduziert", "ja", "nicht reproduziert"),
    ("reproduziert", "nicht geprueft", "vorlaeufig"),
    ("nicht reproduziert", "nicht geprueft", "vorlaeufig"),
    ("Agenten-Fehler vermutet", "ja", "Agenten-Fehler vermutet"),
    ("nicht reproduzierbar (Setup)", "nicht geprueft", "nicht reproduzierbar (Setup)"),
    ("nicht reproduziert", "nein", "nicht bewertbar (Methode nicht umgesetzt)"),
    ("teilweise", "ja", "teilweise reproduziert"),
])
def test_gesamtaussage(urteil, methode, erwartet):
    assert gesamtaussage(urteil, methode) == erwartet


def test_default_axis2_not_checked_and_benchmark_counts_only_two():
    v = Verdict("c", "reproduziert", 0.8, 0.0, "x")
    assert v.methode == "nicht geprueft" and v.gesamtaussage == "vorlaeufig"
    assert v.gesamtaussage not in BENCHMARK_AUSSAGEN
    assert set(BENCHMARK_AUSSAGEN) == {"Paper reproduziert", "nicht reproduziert"}
