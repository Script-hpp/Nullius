# v5 Schritt A: Ergebnisse (Dossier-Treue von Rolle 4)

Vorab-Festlegung: `../v5_step_a.md`. Rohdaten: `runs/v5_step_a/{basis,v1,r1}/` (je 8 Einheiten × 3 Wiederholungen, Rolle 4 allein, gespeicherte Quellen, Token Factory, Super, Cache aus). Auswertung: `scripts/measure_setup_fidelity.py --score <label> --vs basis`, alle Stände mit demselben Skript (Stand `v5-step-a-score2`); Musteränderungen am Auswertungsskript in `v5_step_a_scorer_changes.md`. "unklar" (Thema getroffen, Wert nicht erkennbar) zählt nicht als korrekt.

| Stand | korrekt von 150 | fehlt | falsch | unklar | Ziel 1: Normalisierung/Methode (≥ 5 von 7) | Ziel 2: ≥ 80 % und über Basis | Ziel 3: irreführend als gültig (0) | Kosten brutto |
|---|---|---|---|---|---|---|---|---|
| basis (Rolle 4 wie v4) | 116 (77,3 %) | 8 | 14 | 12 | 4 von 7: nein | – | 43 Treffer: nein | 0,98 $ |
| v1 (`setup_v2`, erste Fassung, `c67a905`) | 113 (75,3 %) | 14 | 12 | 11 | 3 von 7: nein | nein | 6 Treffer: nein | 1,18 $ |
| r1 (Nachbesserungsrunde 1, `ee81e01`) | 116 (77,3 %) | 2 | 19 | 13 | **5 von 7: ja** | nein (gleich Basis) | 4 Treffer: nein | 1,39 $ |

**Schritt A ist nach Runde 1 nicht bestanden** (Ziel 2 und Ziel 3 verfehlt). Eine Nachbesserungsrunde ist noch erlaubt.

Was sich geändert hat (r1 gegen Basis):
- "fehlt" 8 auf 2: DGI liefert jetzt die Werte aus dem Startskript (16 von 18 korrekt, Basis 7).
- Irreführende Angaben 43 auf 4: die GCN-Konfiguration gilt bei Shchur-GAT nicht mehr als gültig. Rest: DGI "Minibatch 256" (induktiv) in allen drei Wiederholungen, SGC-README-Standardisierung in einer.
- "falsch" 14 auf 19: Kipf (5), Shchur-GAT (6: Weight Decay und Dropout aus der Tabellenzeile ohne Spaltenkopf), SGC und GAT je 2, NLinear 2.
- Die Extraktion schwankt zwischen Wiederholungen derselben Quelle (z. B. Shchur-GCN, SGC: Normalisierung in einer von drei Wiederholungen falsch).

Offen für Runde 2 (letzte): Rangfolge (Paper vor `argparse`-Default und README-Text), Modellabgleich mit Akronymen, Paper im gezielten zweiten Durchgang, gezielte Fragen je Pflichtthema über die relevanten Abschnitte mit Tabellenkopf.
