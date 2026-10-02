# v5 Schritt C: Ergebnis des Probelaufs

Vorab-Festlegung: `../v5_step_c.md` (mit Nachtrag vom 2026-10-02). Rohdaten: `runs/v5_step_c.jsonl` (18 Urteile aus 9 Läufen), je Lauf `dossier.json` und `events.jsonl` unter `runs/<paper>/<zeitstempel>/`. Lauf am 2026-10-02 von 14:34 bis etwa 19:36, Token Factory, Arm B (gespeicherte Tavily-Ergebnisse), `setup_v2`, LLM-Cache aus. Gestartet von einem zweiten Agenten, nicht vom Reviewer; der Reviewer hat den Lauf nicht abgebrochen und nicht verändert und wertet ihn hier aus.

**Ergebnis nach der festgelegten Regel: Bedingung 2 ist verfehlt.** In Generierung 0 ist der echte Claim bei 0 von 3 Einheiten „reproduziert" (nötig: mindestens 2 von 3). Kein Placebo ist „reproduziert". Bedingung 1 (Schritt A) war schon vorher verfehlt. **Folge: kein v5.** Der Nutzer hat das am 2026-10-02 bestätigt und auch auf den erlaubten zweiten Probelauf verzichtet. Der Benchmark bleibt bei v4.

## Messwerte

Zielwert und Toleranz aus `benchmark/papers.yaml`. „rep" = reproduziert, „teil" = teilweise, „nicht" = nicht reproduziert.

| Einheit | Zielwert | Gen 0 | Gen 1 | Gen 2 | Placebo (Gen 0 / 1 / 2) |
|---|---|---|---|---|---|
| DLinear (`zeng_dlinear_2023`), MSE | 0,375 | 0,3894 ± 0,0095, teil | 0,3734 ± 0,0033, rep | 0,3787 ± 0,0067, rep | nicht / nicht / nicht |
| SGC (`wu_sgc_2019`), Accuracy | 0,810 | 0,7993 ± 0,0022, teil | 0,8054 ± 0,0021, rep | 0,7991 ± 0,0021, teil | nicht / nicht / nicht |
| Shchur-GAT (`shchur_gat_2018`), Accuracy | 0,818 | 0,7257 ± 0,0172, nicht | 0,7855 ± 0,0192, nicht | 0,7890 ± 0,0145, teil | nicht / nicht / nicht |

## Lesart 1: festgelegt (nur Generierung 0)

| Bedingung | Ergebnis |
|---|---|
| Echter Claim bei mindestens 2 der 3 Einheiten „reproduziert" | **0 von 3** (DLinear teilweise, SGC teilweise, Shchur-GAT nicht reproduziert): verfehlt |
| Kein Placebo „reproduziert" | erfüllt (3 von 3 „nicht reproduziert") |

Diese Lesart entscheidet. Bei DLinear fehlen 0,0044 und bei SGC 0,0007 zur Toleranzgrenze; die Regel kennt kein „knapp".

## Lesart 2: post hoc (Mehrheit über drei Generierungen)

Nicht festgelegt, nur zur Einordnung, und nur möglich, weil der Lauf abweichend drei Generierungen erzeugt hat.

| Einheit | Urteile | Mehrheit |
|---|---|---|
| DLinear | teil / rep / rep | reproduziert |
| SGC | teil / rep / teil | teilweise |
| Shchur-GAT | nicht / nicht / teil | nicht reproduziert |

1 von 3 Einheiten mit Mehrheit „reproduziert"; alle 9 Placebo-Urteile „nicht reproduziert". Auch nach dieser Lesart wäre die Schwelle „2 von 3" nicht erreicht.

Vergleich mit v4, Arm B, dieselben drei Einheiten (`results/benchmark_v4.md`): dort 0 von 9 Generierungen „reproduziert", hier 3 von 9 (DLinear 2, SGC 1). Shchur-GAT ist nicht besser geworden (v4: 0,8043 / 0,7938 / 0,7062). Das ist eine Beobachtung an 9 Generierungen ohne Wiederholung, kein Beleg für eine Verbesserung, und der Code des Probelaufs enthält die unten genannten Fehler.

## Abweichungen von der Vorab-Festlegung

1. **Schritt A nicht bestanden** (Ziel 3 verfehlt). Der Probelauf lief auf Entscheidung des Nutzers als offen deklarierte Abweichung (Nachtrag in `../v5_step_c.md`, vor dem Lauf).
2. **Drei Generierungen statt einer** je Einheit. Für die Regel zählt nur Generierung 0.
3. **Kostenstopp überschritten.** Festgelegt: rund 1 $ erwartet, Stopp bei 2 $ brutto. Tatsächlich **3,11 $ brutto** (2,62 $ netto; Summe von `cost_usd_run` der 9 Läufe, gleich der Summe aller Token-Factory-Aufrufe in `logs/costs.jsonl` ab 14:34). Der Stopp hat den Lauf nicht beendet; die Ursache im Skript ist nicht untersucht.
4. **`--v5c` war beim Start nicht committet.** Der Schalter kam erst mit `7052799` ins Repo; die Urteile tragen `code_commit` `9557def`. Das Feld `config` im Store lautet `v4_block_a`.
5. **Merge `9557def` war lokal und ungeprüft:** Runde 2 an Rolle 4 war zum Zeitpunkt des Laufs weder reviewt noch mit eigenen Tests versehen.
6. **Code-Stand mit bekannten Fehlern** (Review `docs/review_round2.md`, nach dem Start erstellt): R1 (Angaben wie „L2 5e-4" galten als Merkmals-Normalisierung, der gezielte zweite Durchgang konnte ausbleiben) und R2 (optionale README-Angaben wurden aus dem Plan entfernt, zählten aber als Abdeckung); dazu R3 (Rangfolge weicht von V2 ab: `argparse`-Default hinter README-Text), R4, R5 und R6 (fester Prompt-Satz zur transduktiven Einstellung, auf einen Messfall zugeschnitten). R1 und R2 sind nach dem Lauf behoben (`round2-fix`), nicht neu gemessen.
7. **Die im Nachtrag verlangte Korrektur „optional advice"** war im Code (`073a442`), aber mit zwei abgeschwächten Prüfungen in `tests/test_setup_v2.py`; beide sind laut Review Regressionen.
8. Der Lauf wurde nicht vom Reviewer gestartet (Arbeitsvereinbarung: echte Läufe macht der Reviewer).

## Beobachtungen aus den Dossiers (post hoc, ohne Gegenprobe)

- **SGC:** Der README-Tipp zur Standardisierung („normalize the features to have zero mean with standard deviation one") steht in Generierung 0 und 2 weiterhin auch als **nicht** optionale Angabe im Dossier (neben einer als optional markierten Fassung), in Generierung 1 gar nicht. Generierung 1 ist die einzige „reproduziert". Alle drei Dossiers enthalten `row_normalize(features)`. Ob der Tipp die 0,799 erklärt, ist nicht geprüft: Die Gegenprobe S2 aus `failure_analysis_v4.md` (Standardisierung) liegt bei 0,730, also weit darunter; 0,8054 entspricht der Gegenprobe S1 (fester Weight Decay 5e-6).
- **SGC-Placebo:** In allen drei Generierungen derselbe Mittelwert 0,8054 und dieselbe Streuung wie beim echten Claim von Generierung 1, bei verschiedenen Code-Hashes. Bei einem fast deterministischen Modell ist das plausibel, geprüft ist es nicht.
- **Shchur-GAT, Generierung 0 (0,7257):** Der Wert entspricht der Gegenprobe L4 „Kanten nur in einer Richtung" (0,7257 ± 0,0189). Das Dossier hat für diesen Lauf keinen Eintrag der Kantenrichtungs-Prüfung (`edge_direction` leer). Ob die Prüfung nicht lief oder nichts fand, ist nicht untersucht.
- **Pflichtthemen:** „feature normalisation" wird in keinem der neun Läufe als fehlend gemeldet. Bei SGC liegt eine echte Angabe vor; für Shchur-GAT ist nicht geprüft, ob die Abdeckung echt ist oder aus R1 stammt.

## Kosten

| Einheit | Gen 0 | Gen 1 | Gen 2 |
|---|---|---|---|
| DLinear | 0,41 | 0,49 | 0,24 |
| SGC | 0,25 | 0,35 | 0,37 |
| Shchur-GAT | 0,33 | 0,33 | 0,35 |

Brutto in $, Summe 3,11. Je Lauf (echter Claim und Placebo) im Mittel 0,35 $; in v4, Arm B, waren es 0,24 $.

## Was daraus folgt

- Kein v5, kein zweiter Probelauf (Entscheidung des Nutzers). Schritt A und Schritt C stehen im README als Entwicklungsergebnis mit dieser Regel und diesem Ausgang.
- `setup_v2` bleibt im Produktpfad (`api.py`), mit behobenem R1 und R2. Die Wirkung von `setup_v2` auf die Benchmark-Kennzahl ist **nicht belegt**.
- Offen und nicht untersucht: warum der Kostenstopp nicht griff; warum die Kantenrichtungs-Prüfung bei Shchur-GAT Generierung 0 keinen Eintrag hat.
