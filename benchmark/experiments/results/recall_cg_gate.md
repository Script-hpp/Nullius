# Code-grounded Gate G0: Ergebnis (UNVOLLSTAENDIG, 50 von 54 Proben)

Stand 2026-10-09, Branch `Script-hpp/cto`. Vorab-Festlegung: `benchmark/experiments/recall_cg_gate.md` (plus Nachtrag Host-OOM, Commit 7224df3). Auswertung ohne Modellaufruf mit `classify`/`hit`/`decide` aus `scripts/review/cg_gate.py` auf den 50 vorliegenden `final_*.json`; Rohdaten `results/recall_cg_gate_samples/` (inkl. `results.jsonl`, `spend.jsonl`, `log.txt`). Keine Benchmark-Zahl, nichts davon ins README oder Video.

## Status

**UNVOLLSTAENDIG (50/54).** Nach der Vorab-Regel ist ein unvollstaendiges Gate weder bestanden noch nicht bestanden. Es gibt keine nachtraegliche Anpassung der Schwellen. `decide()` meldet `"decision": "unvollstaendig"`.

## Stopp-Grund

Nutzerentscheidung 2026-10-09 gegen 02:15: G0 stoppen. G1 (Branch `Script-hpp/gat-analysis`, Ergebnis 64e7d40) liegt vor und ist bestanden. Die letzten vier Proben (alle Shchur-GAT, lokal zusammen noch 6 bis 10 h laut Koordination) haetten G0 nur verzoegert. Der laufende Prozess wurde beendet (SIGTERM), alle `gate-*`-Container entfernt. Der bisherige G0-Worker reagierte nicht.

## Fehlende Proben (4)

Alle `shchur_gat_2018` (Gruppe Entwicklung, Regel a): `heute` T=0,2, `heute` T=0,8, `cg` T=0,2, `cg` T=0,5. Code und Plaene liegen vor (`code_*.py`, `final_code_*.py`), die Messung lief nicht zu Ende. Vorhanden: `heute` T=0,5 und `cg` T=0,8, beide "nicht reproduzierbar (Setup)" (0/30 gueltige Seeds, 4 Versuche).

## Teilstand je Einheit und Arm (Treffer von 3; fehlende Proben zaehlen als 0)


| Einheit | Gruppe | Referenzurteil | heute (Treffer/3) | cg (Treffer/3) |
|---|---|---|---|---|
| zeng_nlinear_2023 | fail | reproduziert | 0 | 1 |
| wu_sgc_2019 | fail | reproduziert | 3 | 3 |
| zeng_dlinear_2023 | fail | reproduziert | 1 | 2 |
| velickovic_gat_2018 | fail | reproduziert | 0 | 0 |
| velickovic_dgi_2019 | fail | teilweise | 0 | 2 |
| shchur_gat_2018 | fail | reproduziert | 0 | 0 |
| kipf_gcn_2017 | nonreg | reproduziert | 0 | 1 |
| shchur_pitfalls_2018 | nonreg | reproduziert | 0 | 3 |
| rlinear_2023 | new | reproduziert | 0 | 3 |

## Regeln (a) bis (d) auf dem Teilstand

| Regel | Schwelle | Teilstand | Wert |
|---|---|---|---|
| (a) Entwicklungseinheiten (6 x 3 = 18 Proben) | cg >= 8 **und** cg >= heute + 5 | cg 8, heute 4, Abstand 4 | nicht erfuellt (Abstand 4 statt 5; cg-Schwelle 8 erfuellt) |
| (b) Kipf + Shchur-GCN | cg >= heute - 1 | cg 4, heute 0 | erfuellt (vollstaendig, keine fehlende Probe) |
| (c) RLinear | cg >= 2 von 3 | cg 3 von 3 (heute 0) | erfuellt (vollstaendig) |
| (d) leere oder abgeschnittene cg-Code-Aufrufe | <= 2 von 27 | 0 | erfuellt (alle 27 cg-Codes vorhanden) |

**Ob G0 mit allen 54 Proben bestanden haette, ist offen.** Nur Regel (a) haengt an den fehlenden Proben: je zwei fehlende `heute`- und `cg`-Proben (Shchur-GAT). Rechnerisch bleibt (a) in beide Richtungen offen: es genuegt, dass die zwei fehlenden cg-Proben zusammen mindestens einen Treffer mehr bringen als die zwei fehlenden heute-Proben (cg 8 + x, heute 4 + y, Bedingung x - y >= 1). Beide gemessenen Shchur-GAT-Proben (je eine pro Arm) endeten als Setup-Ausfall; das ist ein Hinweis, kein Beleg fuer die vier fehlenden. Die Regeln (b), (c), (d) sind vollstaendig gemessen und aendern sich nicht mehr.

## Befunde (beschreibend, nicht Teil der Regel)

- Treffer insgesamt auf den 50 Proben: cg 15 Treffer aus 25 gemessenen Proben, heute 4 aus 25 (alle Einheiten, Referenzurteil). Einheiten mit cg-Vorsprung: NLinear, DLinear, DGI, Kipf, Shchur-GCN, RLinear. GAT (Velickovic): beide Arme 0 Treffer, cg 3 von 3 Setup-Ausfall; Shchur-GAT: bisher 0 von 2 gemessenen Proben.
- Streuung ist gross: Kipf cg T=0,2/0,8 liefern Mittel 0,62/0,78 mit Std 0,30/0,09 (einzelne Seeds divergiert), T=0,5 trifft. Die drei Proben je Einheit teilen einen Plan und sind nicht unabhaengig.
- Setup-Ausfaelle ueberwiegend bei den GAT-Einheiten (Laufzeit, siehe Pilot und `docs/recall_analysis_3.md`); in der Messung sind das Fehlschlaege in beiden Armen.
- DGI cg T=0,8 meldet "reproduziert" (Referenz "teilweise"): kein Treffer nach Zaehlregel.

## Host-OOM-Unterbrechung

Der erste Lauf (Start 2026-10-08 19:11) endete um 19:25:51 durch einen Host-OOM (Container mit 10,6 GB RSS, vermutlich dichte GAT-Aufmerksamkeit). Ursache: bis zu 48 GB Containerlimits auf einem 30-GB-Host. Infrastrukturfehler, kein Modell- oder Codebefund. Fix (Speicher-Semaphor 20 GB, 6 g leichte / 12 g GAT-Container) und Wiederaufnahme sind im Nachtrag der Vorab-Festlegung dokumentiert; 8 Proben wurden unveraendert uebernommen, unvollstaendige Messungen komplett neu gemessen. Zwischenstaende der Unterbrechung: `results/recall_cg_gate_samples/interrupted_oom/` (fliessen in nichts ein). Die acht vor dem Nachtrag sichtbaren Ergebnisse sind dort offen genannt.

## Kosten

Modellaufrufe (alle Super auf Token Factory, `spend.jsonl`): 291 Aufrufe (264 `code`, 18 `reasoning`, 9 `extract`), netto 1,304 $, **brutto 1,552 $** (Deckel 2,00 $ brutto, nicht erreicht). Keine weiteren Modellaufrufe bei dieser Auswertung. Messung lief lokal in Docker (keine Nebius-Kosten).

## Verweis

G1 (`Script-hpp/gat-analysis`, 64e7d40) ersetzt die Entscheidungsgrundlage fuer die weitere Arbeit; G0 wird nicht fortgesetzt. Dieser Teilstand soll nicht als bestanden oder nicht bestanden zitiert werden.

## Vollstaendige Probentabelle (50)

| Einheit | Arm | T | Urteil | Mittel | Std | gueltige Seeds | Treffer | Ziel erreicht | Versuche Seed 0 | Dauer s |
|---|---|---|---|---|---|---|---|---|---|---|
| zeng_nlinear_2023 | heute | 0.2 | teilweise | 0.3860 | 0.0014 | 5/5 | nein | nein | 1 | 102 |
| zeng_nlinear_2023 | heute | 0.5 | teilweise | 0.3900 | 0.0023 | 5/5 | nein | nein | 1 | 89 |
| zeng_nlinear_2023 | heute | 0.8 | nicht reproduzierbar (Setup) | - | - | 0/5 | nein | - | 4 | 772 |
| zeng_nlinear_2023 | cg | 0.2 | reproduziert | 0.3711 | 0.0002 | 5/5 | ja | ja | 1 | 193 |
| zeng_nlinear_2023 | cg | 0.5 | nicht reproduziert | 0.4234 | 0.0274 | 5/5 | nein | nein | 1 | 326 |
| zeng_nlinear_2023 | cg | 0.8 | nicht reproduzierbar (Setup) | - | - | 0/5 | nein | - | 4 | 686 |
| wu_sgc_2019 | heute | 0.2 | reproduziert | 0.8099 | 0.0014 | 10/10 | ja | ja | 1 | 54 |
| wu_sgc_2019 | heute | 0.5 | reproduziert | 0.8088 | 0.0011 | 10/10 | ja | ja | 1 | 78 |
| wu_sgc_2019 | heute | 0.8 | reproduziert | 0.8088 | 0.0011 | 10/10 | ja | ja | 1 | 65 |
| wu_sgc_2019 | cg | 0.2 | reproduziert | 0.8069 | 0.0017 | 10/10 | ja | ja | 1 | 4050 |
| wu_sgc_2019 | cg | 0.5 | reproduziert | 0.8098 | 0.0025 | 10/10 | ja | ja | 1 | 1971 |
| wu_sgc_2019 | cg | 0.8 | reproduziert | 0.8069 | 0.0017 | 10/10 | ja | ja | 1 | 3466 |
| zeng_dlinear_2023 | heute | 0.2 | reproduziert | 0.3836 | 0.0024 | 5/5 | ja | ja | 2 | 2432 |
| zeng_dlinear_2023 | heute | 0.5 | Agenten-Fehler vermutet | 1.1599 | 0.0417 | 5/5 | nein | nein | 1 | 765 |
| zeng_dlinear_2023 | heute | 0.8 | Agenten-Fehler vermutet | 1.3543 | 0.0337 | 5/5 | nein | nein | 1 | 486 |
| zeng_dlinear_2023 | cg | 0.2 | nicht reproduziert | 0.4943 | 0.0035 | 5/5 | nein | nein | 2 | 2050 |
| zeng_dlinear_2023 | cg | 0.5 | reproduziert | 0.3705 | 0.0002 | 5/5 | ja | ja | 1 | 386 |
| zeng_dlinear_2023 | cg | 0.8 | reproduziert | 0.3708 | 0.0005 | 5/5 | ja | ja | 1 | 1678 |
| velickovic_gat_2018 | heute | 0.2 | nicht reproduziert | 0.7892 | 0.0055 | 10/10 | nein | nein | 2 | 9963 |
| velickovic_gat_2018 | heute | 0.5 | nicht reproduzierbar (Setup) | - | - | 0/10 | nein | - | 4 | 9174 |
| velickovic_gat_2018 | heute | 0.8 | Agenten-Fehler vermutet | 0.1909 | 0.0796 | 10/10 | nein | nein | 3 | 12237 |
| velickovic_gat_2018 | cg | 0.2 | nicht reproduzierbar (Setup) | - | - | 0/10 | nein | - | 4 | 8523 |
| velickovic_gat_2018 | cg | 0.5 | nicht reproduzierbar (Setup) | - | - | 0/10 | nein | - | 4 | 7037 |
| velickovic_gat_2018 | cg | 0.8 | nicht reproduzierbar (Setup) | - | - | 0/10 | nein | - | 4 | 6274 |
| velickovic_dgi_2019 | heute | 0.2 | nicht reproduziert | 0.7739 | 0.0120 | 10/10 | nein | nein | 2 | 4060 |
| velickovic_dgi_2019 | heute | 0.5 | nicht reproduziert | 0.7257 | 0.0137 | 10/10 | nein | nein | 1 | 743 |
| velickovic_dgi_2019 | heute | 0.8 | nicht reproduzierbar (Setup) | - | - | 0/10 | nein | - | 4 | 2762 |
| velickovic_dgi_2019 | cg | 0.2 | teilweise | 0.8054 | 0.0076 | 10/10 | ja | nein | 1 | 4035 |
| velickovic_dgi_2019 | cg | 0.5 | teilweise | 0.8081 | 0.0091 | 10/10 | ja | nein | 2 | 4050 |
| velickovic_dgi_2019 | cg | 0.8 | reproduziert | 0.8180 | 0.0092 | 10/10 | nein | ja | 3 | 4346 |
| shchur_gat_2018 | heute | 0.5 | nicht reproduzierbar (Setup) | - | - | 0/30 | nein | - | 4 | 11256 |
| shchur_gat_2018 | cg | 0.8 | nicht reproduzierbar (Setup) | - | - | 0/30 | nein | - | 4 | 10057 |
| kipf_gcn_2017 | heute | 0.2 | teilweise | 0.8014 | 0.0076 | 10/10 | nein | nein | 1 | 3152 |
| kipf_gcn_2017 | heute | 0.5 | teilweise | 0.8014 | 0.0076 | 10/10 | nein | nein | 1 | 2426 |
| kipf_gcn_2017 | heute | 0.8 | teilweise | 0.7981 | 0.0057 | 10/10 | nein | nein | 1 | 810 |
| kipf_gcn_2017 | cg | 0.2 | nicht reproduziert | 0.6205 | 0.3025 | 10/10 | nein | nein | 1 | 3665 |
| kipf_gcn_2017 | cg | 0.5 | reproduziert | 0.8082 | 0.0072 | 10/10 | ja | ja | 1 | 3355 |
| kipf_gcn_2017 | cg | 0.8 | nicht reproduziert | 0.7833 | 0.0896 | 10/10 | nein | nein | 1 | 4046 |
| shchur_pitfalls_2018 | heute | 0.2 | nicht reproduziert | 0.7353 | 0.0208 | 30/30 | nein | nein | 1 | 4592 |
| shchur_pitfalls_2018 | heute | 0.5 | nicht reproduziert | 0.7836 | 0.0224 | 30/30 | nein | nein | 1 | 4940 |
| shchur_pitfalls_2018 | heute | 0.8 | teilweise | 0.7884 | 0.0186 | 30/30 | nein | nein | 1 | 4921 |
| shchur_pitfalls_2018 | cg | 0.2 | reproduziert | 0.8119 | 0.0145 | 30/30 | ja | ja | 2 | 4964 |
| shchur_pitfalls_2018 | cg | 0.5 | reproduziert | 0.8123 | 0.0148 | 30/30 | ja | ja | 1 | 4927 |
| shchur_pitfalls_2018 | cg | 0.8 | reproduziert | 0.8121 | 0.0130 | 30/30 | ja | ja | 2 | 4977 |
| rlinear_2023 | heute | 0.2 | nicht reproduziert | 0.7995 | 0.0217 | 5/5 | nein | nein | 3 | 2908 |
| rlinear_2023 | heute | 0.5 | nicht reproduziert | 0.7957 | 0.0185 | 5/5 | nein | nein | 2 | 1883 |
| rlinear_2023 | heute | 0.8 | nicht reproduziert | 0.7799 | 0.0212 | 5/5 | nein | nein | 1 | 428 |
| rlinear_2023 | cg | 0.2 | reproduziert | 0.3698 | 0.0001 | 5/5 | ja | ja | 2 | 1355 |
| rlinear_2023 | cg | 0.5 | reproduziert | 0.3698 | 0.0001 | 5/5 | ja | ja | 2 | 808 |
| rlinear_2023 | cg | 0.8 | reproduziert | 0.3717 | 0.0010 | 5/5 | ja | ja | 2 | 2436 |
