# V1 Ergebnis: Umsetzungstest des Steckbriefs

Stand 2026-10-08. Vorab-Festlegung: `benchmark/experiments/recall_v1_uptake.md` (inkl. Addendum). Rohdaten: `recall_v1_uptake_samples/` (samples.jsonl, `*_spec.json`, Ereignisse; Code-Proben nicht im Repo).

## Entscheidung: STOPP (go = false)

Läufe: Token Factory, Super für alle Rollen, Arme `heute` und `steckbrief_pruefung`, 7 Einheiten × 3 Temperaturen = 42 Proben, `LLM_CACHE=off`, alle 42 mit echtem Modellaufruf. Kostengrenze (1,20 $ brutto) nicht erreicht, kein Abbruch.

| Bedingung | Wert | Schwelle | Ergebnis |
|---|---|---|---|
| 1 Dev-Anteil `steckbrief_pruefung` | 0,685 (37/54) | ≥ 0,70 | **nicht erfüllt** |
| 1 Zuwachs gegenüber `heute` (0,574; 31/54) | +0,111 | ≥ +0,25 | **nicht erfüllt** |
| 2 Neu-Anteil RLinear | 0,40 vs. 0,20 (`heute`) | nicht darunter | erfüllt |
| 3 Aufrufe ohne Code im Arm `steckbrief_pruefung` | 0 von 21 | ≤ 2 | erfüllt |
| 4 `spec_override` zerstört Code | 0 (und 0 Overrides überhaupt) | 0 | erfüllt |

Bedingung 1 scheitert zweifach. Empfehlung laut Regel: kein bezahltes v6.

## Faktanteil je Klasse (erfüllt / Instanzen)

| Klasse | heute | steckbrief_pruefung |
|---|---|---|
| D geliefert, nicht befolgt | 6/18 | 15/18 |
| E im Repo nicht herausgezogen | 13/21 | 11/21 |
| R Datei nie gelesen | 6/9 | 5/9 |
| R* falsche Datei | 6/6 | 6/6 |
| N neu (RLinear) | 3/15 | 6/15 |

## Je Einheit (erfüllt / Instanzen)

| Einheit | heute | steckbrief_pruefung |
|---|---|---|
| DLinear | 3/9 | 9/9 |
| NLinear | 9/15 | 12/15 |
| SGC | 5/6 | 4/6 |
| Veličković-GAT | 0/3 | 3/3 |
| DGI | 8/12 | 4/12 |
| Shchur-GAT | 6/9 | 5/9 |
| RLinear | 3/15 | 6/15 |

## Je Fakt

| Einheit | Fakt | Klasse | heute | steckbrief_pruefung |
|---|---|---|---|---|
| zeng_dlinear_2023 | seq336 | D | 3/3 | 3/3 |
| zeng_dlinear_2023 | lr0.005 | D | 0/3 | 3/3 |
| zeng_dlinear_2023 | halve_each_epoch | D | 0/3 | 3/3 |
| zeng_nlinear_2023 | seq336 | D | 3/3 | 3/3 |
| zeng_nlinear_2023 | lr0.005 | D | 0/3 | 3/3 |
| zeng_nlinear_2023 | halve_each_epoch | D | 0/3 | 0/3 |
| zeng_nlinear_2023 | subtract_last | R* | 3/3 | 3/3 |
| zeng_nlinear_2023 | no_decomposition | R* | 3/3 | 3/3 |
| wu_sgc_2019 | feat_row_l1 | E | 2/3 | 1/3 |
| wu_sgc_2019 | no_col_standardize | E | 3/3 | 3/3 |
| velickovic_gat_2018 | feat_row_l1 | E | 0/3 | 3/3 |
| velickovic_dgi_2019 | feat_row_l1 | E | 2/3 | 1/3 |
| velickovic_dgi_2019 | graph_conv | E | 2/3 | 3/3 |
| velickovic_dgi_2019 | prelu | E | 2/3 | 0/3 |
| velickovic_dgi_2019 | hid512 | E | 2/3 | 0/3 |
| shchur_gat_2018 | additive_attention | R | 1/3 | 1/3 |
| shchur_gat_2018 | self_loops | R | 2/3 | 2/3 |
| shchur_gat_2018 | both_directions | R | 3/3 | 2/3 |
| rlinear_2023 | seq336 | N | 3/3 | 3/3 |
| rlinear_2023 | lr0.005 | N | 0/3 | 0/3 |
| rlinear_2023 | halve_each_epoch | N | 0/3 | 3/3 |
| rlinear_2023 | batch128 | N | 0/3 | 0/3 |
| rlinear_2023 | revin | N | 0/3 | 0/3 |


## Wo es hängt

- **S1 (Steckbrief enthält den Fakt nicht)** ist die Hauptursache der Fehlstellen: Alle Fehlstellen außer einer stehen laut `SPEC_HAS` nicht im Steckbrief-Text. Betroffen: NLinear `halve_each_epoch` (0/3), SGC und DGI `feat_row_l1`, DGI `prelu` und `hid512` (je 0/3, im Arm `heute` 2/3), Shchur-GAT `additive_attention`/`self_loops`/`both_directions`, RLinear `lr0.005` und `batch128`. Beim RLinear-Steckbrief stehen stattdessen argparse-Defaults (lr 0,001, batch 16) aus `run.py` mit Rang „argparse default“; das Start-Skript (lr 0,005, batch 128) wurde nicht herausgezogen. Das S4-Zusatzmaterial (Start-Skripte, Modellkonfiguration) hat das hier nicht behoben.
- **S2/S3 oder Modell:** nur RLinear `revin` (0/3): Fakt steht im Steckbrief (`input_norm = RevIN`), die Probe-Checks (Heuristik) erkennen die Umsetzung in keiner Probe; mögliche Check-Grenze, nicht ausgeschlossen. Das Zahlenwerte-Problem (D-Klasse) löst die Prüfung dort gut: 15/18 gegenüber 6/18.
- **Netto-Verschlechterung durch den Steckbrief** bei DGI (8/12 → 4/12): der Steckbrief führt `prelu`, `hid512` nicht; der Prompt `heute` mit Repo-Ausschnitt hatte sie meist. Das spricht dafür, dass S1 Fakten gegenüber dem Roh-Ausschnitt **verliert** (Verdichtungsverlust), nicht nur nicht hinzufügt.
- S2 `spec_override`: 0 Aufrufe in allen 21 Proben (CONFIG-Werte stimmten bereits oder das Konstrukt wurde nie benötigt). S3 `spec_check`: Reparaturrunden (Runde ≥ 1, aus den gespeicherten Ereignissen, nur letzte 50 je Einheit): DLinear 5, NLinear 2, Veličković-GAT 5, Shchur-GAT 2, RLinear 6, SGC 0, DGI keine Ereignisse gespeichert. Mehrere Proben endeten mit offenen Fehlstellen (`lr`/`epochs`/`patience` „unused"), was Fakt `lr0.005` bei RLinear erklärt, weil der Spec-Wert 0,001 lautete.

## Hinweise und Grenzen

- Trockenlauf (DLinear, Temp 0,2) lief noch mit Ultra für die Plan-Rolle (Umgebungswert `TF_MODEL_REASONING`); sein Ergebnis ist nicht Teil der Auswertung. Der volle Lauf überschrieb die Rolle mit Super (Addendum).
- Checks sind Heuristiken (siehe Vorab-Festlegung). Das verblindete Gegenlesen (`blind/`) wurde nicht durchgeführt.
- Ereignisdateien enthalten je Einheit nur die letzten 50 Ereignisse; Override-Zählung stammt aus `Sample.n_override` (vollständig).
- Kosten: Trockenlauf 0,049 $ brutto + Hauptlauf 0,870 $ brutto (0,731 $ netto) = **0,92 $ brutto** (Grenze 1,20 $). Laufzeit ca. 50 Minuten. Keine Tavily-, keine Sandbox-Aufrufe.
- Aussage: V1 sagt nichts über Accuracy. Die Umsetzung der D-Fakten verbessert sich deutlich (S2/S3 wirken), die E/R/N-Fakten nicht, weil S1 sie oft nicht liefert.
