# Code-grounded Gate G1: Ergebnis

Lauf 2026-10-09, 01:28 bis 02:25 (ca. 58 Minuten), Branch `Script-hpp/gat-analysis`, Vorab-Festlegung `benchmark/experiments/recall_cg_gate_g1.md` (Commit `436b4b2`, vor jedem Modell-Aufruf). Rohdaten: `benchmark/experiments/results/recall_cg_gate_g1_samples/` (Steckbriefe, Plaene, Codes, `final_*.json`, Versuchsprotokolle `attempts_*.jsonl`, `spend.jsonl`). Auswertung: `uv run python scripts/review/cg_gate_g1.py --score <Ausgabe> --g0 <G0-Ausgabe>`.

## Entscheidung nach der vorab festgelegten Regel

**G1: bestanden** (alle Teilregeln (a), (b), (b′), (c), (d) erfuellt, 18 von 18 Proben mit Urteil, Kostengrenze nicht erreicht).

**Das ist ein schwaches Signal, kein Wirksamkeitsnachweis.** Gruende, alle schon in der Vorab-Festlegung genannt und hier bestaetigt: (1) die Huerde war ein Treffer Vorsprung auf den zwei GAT-Einheiten; (2) das Fix-Paket wurde nach Sicht der G0-Ergebnisse dieser zwei Einheiten entworfen (In-Sample); (3) 3 Proben je Zelle teilen einen Plan; (4) der Messweg ist ein anderer als in G0 (Nebius, 4 GB RAM, 4 vCPU je Branch). Es traegt die Entscheidung, Plan A zu bezahlen, mehr nicht. Nichts davon ins README oder Video.

## Zweigeteilt berichtet

**G0 (Stand dieser Auswertung: 50 von 54 Proben, G0 laeuft im Worktree `cto` noch, letzte vier Shchur-GAT-Proben offen; das G0-Endergebnis ist hier nachzutragen, sobald es vorliegt):** auf den sechs Entwicklungseinheiten `cg` 8 und `heute` 4 Treffer von je 16 fertigen Proben (Veličković-GAT 0 : 0 von je 3, Shchur-GAT 0 : 0 von je einer fertigen Probe; die vier Nicht-GAT-Einheiten NLinear, SGC, DLinear, DGI tragen alle 8 : 4), Regel (a) dort (8 ≥ 8, Differenz 4 < 5) nicht erfuellt. Kipf + Shchur-GCN `cg` 4 von 6, `heute` 0 von 6; RLinear `cg` 3 von 3, `heute` 0 von 3.

**G1 (18 Proben, 6 Zellen, plus uebernommene G0-Zahlen):**

| Einheit | Arm | Treffer G1 | Treffer G0 | Urteile G1 (T = 0,2 / 0,5 / 0,8) |
|---|---|---|---|---|
| Veličković-GAT | `heute` | **0 von 3** | 0 von 3 | nicht reproduziert (0,7892) / Setup / nicht reproduziert (0,5307) |
| Veličković-GAT | `cg` | **0 von 3** | 0 von 3 | Setup / Setup / Setup (je 4 Versuche, überwiegend Timeouts) |
| Shchur-GAT | `heute` | **0 von 3** | 0 von 1 (2 offen) | nicht reproduziert (0,6596) / teilweise (0,7924) / nicht reproduziert (0,7754) |
| Shchur-GAT | `cg` | **2 von 3** | 0 von 1 (2 offen) | reproduziert (0,8166) / reproduziert (0,8145) / Setup (Formfehler in 4 Versuchen) |
| Shchur-GCN | `cg` | 3 von 3 | 3 von 3 | reproduziert (0,8115 / 0,8114 / 0,8117) |
| RLinear (Held-out) | `cg` | 2 von 3 | 3 von 3 | reproduziert (0,3707) / teilweise (0,3782) / reproduziert (0,3696) |

Aggregat (Nicht-GAT-Entwicklungseinheiten aus G0, GAT aus G1; `decide_g1`):

| Regel | Wert | Ergebnis |
|---|---|---|
| (a) `cg` ≥ 8 von 18 und `cg` ≥ `heute` + 5 (Entwicklung) | `cg` 10 (8 + 2), `heute` 4 (4 + 0); 10 ≥ 9 | erfuellt |
| (b) Kipf + Shchur-GCN: `cg` ≥ `heute` − 1 | 4 gegen 0 (Kipf aus G0, Shchur-GCN `cg` aus G1) | erfuellt |
| (b′) Shchur-GCN `cg` ≥ 2 von 3 | 3 von 3 | erfuellt |
| (c) RLinear `cg` ≥ 2 von 3 | 2 von 3 | erfuellt (knapp, G0: 3 von 3) |
| (d) höchstens 2 von 27 leere `cg`-Code-Aufrufe | 0 (alle 12 neuen `cg`-Aufrufe lieferten Code) | erfuellt |

Die Plaene und ersten Codes des Arms `heute` sind byte-identisch aus G0 (`reuse_manifest.json`, 8 von 8 Dateien SHA-256 gleich).

## Was der Gewinn ist und was nicht

- **Der ganze Vorsprung kommt von Shchur-GAT `cg` (2 Treffer).** Dort hat F4 gewirkt wie vorhergesagt: der Steckbrief erzwingt jetzt `weight_decay` 0,01 (Modell-Konfiguration `optimized`, entschieden durch die Zahl im Paper-Zitat), `lr`, `hidden` und `layers` aus den Konfigurationsdateien; das Plateau um 0,78 aus G0 ist weg (0,8166 und 0,8145 gegen Ziel 0,814). Die dritte Probe (T = 0,8) scheiterte an Formfehlern der dichten Aufmerksamkeit (`mat1 and mat2 shapes cannot be multiplied`, `Fan in and fan out …`), nicht an Speicher.
- **Veličković-GAT ist unveraendert 0 von 3 in beiden Armen.** F1 hat den Steckbrief wie geplant korrigiert (`batch_size` 1 aus dem Startskript statt 2 aus dem Paper, in `spec.json` belegt), aber das haben die drei `cg`-Proben nicht in einen Treffer uebersetzt: alle drei enden als Setup-Ausfall nach vier Versuchen, ueberwiegend durch Zeitueberschreitungen bei 660 s (`cg` T = 0,2: Timeout, Timeout, Formfehler, Timeout) und Torch-Idiom-Fehler (`scatter_reduce`, `index_add_`). **F2 (Stack, Pflicht-Hinweis) hat in diesem Lauf keine Probe gerettet**: es gab 9 weiche Timeouts (alle ueber `timeout -s ABRT`, kein harter Sandbox-Timeout), aber die Ketten blieben langsam oder wechselten in andere Fehler.
- **RLinear `cg` verliert eine Probe gegenueber G0** (T = 0,5: „teilweise“, 0,3782, erste Fassung mit `numpy()`-Fehler, danach gueltig). 2 von 3 genuegt der Regel; ob F4 (a) (fuenf nun erzwungene argparse-Defaults) oder Stichprobenrauschen die Probe kostet, ist mit drei Proben nicht trennbar. Shchur-GCN verliert nichts (3 von 3, obwohl drei Werte jetzt erzwungen sind).
- **`heute` bleibt bei GAT bei 0 von 6** (Merkmale roh, Aggregationsfehler, Timeouts). Die 23 ungueltigen Seeds („nur eine einzige Vorhersage-Klasse“) stammen alle aus Shchur-GAT `heute` T = 0,2 (Mittel 0,6596, Std 0,26).
- **Speicher war in diesem Lauf kein Problem:** keine Zeile mit `MemoryError`, `Killed` oder Exit 137 in den 331 Versuchsprotokollen. Das Risiko „4 GB“ ist nicht eingetreten, aber nicht widerlegt (die dichte B = 2-Variante aus G0 wurde durch F1 vermieden).
- **Versuchs-Statistik (alle Zellen):** 275 Laeufe ok, 24 Fehler, 23 ungueltig, 9 weiche Timeouts, 0 harte.

## Offene Grenzen (im Ergebnis zu nennen)

1. **Hinweistext F4 (b) an den fuenf nicht erneut gemessenen `cg`-Einheiten** (Kipf, NLinear, SGC, DLinear, DGI): ihr Prompt hat sich geaendert (`missing`-Hinweis), ihre G0-Zahl gilt ohne Messung der Textaenderung. Sie gehen mit den G0-Zahlen ins Aggregat (NLinear, SGC, DLinear, DGI zusammen `cg` 8 und `heute` 4 von je 12; Kipf `cg` 1 und `heute` 0 von je 3). Gemessen wurden Veličković-GAT, Shchur-GAT, Shchur-GCN, RLinear (vier von neun).
2. **DGI** hat in G0 Timeouts gesehen und wurde in G1 nicht erneut gemessen: ein Effekt von F2/F3 auf DGI ist nicht gemessen.
3. **Messweg anders als G0** (Nebius statt lokales Docker, 4 vCPU/4 GB, andere Zeiten, Zeitlimits aus der Referenz auf Nebius: V 660 s, Shchur-GCN 900 s, sonst 600 s). Zahlen nicht streng mit G0 vergleichbar.
4. **Niedrige Huerde und In-Sample-Fixes** (siehe oben). Das Gate sagt nicht, dass Code-Grounding allgemein die Accuracy hebt; auf dem unabhaengigen Held-out (RLinear) gibt es `cg` 2 von 3 gegen `heute` 0 von 3 (G0), also keine Verbesserung durch G1, nur Nichtverschlechterung.
5. **Fehler im Fix F2 (a), erst nach dem Lauf gefunden:** `faulthandler` listet die innersten Frames zuerst, die Implementierung hat den Stack aber vom Ende gekuerzt (1 200 Zeichen). Bei langen Stacks gingen die innersten Frames verloren, und der Satz „Skript stand in `<Funktion>` bei `train.py` Zeile N“ zeigte dann auf einen aeusseren Frame (z. B. `train` statt `forward`). Die Frames der Abdeckungsmessung (`coverage`) fuellten den Platz zusaetzlich. Betroffen sind die 9 weichen Timeouts dieses Laufs (Seed 0 laeuft unter `coverage`). Der Fehler ist in einem separaten Commit nach dem Lauf behoben (`trim_stack`: vom Anfang kuerzen, `coverage`-Frames weglassen); das Ergebnis oben wurde mit der fehlerhaften Fassung gemessen und wird nicht neu berechnet. Wie stark der Fehler die Reparaturen der Timeout-Ketten beeinflusst hat, ist nicht messbar (der Reparaturtext selbst wird nicht gespeichert, nur der Stack im Versuchsprotokoll).
6. **Kein G2.** Wiederholungen sind vorab ausgeschlossen.

## Kosten und Aufwand

- **0,6053 $ brutto** (85 Modell-Aufrufe: 4 Plaene, 81 Code-Aufrufe davon 12 erste Codes und 69 Reparaturen), Deckel 1,00 $ nicht erreicht; Schaetzung vorab 0,4 bis 0,7 $. Eigene `spend.jsonl`, G0-Verbrauch zaehlt nicht mit.
- Laufzeit ca. 58 Minuten (Schaetzung 1,5 bis 3 Stunden); letzter Modell-Aufruf 02:04, die letzten Proben (Shchur-GAT `heute`, 30 Splits) liefen bis 02:25.
- Konsolenabgleich der Token-Factory-Kosten bleibt offen (Log ist eine Schaetzung).

## Nachtraege

- *(G0-Endergebnis hier eintragen, sobald die letzten vier Shchur-GAT-Proben in `cto` fertig sind.)*
