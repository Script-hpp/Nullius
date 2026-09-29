# Tavily v2: Entwicklungsmenge, Prüfmenge und Auswahl des Prüffalls

Festgelegt **bevor irgendein Agent auf dem neuen Prüffall läuft** (auf DGI lief bisher nur die menschliche Referenzlösung, ohne Tavily-Rolle). Vorab-Datei für die Auswahl; die Regeln des Vergleichs (Version 1 gegen Version 2) folgen in einer eigenen Datei, bevor die Läufe starten.

## Warum getrennt
Version 1 (Benchmark, 48 Urteile) bleibt eingefroren im README. Verbesserungen an Tavily (v2) dürfen sich nicht an denselben Papers messen, an denen sie entwickelt wurden; sonst beweist ein besseres Ergebnis wenig (Zuschnitt auf den Benchmark, wie beim Nachbessern des Prüfers an seinen Eichcodes).
- **Entwicklungsmenge (dort darf ich bauen, zählt nicht als Beleg):** Kipf & Welling (GCN), Wu (SGC), Zeng (DLinear), Shchur (GCN auf zufälligen Splits).
- **Prüfmenge (dort wird v2 nur gemessen):** **DGI** (sauber: bisher kein Agentenlauf, keine Rolle-4-Details gelesen) und **GAT** (nur halb sauber: der Live-Lauf lief schon, ich habe die Setup-Details und Seed-Werte gesehen und die Beobachtung „11 von 24 Details aus Repo-Dateien“ fließt in den Plan von v2 ein). GAT wird deshalb als „halb sauber“ berichtet, DGI als der eigentliche Prüffall.

## Kriterien für den Prüffall und Ergebnis der Prüfung (ein Tavily-Extract, Fixture `benchmark/fixtures/1809.10341.json`)
1. **Zielwert steht im Paper selbst:** ja. Tabelle 2 des DGI-Papers: „X, A DGI (ours) 82,3 ± 0,6 %“ (Cora), Mittel nach 50 Läufen des Trainings mit anschließender logistischer Regression. (Die 82,3 aus der SGC-Tabelle war nur der Hinweis, wo man suchen soll.)
2. **Paper nutzt den Standard-Split:** Tabelle 1 nennt für Cora Train/Val/Test **140/500/1000** (transduktiv); die Zahlen entsprechen dem Split unseres Cora-Tasks. Das Paper nennt den Split nicht „Planetoid“ (es zitiert Planetoid als Vergleichsverfahren); dass es derselbe Standard-Split ist, ist eine Annahme aus den gleichen Größen.
3. **Ein Seed läuft deutlich unter 10 Minuten auf CPU:** ja, gemessen mit einer menschlichen Referenzlösung (`tests/reference/dgi_cora_reference.py`, Ein-Schicht-GCN-Encoder, 512 Einheiten, PReLU, Zeilenmischen als Korruption, Geduld 20, dünn besetzte Adjazenz): **32 bis 52 Sekunden je Seed** durch die Pipeline (inklusive Vorbereitung), Accuracy 0,810 / 0,813 / 0,806 (Mittel 0,8097, Abstand 0,013 zum Paperwert). Mit dichter Adjazenz wäre es deutlich langsamer; ein Agent, der dicht rechnet, kann in das 10-Minuten-Limit laufen.

**MoNet wurde nicht extrahiert** (Tavily-Credits gespart); es bleibt wegen der Gauß-Kerne über Pseudo-Koordinaten als Implementierungstest eher ungeeignet.

## Einschränkung (offen genannt)
Alle Prüffälle laufen wie die Entwicklungsmenge auf **Cora**. Eine Verbesserung durch v2 belegt, dass Tavily bei **neuen Methoden auf bekannten Daten** hilft, nicht bei neuen Datensätzen oder Gebieten.

## Weiteres
- Papers-Eintrag: `velickovic_dgi_2019` mit Echt- und Placebo-Claim in `benchmark/papers.yaml` (10 Seeds statt 50 Läufe; Toleranz ±0,01 wie bei allen Accuracy-Claims).
- Zielwert-Prüfung: Die Spaltenprüfung erkennt die DGI-Zeile (Cora Position 1 von 3, ohne LLM geprüft). Dabei wurde ein Fehler der Spaltenprüfung behoben (Kopf mit Zeilennamen „Raw features“ wurde als fünf Spalten gelesen).
- Für GAT und DGI müssen Referenzurteile vor den Läufen stehen (GAT-Referenzlösung fehlt noch).

## Nachtrag: Referenzlösungen der Prüfmenge (ohne LLM, vor jedem v2-Lauf; `results/cora_references_holdout.txt`)
| Paper | Referenz (Mittel, 10 Seeds) | Zielwert ± Toleranz | Abstand | Referenzurteil |
|---|---|---|---|---|
| GAT (`tests/reference/gat_cora_reference.py`) | **0,8280** (Std 0,0050) | 0,830 ± 0,01 | 0,002 | reproduziert |
| DGI (`tests/reference/dgi_cora_reference.py`) | **0,8047** (Std 0,0077) | 0,823 ± 0,01 | 0,0183 | teilweise |

- **GAT:** Die 83,0 % sind im Harness auf CPU erreichbar: Eine sorgfältige menschliche Referenz nach Paper und Original-Code (8 Köpfe × 8 Merkmale, ELU, Dropout 0,6 auf Eingaben und Attention-Koeffizienten, L2 5e-4, Adam 0,005, Early Stopping auf Validierungsloss und -accuracy mit Geduld 100) landet bei 0,828. Die Lücke des Agentenlaufs (0,8036) liegt damit am Agenten-Code, nicht an Protokoll oder Umgebung. GAT ist als Prüffall brauchbar (halb sauber, siehe oben).
- **DGI:** Meine Referenz erreicht den Paperwert nicht ganz (0,8047 gegen 0,823, Referenzurteil „teilweise“). Das Referenzurteil ist damit „teilweise“, nicht „reproduziert“; ein Agent, der „teilweise“ erreicht, stimmt überein. Ich habe die Referenz nicht nachträglich auf den Paperwert hingetrimmt (Lernschritte der logistischen Regression stehen nicht im Paper-Text). Ob die Lücke an meiner Referenz oder am Protokoll liegt, ist offen; das ist eine Schwäche des DGI-Prüffalls und wird so berichtet.
