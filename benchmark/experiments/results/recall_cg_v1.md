# recall cg v1: Ergebnis

Lauf 2026-10-09, Start Stufe 1 (zweiter Anlauf) 02:02Z, Ende 05:49Z (ca. 3 h 47 min), Branch `Script-hpp/recall-cg-v1-run`, Vorab-Festlegung `../recall_cg_v1.md` (Commit `1697458`, vor jedem Modell-Aufruf; Nachtrag 1 `568b9f5`, vor dem Neustart). Rohdaten dieses Laufs (Store, Gate, Planer-Log, Infrastruktur-Abbrueche): `recall_cg_v1_data/`. Fixtures der Vorbereitung: `../recall_cg_v1_fixtures/`. Auswertung: `uv run python scripts/eval_recall_cg_v1.py`. **method fidelity not checked.** Nichts davon ins README oder Video, bevor die Nutzerin/der Nutzer entschieden hat.

## Entscheidung nach den vorab festgelegten Regeln

- **Stufen-Gate: bestanden** (F = 4, U = 4; Regel F ≥ 2 und F + U ≥ 4). Stufe 2 lief vollstaendig, **kein Kostenstopp, nicht pausiert**, 16 + 6 Laeufe.
- **Lesregel (`recall_cg_v1.md`, „Zaehlregel und Kennzahlen“):** S1 (Entwicklung ≥ 4 von 8): **erfuellt (7 von 8)**. S2 (RLinear 1 von 1): **NICHT erfuellt (0 von 1)**. S3 (kein Placebo „reproduziert“, DGI nicht mehrheitlich „reproduziert“): **erfuellt**. Damit ist die Aussage „Recall verbessert“ **nach der vorab festgelegten Regel nicht erlaubt**.
- Erlaubt ist nur: „Im Modus Nachbau aus Paper und Autorencode (Super in allen Rollen) stimmen 7 von 8 Entwicklungseinheiten (In-Sample, Fixes nach Sicht der Daten entworfen) mit dem Referenzurteil ueberein (v4: 2 von 8, anderer Modus); auf dem einen Paper, auf das nichts abgestimmt wurde (RLinear), stimmen 0 von 1 ueberein; kein Placebo wurde bestaetigt (0 von 9). Method fidelity not checked.“ Die drei Zahlen werden nie summiert oder zu einem Gewinn gegenueber v4 verrechnet.

## Was dieser Lauf nicht ist (Grenzen, vorab genannt und hier bestaetigt)

1. **In-Sample.** Die 7 von 8 stehen auf den Einheiten, nach deren Sicht die Fixes entworfen wurden. Das einzige Held-out verfehlt das Urteil in beiden Generationen.
2. **Anderer Modus und andere Modelle als v4.** „Nachbau aus Paper und Autorencode“, in allen Rollen Super (Nachtrag 1: Ultra wurde von Token Factory entfernt; in keinem gezaehlten Lauf verwendet). Nicht unter exakt den v2-Regeln gezaehlt (nur Arm B, Placebo nur in Generation 0, adaptive Generation 2, Rolle 5 nicht live, Kappe je Lauf 0,75 $, Fixtures aus einem Overlay).
3. **Der Gewinn kommt nicht aus „mehr Tavily“.** Tavily lief nicht (Fixtures); die Zahlen sagen nichts ueber den Tavily-Beitrag.
4. **Fixture-Quellen von RLinear** sind gespeichert bzw. aus dem Klon (README), nicht frisch per Tavily.
5. **Ein gezaehlter Lauf**, keine Wiederholung; Einheiten mit Gen 0 ≠ Gen 1 zeigen die Streuung zwischen Generationen (siehe unten).

## Infrastruktur-Vorfall (Nachtrag 1)

Ultra verschwand um ca. 01:52Z aus Token Factory (HTTP 404). Der erste Anlauf (01:29Z) wurde angehalten, Kipf g0 (Ultra-Plan) in die Infrastruktur-Datei verschoben, DGI g0 und SGC g0 brachen ab, fuenf Worker wurden angehalten. **Verlust 1,96 $ brutto** (davon Ultra 0,215 $ netto), gezaehlt im Deckel. Der zweite Anlauf lief ohne Infrastruktur-Abbruch (`recall_cg_v1_data/recall_cg_v1_incidents.jsonl` enthaelt nur die fuenf Eintraege des Haltens vom ersten Anlauf).

## Kosten

- Laut Aufruf-Log seit Zaehlbeginn (2026-10-09T01:21:24Z): **7,45 $ brutto** (Super 6,048 $ netto, Ultra 0,215 $ netto, Lightning 0,001 $ netto); Stopp 8,75 $ und Deckel 9,00 $ **nicht erreicht**.
- Davon verloren (erster Anlauf, nicht in der Auswertung): **1,96 $**. In der Auswertung: **5,49 $ brutto** fuer 22 Laeufe (Stufe 1 = 16 Laeufe 4,09 $; Stufe 2 = 6 Laeufe 1,41 $). Schritt 0 (Vorbereitung): 0,008 $.
- Kontostand Token Factory laut Log (`scripts/cost_report.py`): 13,35 $ brutto seit Stichtag 2026-10-02 von 30,74 $, **Rest 17,39 $**. Die Konsole ist nicht verifiziert (Log ist eine Schaetzung).

## Beobachtungen (post hoc, beschreibend, nicht vorab festgelegt)

- **RLinear (Held-out) verfehlt in beiden Generationen** (0,4871 und 0,4321 gegen Ziel 0,366, Referenz 0,3704), obwohl G0 (cg 3 von 3) und G1 (cg 2 von 3) dort trafen. Unterschiede zu G0/G1, nicht getrennt messbar: voller Produkt-Loop mit drei Kandidaten und Auswahl nach Validierung, Dateiwahl des S4-Auszugs durch `select_files` statt der gespeicherten v4-Liste, README aus dem Klon, kein Ultra-Plan (G1: Super), Steckbrief-Werte je Generation verschieden (Gen 0: epochs 20, seq_len 336; Gen 1: epochs 15, batch_size 16; je ein anderer Teil der Konfiguration landete im erzwungenen Teil). Beide Generationen enden mit „Rueckfall ohne Val-Wert“ bzw. hoher Validierungs-MSE (0,70 bis 0,89), also ohne brauchbares Auswahlsignal.
- **NLinear kippt von „reproduziert“ (Gen 0: 0,3744) zu „nicht reproduziert“ (Gen 1: 0,4131; Gen 2: 0,4040)** — die Einheit verfehlt per Mehrheit. In allen drei Generationen erzwingt der Steckbrief lr 0,005, epochs 10, batch_size 32, patience 7, seq_len 336 (aus dem Start-Skript); Gen 1 und 2 landen trotzdem 0,03 bis 0,04 darueber. Ursache nicht untersucht.
- **Veličković-GAT Gen 0 = Ausfall (Setup)** nach vier Versuchen (Formfehler in `index_add_`/`scatter`, zwei Zeitueberschreitungen: eine mit 677 s, eine mit **1097,7 s bei einem Limit von 660 s**, d. h. der weiche Timeout hat dort nicht rechtzeitig beendet); der zugehoerige Placebo hat deshalb **kein Urteil** (8 von 9 Placebos als „nicht reproduziert“ erkannt, 1 ohne Urteil, 0 bestaetigt). Gen 1 (0,8308) und Gen 2 (0,8223) reproduziert; die Einheit steht per Mehrheit (2 von 3) auf „reproduziert“.
- **Shchur-GAT**: Gen 0 „teilweise“ (0,8065), Gen 1 und 2 „reproduziert“ (0,8138; 0,8099). **DGI**: Gen 0 teilweise, Gen 1 reproduziert (0,8191), Gen 2 teilweise; Mehrheit „teilweise“ = Referenz. **Kipf, Shchur-GCN, SGC, DLinear**: beide Generationen reproduziert, Gen 0 = Gen 1.
- **Vier Einheiten brauchten Generation 2** (Shchur-GAT, V-GAT, DGI, NLinear); ihr Urteil haengt an einer einzigen dritten Generation. Die Streuung zwischen Generationen derselben Einheit ist gross.
- **Rohzaehlung je Generation: 15 von 22 Einzelurteile = Referenzurteil**.
- Graph-Ablation: „Graph genutzt“ in 13 von 15 gemessenen Graph-Laeufen (ein „nicht messbar“, ein leer wegen Ausfall).

## Auswertungsskript (unveraendert, Rohausgabe)

- **Entwicklungseinheiten (In-Sample, „code-grounded rebuild“): 7 von 8** stimmen mit dem Referenzurteil überein (v4: 2 von 8).
- **Neues Paper (RLinear, nicht beim Entwurf der Fixes betrachtet): 0 von 1.**
- **Placebos bestätigt („reproduziert“): 0 von 9** (Ziel 0; v2, v3, v4 je 0 von 48).
- Stufen-Gate (nach Stufe 1): F = 4, U = 4, vollständig = True, weiter = True (Regel: F ≥ 2 und F + U ≥ 4).

## Je Einheit (echter Claim; Klassen der Generationen 0/1/2)

| Einheit | Referenzurteil (Mittel) | Klassen Gen 0 / 1 / 2 | Einheitsklasse | Treffer | Mittel je Generation | Placebo (Gen 0) |
|---|---|---|---|---|---|---|
| shchur_gat_2018 | reproduziert (0.814) | teil / rep / rep | reproduziert | ja | 0.8065 / 0.8138 / 0.8099 | nicht (0.8018, Ziel 0.88) |
| shchur_pitfalls_2018 | reproduziert (0.8107) | rep / rep | reproduziert | ja | 0.8106 / 0.8096 | nicht (0.8108, Ziel 0.88) |
| velickovic_gat_2018 | reproduziert (0.828) | Ausfall (nicht reproduzierbar (Setup)) / rep / rep | reproduziert | ja | – / 0.8308 / 0.8223 | Ausfall (nicht reproduzierbar (Setup)) (–, Ziel 0.88) |
| velickovic_dgi_2019 | teilweise (0.8047) | teil / rep / teil | teilweise | ja | 0.8125 / 0.8191 / 0.8127 | nicht (0.8098, Ziel 0.88) |
| kipf_gcn_2017 | reproduziert (0.8136) | rep / rep | reproduziert | ja | 0.8151 / 0.8079 | nicht (0.8144, Ziel 0.88) |
| wu_sgc_2019 | reproduziert (0.8081) | rep / rep | reproduziert | ja | 0.8054 / 0.8054 | nicht (0.8054, Ziel 0.88) |
| zeng_nlinear_2023 | reproduziert (0.3739) | rep / nicht / nicht | nicht reproduziert | nein | 0.3744 / 0.4131 / 0.4040 | nicht (0.3744, Ziel 0.3) |
| zeng_dlinear_2023 | reproduziert (0.3809) | rep / rep | reproduziert | ja | 0.3703 / 0.3703 | nicht (0.3703, Ziel 0.3) |
| rlinear_2023 (neu) | reproduziert (0.3704) | nicht / nicht | nicht reproduziert | nein | 0.4871 / 0.4321 | nicht (0.3725, Ziel 0.3) |

## Sekundär: Rohzählung je Generation (post hoc beschreibend)

- Einzelurteile = Referenzurteil: 15 von 22; „nah“ (reproduziert oder teilweise): 17 von 22.
- Urteilsverteilung: reproduziert 14, nicht reproduziert 4, teilweise 3, nicht reproduzierbar (Setup) 1
- Median Abstand zum Zielwert (gültige Generationen): 0.0054 (n=21)
- Ausfälle nach Ursache (alle Zeilen): {'nicht reproduzierbar (Setup)': 2}
- Graph-Ablation (echte Claims, Graph-Einheiten): {'Graph genutzt': 13, None: 1, 'nicht messbar': 1}
- Belegter Zielwert weicht von `papers.yaml` ab: 0 Fälle 

## Placebos

- shchur_gat_2018: nicht reproduziert (Mittel 0.8018, Ziel 0.88, Referenzurteil Placebo nicht reproduziert)
- shchur_pitfalls_2018: nicht reproduziert (Mittel 0.8108, Ziel 0.88, Referenzurteil Placebo nicht reproduziert)
- velickovic_gat_2018: nicht reproduzierbar (Setup) (Mittel –, Ziel 0.88, Referenzurteil Placebo nicht reproduziert)
- velickovic_dgi_2019: nicht reproduziert (Mittel 0.8098, Ziel 0.88, Referenzurteil Placebo nicht reproduziert)
- kipf_gcn_2017: nicht reproduziert (Mittel 0.8144, Ziel 0.88, Referenzurteil Placebo nicht reproduziert)
- wu_sgc_2019: nicht reproduziert (Mittel 0.8054, Ziel 0.88, Referenzurteil Placebo nicht reproduziert)
- zeng_nlinear_2023: nicht reproduziert (Mittel 0.3744, Ziel 0.3, Referenzurteil Placebo nicht reproduziert)
- zeng_dlinear_2023: nicht reproduziert (Mittel 0.3703, Ziel 0.3, Referenzurteil Placebo nicht reproduziert)
- rlinear_2023: nicht reproduziert (Mittel 0.3725, Ziel 0.3, Referenzurteil Placebo nicht reproduziert)

## Kosten und Code-Stand

- 22 Läufe, 4.617 $ netto, 5.494 $ brutto (Summe der Laufkosten; maßgeblich für die Kappe ist das Aufruf-Log, siehe Bericht).
- Code-Stand (`code_commit`): {'568b9f5': 31}
