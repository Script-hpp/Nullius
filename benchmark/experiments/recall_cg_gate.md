# Code-grounded Gate G0 (Vorab-Festlegung)

Stand 2026-10-08, Branch `Script-hpp/cto`. Diese Datei, `scripts/review/cg_gate.py` (Zaehlregel `classify`/`hit`, Entscheidungsregel `decide`, Aufruf-Kostenbremse), `scripts/review/gate_sandbox.py`, der Ausgabe-Helfer (`src/repro_agent/harness/nullius_io.py`, Flag `io_helper`) und `tests/test_io_helper.py` werden **vor dem ersten Modell-Aufruf** committet. Danach aendern sich Einheiten, Arme, Proben, Regeln und Grenzen nicht mehr. Bauplan: `docs/recall_analysis_3.md` Abschnitt 4 Hebel 3 und 4, Abschnitt 5 (Gate G0). Vorgaenger: Pilot `recall_cg_pilot.md` / `results/recall_cg_pilot.md` (bestanden, aber 9 von 16 Proben endeten als Setup-Ausfall in beiden Armen, ueberwiegend am Ein-/Ausgabevertrag).

**Einordnung (wichtig):** Das Gate misst, ob `code_grounded` die Accuracy hebt, **bevor** Geld fuer Plan A/B ausgegeben wird. Die acht Entwicklungseinheiten sind In-Sample (mehrfach von Hand analysiert, `code_grounded` ist auf ihre Fehlerbilder gebaut); RLinear ist die einzige Held-out-Einheit (im Bauplan `docs/recall_analysis_3.md` als solche vorgesehen; nicht Teil der Fehleranalysen, auf denen `code_grounded` gebaut ist; sie kam allerdings im Nachlauf-Vorversuch vor). Keine Benchmark-Zahl, nichts davon ins README oder Video. Modus: Nachbau aus Paper **und** Autorencode, unabhaengig gemessen (Entscheidung der Nutzerin/des Nutzers vom 2026-10-08).

## Frage

Treffen Generierungen das Referenzurteil deutlich oefter, wenn der Agent den entscheidenden Autorencode **roh** sieht (`code_grounded`) und die Zahlenwerte per `CONFIG` deterministisch erzwungen werden, **nachdem** die haeufigste Ausfallklasse des Pilots (Vertragsfehler bei Ausgabe und Smoke-Teilmenge) fuer beide Arme beseitigt ist?

## Aenderungen seit dem Pilot (beide Arme gleich, kein Paper-Wissen, nichts in Richtung Zielwerte)

1. **Ausgabe-Helfer (Hebel 4, Flag `io_helper`, Default aus):** `harness/nullius_io.py` liegt im Branch neben `train.py`. Der Agent liefert Arrays (`nio.write_predictions(pred)` bzw. `(test_pred, val_pred)`), der Helfer schreibt `predictions.csv` / `val_predictions.csv`, das Id-Schema und die SMOKE-Teilmengen (Cora: ersten 20 sortierten Test-/Val-Knoten, ETT: Fenster 0..7); `nio.load()` liefert die Arrays der Datendatei, `nio.SMOKE` das Flag. Knoten-Tasks: Vorhersage fuer **alle** N Knoten (oder Logits), kein Ausschneiden auf `test_idx`; Torch-Tensoren und Numpy-Arrays werden akzeptiert; falsche Formen liefern eine eindeutige Fehlermeldung. Der Helfer kennt nur Dateiformat und Id-Schema des Tasks. Der Code-/Reparatur-Prompt ersetzt den alten Ausgabeabschnitt, die Smoke-Anweisung und den Val-Hinweis durch den Helfer-Vertrag (`Task.with_io_helper`); der **Plan-Prompt** bleibt unveraendert. Flag aus: Prompts und Branch byte-identisch (getestet). Messung (Harness, Metrik, Toleranz) unveraendert. Validiert in `tests/test_io_helper.py` und mit Helfer-Varianten von GCN (Cora) und NLinear (ETTh1) durch den echten Messweg samt Smoke-Pruefung und Val-Auswertung (Seeds 0 und 1: gueltig).
2. **Speicher-Hinweis bei Reparatur:** meldet der Lauf „Killed“ ohne Traceback oder „can't allocate memory“, bekommt `repair_code` einen festen Hinweis (dichte Knoten x Knoten x Koepfe-Tensoren vermeiden). Gilt fuer beide Arme.
3. **Messweg (`GateSandbox`, nur Infrastruktur):** lokale Docker-Container, **vier Kerngruppen zu je vier Kernen** (`docker update --cpuset-cpus` auf die Gruppe, solange ein Kommando laeuft; hoechstens vier Kommandos gleichzeitig, also <= 16 Threads), **12 GB Speicher je Container**, Container waehrend des Trainings vom Netz getrennt (statt `unshare -n`, das im Container Rechte braucht), Harness frisch mit Netz fuer die Testlabels. `LocalSandbox.run` beendet bei Timeout alle Prozesse im Container (`kill -9 -1`): im Pilot rechnete ein per Timeout abgebrochener `docker exec` weiter und stahl den naechsten Laeufen die Kerne. Der Branch wird nach jedem Versuch entfernt.
4. **Zeitlimit je Seed (generisch):** `timeouts.train_timeout(ref_s) = max(600, 10 x ref_s)`, gedeckelt bei 3600 s, ref_s = langsamster Seed der Referenzloesung dieser Einheit **auf diesem Messweg**. Alle Einheiten ergeben 600 s (Referenz hoechstens 54 s), siehe Tabelle. Der Pilot-Befund „Training > 600 s“ bei GAT bleibt in der sauberen Neumessung bestehen (3 von 3 Proben): der Agenten-Code (Python-Schleifen ueber Knoten) ist zu langsam, kein Infrastrukturfehler. Die Reparaturschleife des Produkts (Timeout-Hinweis) behandelt das.

### Beweis des Messwegs (Schritt 2, vor jedem Modell-Aufruf, kein LLM)

`tests/reference`-Loesungen durch den genauen Messweg (GateSandbox, Produkt-`_attempt`: statische Pruefung, Training, frischer Harness, Urteil des Orchestrators, volle Seed-Zahl). Die Formpruefung entfaellt, weil die Referenzen kein SMOKE kennen (nicht Agenten-Code).

| Einheit | gueltige Seeds | Mittel (Messweg) | Referenzmittel (frueher) | Urteil (Messweg) | Referenzurteil | langsamster Seed s | Zeitlimit s |
|---|---|---|---|---|---|---|---|
| velickovic_gat_2018 | 10/10 | 0.8280 | 0.828 | reproduziert | reproduziert | 45 | 600 |
| kipf_gcn_2017 | 10/10 | 0.8136 | 0.8136 | reproduziert | reproduziert | 16 | 600 |
| wu_sgc_2019 | 10/10 | 0.8081 | 0.8081 | reproduziert | reproduziert | 3 | 600 |
| zeng_nlinear_2023 | 5/5 | 0.3739 | 0.3739 | reproduziert | reproduziert | 6 | 600 |
| rlinear_2023 | 5/5 | 0.3704 | 0.3704 | reproduziert | reproduziert | 14 | 600 |
| zeng_dlinear_2023 | 5/5 | 0.3809 | 0.3809 | reproduziert | reproduziert | 8 | 600 |
| velickovic_dgi_2019 | 10/10 | 0.8037 | 0.8047 | teilweise | teilweise | 52 | 600 |
| shchur_pitfalls_2018 | 30/30 | 0.8107 | 0.8107 | reproduziert | reproduziert | 54 | 600 |
| shchur_gat_2018 | 30/30 | 0.8140 | 0.814 | reproduziert | reproduziert | 22 | 600 |

Alle neun Referenzen enden mit gueltigen Seeds, unter dem Zeitlimit und mit dem frueher bekannten Urteil (DGI: 0,8037 statt 0,8047, Abstand 0,001 durch Thread-Zahl, gleiche Klasse). DLinear gilt mit `LR_HALVE=1` (Lernrate pro Epoche halbiert, `etth1_harness_reference.txt`: 0,3809); ohne den Schalter misst derselbe Weg 0,4278, genau der frueher dokumentierte Wert „konstante Lernrate“. Rohdaten: `results/recall_cg_gate_prep/reference.jsonl`. Ein Seed laeuft hoechstens 54 s; die Pilot-Messung des Agenten-Codes vorher: NLinear `heute` T=0,2 liefert im neuen Weg exakt dasselbe Mittel wie im Pilot (0,3856).

**Neumessung der 9 Setup-Ausfaelle des Pilots** (Code unveraendert, endgueltiger Code der Probe = Reparatur, falls vorhanden; Pilotergebnis bleibt wie aufgezeichnet): 
| Einheit | Arm | T | Pilot (as run) | neu gemessen | Seed 0 | letzte Fehlerzeile |
|---|---|---|---|---|---|---|
| kipf_gcn_2017 | cg | 0.2 | nicht reproduzierbar (Setup) | nicht reproduzierbar (Setup) | fehler | `IndexError: index 1708 is out of bounds for axis 0 with size 1000` |
| kipf_gcn_2017 | cg | 0.8 | nicht reproduzierbar (Setup) | nicht reproduzierbar (Setup) | fehler | `NameError: name 'test_idx' is not defined. Did you mean: 'test_ids'?` |
| kipf_gcn_2017 | heute | 0.8 | nicht reproduzierbar (Setup) | nicht reproduzierbar (Setup) | fehler | `Smoke run (SMOKE=1) output has the wrong form: expected 20 unique ids, got 1000 rows (1000 unique). ` |
| velickovic_gat_2018 | cg | 0.2 | nicht reproduzierbar (Setup) | nicht reproduzierbar (Setup) | fehler | `AttributeError: 'numpy.ndarray' object has no attribute 'cpu'` |
| velickovic_gat_2018 | cg | 0.8 | nicht reproduzierbar (Setup) | nicht reproduzierbar (Setup) | timeout | `Training > 600s` |
| velickovic_gat_2018 | heute | 0.2 | nicht reproduzierbar (Setup) | nicht reproduzierbar (Setup) | timeout | `Training > 600s` |
| velickovic_gat_2018 | heute | 0.8 | nicht reproduzierbar (Setup) | nicht reproduzierbar (Setup) | timeout | `Training > 600s` |
| wu_sgc_2019 | cg | 0.8 | nicht reproduzierbar (Setup) | nicht reproduzierbar (Setup) | fehler | `IndexError: index 1708 is out of bounds for dimension 0 with size 1000` |
| wu_sgc_2019 | heute | 0.2 | nicht reproduzierbar (Setup) | nicht reproduzierbar (Setup) | fehler | `IndexError: index 1708 is out of bounds for axis 0 with size 1000` |

**Alle neun bleiben Setup-Ausfaelle**: sechs am Vertrag/Code (dreimal `IndexError … size 1000`, falsche Smoke-Form, `.cpu` auf Numpy, `NameError`), drei bei GAT durch echte Langsamkeit des Agenten-Codes (`Training > 600s`, Python-Schleifen ueber Knoten; die Referenz braucht 45 s). Das Pilotergebnis war also **kein Messartefakt**; vor der Neumessung wurde bei der Speicherobergrenze (5 g, dann 8 g) ein GAT-Smoke-Lauf mit einer Spitze ueber 7 GB getoetet, mit 12 g laeuft er bis zum Vertragsfehler. Der Ausgabe-Helfer adressiert die sechs Vertragsfehler (bis auf den `NameError`, einen reinen Code-Fehler), die Produkt-Reparatur den Timeout. Rohdaten: `results/recall_cg_gate_prep/remeasure.jsonl`.

## Einheiten (9) und Referenzurteile

Klasse des Zahl-Urteils (Achse 1) gegen das **vorab festgelegte Referenzurteil** (Quelle `benchmark/experiments/benchmark_v2.md`, RLinear `results/b1_reference_rlinear.txt`; Toleranz je Einheit 0,01; Klassen: <= Toleranz reproduziert, <= 3 x teilweise, sonst nicht reproduziert):

| Einheit | Gruppe | Claim | Seeds | Referenzmittel | Referenzurteil |
|---|---|---|---|---|---|
| Zeng NLinear | Entwicklung (v4 B: 0/3) | `nlinear_etth1_96_mse` | 5 | 0,3739 | reproduziert |
| Wu SGC | Entwicklung (v4 B: 0/3) | `sgc_cora_acc` | 10 | 0,8081 | reproduziert |
| Zeng DLinear | Entwicklung | `dlinear_etth1_96_mse` | 5 | 0,3809 | reproduziert |
| Veličković GAT | Entwicklung (v4 B: 0/3) | `gat_cora_acc` | 10 | 0,8280 | reproduziert |
| Veličković DGI | Entwicklung | `dgi_cora_acc` | 10 | 0,8047 | **teilweise** (Abstand 0,0183) |
| Shchur GAT (Cora-LCC) | Entwicklung | `shchur_gat_cora_acc` | 30 Splits x 1 | 0,8140 | reproduziert |
| Kipf GCN | Nicht-Verschlechterung | `gcn_cora_acc` | 10 | 0,8136 | reproduziert |
| Shchur GCN (Cora-LCC) | Nicht-Verschlechterung | `shchur_gcn_cora_acc` | 30 Splits x 1 | 0,8107 | reproduziert |
| Zeng RLinear | **Held-out (neu)** | `rlinear_etth1_96_mse` | 5 | 0,3704 | reproduziert |

„Entwicklung“ sind die sechs Einheiten der Regel (a): NLinear, SGC, DLinear, GAT, DGI, Shchur-GAT. DGI zaehlt nur als Treffer, wenn die Probe **„teilweise“** liefert (wie in Benchmark v2 bis v4: gleich Referenzurteil, nicht „Zielwert erreicht“). Beschreibend wird zusaetzlich gemeldet, ob der Zielwert innerhalb der Toleranz liegt.

## Arme (je Einheit)

Beide Arme: Produktkonfiguration wie Benchmark v4 (`graph_repair=True`, `setup_v2`), `io_helper=True`, gespeicherte belegte Setup-Details und Paper-Text aus dem Dossier (kein Tavily), Repo-Dateien aus lokalen Klonen am Commit des gespeicherten Laufs ueber den echten `read_repo` (Dateiwahl des gespeicherten Laufs als Proxy). Kein Tavily-, kein Nebius-Aufruf.

1. `heute`: Produktpfad wie Benchmark v4 Arm B (Rolle-4-Zitate im Prompt, Konflikthinweis, Plan, Code).
2. `cg`: `code_grounded=True` wie im Pilot (numerischer Steckbrief S1 mit einem Extraktionsaufruf je Einheit, CONFIG-Pflicht mit S2-Override, **roher** fokussierter Repo-Ausschnitt S4 (<= 15 000 Zeichen) in Plan-, Code- und Reparatur-Prompt, argparse-Defaults entfernt, wo ein Startskript das Flag setzt, Modellnamen-Regel).

**Proben:** je Einheit und Arm **1 Plan** (Rolle `reasoning`) und **3 Code-Proben** (Temperatur 0,2 / 0,5 / 0,8, gleicher Plan, wie die drei Kandidaten des Produkts). Insgesamt 18 Plaene, 54 Code-Proben, 9 S1-Aufrufe (nur `cg`).

**Reparaturen wie im Produkt, in beiden Armen** (Produktfunktionen): (i) nur `cg`: CONFIG-Befunde (`_spec_conform`, deterministische Pruefung, hoechstens `MAX_SPEC_REPAIRS` = 2 Reparaturen, danach S2-Override); (ii) Seed 0 mit Reparaturschleife (`_run_seed`: Formpruefung `SMOKE=1`, statische Pruefung, Training, Harness; **bis zu `MAX_ATTEMPTS_PER_CLAIM` = 4 Versuche**, also bis zu 3 Reparaturen, Timeout-Hinweis bei Zeitueberschreitung); (iii) die uebrigen Seeds laufen mit dem Code von Seed 0 parallel (4 gleichzeitig), jeder Seed ebenfalls ueber `_run_seed`. **Nicht enthalten, beide Arme gleich:** Kandidatenauswahl, Fehler- und Graph-Reparatur nach Seed 0, Repo-Abgleich, Fehlersuche per Tavily. Das Gate unterschaetzt das Produkt, aber in beiden Armen gleich.

## Modell, Kosten, Nebenbedingungen

- **Anbieter:** Token Factory (`LLM_PROVIDER=tokenfactory`), **Super (`nvidia/nemotron-3-super-120b-a12b`) fuer alle Rollen** (`TF_MODEL_FAST`, `TF_MODEL_CODE`, `TF_MODEL_REASONING` alle auf Super), **kein Ultra**, `LLM_CACHE=off`. Nur oeffentliche Papers und Repos im Prompt. `.env` und `logs/` zeigen auf den Haupt-Checkout.
- **Harte Kostengrenze 2,00 $ brutto** fuer das gesamte Gate (`CG_GATE_COST_CAP_GROSS_USD=2.00`), auf Aufruf-Ebene erzwungen: vor jedem Aufruf wird die Summe aus `out/spend.jsonl` (netto x 1,19, ueber alle Prozesse und Wiederaufnahmen) geprueft, bei Erreichen Stopp mit Teilergebnis (`CostStop`). Schaetzung aus den Pilot-Aufrufen (Reparatur ca. 0,007 bis 0,010 $ brutto, `cg` mit rohem Ausschnitt teurer): ca. 1,0 bis 1,6 $ brutto, unter der Grenze. Wird die Grenze trotzdem erreicht, ist das Gate **unvollstaendig** (weder bestanden noch nicht bestanden) und wird gemeldet, nicht stillschweigend gewertet.
- **Parallelitaet:** bis zu 8 gleichzeitige Modell-Aufrufe; jede Probe wird gemessen, sobald ihr Code vorliegt; Messung hoechstens vier Kommandos gleichzeitig auf je vier Kernen (Summe <= 16 Threads).
- **Messung:** lokales Docker (`python:3.11-slim`, CPU) mit unserem Harness (Testlabels frisch aus dem Original-Datensatz) und unserer Metrik, **volle Seed-Zahl** der Einheit (5, 10 oder 30). Agenten-Code laeuft nie auf dem Host. Zeitlimit 600 s je Seed (alle Einheiten, Tabelle oben).
- **Wiederaufnahme:** alle Zwischenstaende (Steckbrief, Plaene, Code, `final_*.json`) liegen in `out/`. Ein Absturz der Infrastruktur (Docker, Netz) wird durch denselben Befehl fortgesetzt; es entsteht keine Anfrage doppelt, soweit die Datei schon existiert.

## Zaehlregel

Klasse je Probe = Urteil des Orchestrators (`Orchestrator._verdict`: `judge(Mittel, Ziel, Toleranz)` samt Agenten-Fehler-Pruefung gegen die Baselines) ueber den Mittelwert der gueltigen Seeds; mindestens die Haelfte der Seeds muss einen gueltigen Wert liefern, sonst „nicht reproduzierbar (Setup)“. **Treffer** = Klasse gleich dem Referenzurteil der Einheit. „Agenten-Fehler vermutet“, jede andere Klasse, Setup-Ausfall und fehlender Code sind keine Treffer; ein Ausfall zaehlt als Fehlschlag.

**Code-Aufruf „leer oder abgeschnitten“ (Regel d):** ein `cg`-Aufruf von `write_code` (je Probe einer, mit seinen internen drei Wiederholungen), der **keinen Code** liefert (kein Codeblock oder `finish_reason: length` in allen Versuchen). Gezaehlt werden die 27 Code-Aufrufe des Arms `cg`; Reparaturaufrufe zaehlen nicht.

## Entscheidungsregel (vorab; aus `docs/recall_analysis_3.md` Abschnitt 5, auf 3 Proben je Einheit und Arm angepasst)

Das Gate **besteht**, wenn alle vier gelten:

- **(a)** `cg` trifft auf den sechs Entwicklungseinheiten **mindestens 8 von 18** Proben **und** mindestens `heute` + 5;
- **(b)** Kipf + Shchur-GCN (6 Proben je Arm): `cg`-Treffer >= `heute`-Treffer - 1;
- **(c)** RLinear: `cg` trifft **mindestens 2 von 3** Proben (Klasse „reproduziert“, also im Toleranzband);
- **(d)** hoechstens **2 von 27** `cg`-Code-Aufrufen enden leer oder abgeschnitten.

Sonst: **nicht bestanden**, Stopp, Bericht „Code-Grounding hat die Accuracy im Gate nicht angehoben“ ohne Geld fuer Plan A/B; v4 bleibt. Ein unvollstaendiges Gate (nicht alle 54 Proben gemessen, Kostengrenze) ist weder bestanden noch nicht bestanden und wird als solches gemeldet. Es gibt keine nachtraegliche Anpassung der Schwellen.

**Grenzen der Regel:** 3 Proben je Einheit und Arm teilen einen Plan, sind also nicht unabhaengig; 18 Proben je Arm auf den Entwicklungseinheiten sind ein Signal, kein Test. Seeds und Modellantworten schwanken; ein Neulauf kann anders ausgehen. Die Entwicklungseinheiten sind In-Sample; nur RLinear (3 Proben) ist held-out. Ein Bestehen gibt keine Wirksamkeitsaussage, sondern die Empfehlung an die Nutzerin/den Nutzer, Plan A (arm B only, product configuration + code-grounding) zu bezahlen.

## Zusaetzlich berichtet (beschreibend, nicht Teil der Regel)

Je Probe: Klasse, Mittel, Std, gueltige Seeds, Versuche bei Seed 0, Dauer, ob der Zielwert innerhalb der Toleranz liegt, Fakt-Checks (`uptake_probe.FACTS`, Heuristik). Treffer je Einheit und Arm. Aufrufzahl und Kosten. Die Setup-Ausfaelle werden nach Fehlerklasse (Vertrag, Zeit, Speicher, Sonstiges) aufgeschluesselt.

## Bekannte Abweichungen vom Produkt und von der Analyse (offen genannt)

- Keine Kandidatenauswahl (die drei Proben sind drei einzelne Proben, keine Auswahl), keine Graph-Ablation oder Fehler-Reparatur nach Seed 0, kein Repo-Abgleich: unterschaetzt das Produkt in beiden Armen.
- Analyse: „hoechstens 1 Smoke-Reparatur“; hier Produkt-Reparaturschleife (bis 4 Versuche) auf Anweisung der Koordination.
- Der Arm `heute` nutzt die gespeicherten Rolle-4-Details des v4-Laufs (RLinear: neuester gespeicherter Lauf), nicht eine neue Extraktion; alle Rollen laufen auf Super statt Ultra beim Plan.
- Die Dateiwahl des Repos ist die des gespeicherten Laufs. Entscheidende Angaben, die in keiner gelesenen Datei stehen (z. B. das getunte SGC-Weight-Decay in einer Pickle-Datei), kann auch `cg` nicht liefern.
- Messweg lokal (Docker, vier Kerngruppen) statt Nebius; Laufzeiten sind nicht mit Nebius vergleichbar.

## Reihenfolge

1. Diese Datei und alle Skripte/Tests committen (**vor dem ersten Modell-Aufruf**).
2. Gate starten: `LLM_PROVIDER=tokenfactory TF_MODEL_FAST=… TF_MODEL_CODE=… TF_MODEL_REASONING=… LLM_CACHE=off uv run python scripts/review/cg_gate.py --run --main <Checkout mit runs/> --clones <Klone> --out benchmark/experiments/results/recall_cg_gate_samples`.
3. Auswertung `--score`, Ergebnis nach `benchmark/experiments/results/recall_cg_gate.md`, committen, nicht pushen, nicht mergen.
