# Code-grounded Gate G1 (Vorab-Festlegung)

Stand 2026-10-09, Branch `Script-hpp/gat-analysis`. Diese Datei, `scripts/review/cg_gate_g1.py` (Entscheidungsregel `decide_g1`, Offline-Pruefung, Lauf), die Fixes F1 bis F4 und F8 in `src/repro_agent/{loop,spec}.py` und `scripts/review/{cg_gate,gate_sandbox}.py`, `scripts/review/nebius_probe.py`, die Tests (`tests/test_spec_raw.py`, `tests/test_timeout_exit.py`, `tests/test_cg_gate_g1.py`), `benchmark/experiments/recall_cg_gate_g1_timeouts.json` und die Dateien unter `benchmark/experiments/results/recall_cg_gate_g1_prep/` und `recall_cg_gate_g1_audit.*` werden **vor dem ersten Modell-Aufruf** committet. Danach aendern sich Einheiten, Arme, Proben, Regeln, Schwellen und Zeitlimits nicht mehr. Vorgaenger: G0 (`recall_cg_gate.md`, Nachtrag eingeschlossen), Analyse der GAT-Ausfaelle `docs/gat_failure_analysis.md` (Commit `6f674e3`).

## Einordnung (wichtig)

- **G1 ist der einzige zweite Versuch nach G0.** Es gibt kein G2. Faellt G1 durch, gilt: „Code-Grounding hat die Accuracy im Gate nicht angehoben“, v4 bleibt, kein Geld fuer Plan A/B.
- **G0 bleibt mit seinem tatsaechlichen Ergebnis stehen und wird nicht ersetzt.** Stand dieser Festlegung: **50 von 54 Proben fertig, G0 laeuft noch** (letzte vier Shchur-GAT-Proben im Worktree `cto`, Worker `ctx_bd4cf1c8dec2`). Auf den Einheiten ausser den zwei GAT-Einheiten ist G0 vollstaendig. Das endgueltige G0-Ergebnis wird hier nachgetragen, sobald es vorliegt (Abschnitt „Nachtraege“), und im Ergebnisbericht von G1 immer **zweigeteilt** berichtet („G0 (54 Proben): …; G1 (18 Proben + uebernommene G0-Zahlen): …“).
- **Das Fix-Paket wurde nach Sicht der G0-Ergebnisse der zwei GAT-Einheiten entworfen (In-Sample).** Ein Bestehen von G1 ist deshalb ein **schwaecheres Signal** als ein Bestehen von G0. Es ist ein Signal fuer die Entscheidung, Plan A zu bezahlen, keine Wirksamkeitsaussage.
- **Niedrige Huerde.** Weil G0 auf den vier Nicht-GAT-Entwicklungseinheiten bei `cg` 8 und `heute` 4 Treffer steht, genuegt in G1 auf den zwei GAT-Einheiten ein **Treffer Vorsprung** von `cg` ueber `heute` (Regel (a), Rechnung unten). Bei 6 gegen 6 Proben, die sich je Einheit einen Plan teilen, ist das ein sehr schwacher Test. Das Ergebnis wird so benannt.
- **Der Messweg ist ein anderer als in G0** (Nebius-Sandbox statt lokalem Docker, siehe „Messweg“). Insbesondere hat ein Nebius-Branch nur ca. 4 GB Arbeitsspeicher (G0: 12 GB je GAT-Container). Das ist die Produktbedingung, aber es verschiebt Zahlen gegenueber G0 und kann dichte Aufmerksamkeit (N × N × Koepfe) als Speicherfehler scheitern lassen. Das trifft beide Arme.
- Keine Benchmark-Zahl, nichts davon ins README oder Video. Modus: Nachbau aus Paper und Autorencode, unabhaengig gemessen (wie G0).

## Was sich gegenueber G0 aendert (nur dies)

Alle Fixes sind **generisch** (kein GAT-, Cora- oder Shchur-Sonderfall; Namenstabellen enthalten nur Schluesselnamen, nie Werte oder Methodenwissen). Keine Aenderung an Modellen, Temperaturen, Plan-Prompts der uebrigen Einheiten, Zaehlregel, Schwellen, Reparaturzahl (4), Kostenfuehrung der Rollen. Modell: Super fuer alle Rollen, Token Factory, `LLM_CACHE=off`, wie G0.

| Fix | Wo | Inhalt | Gilt fuer |
|---|---|---|---|
| **F1** Startskript schlaegt Paper | `spec.py: raw_items`, `resolve` | Aus Startskripten und Konfigurationsdateien des Repo-Ausschnitts werden Zuweisungen `name = Zahl`, `name: Zahl` (nur Zeilen ohne Einrueckung) und Shell-Flags `--name Zahl` deterministisch gelesen und per fester Namenstabelle (`KEY_ALIASES`) den Steckbrief-Schluesseln zugeordnet. Mehrere verschiedene Werte desselben Schluessels in einer Datei (mehrere Laeufe in einem Shell-Skript) werden dort nicht gelesen. Die bestehende Rangregel (Startskript > Config > Paper > Code > README > argparse-Default) entscheidet dann, auch wenn der Extraktor den Repo-Wert nicht erfasst hat. | nur `code_grounded` (`numeric_only`) |
| **F2 (a)** Timeout mit Stack | `loop.py: _run_and_eval`, `timeout_report` | Training laeuft als `PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 unshare -n timeout -s ABRT <Limit−15> python train.py`. Bei Zeitueberschreitung schreibt `faulthandler` die Python-Stacks nach stderr (Exit 124 von `timeout`). Der Reparaturtext nennt Funktion und Zeile in `train.py` samt Quellzeile, den Stack und die letzten 400 Zeichen der Ausgabe („Messung“: wo das Skript stand und wie weit es kam). Auf Nebius toetet ein harter Sandbox-Timeout den ganzen Branch (Sondierung), der weiche Timeout davor ist der einzige Weg, Stack und Ausgabe zu behalten. Der harte Timeout bleibt als Rueckfall. | alle Pfade (Training); Reparaturtext nur Produktpfad (`graph_repair`) |
| **F2 (c)** Pflicht-Hinweis | `loop.py: _run_seed`, `STAY_FAST_NOTE` | Nach einem Timeout haengt jede weitere Reparatur desselben Seeds den Hinweis „muss schnell bleiben: keine Python-Schleifen ueber Knoten, Kanten oder Stichproben“ an. | Produktpfad |
| **F3** Exit-Code | `loop.py: describe_exit`, `repair_hint` | Fehlertext endet immer mit `(exit code N)` (am Ende, damit er im 1500-Zeichen-Fenster bleibt); leerer Text wird nie leer weitergegeben („The script printed nothing …“); Exit 137 oder −9 loest den Speicher-Hinweis (`OOM_HINT`) auch ohne das Wort „Killed“ aus. Gilt auch fuer den Smoke-Lauf. | alle Pfade (Text), Hinweis nur Produktpfad |
| **F4 (a)** Quelle ohne `@commit` | `spec.py: normalize_source`, `verify_entries(normalize=True)` | Nennt ein Steckbrief-Eintrag die Quelle ohne `@commit` (oder ohne `repo:`), wird sie auf die eindeutige Dokumentzeile abgebildet (G0: „Quelle unbekannt“ bei Shchur-GAT `layers` und sechs Eintraegen bei RLinear). Mehrdeutig oder unbekannt: unveraendert. Die Pruefregeln werden nur gelockert, nie verschaerft. | nur `code_grounded` |
| **F4 (b)** Hinweistext | `spec.py: format_spec` | „Not stated in any source (choose a standard value …)“ wird, wenn der rohe Ausschnitt im Prompt steht, zu „Not extracted into this spec (if the authors' code excerpt below defines one of them, use that value; otherwise choose …)“. Widersprechen sich Quellen gleichen Rangs ohne Entscheidung, steht im Prompt statt eines Wertes „sources of equal rank disagree (…)“. | nur `code_grounded` |
| **F4 (c)** Konfigurationswerte aus Rohdateien | `spec.py: raw_items`, `resolve`, `paper_numbers` | Wie F1, fuer Konfigurationsdateien (`.yaml`, `.json`, `.conf` …). Eine Modell-Konfiguration (Modellname im Dateipfad) schlaegt die allgemeine (`train.conf`). Widersprechen sich zwei gleichrangige Modell-Konfigurationen (`reference` und `optimized`), gewinnt der Wert, dessen Zahl in einem belegten Paper-Zitat steht, **wenn genau ein Wert so gestuetzt ist**; sonst bleibt der Konflikt ungeloest und nichts wird erzwungen. Die Paper-Zahl wird schluesselunabhaengig verglichen (schwach, offen genannt). | nur `code_grounded` |
| **F8** Versuchsprotokoll | `cg_gate.py: log_attempt` | Je Versuch eine Zeile in `attempts_<arm>_<T>.jsonl` (Status, Sekunden, Exit-Code, letzte 1500 Zeichen Fehlertext, Stack, Code-Hash). Reine Beobachtung. | Gate-Skripte |

**Nicht** in G1: F2 (b) (Schrittzaehler per `sitecustomize`), F5 (getrenntes Versuchsbudget), F6 (statische Schleifenpruefung), F7 (`state_dict()`-Regel). Sie wirken auf andere Einheiten oder beide Arme gleich.

**Aenderung an allen Pfaden (auch Produkt, Benchmark):** der Trainingsbefehl laeuft ueber `timeout -s ABRT`, und Fehlertexte tragen die Zeile `(exit code N)`. F1/F4 gelten nur mit `code_grounded=True`; Benchmark v4 und die Produktkonfiguration ohne dieses Flag sind in Prompts und Steckbrief unveraendert. `tests/` deckt beides ab.

## Offline-Pruefung: welche Einheiten beruehren die Fixes? (`results/recall_cg_gate_g1_audit.md`, kein Modell-Aufruf)

Der deterministische Teil von F1/F4 wurde auf den gespeicherten G0-Steckbriefen aller neun Einheiten ohne neuen Extraktor-Aufruf wiederholt (`spec.rebuild_spec`). Ergebnis:

- **F1:** ein Konflikt Paper gegen Startskript, in Veličković-GAT (`batch_size` 2 → 1). Dazu `lr` und `weight_decay` dort mit gleichem Wert, aber Rang „Startskript“ statt „Paper“.
- **F4 (a) + (c):** Eintraege aendern sich bei Shchur-GAT (`weight_decay` 0,01, `lr`, `hidden`, `layers` neu erzwungen, `epochs` aus der Konfiguration), Shchur-GCN (`dropout` 0,8, `hidden` 64, `weight_decay` 0,001 jetzt erzwungen, vorher nur genannt) und RLinear (`lr`, `epochs`, `batch_size`, `dropout`, `patience`, bisher durch „Quelle unbekannt“ verworfen, jetzt als argparse-Defaults des Repos erzwungen). Keine Aenderung der Eintraege bei Kipf, NLinear, SGC, DLinear, DGI.
- **F4 (b) (Hinweistext):** aendert den Prompt **aller neun** Einheiten mit nicht leerer `missing`-Liste, also aller `cg`-Prompts. Nur zwei der neun werden in G1 erneut gemessen. Fuer Kipf, NLinear, SGC, DLinear und DGI gilt ihre G0-Zahl **ohne Messung der Textaenderung** (offen genannt).
- **F2/F3:** im G0-Log Zeitueberschreitungen nur bei GAT (8 + 2 Zeilen: Veličković, Shchur) und DGI (2 Zeilen). Wirkt nur nach einem Timeout oder bei Exit 137/leerem Text. DGI wird in G1 **nicht** erneut gemessen (Aufwand und Deckel), der Bericht schreibt: „DGI-Ergebnisse aus G0; ein Timeout-Hinweis im Pfad kann sie aendern, nicht gemessen.“
- **F4 (b) und (c) im Arm `heute`:** keine Wirkung (der Arm nutzt keinen Steckbrief). Pläne und erste Codes von `heute` werden deshalb byte-identisch aus G0 uebernommen.

## Einheiten, Arme, Proben

| Einheit | Arm | Proben | Plan / erster Code | Zweck |
|---|---|---|---|---|
| `velickovic_gat_2018` | `heute` | T = 0,2 / 0,5 / 0,8 | Plan und erste Codes **byte-identisch aus G0** (`reuse_manifest.json` belegt SHA-256 beider Seiten); Reparaturen neu | Regel (a) |
| `velickovic_gat_2018` | `cg` | 3 | **neu**: neuer Steckbrief (F1, F4) → 1 neuer Plan, 3 neue Code-Proben | Regel (a) |
| `shchur_gat_2018` | `heute` | 3 | wie V-`heute` (aus G0) | Regel (a) |
| `shchur_gat_2018` | `cg` | 3 | neu (F4 aendert Steckbrief und Hinweistext) | Regel (a) |
| `shchur_pitfalls_2018` (Shchur-GCN) | `cg` | 3 | neu (F4 aendert drei erzwungene Werte und den Hinweistext) | Nichtverschlechterung |
| `rlinear_2023` | `cg` | 3 | neu (F4 (a) erzwingt fuenf bisher verworfene Werte) | Nichtverschlechterung, Held-out-Einheit |

**18 Proben**, 4 Plaene + 12 Code-Aufrufe neu (`cg`), Steckbrief ohne Extraktor-Aufruf neu berechnet (S1 aus G0 wiederverwendet, nur der deterministische Teil wiederholt; in G0 war S1 ein eigener Modell-Aufruf je Einheit). Alle anderen Zahlen werden aus G0 **uebernommen und offen als uebernommen benannt**: NLinear, SGC, DLinear, DGI (beide Arme), Kipf (beide Arme), Shchur-GCN und RLinear Arm `heute`. Reparaturen laufen neu (neue Aufrufe). Messung: volle Seed-Zahl der Einheit (V 10, Shchur 30 Splits, Shchur-GCN 30, RLinear 5), Seed 0 mit Reparaturschleife (bis 4 Versuche), uebrige Seeds mit dem Code von Seed 0, Urteil wie der Orchestrator (`measure_sample` aus `cg_gate.py`, unveraendert).

## Messweg: Nebius-Sandbox (anders als G0)

Sondierung am 2026-10-09 (`results/recall_cg_gate_g1_prep/nebius_probe.json`, kein Modell-Aufruf, nur feste Kommandos):

- **Je Branch 4 vCPU und 4,0 GB RAM** (`MemTotal` 4 034 744 kB, kein Swap, `bytearray(4 GB)` → `MemoryError`, 3 GB laufen). G0 lokal: 12 GB je GAT-Container, 6 GB sonst.
- **16 Branches gleichzeitig** ohne Fehler, gleicher Durchsatz je Branch wie bei einem (kein Teilen der Kerne sichtbar). Hoeher nicht getestet; der Messweg begrenzt auf 16 gleichzeitige Kommandos.
- **Harter Sandbox-Timeout toetet den Branch** (Session `FAILED`, Dateien und weitere Kommandos danach nicht mehr moeglich). Deshalb F2 (a) im Kommando selbst: `timeout -s ABRT` mit `faulthandler`, getestet: Stack in stderr, Exit 124.
- Netzsperre `unshare -n` und Harness (frisch kopiert, Testlabels nur im Harness) laufen wie im Produkt; der Messweg ist `NebiusGateSandbox` (duenne Huelle um `NebiusSandbox`).

**Referenzloesungen auf Nebius** (`results/recall_cg_gate_g1_prep/reference.jsonl`, Messweg wie oben, volle Seed-Zahl):

| Einheit | gueltige Seeds | Mittel (Nebius) | Referenzmittel | Urteil | langsamster Seed s | Zeitlimit s (`timeouts.train_timeout`) |
|---|---|---|---|---|---|---|
| velickovic_gat_2018 | 10/10 | 0,8280 | 0,828 | reproduziert | 64,6 | **660** |
| shchur_gat_2018 | 30/30 | 0,8142 | 0,814 | reproduziert | 25,5 | 600 |
| shchur_pitfalls_2018 | 30/30 | 0,8107 | 0,8107 | reproduziert | 86,7 | **900** |
| rlinear_2023 | 5/5 | 0,3704 | 0,3704 | reproduziert | 15,2 | 600 |

Alle vier Referenzen enden mit gueltigen Seeds, dem frueher bekannten Urteil und innerhalb von 4 GB. Zeitlimits je Seed aus `recall_cg_gate_g1_timeouts.json`, fuer beide Arme einer Einheit gleich. Unterschiede zu G0: andere CPU (G0: 4 Kerngruppen à 4 Kerne auf einem Laptop), 4 GB statt 12 GB, keine Wartezeit auf Speicher oder Kerne in den Zeiten, F2 (a) im Trainingsbefehl. **Die Zahlen sind mit G0 nicht streng vergleichbar** (Abstand der Referenzen zu G0: ≤ 0,0003).

## Zaehlregel und Entscheidungsregel (vorab, unveraendert gegenueber G0, auf das G1-Aggregat angewandt)

Klasse und Treffer wie in G0 (`classify`, `hit` aus `cg_gate.py`: Treffer = Urteilsklasse gleich Referenzurteil, DGI „teilweise“; Setup-Ausfall = Fehlschlag). Aggregat (`decide_g1`): Zahlen der vier Nicht-GAT-Entwicklungseinheiten (NLinear, SGC, DLinear, DGI) aus G0, Zahlen der zwei GAT-Einheiten aus G1. G0-Stand dieser Einheiten: `cg` 8, `heute` 4 Treffer. Das G1-Aggregat **besteht**, wenn alle gelten:

- **(a)** `cg` ≥ 8 von 18 auf den sechs Entwicklungseinheiten **und** `cg` ≥ `heute` + 5. Mit 8 : 4 aus G0 heisst das auf den zwei GAT-Einheiten: **`cg`-Treffer − `heute`-Treffer ≥ 1.**
- **(b)** Kipf + Shchur-GCN: `cg`-Treffer ≥ `heute`-Treffer − 1 (Kipf `cg`/`heute` aus G0, Shchur-GCN `cg` aus G1, Shchur-GCN `heute` aus G0).
- **(b′, neu, Nichtverschlechterung)** Shchur-GCN `cg` in G1: **mindestens 2 von 3** (G0: 3 von 3).
- **(c)** RLinear `cg` in G1: **mindestens 2 von 3** (Klasse „reproduziert“; G0: 3 von 3).
- **(d)** hoechstens **2 von 27** `cg`-Code-Aufrufen enden leer oder abgeschnitten (G0-Zaehlung fuer die fuenf nicht erneut gemessenen Einheiten Kipf, NLinear, SGC, DLinear, DGI = 15 Aufrufe, plus die 12 neuen G1-Aufrufe).

Sonst: **nicht bestanden**, Stopp, Bericht „Code-Grounding hat die Accuracy im Gate nicht angehoben“ ohne Geld fuer Plan A/B. Ein Lauf, der die Kostengrenze erreicht oder in dem nicht alle 18 Proben ein Urteil haben, ist **unvollstaendig** (weder bestanden noch nicht bestanden). Es gibt keine nachtraegliche Anpassung von Schwellen, Einheiten oder Proben.

Grenzen der Regel (zusaetzlich zu G0): niedrige Huerde (ein Treffer Vorsprung), 3 Proben je Zelle teilen einen Plan, Fix-Paket In-Sample, Messweg anders als G0 (4 GB), 5 Einheiten ohne Wiederholung der Textaenderung F4 (b), DGI ohne Pruefung von F2/F3. Nichtverschlechterung wird nur an zwei Einheiten geprueft.

## Kosten, Ablauf, Zeit

- **Kostendeckel 1,00 $ brutto fuer ganz G1** (`CG_GATE_G1_COST_CAP_GROSS_USD`), auf Aufruf-Ebene, eigene `spend.jsonl` im G1-Ausgabeverzeichnis (G0-Verbrauch zaehlt nicht mit). Bei Erreichen Stopp (`CostStop`), Ergebnis „unvollstaendig“. Schaetzung aus dem G0-Verbrauch (0,0129 $ je Probe nach der Wiederaufnahme, Plan und Code je ca. 0,01 $): 0,4 bis 0,7 $.
- **Kontostand Token Factory** (`scripts/cost_report.py`, 2026-10-09): laut Log seit Stichtag 2026-10-02 19:43Z **5,26 $ brutto verbraucht, Rest 25,48 $ von 30,74 $**. Die Konsole ist **nicht verifiziert** (letzter Abgleich 2026-09-29, Abweichung Log gegen Konsole dort sehr gross); das Log ist eine Schaetzung, massgeblich ist die Konsole.
- **Parallelitaet:** bis 8 gleichzeitige Modell-Aufrufe, bis 16 gleichzeitige Sandbox-Kommandos. **ETA 1,5 bis 3 Stunden** (Referenzlaufzeiten: V 65 s, Shchur-GAT 26 s, Shchur-GCN 87 s, RLinear 15 s je Seed fuer die Referenz; Agenten-Code ist erfahrungsgemaess 2- bis 10-mal langsamer, GAT-Seed-0-Ketten bis 4 × 660 s).
- Start: `LLM_PROVIDER=tokenfactory` mit Super auf allen drei Rollen, `LLM_CACHE=off`, `--sandbox nebius`. Wiederaufnahme mit demselben Befehl (fertige `final_*.json` bleiben).

## Reihenfolge

1. Diese Datei und alle Skripte, Tests, Zeitlimits, Sondierungs- und Referenzdaten committen (**vor dem ersten Modell-Aufruf**).
2. Koordinator bestaetigt; dann Start:
   `uv run python scripts/review/cg_gate_g1.py --run --g0 <G0-Ausgabe> --main <Checkout mit runs/> --out benchmark/experiments/results/recall_cg_gate_g1_samples --timeouts benchmark/experiments/recall_cg_gate_g1_timeouts.json`
3. Auswertung `--score`, Ergebnis nach `benchmark/experiments/results/recall_cg_gate_g1.md`, committen, nicht pushen, nicht mergen.

## Nachtraege

- *(G0-Endergebnis, sobald die letzten vier Shchur-GAT-Proben fertig sind: hier eintragen; bis dahin „50 von 54, laufend“.)*
