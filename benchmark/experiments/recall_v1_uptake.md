# V1: Umsetzungstest des Steckbriefs (Vorab-Festlegung)

Stand 2026-10-08, Branch `Script-hpp/cto`. Diese Datei, `scripts/review/uptake_probe.py` (Fakt-Checks, `decide`) und `tests/test_uptake_probe.py` werden **vor dem ersten Probe-Aufruf** committet. Danach ändern sich Fakten, Checks, Schwellen und Regel nicht mehr. Plan: `docs/recall_steckbrief_plan.md` Abschnitt 5; V0 (Lieferung): `results/recall_v0_audit.md`.

**Frage:** Kommen die entscheidenden Fakten im Agenten-Code an, wenn der Prompt statt der Zitatliste den Steckbrief enthält (und S2/S3 ihn durchsetzen)? V1 sagt **nichts über Accuracy**, nur über Umsetzung.

## Aufbau

- **Einheiten (7):** Dev-Einheiten DLinear, NLinear, SGC, Veličković-GAT, DGI, Shchur-GAT (6) und die neue Einheit RLinear. Die Fakten der Dev-Einheiten stammen aus der Fehleranalyse (post hoc, `failure_analysis_v4.md`, `recall_root_cause.md`). Die Fakten von RLinear stammen aus `script.md` und `run.py` des Repos und wurden vor V1 festgelegt, ohne einen RLinear-Code zu sehen.
- **SparseTSF entfällt** (Abweichung vom Plan): Für SparseTSF liegen weder Paper-Text noch Setup-Details gespeichert vor. Beides neu zu holen bräuchte Tavily-Extract-Aufrufe. Damit stützt die Regel 2 nur eine neue Einheit; das steht im Bericht.
- **Eingabe je Einheit (gespeichert, keine neue Extraktion außer S1):** Paper-Text und belegte Setup-Details aus dem v4-Arm-B-Lauf der Einheit (RLinear: Lauf `runs/rlinear_2023`), Repo-Dateien aus einem lokalen Klon am Commit des Laufs über den echten `read_repo` (Unit-Modus: Pflicht-Modelldatei, Modell-Konfiguration, relative Imports, Quelltextfilter), README aus dem Klon. Die LLM-Dateiwahl ist durch die Dateien des gespeicherten Laufs ersetzt (Proxy).
- **Arme (je Einheit, Produktkonfiguration `graph_repair=True`, `setup_v2`):**
  1. `heute`: `format_for_prompt_v2` + Plan + Code (v4-Pfad).
  2. `steckbrief`: S1 im Plan- und Code-Prompt, kein CONFIG-Pflichtblock, kein Override, keine Prüfung.
  3. `steckbrief_pruefung`: S1 + CONFIG-Pflicht + S2 (Override) + S3 (Einzelfragen, höchstens 2 Reparaturrunden).
- **Proben:** je Arm und Einheit 1 Plan, dann 3 Code-Proben bei Temperatur 0,2 / 0,5 / 0,8. Das sind 21 Code-Aufrufe je Arm (7 × 3). Der Plan-Schritt und S1 laufen einmal je Einheit/Arm.
- **Modell:** Prototyp, `LLM_PROVIDER=openrouter`, `nvidia/nemotron-3-super-120b-a12b:free` für `code`, `extract` und `reasoning`. 0 $. Keine Token-Factory-, keine Tavily-, keine Sandbox-Aufrufe. Es gelten die Prototyp-Regeln aus `CLAUDE.md` (nur öffentliche Papers und Repos, keine Schlüssel/Pfade im Prompt, Retries mit Backoff höchstens 5). Die Zahlen werden nicht veröffentlicht.

## Fakten und Checks

Alle Checks laufen auf dem Code ohne Kommentare und Docstrings (`ast.unparse`), ein Syntaxfehler im Code erfüllt keinen Fakt. Definition im Skript (`FACTS`); Klassen: D = geliefert, nicht befolgt; E = im Repo, nicht herausgezogen; R = Datei nie gelesen; R* = falsche Datei übernommen; N = neue Einheit.

| Einheit | Fakt | Klasse | Check (Kurzform) |
|---|---|---|---|
| DLinear, NLinear | seq336 | D | Konstante 336 im Code |
| DLinear, NLinear | lr0.005 | D | Konstante 0.005 (auch als `5e-3`) |
| DLinear, NLinear | halve_each_epoch | D | `0.5 **`, `* 0.5`, `*= 0.5` oder `gamma=0.5` |
| NLinear | subtract_last | R* | AST: Variable aus `x[:, -1…]` wird abgezogen und später wieder addiert |
| NLinear | no_decomposition | R* | kein `series_decomp`, `moving_avg`, `trend`, `seasonal` |
| SGC | feat_row_l1 | E | Zeilensumme/`row_normalize`/`norm='l1'`/`preprocess_features` |
| SGC | no_col_standardize | E | kein `StandardScaler`, `.std(`, `scale(`, `.mean(0)` |
| Veličković-GAT, DGI | feat_row_l1 | E | wie oben |
| DGI | graph_conv, prelu, hid512 | E | Matrixprodukt mit `adj`/`spmm`; `PReLU`; Konstante 512 |
| Shchur-GAT | additive_attention | R | `leaky_relu` und `softmax` |
| Shchur-GAT | self_loops | R | `eye(`, `add_self_loops`, `fill_diagonal`, `setdiag` |
| Shchur-GAT | both_directions | R | `.T`, `transpose`, `maximum(`, `to_undirected`, `adj + adj` |
| RLinear | seq336, lr0.005, halve_each_epoch | N | wie oben |
| RLinear | batch128 | N | Konstante 128 |
| RLinear | revin | N | Mittelwert, Std/Var, Division durch und Multiplikation mit der Std, Addition des Mittelwerts |

**Grenzen:** Die Checks sind Heuristiken (Regex). Ein Treffer belegt, dass der Fakt im Code steht, nicht dass er richtig verdrahtet ist (eine Konstante 336 kann ungenutzt sein). Sie sind in allen drei Armen gleich, verzerren also den Vergleich nicht systematisch. `no_decomposition` und `no_col_standardize` sind Negativchecks. Zusätzlich schreibt das Skript verblindete Code-Kopien (`blind/`, Schlüssel getrennt), damit eine Person, die den Arm nicht kennt, die Checks stichprobenweise gegenliest. Dieses Gegenlesen ist nicht Teil der Regel.

## Entscheidungsregel (alle vier Bedingungen müssen gelten)

**Faktanteil** eines Arms = erfüllte Fakt-Instanzen geteilt durch alle Fakt-Instanzen (Einheit × Probe × Fakt). Ein Aufruf ohne Code (leer oder abgeschnitten nach drei Antworten) zählt alle seine Fakten als nicht erfüllt. Dev-Anteil über die 6 Dev-Einheiten (18 Fakten × 3 Proben = 54 Instanzen je Arm), Neu-Anteil über RLinear (5 × 3 = 15).

1. Dev-Anteil im Arm `steckbrief_pruefung` ≥ **0,7** **und** mindestens **0,25** über dem Dev-Anteil im Arm `heute`.
2. Neu-Anteil (RLinear) im Arm `steckbrief_pruefung` ≥ Neu-Anteil im Arm `heute`.
3. Höchstens **2 von 21** Code-Aufrufen im Arm `steckbrief_pruefung` enden ohne Code (der Plan nannte 2 von 24 bei 8 Einheiten).
4. Kein `spec_override` führt in einer Probe zu einem Syntaxfehler oder fehlendem CONFIG-Block.

Gilt alles: Empfehlung, ein vorab festgelegtes bezahltes v6 vorzubereiten (Gate und Budget separat, Zustimmung der Nutzerin oder des Nutzers nötig). Sonst Stopp mit Bericht, **an welchem Baustein es hängt**: Fehlt der Fakt schon im Steckbrief-Text (Muster `SPEC_HAS` im Skript, Dokument `*_spec.json`), liegt es an S1; steht er im Steckbrief, aber nicht im Code des Arms `steckbrief_pruefung`, liegt es an S2/S3 oder am Modell. Zusätzlich berichtet: Anteil je Klasse, je Einheit, Arm `steckbrief` allein, Anzahl Overrides, Reparaturrunden.

## Nicht Teil von V1

Kein Sandbox-Smoke (`SMOKE=1`), keine Accuracy, kein Vergleich mit Arm A. Keine Änderung an Benchmark-v4-Ergebnissen oder -Protokollen. Das Ergebnis ist ein Gate für die Frage „bezahlter Lauf ja/nein“, kein Befund über die Wirksamkeit.

## Addendum 2026-10-08 (vor dem ersten Probe-Aufruf; Text oben unverändert)

Entscheidung der Nutzerin/des Nutzers: Der OpenRouter-Schlüssel ist Free-Tier (ca. 50 Anfragen pro Tag), V1 läuft deshalb auf Token Factory, verkleinert. Die Abweichungen im Einzelnen:

- **Anbieter und Modell:** `LLM_PROVIDER=tokenfactory`, Modell `nvidia/nemotron-3-super-120b-a12b` (Super) für `code`, `extract` und `reasoning`; **kein Ultra**. Plan und S1 laufen also auf Super. Keine Tavily- und keine Sandbox-Aufrufe (unverändert). Die Prototyp-Regel „0 $" entfällt; die Kosten zählen zum globalen Budget (`.env` und `logs/` sind die des Haupt-Checkouts).
- **Arme:** nur `heute` und `steckbrief_pruefung`. Der Arm `steckbrief` (S1 allein) entfällt; die Regeln 1–4 nutzen nur diese zwei Arme und bleiben unverändert. „Arm `steckbrief` allein" wird nicht berichtet. Aufrufe je Arm: 7 Einheiten × (1 Plan + 3 Code-Proben); dazu je Einheit ein S1-Aufruf. Regel 3 bleibt „höchstens 2 von 21".
- **Einheiten und Temperaturen:** wie oben, 7 Einheiten × 3 Temperaturen (0,2 / 0,5 / 0,8). SparseTSF entfällt (siehe oben, unverändert).
- **S4-Erweiterungen über den Plan hinaus akzeptiert:** Der Repo-Ausschnitt (S4) enthält gegenüber dem Plan zusätzlich (a) Modell-Konfigurationsdateien, (b) Rümpfe von Vorverarbeitungsfunktionen aus Skripten, (c) Stern-Importe. Mit ihnen stieg die Liefer-Abdeckung in V0 von 13/22 auf 21/22 Fakten (`results/recall_v0_audit.md`). Diese Erweiterungen sind Teil dessen, was V1 testet; ein Bestehen von V1 gilt für S1–S4 einschließlich dieser Zusätze.
- **Kostengrenze:** hart **1,20 $ brutto** für ganz V1 (inkl. Trockenlauf), im Skript durchgesetzt: vor jedem Aufruf wird `LLMClient.total_cost_usd × (1 + VAT_RATE)` geprüft; ab 1,20 $ minus 0,03 $ Marge bricht das Skript ab (`CostCapReached`), speichert die bis dahin erzeugten Proben und berichtet das Teilergebnis. Bei Abbruch wird die Regel nur auf den vorhandenen Proben ausgewertet und als unvollständig gekennzeichnet; sie wird nicht als bestanden gewertet.
- **Echte Proben:** `LLM_CACHE=off`. Kein Cache-Treffer über Temperaturen oder Wiederholungen hinweg; jeder Code-Aufruf ist ein frischer Modellaufruf.
- **Trockenlauf zuerst:** 1 Einheit, 1 Temperatur (0,2), beide Arme (`--units`, `--temps`, `--arms`). Die Kosten werden hochgerechnet; liegt die Hochrechnung der Gesamtkosten über 1,20 $, wird nicht weitergelaufen, sondern die Koordination gefragt. Die Skriptänderungen (`--arms`, `--temps`, Kostengrenze, Teilspeicherung) ändern weder `FACTS` noch `decide` noch die Checks.
