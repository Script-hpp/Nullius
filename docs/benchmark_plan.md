# Benchmark-Plan (Roadmap W2): Entwurf zur Vorab-Festlegung

Stand 2026-09-30, Auftrag 7. Nur Analyse. Kein Code in `src/`, keine echten Aufrufe (auch kein Tavily). Jede Zahl trägt eine Quelle im Repo. Was aus dem Gedächtnis stammt, steht als **[zu prüfen]** (CLAUDE.md: Zielwerte nie aus dem Gedächtnis übernehmen). Der Reviewer prüft diese Punkte später mit Tavily.

## 1. Bestand

### 1.1 Papers in `benchmark/papers.yaml`

| Paper (Zeile) | Task (`tasks.py:189`) | Claims | Seeds | Toleranz | Referenzurteil und Quelle | Stand |
|---|---|---|---|---|---|---|
| Kipf GCN (`papers.yaml:3`) | Cora | `gcn_cora_acc` 0,815 (:14) + Placebo 0,88 | 10 | ±0,01 | reproduziert; eigene Referenzlösung 0,8136 (`results/cora_references.txt`) | im Benchmark v1 gemessen |
| Shchur (:29) | Cora-LCC | `shchur_gcn_cora_acc` 0,815 (:38) + Placebo | 30 Splits × 1 Init. | ±0,01 | reproduziert; Referenz 0,8107 (`results/lcc_harness_reference_30x1.txt`); menschlicher Wert im Paper 81,5 ± 1,3 (Tabelle 1, Fixture `1811.05868.json`) | v1 |
| Zeng DLinear (:53) | ETTh1 (Horizont 96, Fenster 336) | `dlinear_etth1_96_mse` 0,375 (:65), `_mae` 0,399 (:79, nicht im Benchmark), Placebo 0,300 | 5 | ±0,01 absolut | reproduziert; Referenz 0,3809 (`results/etth1_harness_reference.txt`) | v1 (nur MSE) |
| Dacrema (:87) | fehlt | keine | – | – | Erwartung „Baselines schlagen neuronale Modelle“, nur Text | nicht baubar heute |
| Wu SGC (:93) | Cora | `sgc_cora_acc` 0,810 (:102) + Placebo | 10 | ±0,01 | reproduziert; Referenz 0,8081 | v1 |
| Veličković GAT (:117) | Cora | `gat_cora_acc` 0,830 (:125), kein Placebo | 10 | ±0,01 | reproduziert; Referenz 0,8280 (`tavily_v2_holdout.md`, Nachtrag) | nur Demo-Lauf live (`gat_demo_run.md`: teilweise 0,8036), nicht im Benchmark |
| Veličković DGI (:133) | Cora | `dgi_cora_acc` 0,823 (:141) + Placebo | 10 | ±0,01 | **teilweise**; Referenz 0,8047 (Abstand 0,0183) | noch kein Agentenlauf; reserviert als Prüfmenge für Tavily v2 (`tavily_v2_holdout.md`), v2 ist pausiert |

Nicht in `papers.yaml`: Shwartz-Ziv (nur `shwartz_ziv_scoping.md`), Katalog-Kandidaten (`benchmark/catalog.yaml`: SL-GCN geprüft „nicht reproduziert“ 0,783, MoHETS „Agenten-Fehler vermutet“, zwei ungeprüft). Für diese gibt es **kein** menschliches Referenzurteil, sie sind keine Benchmark-Kandidaten.

### 1.2 Ergebnisse (alles Token Factory, Prod-Modelle; nichts davon ist Prototyp)

Benchmark v1 (`results/benchmark_results.md`, Rohdaten `results/benchmark.jsonl`, 48 Urteile aus 24 Läufen, Protokoll `benchmark_protocol.md`):

| | Mit Tavily (B) | Ohne Tavily (A) |
|---|---|---|
| echte Claims, Übereinstimmung mit Referenzurteil | 6 von 12 | 3 von 12 |
| „nah“ (reproduziert oder teilweise) | 10 von 12 | 10 von 12 |
| Placebos als „nicht reproduziert“ erkannt | 12 von 12 | 12 von 12 |
| Ausfälle | 1 | 0 |

Weitere Messungen: Referenzläufe der Harnesses (`results/*_reference*.txt`, `cora_references_holdout.txt`), Graph-Ablation-Validierung 9/10, Role-3/4-Experimente DLinear/Shchur/GCN, GAT-Demo-Lauf live, Katalog-Läufe SL-GCN und MoHETS (nicht Teil des Benchmarks).

**Befund für den Plan:** In v1 war der Zielwert in beiden Armen gleich (`target_matches_yaml`: 0 Abweichungen, `benchmark_results.md`). Rolle 2 ändert im Benchmark also nichts am Urteil. Der Unterschied zwischen den Armen entsteht allein durch Rolle 4 (Setup-Details) und die längere Prompt-Länge. Das gehört ins README.

## 2. Lücke zu 8 bis 10 Papers

Heute: 4 Papers mit Benchmark-Ergebnis (Kipf, Wu, Zeng, Shchur). Das sind 4, nicht 8 bis 10.

| Kandidat (CLAUDE.md) | Mit vorhandenem Task machbar? | Aufwand | Anmerkung |
|---|---|---|---|
| Kipf GCN | ja, in v1 | 0 | |
| Shchur | ja, in v1 | 0 | Paper prüft Rangfolge, Task prüft nur die GCN-Zahl (`shchur_task.md`, CLAUDE.md) |
| Zeng DLinear | ja, in v1 (nur MSE) | 0 | Die Kernaussage „schlägt Transformer“ prüft der Task nicht |
| Wu SGC (zusätzlich) | ja, in v1 | 0 | |
| GAT, DGI (zusätzlich, nicht in CLAUDE.md-Liste) | ja (Cora) | klein: Placebo-Claim GAT ergänzen, `CLAIMS`-Eintrag | Referenzen liegen vor; DGI-Referenz nur „teilweise“ |
| **Dacrema (MovieLens)** | **nein** | **groß, siehe 2.1** | einziger echter negativer Fall |
| **Shwartz-Ziv (Tabellen)** | **nein** | **sehr groß, Empfehlung: weglassen, siehe 2.2** | |

### 2.1 Dacrema / MovieLens-Recsys (neuer Task)

Belegt im Repo: nur die Erwartung in `papers.yaml:87-91` und CLAUDE.md („Bei Recsys exaktes Auswertungsprotokoll des Papers umsetzen, Anzahl Ranking-Kandidaten usw.“). Es gibt keine Fixture für 1907.06902, das Protokoll des Papers ist **nicht gelesen** (ein Tavily-Extract nötig, vom Reviewer).

Aufwand (Schätzung, kein Messwert):
- **Harness-Vertrag:** `id,prediction` liefert eine Zahl pro Zeile. Ranking braucht Scores pro (Nutzer, Kandidat); `id = user_item` ist machbar, Vertrag aus `harness/verify.py` bleibt, Plausibilitätsprüfung (IDs vollständig, nicht konstant) muss angepasst werden. Kandidatenlisten muss der Harness festlegen (Seed), nie der Agent.
- **Metrik:** HR@K und NDCG@K mit getestetem Code in `harness/metrics.py` (Recall@K/NDCG sind in CLAUDE.md als benötigt genannt). Ein Drittel des Aufwands ist der Test gegen die Referenzimplementierung.
- **Split und Negativ-Stichprobe:** Leave-one-out, 100 Negative je Nutzer **[zu prüfen: Protokoll von Dacrema und der Originalpaper]**.
- **Datensatz-Lizenz:** GroupLens-MovieLens erlaubt Forschungsnutzung, aber keine Weitergabe ohne Erlaubnis **[zu prüfen, Lizenztext]**. Folge: nicht ins Repo legen; der Harness lädt die Datei zur Vorbereitung (mit Netz) wie bei Cora (`prepare_cmd`, `tasks.py:185`), das Training läuft ohne Netz.
- **CPU unter 10 Minuten:** ML-1M ist klein **[zu prüfen]**; ein NCF-artiges Modell auf CPU plausibel, nicht gemessen.
- **Claim-Form:** Dacremas Kernaussage („einfache Baselines schlagen neuronale Modelle“) ist ein Vergleich zweier Zahlen, nicht eine Zielzahl. Zwei numerische Claims (neuronaler Wert aus dem Originalpaper, Baseline-Wert aus Dacrema) sind machbar; ein Vergleichsclaim bräuchte eine neue Urteilsform. Empfehlung: zwei Zahlen-Claims.
- **Referenzlösungen** für beide Modelle (Selbsttest der Pipeline, wie `tests/reference/`) sind nötig, sonst gibt es kein Referenzurteil.
- **Geschätzt: 3 bis 5 Personentage** (Harness 1, Metrik+Tests 0,5, zwei Referenzlösungen 1 bis 1,5, Task/Prompt/Formprüfung 1, Fixtures und Claim-Belege 0,5). Hohes Risiko: Protokollraten wie bei Shwartz-Ziv.

### 2.2 Shwartz-Ziv / Tabellen

`shwartz_ziv_scoping.md` benennt fünf Blocker: Protokoll steht nicht im Paper, 1000 HyperOpt-Schritte nicht in 10 Minuten CPU, Metrik (Kreuzentropie × 100) unklar, Vertrag braucht Wahrscheinlichkeiten, Kernaussage verlangt tiefe Modelle. **Empfehlung: weglassen** (kein Task, dessen Urteil Protokollraten misst). Falls ein Tabellen-Paper gewünscht ist, Ersatz mit im Text stehendem Protokoll suchen.

## 3. Weitere Kandidaten auf vorhandenen Tasks

Ehrliche Einordnung vorweg: Ein „menschliches Reproduktionsergebnis“ im Sinn einer MLRC-Studie ist im Repo **nicht belegt**. Belegt sind unabhängige Neumessungen in Kritik-Papers (Shchur, Wu) und unsere eigenen Referenzlösungen. Die Herkunft des Referenzurteils muss pro Claim im README stehen: **(H)** unabhängige Neumessung aus Fixture, **(R)** eigene Referenzlösung im eigenen Harness.

### 3.1 Aus dem Repo belegt (Fixtures, Zielwerte im Text)

| Kandidat | Task | Claim, Zielwert | Quelle Zielwert | Quelle Referenzurteil | Arbeit |
|---|---|---|---|---|---|
| GAT (Planetoid) | Cora | 83,0 ± 0,7 % | `papers.yaml:125`, Wu Tabelle 2 (`sgc_task.md`) | (R) 0,8280 → reproduziert | Placebo ergänzen |
| DGI | Cora | 82,3 ± 0,6 % | Tabelle 2 des DGI-Papers, Fixture `1809.10341.json` (`tavily_v2_holdout.md`) | (R) 0,8047 → teilweise | Repo-Fixture für Rolle 4 fehlt (`benchmark/fixtures/repos/` hat kein DGI) |
| GAT auf zufälligen Splits | Cora-LCC | 81,8 ± 1,3 % | Shchur Tabelle 1, Fixture `1811.05868.json` (Zeile „GAT 81.8 ± 1.3“) | (H) Shchur selbst; (R) fehlt, `tests/reference/gat_cora_reference.py` müsste auf LCC-Splits laufen | neuer Claim, Referenzlauf, Toleranz vorab |
| NLinear / Linear | ETTh1 | NLinear 0,374 / 0,394; Linear 0,375 / 0,397 (MSE/MAE, Horizont 96) | Zeng Tabelle 2, Fixture `2205.13504.json`; **Spaltenzuordnung Linear/NLinear/DLinear aus der Reihenfolge abgelesen, nicht geprüft** (DLinear 0,375/0,399 stimmt mit `papers.yaml:65,79`) | (R) fehlt | gleicher Paper wie DLinear, wenig unabhängig |
| Kipf Tabelle 3: Chebyshev K=2 81,2; 1st-order 80,0; MLP 55,1 | Cora | je ein Zahlen-Claim | Fixture `1609.02907.json` (Tabelle 3) | (R) fehlt | gleicher Paper wie GCN; MLP kollidiert mit Graph-Ablation (Claim ohne Graph würde „Graph nicht genutzt“ melden), weglassen |
| SGC auf zufälligen Splits | Cora (nicht LCC) | 80,62 ± 1,21 | Wu Tabelle 10, Fixture `1902.07153.json` | (H) | Split-Variante (Cora voll, zufällig) ist kein vorhandener Task, nicht machbar ohne Task-Code |

Nicht machbar ohne Großmodelle oder Zeitverletzung: Informer (0,865), Autoformer (0,449), FEDformer (0,376) auf ETTh1 (Werte aus Fixture `2205.13504.json`, Zuordnung ungeprüft). Der Demo-Akt 2 („DLinear schlägt Transformer“) bräuchte diese Implementierungen; Risiko CPU-Limit, Nicht-Ziel „keine großen Modelle“. Nicht empfohlen.

### 3.2 Aus dem Gedächtnis, alles **[zu prüfen]** (Reviewer via Tavily)

- APPNP, GCNII, GraphSAGE, ChebNet auf Planetoid-Cora: Zielwerte und ob ein Kritik-Paper sie neu misst.
- PatchTST / TimesNet auf ETTh1, Horizont 96: Zielwerte, Fenstergröße (unser Harness fixiert 336), Modellgröße auf CPU.
- MLRC-Berichte („[Re] …“) zu GNN- und Zeitreihenpapers: Suchanfragen „ML Reproducibility Challenge [Re] graph attention / GNN Cora“, „reproducibility report time series forecasting ETTh1“. Kandidat nur aufnehmen, wenn (1) der Datensatz Cora oder ETTh1 ist, (2) der Bericht ein Urteil und Zahlen nennt, (3) ein Seed unter 10 Minuten CPU läuft.
- Ergebnis der Suche ist nicht vorhersehbar. Plan: nicht auf diese Suche bauen.

## 4. Ablauf und Regeln (Entwurf für die Vorab-Festlegung)

### 4.1 Was „ohne Tavily“ tatsächlich abschaltet (geprüft im Code)

Reviewer-Festlegung: Ohne Tavily = keinerlei Tavily zur Laufzeit. Das gilt für die Rollen, die im Loop verdrahtet sind:

- `use_tavily=False` (`loop.py:146`) überspringt `resolve_targets` (Rolle 2: Paper-Text, Zielwert-Extraktion, Spaltenprüfung, `target_check`) bei `loop.py:788`.
- `use_setup_details = use_tavily and setup_details` (`loop.py:147`) schaltet Rolle 4 ab (Block `loop.py:792`: `_gather_docs` mit README-Extract und Repo-Klon, `extract_setup_details`).
- `paper_text` bleibt `None`; `Dossier.sources` ist leer.
- Die einzigen Tavily-Aufrufstellen im Loop sind `fetch_paper_text` (`loop.py:461`) und `fetch_readme` (`loop.py:505`). **Rollen 3, 5, 6, 7 sind nicht im Loop verdrahtet** (keine Search/Crawl-Aufrufe in `loop.py`; `tavily_tools` wird nur in `loop.py`, `claims_source.py`, `discovery.py` und `scripts/experiment_role3_stage1.py` importiert). Rolle 1 (Discovery) betrifft Benchmark-Papers nicht (fest in `papers.yaml`).

**Abweichungen vom Ideal „ohne jede Quelle“:**
1. Der Zielwert in `papers.yaml` stammt teils aus Tavily-Fixtures: Shchur, SGC, DGI „manuell aus Fixture“ (`papers.yaml:43,107,146`), GCN und DLinear „Gedächtnis, ungeprüft“ (:19,:70). Arm A nutzt also für drei Papers Werte, die früher mit Tavily belegt wurden. Das ist gewollt (gleiche Zielwerte in beiden Armen, `benchmark_protocol.md`), muss aber im README stehen.
2. Das Modell kennt die Papers aus dem Training (GCN, GAT sind bekannt). „Ohne Tavily“ heißt „ohne Laufzeitquellen“, nicht „ohne Vorwissen“.
3. Fehlerreparatur nutzt lokale Hinweise (`harness/repair_hints.py`), kein Tavily; sie wäre für beide Arme gleich.
4. Arm B hat längere Prompts (Setup-Details), das ist eine Störgröße (steht schon in den Grenzen von `benchmark_protocol.md`).

### 4.2 Tavily-Modus: Fixture, Stand und Begründung

Stand (belegt): `scripts/run_benchmark.py:41-42` setzt für beide Arme `tavily_mode="fixture"`. Das entspricht CLAUDE.md („Benchmark: `tavily_mode="fixture"` in den Skripten“). Beide Arme und alle Wiederholungen sehen denselben gespeicherten Text.

- **Vorteil:** Vergleichbarkeit (kein Drift der Seite, gleicher Text für 3 Generierungen), keine Tavily-Credits, reproduzierbar, `src/` eingefroren.
- **Nachteil:** Der Benchmark belegt keinen Live-Aufruf; der Preis „Best Use of Tavily“ verlangt den Live-Aufruf (steht in CLAUDE.md). Den liefern die Demo-/Katalog-Läufe (`gat_demo_run.md`: Live-Aufrufe mit Zeitstempel und Hash), nicht der Benchmark. Im README trennen: „Benchmark: gespeicherte Tavily-Ergebnisse; Produktläufe: live“.
- Neue Papers brauchen vor dem Lauf je 1 Fixture für Paper-Text (und README/Repo für Rolle 4): einmalige Live-Aufrufe durch den Reviewer, danach eingefroren (Tavily sparsam, CLAUDE.md).

### 4.3 Flags: `graph_repair`

Stand: Benchmark v1 lief ohne `graph_repair` (`run_benchmark.py:41-42` setzt es nicht; Default `False`, `loop.py:134`). Das Flag aktiviert Plan-Validator, Fehler-Reparatur, Graph-Ablation mit Reparatur (`worker_handover.md`).

| | Vorteil | Nachteil |
|---|---|---|
| ohne (wie v1) | v1-Ergebnisse bleiben gültig und vergleichbar, kein Neulauf der 4 Papers, Experiment-Disziplin (src einfrieren) | mehr Agenten-Fehler (GCN Arm A Gen. 0: Mittel 0,508; DLinear Ausfall) |
| mit | weniger Agenten-Fehler, weniger Rauschen zwischen den Armen, entspricht dem Produkt | Neulauf aller Papers nötig (~2 $), nur für Graph-Tasks relevant (Ablation), ändert den Code-Stand gegenüber v1, Tavily-Effekt schwerer zu trennen |

**Empfehlung:** Primärmessung **ohne** `graph_repair`, alle 10 Papers gleiches Protokoll (v1 bleibt eingefroren im README, neue Papers ergänzen sie). Optional sekundär (nur wenn Budget und Zeit bleiben): ein Zusatzlauf der Graph-Papers mit `graph_repair`, klar als „Produktkonfiguration“ getrennt berichtet.

### 4.4 Seeds, Toleranz, Generierungen

- Seeds und Toleranz je Paper wie in `papers.yaml`, **vor** dem Lauf festgelegt und im Bericht genannt. Neue Claims: feste Regel ±0,01 bei Accuracy (CLAUDE.md), bei MSE ±0,01 absolut; Shchur-GAT mit 30 Splits wie Shchur-GCN. Kein Anpassen nach Sicht.
- 3 frische Generierungen je Paper und Arm, LLM-Cache aus, je Generierung echter Claim und Placebo (wie `benchmark_protocol.md`). Infrastruktur-Abbrüche werden vollständig wiederholt (Regel aus `benchmark_incidents.md`), `MAX_PARALLEL_SEEDS=3`.
- Placebo pro neuem Paper (GAT, DGI haben es, GAT fehlt noch).

### 4.5 Zählregel und Kennzahl (Entscheidung nötig)

1. **Zähleinheit:** CLAUDE.md und Auftrag sagen „X von 10“; das v1-Protokoll zählt „von 12“ (4 Papers × 3 Generierungen). Vorschlag: **Primär je Paper das Mehrheitsurteil der 3 Generierungen** (mindestens 2 von 3 gleich dem Referenzurteil) → „X von N Papers“, sekundär die Rohzählung „von 3N Urteilen“. Muss vor dem Lauf feststehen, auch rückwirkend für v1 (v1-Daten sind schon gesehen; Mehrheit aus `benchmark.jsonl` ist dort aber eine neue Auswertung, als post hoc kennzeichnen).
2. **Dreistufig exakt** wie v1; „nicht reproduzierbar (Setup)“, „Agenten-Fehler vermutet“, „abgebrochen“ sind Ausfälle.
3. **Achse 2 „nicht geprüft“ (endgültig, CLAUDE.md):** Jedes Urteil ist „vorläufig“. Strikt nach CLAUDE.md („Benchmark zählt nur ‚Paper reproduziert‘ und ‚nicht reproduziert‘“) zählte dann **kein** Urteil, weil „Paper reproduziert“ Achse 2 = ja verlangt. Das v1-Protokoll löst das, indem es Achse 1 zählt. Ehrliche Formulierung fürs README: Kennzahl misst **„Zielwert erreicht laut Harness“** (Achse 1), nicht „Methode reproduziert“. Der Wortlaut „reproduziert“ darf nicht als Paper-Urteil erscheinen. Offene Entscheidung für den Reviewer: CLAUDE.md-Zählregel anpassen oder README so formulieren.
4. **Referenzurteil ist kein Fremdurteil:** Bei (R)-Claims stammt es aus unserer Referenzlösung im eigenen Harness (nicht unabhängig von unseren Protokollentscheidungen). Die Formulierung „stimmten mit menschlichen Reproduktionsstudien überein“ ist nur für (H)-Claims haltbar. Vorschlag für den Zielsatz: „… stimmten mit dem Urteil einer sorgfältigen menschlichen Referenzimplementierung im selben Harness überein; für X Claims liegt zusätzlich eine unabhängige Neumessung aus einem Kritik-Paper vor.“
5. **Placebos** getrennt berichten (konstruiert, kein echter negativer Fall). Der echte negative Fall fehlt ohne Dacrema; das README sagt das offen.
6. Keine Signifikanzaussagen; 6 bis 10 Papers sind Zählwerte mit Vorbehalt.

### 4.6 Agenten-Fehler

Wie v1: Agenten-Fehler bleiben in der Zählung (Ausfall, getrennt berichtet), nie stillschweigend wiederholt. Nur Infrastruktur-Abbrüche (Sandbox-Timeout, Billing) werden wiederholt. Einzelne Generierungen mit instabilem Code (GCN Arm A Gen. 0) bleiben drin.

## 5. Kosten und Zeit

Aus `results/benchmark.jsonl` (Feld `cost_usd_run`, netto, je Lauf beider Claims; `cost_usd_run` ist `spent_usd` seit Laufstart, `loop.py:169`): 24 Läufe, 0,862 $ netto, 1,026 $ brutto.

| Paper | Arm A je Lauf (netto) | Arm B je Lauf (netto) |
|---|---|---|
| Kipf GCN | 0,016 $ | 0,052 $ |
| Shchur | 0,018 $ | 0,029 $ |
| Wu SGC | 0,018 $ | 0,026 $ |
| Zeng DLinear | 0,022 $ | 0,106 $ |

Katalog-Läufe (Auftrag): SL-GCN 0,025 bis 0,028 $, MoHETS 0,093 $ brutto. GAT-Demo 0,041 $ netto.

**Hochrechnung** (Mittel Arm A ≈ 0,019 $, Arm B ≈ 0,053 $ je Lauf, also ≈ 0,072 $ je Generierungspaar):
- 10 Papers × 3 Generierungen × 2 Arme = 60 Läufe ≈ 10 × 3 × 0,072 = **2,2 $ netto, ≈ 2,6 $ brutto** (Obergrenze mit DLinear-ähnlichen Ausreißern ≈ 4 $ brutto). Nur die 6 neuen Papers: ≈ 1,3 $ netto.
- Gegen `BUDGET_TOTAL_USD=29,75` (Kontostand zum Stichtag, aktueller Verbrauch nicht geprüft, Reviewer). Geld ist nicht der Engpass.
- Tavily: Fixture-Modus, 0 Credits im Lauf; einmalige Fixture-Abrufe je neuem Paper (Paper-Text, README), Anzahl Credits nicht bekannt.
- **Zeit (Schätzung, nicht gemessen):** Seeds 3 bis 7 Minuten parallel auf der Nebius-Sandbox (Auftrag). Je Lauf mit 2 Claims, Formprüfung, Ablation, Reparaturversuchen grob 15 bis 40 Minuten. 60 Läufe seriell 15 bis 40 Stunden; mit 2 bis 3 parallelen Prozessen (wie v1) grob 6 bis 15 Stunden Wanduhrzeit. v1 brach viermal mit `ApiTimeoutError` ab (`benchmark_incidents.md`); mit 60 s Timeout und `MAX_PARALLEL_SEEDS=3` eingeplant, aber Puffer für Neustarts nötig.

## 6. Empfehlung

1. **Ehrliche Zielgröße:** Mit vorhandenen Tasks sind **6 Papers** machbar (Kipf, Wu, Zeng, Shchur, GAT, DGI). 8 bis 10 Papers gibt es nur mit Dacrema (7) und zusätzlichen Claims desselben Papers; „8 bis 10 Claims aus 6 bis 7 Papers“ ist ehrlicher als „10 Papers“. Die README-Kennzahl nennt Papers und Claims getrennt.
2. **Zuerst bauen (billig, vorhandene Tasks):** GAT-Placebo in `papers.yaml`; `CLAIMS` in `run_benchmark.py` für GAT, DGI; Shchur-GAT-Claim (81,8 ± 1,3, Tabelle 1) mit Referenzlauf auf LCC-Splits; ggf. NLinear-Claim. Vorab-Festlegung (dieses Dokument, dann Commit) **vor** dem ersten Lauf, `src/` einfrieren.
3. **Dacrema: Entscheidung des Reviewers bis zu einem Stichtag.** 3 bis 5 Personentage, Risiko Protokollraten und Lizenz. Er liefert den einzigen echten negativen Fall und den Demo-Akt 3. Falls nicht gebaut: README sagt offen, dass alle Benchmark-Papers „reproduziert/teilweise“ erwarten und Placebos die einzige Negativkontrolle sind. Vorher ein Tavily-Extract für 1907.06902 (Protokoll lesen), dann Aufwand neu schätzen.
4. **Weglassen:** Shwartz-Ziv (Blocker in `shwartz_ziv_scoping.md`), Informer/Autoformer/FEDformer, SGC auf zufälligen Splits (kein Task), MLP-Claim (Ablation-Konflikt), Benchmark-Erweiterung auf Papers ohne belegte Referenz.
5. **Protokoll:** Fixture-Modus beider Arme, ohne `graph_repair`, 3 Generierungen, Mehrheitsurteil je Paper als Primärzahl, Rohzählung sekundär, Placebos getrennt. Referenzherkunft (H)/(R) pro Claim im README. Kein Anspruch „Paper reproduziert“ wegen Achse 2.
6. **Kennzahl offen benennen:** Der Effekt von Tavily im Benchmark kommt aus Rolle 4 (Zielwerte waren gleich) und ist an Shchur und SGC gebunden (`benchmark_interpretation.md`: Differenz von 3 stammt aus zwei Papers; GCN und DLinear ohne Unterschied). Mit 6 Papers bleibt die Aussage „Tendenz“, kein Test.
7. **DGI-Hinweis:** Wird DGI im Benchmark benutzt, ist es als Prüffall für Tavily v2 verbraucht (`tavily_v2_holdout.md`). Vorschlag: v2 bleibt pausiert bis nach Abgabe, dann ist das unkritisch; im README vermerken.
8. **Risiken:** Sandbox-Timeouts (Puffer für Neustarts), Referenzlösungen für neue Claims fehlen (vor den Läufen nötig, sonst kein Referenzurteil), Budget nicht geprüft, Spaltenzuordnung bei Zeng Tabelle 2 ungeprüft (Rolle-2-Spaltenprüfung läuft vor dem Lauf, `table_check.py`), Protokoll-Nachträge nach Sicht der Ergebnisse vermeiden.

## Offene Fragen an den Reviewer

1. Zähleinheit: Mehrheitsurteil je Paper (X von N Papers) plus Rohzählung, einverstanden?
2. Achse-2-Konsequenz: CLAUDE.md-Zählregel anpassen oder README-Formulierung „Zielwert erreicht“?
3. Dacrema bauen: ja/nein, bis wann? Tavily-Extract für 1907.06902 freigeben?
4. Shchur-GAT (81,8) und NLinear als Zusatzclaims aufnehmen, oder nur GAT/DGI?
5. Ob DGI trotz Holdout-Reservierung in den Benchmark geht.
