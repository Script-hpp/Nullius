# V0: Liefer-Audit des Steckbrief-Ausschnitts (offline, ohne LLM)

Stand 2026-10-08, Branch `Script-hpp/cto`. Plan: `docs/recall_steckbrief_plan.md` Abschnitt 5. Kein LLM-, Tavily- oder Token-Factory-Aufruf; die sechs öffentlichen Repos wurden nur lesend per `git clone` am Commit der Fixtures geholt.

**Frage:** Steht jeder vorab gelistete entscheidende Fakt (Liste aus `scripts/review/fact_audit.py`, entstanden aus `failure_analysis_v4.md` und `recall_root_cause.md`, post hoc auf den Dev-Einheiten) im S4-Ausschnitt, der in den Steckbrief-Aufruf (S1) geht?

**Aufruf:** `uv run python scripts/review/fact_audit.py <Haupt-Checkout mit runs/> --s4 <Verzeichnis mit Klonen>`

**Methode:** `repo_source.read_repo` läuft unverändert über eine Sandbox-Attrappe auf dem Klon. Die LLM-Dateiwahl (`select_files`) ist durch die Dateien der gespeicherten Fixture ersetzt (Proxy). Danach laufen die deterministischen S4-Schritte (Pflicht-Modelldatei, Modell-Konfiguration, relative Imports zwei Ebenen, Quelltext-Filter, Zeichenbudget) und `spec.focused_excerpt` mit den Setup-Details eines v4-Arm-B-Laufs der Einheit.

**Grenzen:** Die echte Dateiwahl des LLM kann abweichen (Proxy). Die Fakt-Regexe prüfen, dass der Fakt im Text steht, nicht dass S1 ihn herauszieht. SparseTSF und RLinear haben keine vorab gelistete Fakten und sind nicht geprüft. Die Fakten stammen von den Dev-Einheiten, die S4-Regeln wurden an deren Befunden nachgebessert (Überanpassungsrisiko, siehe unten).

## Ergebnis

| Stand | Fakten im Ausschnitt |
|---|---|
| S4 wie vom Vorgänger übernommen (nach Merge, mit `_paper_id`-Fehler im Loop, Lauf des Audits direkt über `read_repo`) | 13 von 22 |
| nach den Korrekturen unten | **21 von 22** (ohne die `MISLEADING`-Einträge, die nicht im Ausschnitt stehen müssen) |

Offen: `kipf_gcn_2017 / glorot_init` (`gcn/layers.py`, `glorot(`): die Datei steht als importierte Schicht im Ausschnitt, wird aber durch das Zeichenbudget (15 000) abgeschnitten. Kipf besteht heute (3 von 3), der Fakt ist kein R-/E-Fehlschlag.

## Fehler, die V0 gefunden hat (alle korrigiert, mit Tests in `tests/test_spec_loop.py`)

1. `_paper_id` wurde nie gesetzt, die Fixture je Einheit (S4) war damit totes Code. Behoben in `loop.run`.
2. `focused_excerpt` brach beim ersten kurzen Stück (< 200 Zeichen, z. B. ein `__init__.py` oder eine kurze Konfiguration) ab und verlor alle folgenden Stücke (NLinear: Startskript fehlte, damit lr und seq_len).
3. Stücke wurden nach Dateiname dedupliziert: zweite Funktionsrumpf derselben Datei (`load_citation`, `preprocess_citation` in `utils.py`) fiel weg (SGC).
4. `from gcn.layers import *` wurde vom Import-Muster nicht erkannt (Kipf). Neues Muster nur im Steckbrief-Pfad; das alte `IMPORT_RE` bleibt für v4 unverändert.
5. Konfigurationsdatei des Modells (`config/optimized/gat.conf.yaml`) wurde nie gelesen oder zeigte sich nicht im Ausschnitt (Shchur). Neu: `model_config_in_listing` (Pflichtlesen) und Konfigurationsstücke im Ausschnitt.
6. Rümpfe von Vorverarbeitungs- und Schedule-Funktionen kamen nur aus den Zitaten der Setup-Details. Neu: auch aus Startskripten, Modell- und Trainingsdatei (nur Namen, die `norm|preprocess|schedul|adjust_|scale|standardi|split|load_` enthalten), mit einer zweiten Stufe.

## Überanpassung

Die Korrekturen 5 und 6 sind aus den Befunden der Dev-Einheiten entstanden. Sie hängen an Namensmustern, nicht an Werten oder Paper-Namen. Ob sie auf SparseTSF und RLinear tragen, zeigt erst V1.

## Rohausgabe

```
Auswahl der LLM-Dateiwahl (select_files) = die Dateien der gespeicherten Fixture (Proxy, offline); dazu die deterministischen S4-Schritte aus repo_source.read_repo: Pflicht-Modelldatei, relative Imports zwei Ebenen, Quelltext-Filter, Zeichenbudget. 

zeng_dlinear_2023  model=DLinear  read=20 files, excerpt=13212 chars, pieces=10
   seq336                           scripts/EXP-LongForecasting/Linear/etth1.sh        IN EXCERPT
   lr0.005                          scripts/EXP-LongForecasting/Linear/etth1.sh        IN EXCERPT
   halve_type1                      utils/tools.py                                     IN EXCERPT
   Ausschnitt-Stuecke: repo:models/DLinear.py@0c11366, repo:scripts/EXP-LongForecasting/Linear/etth1.sh@0c11366, repo:scripts/EXP-LongForecasting/Linear/univariate/etth1.sh@0c11366, repo:utils/tools.py@0c11366 (function body), repo:data_provider/data_loader.py@0c11366 (function body), repo:data_provider/data_factory.py@0c11366 (function body), repo:exp/exp_main.py@0c11366 (function body), repo:utils/tools.py@0c11366 (function body), repo:utils/metrics.py@0c11366 (function body), repo:exp/exp_main.py@0c11366 (function body) 

zeng_nlinear_2023  model=NLinear  read=21 files, excerpt=7948 chars, pieces=8
   seq336                           scripts/EXP-LongForecasting/Linear/etth1.sh        IN EXCERPT
   lr0.005                          scripts/EXP-LongForecasting/Linear/etth1.sh        IN EXCERPT
   halve_type1                      utils/tools.py                                     IN EXCERPT
   subtract_last                    models/NLinear.py                                  IN EXCERPT
   Ausschnitt-Stuecke: repo:models/NLinear.py@0c11366, repo:scripts/EXP-LongForecasting/Linear/etth1.sh@0c11366, repo:scripts/EXP-LongForecasting/Linear/univariate/etth1.sh@0c11366, repo:utils/tools.py@0c11366 (function body), repo:data_provider/data_loader.py@0c11366 (function body), repo:utils/timefeatures.py@0c11366 (function body), repo:utils/tools.py@0c11366 (function body), repo:models/NLinear.py@0c11366 (function body) 

wu_sgc_2019  model=SGC  read=8 files, excerpt=9170 chars, pieces=10
   row_normalize                    utils.py                                           IN EXCERPT
   Ausschnitt-Stuecke: repo:models.py@2c7a272, repo:utils.py@2c7a272 (function body), repo:utils.py@2c7a272 (function body), repo:args.py@2c7a272 (function body), repo:utils.py@2c7a272 (function body), repo:models.py@2c7a272 (function body), repo:utils.py@2c7a272 (function body), repo:metrics.py@2c7a272 (function body), repo:citation.py@2c7a272 (function body), repo:citation.py@2c7a272 (function body) 

velickovic_gat_2018  model=GAT  read=10 files, excerpt=15327 chars, pieces=7
   feat_rownorm                     execute_cora.py                                    IN EXCERPT
   Ausschnitt-Stuecke: repo:models/gat.py@5af87e7, repo:execute_cora.py@5af87e7, repo:execute_cora_sparse.py@5af87e7, repo:utils/process.py@5af87e7 (function body), repo:utils/process.py@5af87e7 (function body), repo:utils/process.py@5af87e7 (function body), repo:utils/layers.py@5af87e7 

velickovic_dgi_2019  model=DGI  read=11 files, excerpt=9095 chars, pieces=7
   one_gcn_encoder                  models/dgi.py                                      IN EXCERPT
   hid512                           execute.py                                         IN EXCERPT
   lr0.001                          execute.py                                         IN EXCERPT
   prelu                            execute.py                                         IN EXCERPT
   feat_rownorm                     execute.py                                         IN EXCERPT
   gcn_layer_body                   layers/gcn.py                                      IN EXCERPT
   Ausschnitt-Stuecke: repo:models/dgi.py@61baf67, repo:execute.py@61baf67, repo:utils/process.py@61baf67 (function body), repo:utils/process.py@61baf67 (function body), repo:utils/process.py@61baf67 (function body), repo:layers/gcn.py@61baf67, repo:layers/discriminator.py@61baf67 

shchur_gat_2018  model=GAT  read=18 files, excerpt=15239 chars, pieces=5
   gat_model_file                   gnnbench/models/gat.py                             IN EXCERPT
   gat_config                       config/optimized/gat.conf.yaml                     IN EXCERPT
   patience50                       config/train.conf.yaml                             IN EXCERPT
   Ausschnitt-Stuecke: repo:gnnbench/models/gat.py@1e72912, repo:gnnbench/train.py@1e72912, repo:config/reference/gat.conf.yaml@1e72912, repo:config/optimized/gat.conf.yaml@1e72912, repo:config/train.conf.yaml@1e72912 

shchur_pitfalls_2018  model=GCN  read=17 files, excerpt=15416 chars, pieces=8
   glorot_init                      gnnbench/models/gcn.py                             IN EXCERPT
   hid64                            config/optimized/gcn.conf.yaml                     IN EXCERPT
   Ausschnitt-Stuecke: repo:gnnbench/models/gcn.py@1e72912, repo:gnnbench/train.py@1e72912, repo:config/optimized/gcn.conf.yaml@1e72912, repo:config/reference/gcn.conf.yaml@1e72912, repo:config/train.conf.yaml@1e72912, repo:gnnbench/models/gcn.py@1e72912 (function body), repo:gnnbench/models/gcn.py@1e72912 (function body), repo:gnnbench/models/base_model.py@1e72912 

kipf_gcn_2017  model=GCN  read=7 files, excerpt=15264 chars, pieces=6
   feat_rownorm                     gcn/utils.py                                       IN EXCERPT
   glorot_init                      gcn/layers.py                                      MISSING (Fakt abgeschnitten)
   Ausschnitt-Stuecke: repo:gcn/models.py@39a4089, repo:gcn/train.py@39a4089, repo:gcn/utils.py@39a4089 (function body), repo:gcn/utils.py@39a4089 (function body), repo:gcn/utils.py@39a4089 (function body), repo:gcn/layers.py@39a4089 

Fehlende Fakten im Ausschnitt: 1
```
