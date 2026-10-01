# Benchmark v4: Vorab-Festlegung (final, vor dem ersten gezählten Lauf)

Stand 2026-10-01. **Alles wie Benchmark v3** (`benchmark_v3.md`, dort wiederum wie `benchmark_v2.md`: 8 Einheiten, Referenzurteile, Toleranzen, Seeds, Placebos, Zählregel "Mehrheit über 3 Generierungen", Arme A ohne / B mit Tavily, Rollen 2/4 aus Fixtures, Rolle 5 live in Arm B, LLM-Cache aus). Unterschied: Code-Stand mit Block A komplett, A1-A3 (`docs/roadmap_quality.md`), Budgetstopp 8 $ brutto statt 6 $ (drei Code-Kandidaten je Claim).

## Änderungen gegenüber v3 (nur Produktpfad `graph_repair=True`, beide Arme)

Grundlage: Fehlermuster `failure_patterns_v3.md` (M1-M7) und `results/benchmark_v3.md` (2 von 8 Einheiten in beiden Armen, Hauptursache "keine Mehrheit" und knapp daneben).

1. **A1, drei Code-Kandidaten mit Auswahl über Validierung** (`fa39c48`): Nach dem Plan schreibt `code` drei Kandidaten (gleicher Prompt, Temperatur 0,2/0,5/0,8). Jeder läuft Seed 0 mit Reparatur- und Fehler-Check. Der Agenten-Code schreibt zusätzlich `/output/val_predictions.csv`; **unser Harness** rechnet die Validierungs-Metrik mit selbst geladenen Validierungslabels (Cora `val_idx`, Cora-LCC Validierungsknoten des Splits, ETTh1 117 Validierungsfenster mit Schrittweite 24, Zielbereich ganz im Validierungsteil). Gewählt wird die beste Validierungs-Metrik (Genauigkeit max, MSE min, Gleichstand kleinerer Index). Testwerte und Zielwert des Papers gehen nie in die Auswahl. Nur der Sieger bekommt Graph-Ablation und die übrigen Seeds; Seed 0 des Siegers wird wiederverwendet. Verlierer bleiben als Versuche im Dossier.
2. **A3-M1, Vorhersage-Zuordnung** (`5ab9141`): Checkliste "pred[test_ids]"; liegt die Validierungs-Genauigkeit unter der Mehrheitsklasse-Baseline auf Validierung (Harness), Reparatur mit festem Hinweis ohne Zahlen (teilt das Budget der Fehler-Reparaturen).
3. **A3-M3, verbotene Bibliotheken** (`e124892`): Hinweis "torch_geometric, dgl, ogb usw. nicht installiert, reines PyTorch" schon im ersten Code-Prompt.
4. **A3-M7, Repo-Standardwerte** (`a890ef0`): Setup-Angaben aus `argparse`-Defaults im Prompt als "repository default (the paper may tune this per dataset)" markiert; widersprechen sich Paper und Repo beim selben Thema, nennt der Plan-Prompt beide mit Quelle (wirkt nur in Arm B, Arm A hat keine Setup-Angaben).

5. **A2, Abgleich mit dem Original-Code** (`43dfffb`, nur wenn Repo-Dateien vorliegen, also nur Arm B): Nach Auswahl und Graph-Check vergleicht ein Aufruf (Rolle `extract`, Super, Temperatur 0) unseren Code mit den Repo-Dateien (höchstens 60 000 Zeichen, deterministisch gekürzt) und listet nur Abweichungen in festen Aspekten (Dropout, Normalisierung, Initialisierung, Weight Decay, Abbruchkriterium). Jede Abweichung braucht ein wörtliches Zitat aus der Repo-Datei und eines aus unserem Code (Prüfung wie Rolle 4), sonst verworfen. Geprüfte Abweichungen führen zu genau einer Reparaturrunde; der neue Code ersetzt den alten nur, wenn er Fehler-Check und Graph-Ablation besteht und die Validierungs-Metrik gleich oder besser ist. Modus bleibt "Nachbau" (Repo nie im Branch). Kein Zielwert, kein Claim-Text, keine Messzahl im Prompt.

Keine Änderung an Test-Harness, Toleranzen, Referenzurteilen, Zielwerten, Fixtures. Leak-Schutz: Die Validierungslabels lädt nur der Harness; die Validierungsdaten selbst hat der Agent schon immer (Standard-Split). Keine Test- oder Validierungszahl geht in einen Prompt.

## Kennzahlen und Erwartung (vorab)

- Primär wie v2/v3: Übereinstimmung nach Mehrheitsregel je Arm (von 8), Placebos getrennt (Anteil "reproduziert" muss 0 bleiben oder wird offen berichtet).
- Sekundär: Zahl der Einheiten mit "keine Mehrheit" (v3 als Vergleich), Ausfall-Urteile je Arm, Anteil der Läufe, in denen der gewählte Kandidat nicht Kandidat 0 ist (aus Ereignis `candidates`).
- Erwartung: mehr Übereinstimmungen als v3 (2 von 8) in mindestens einem Arm. Ehrlich offen; keine Signifikanzaussage bei 8 Einheiten.
- Kosten: v3 kostete 3,10 $ brutto für 48 Läufe; erwartet 4-6 $ brutto. Stopp bei 8 $ brutto im Store.
- **v2, v3 und v4 stehen alle im README**, auch wenn v4 nicht besser ist.

## Durchführung

Code-Stand: Commit dieser Datei (enthält Flag `--v4`, Store `runs/benchmark_v4.jsonl`, Konfiguration `v4_block_a`). Drei Prozesse:

```
LLM_PROVIDER=tokenfactory LLM_CACHE=off MAX_PARALLEL_SEEDS=3 .venv/bin/python scripts/run_benchmark.py --v4 shchur_pitfalls_2018 kipf_gcn_2017 wu_sgc_2019
LLM_PROVIDER=tokenfactory LLM_CACHE=off MAX_PARALLEL_SEEDS=3 .venv/bin/python scripts/run_benchmark.py --v4 shchur_gat_2018 velickovic_gat_2018 velickovic_dgi_2019
LLM_PROVIDER=tokenfactory LLM_CACHE=off MAX_PARALLEL_SEEDS=3 .venv/bin/python scripts/run_benchmark.py --v4 zeng_dlinear_2023 zeng_nlinear_2023
```

Auswertung: `.venv/bin/python scripts/eval_benchmark_v2.py runs/benchmark_v4.jsonl > benchmark/experiments/results/benchmark_v4.md`, Fehlerprotokoll `scripts/failure_log.py runs/benchmark_v4.jsonl`. `src/` auf `main` eingefroren, bis alle 48 Läufe fertig sind. Probeläufe vor dem Start (Kipf und DLinear Arm A, GAT Arm B mit A2, je 2 Seeds, Rolle 5 aus, außerhalb des Stores, `logs/v4_probe/`) prüfen nur, dass der neue Pfad gegen die echte Sandbox läuft; sie zählen nicht.
