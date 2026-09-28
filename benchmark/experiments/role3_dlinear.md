# Experiment: Findet Rolle 3 (Einwände sammeln) einen Einwand, und verändert er das Ergebnis? — DLinear/ETTh1

Vorab festgelegt, vor jedem Rolle-3-Aufruf. Nichts wird nach Sicht der Ergebnisse geändert.
Auswertung mit **Median**, nicht Mittelwert (Lehre aus `role4_dlinear.md`: einzelne kaputte
Generierungen dominieren Mittelwerte).

## Ausgangslage (belegt, vor dem Experiment)
- Im DLinear-Repo (`cure-lab/LTSF-Linear`, Commit `0c11366`) steht in `data_provider/data_factory.py`
  beim Test `drop_last = False`, bei Training/Validierung `True` (Rolle 4, wörtliches Zitat).
- Der Harness wertet alle 2785 Testfenster aus; seine Repeat-Baseline (1,2944 / 0,7132) trifft die
  Paper-Werte (1,295 / 0,713).
- Die Vermutung, dass Repos dieser Familie beim Test unvollständige Batches verwerfen (TFB, 2024),
  ist eine Erinnerung und **nicht belegt**. Ob Tavily sie findet, ist Teil der Frage.

## Stufe 1: Entdeckung (kein Agent)
- Ein Aufruf `TavilyTools.collect_objections("Are Transformers Effective for Time Series
  Forecasting?", limit=5)` (Search plus Crawl mit `limit`), Ergebnis wird als Fixture gespeichert.
- Extraktion (Rolle `extract`, abschnittsweise wie Rolle 4): Aussagen über Auswertungsprotokoll oder
  Reproduzierbarkeit der Paper-Ergebnisse auf ETTh1. Jede Aussage braucht ein wörtliches Zitat, das im
  Quelltext steht; der Quelltext muss „DLinear“ oder „LTSF“ nennen.
- Zusätzlich deterministischer Abgleich mit dem Repo: passt der Einwand zum Code
  (z. B. behauptet er `drop_last=True` beim Test, das Repo hat `False`)? Ergebnis: bestätigt,
  widerlegt oder nicht prüfbar.
- **Erfolg Stufe 1:** mindestens ein verifizierter Einwand zum Auswertungsprotokoll (Stichwörter:
  drop_last, letzter Batch, Testset, Leakage, Hyperparameter-Suche auf Testdaten). Kein Fund ist ein
  gültiges Ergebnis und wird so berichtet.

## Stufe 2: Test (nur wenn Stufe 1 einen Protokoll-Einwand findet)
- Kein Agent, sondern die menschlich geschriebene Referenz-DLinear (`tests/reference/dlinear_reference.py`,
  mit Lernraten-Halbierung), 5 Seeds, dieselben Vorhersagen zweimal ausgewertet:
  vollständig (2785 Fenster) und mit verworfenem letzten unvollständigen Batch (Batchgröße 32,
  also die ersten 2768 Fenster).
- Effekt Δ = MSE(verworfen) − MSE(vollständig), **Median über die Seeds**.
- **Der Einwand zählt für diesen Claim**, wenn |Δ| > 0,01 (die vorab festgelegte Toleranz).
  Sonst: „Einwand geprüft, verändert das Ergebnis nicht“. Beides wird berichtet.

## Einschränkungen (vorab)
- Stufe 2 misst die Empfindlichkeit der Metrik, nicht die Fähigkeit des Agenten.
- Ein Paper, ein Einwand-Typ, ein Tavily-Aufruf; Suchergebnisse sind nicht deterministisch.
- Der Effekt betrifft nur diesen Harness, nicht die Auswertung im Original-Code.
