# Eichung des Methodentreue-Prüfers v3 (letzte Eichung, neue Codes)

Vorab festgelegt und committet **vor dem Lauf**. Nichts wird nach Sicht der Ergebnisse geändert.
Voraussetzung laut `results/calibration_fidelity_v2.md`: neue Codes, v1 und v2 sind verbraucht.

## Änderungen am Prüfer seit v2 (vor Erstellung der Codes eingefroren)
- Prompt: Bei verlangtem Wert muss `code_quote` die Zeile enthalten, die den Wert setzt (Zuweisung, Standardargument,
  Aufrufargument), nicht nur eine Zeile, die eine Variable benutzt (Ursache des v2-Fehlers `moving_average_kernel_25`, vermutet).
- Code prüft zusätzlich per AST, ob ein Standardargument `param=WERT` beim Aufruf mit einer Konstante überschrieben wird
  (`Model(a, b, 5)`, `Model(a, b, kernel_size=49)`) → Baustein `teilweise`. Variablen als Argument sind nicht entscheidbar
  und zählen nicht als Überschreibung.
- Temperatur: Hauptaufruf bleibt 0,0 (war schon so; die Streuung des v2-Falls `scatter` kommt aus dem Serving, nicht
  aus der Temperatur). Die abweichende Temperatur 0,2 gilt nur für den Wiederholungsversuch nach einem technischen Ausfall.
- **Der Prompt des Prüfers bleibt ab jetzt unverändert**, bis das Ergebnis vorliegt.

## Eichset
`tests/fixtures/agent_codes_v3/` (13 Codes), Labels aus der Konstruktion in `benchmark/components/gold_labels_v3.json`
(Begründung je Code dort).

| Klasse | Codes |
|---|---|
| ja (4) | dl3_module_constant_kernel, dl3_default_kernel_effective, dl3_cumsum_moving_average, gcn3_sparse_tensor_faithful |
| teilweise (5) | dl3_default_overridden_positional (`Model(.., 5)`), dl3_default_overridden_keyword (`kernel_size=49`), dl3_trend_only, gcn3_no_selfloops_symmetric, gcn3_three_layers |
| nein (2) | dl3_tiny_transformer, dl3_gru |
| nein, `dead_code` (2) | dl3_decomposition_overwritten (Ergebnis in der nächsten Zeile überschrieben), gcn3_propagation_result_discarded (erste Schicht berechnet und überschrieben) |

## Aufbau des Laufs
Wie v2: Token Factory, Rolle `extract` (Super), Temperatur 0,0, Cache aus, jeder Code 3×, Mehrheitsurteil pro Code,
kein Mehrheitsurteil zählt als falsch, ein Lauf, `src/` unverändert.
Aufruf: `LLM_PROVIDER=tokenfactory uv run python scripts/calibrate_fidelity.py` (Standard v3).

## Erfolgsregel (nur über die 11 Codes ohne `dead_code`)
1. mindestens **9 von 11** Mehrheitsurteile entsprechen dem Gold-Label (≈ 82 %, ähnlich streng wie 8/10) **und**
2. bei keinem dieser Codes sind ja und nein vertauscht.
Die `dead_code`-Fälle werden nur berichtet; ausgeführten, aber verworfenen Code sieht ein Existenzprüfer nicht
(bei GCN deckt das die Graph-Ablation ab, `ablation_gcn.md`).

## Konsequenz bei Scheitern (jetzt festgelegt, endgültig)
**Achse 2 (Methodentreue durch den Prüfer) bleibt für den Benchmark „nicht geprüft“, und das steht offen im README**
(Eichung v1, v2, v3 gescheitert, mit Ergebnissen). Der Prüfer ist dann ein dokumentierter offener Punkt, keine Baustelle mehr.
Für GCN bleibt die Graph-Ablation als deterministische Methodenprüfung (nur herabstufend).
Es gibt keine weitere Eichung vor der Abgabe.

## Konsequenz bei Erfolg
Achse 2 wird für Papers mit eingefrorener Bausteinliste aus dem Prüfer befüllt, mit dem Hinweis „Existenzprüfung,
keine Nutzungsprüfung“; Kombination mit Abdeckung/Ablation nur herabstufend (`calibration_fidelity_v2.md`).

## Grenzen
Selbst geschriebene Varianten können von echten Agenten-Codes abweichen; Labels von Claude Code, nicht vom Team gegengelesen.
