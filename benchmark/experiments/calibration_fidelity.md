# Eichung des Methodentreue-Prüfers (`fidelity.check_fidelity`)

Vorab festgelegt, vor dem ersten Eichlauf. Nichts wird nach Sicht der Ergebnisse geändert.

## Aufbau
- Prüfling: `check_fidelity` mit Rolle `extract` (Super; Ultra nur für Plan und Endurteil), Temperatur 0,0,
  Token Factory (`LLM_PROVIDER=tokenfactory`), LLM-Cache aus (sonst wären die drei Wiederholungen identisch).
- Eichset: die 10 Codes in `tests/fixtures/agent_codes/` mit Gold-Labels in
  `benchmark/components/gold_labels.json` (von Hand eingeordnet, vor dem Lauf eingefroren).
  Der schwere Fall (`gcn_variant_adjacency_unused`) zählt mit.
- Jeder Code wird **3×** geprüft. Pro Code zählt das **Mehrheitsurteil** der drei (ja/teilweise/nein).
  Kein Mehrheitsurteil (alle drei verschieden) zählt als falsch.
- Ein Lauf. `src/` bleibt währenddessen unverändert.

## Erfolgsregel
Der Prüfer gilt als geeicht, wenn
1. mindestens **8 von 10** Mehrheitsurteile dem Gold-Label entsprechen **und**
2. bei keinem Code ja und nein vertauscht sind (Gold ja → Mehrheit nein oder Gold nein → Mehrheit ja).

Zusätzlich berichtet (keine Regelbedingung): bei wie vielen Codes die drei Urteile nicht übereinstimmen,
Kosten, und die Bausteinergebnisse je Code.

## Konsequenz bei Scheitern (jetzt festgelegt)
- Achse 2 (Methodentreue) des Zeugnisses bleibt „nicht geprüft“, die Gesamtaussage „vorläufig“.
- Wird der Prüfer danach verbessert, läuft die neue Eichung auf **neuen** Codes, nicht auf denselben 10.
  Sonst wäre der Prüfer auf diese Fälle zugeschnitten und die Eichung beweist nichts mehr.

## Konsequenz bei Erfolg
Achse 2 wird aus dem Prüfer befüllt (ja/teilweise/nein). Ein Erfolg gilt nur für Papers mit
eingefrorener Bausteinliste (`benchmark/components/*.json`), nicht allgemein.
