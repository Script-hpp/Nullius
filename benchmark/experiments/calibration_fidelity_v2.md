# Eichung des Methodentreue-Prüfers v2 (neue Codes)

Vorab festgelegt und committet **vor dem Lauf**. Nichts wird nach Sicht der Ergebnisse geändert.
Voraussetzung laut `results/calibration_fidelity_v1.md`: neue Codes, die 10 aus v1 sind verbraucht.

## Was sich seit v1 geändert hat (Prüfer, vor Erstellung der Codes eingefroren)
- Technischer Ausfall ist nie ein Urteil: unlesbare/abgeschnittene Antwort → ein Wiederholungsversuch
  (doppelte Ausgabegrenze, andere Temperatur), danach `nicht geprueft`; fehlt ein tragender Baustein in der
  Antwort → Gesamtergebnis `nicht geprueft`, nicht `nein`.
- Code-Zitate: Leerraum, Zeilenumbrüche, typografische Striche/Anführungszeichen und das Zeilennummern-Präfix
  werden vereinheitlicht, Bezeichner (case-sensitiv) und Zahlen bleiben exakt. Der Wert-Check verlangt den Wert
  als ganzes Token (`5` steht nicht in `kernel_size=25`).
- Kein strukturierter Ausgabemodus (nicht geprüft, ob Token Factory ihn anbietet).
- **Der Prompt des Prüfers bleibt ab jetzt unverändert**, bis das Ergebnis vorliegt.

## Eichset
`tests/fixtures/agent_codes_v2/` (12 Codes), Labels in `benchmark/components/gold_labels_v2.json`.
Die Labels entstehen **aus der Konstruktion** (die Variante wurde absichtlich so geschrieben), nicht aus
einer nachträglichen Einschätzung; Begründung je Code im JSON.

| Klasse | Codes |
|---|---|
| ja (4) | dlinear_new_faithful_replicate, dlinear_new_faithful_conv_kernel, gcn_new_faithful_dense, gcn_new_faithful_scatter |
| teilweise (4) | dlinear_new_kernel_13 (falscher Kernel), dlinear_new_shared_linear (ein Zweig statt zwei), gcn_new_row_normalised, gcn_new_one_layer |
| nein (2) | dlinear_new_mlp_no_decomposition, dlinear_new_nlinear |
| nein, `dead_code` (2) | dlinear_new_decomposition_discarded (Zerlegung berechnet, verworfen), gcn_new_adjacency_stored_unused (Adjazenz gebaut, nie benutzt) |

Subjektiver Fall, offen benannt: `gcn_new_row_normalised` ist `teilweise`, weil die Renormalisierung im Paper
symmetrisch ist (D̃^-1/2 Ã D̃^-1/2); Zeilennormalisierung ist eine andere Normalisierung.

## Aufbau des Laufs
Wie v1: Token Factory, Rolle `extract` (Super), Temperatur 0,0 (Wiederholungsversuch bei Ausfall intern anders),
Cache aus, jeder Code 3×, Mehrheitsurteil pro Code, kein Mehrheitsurteil zählt als falsch, ein Lauf,
`src/` unverändert. Aufruf: `LLM_PROVIDER=tokenfactory uv run python scripts/calibrate_fidelity.py`.

## Erfolgsregel (nur über die 10 Codes ohne `dead_code`)
1. mindestens **8 von 10** Mehrheitsurteile entsprechen dem Gold-Label **und**
2. bei keinem dieser Codes sind ja und nein vertauscht.

Begründung der Trennung: Der Prüfer belegt nur, dass Code **existiert**, nicht dass er **benutzt** wird
(siehe Docstring `fidelity.py`). Die beiden `dead_code`-Fälle liegen absichtlich außerhalb dessen, was ein
Existenzprüfer sehen kann; sie gehen **nicht** in die Regel ein, werden aber berichtet, damit sichtbar bleibt,
dass ein Erfolg keine Aussage über tote Codes ist. Sie sind der Testfall für die spätere Ausführungsabdeckung.

Zusätzlich berichtet (keine Regelbedingung): Codes mit uneinigen Urteilen (alle 12), Kosten, Bausteinergebnisse.

## Konsequenzen
- **Erfolg:** Achse 2 wird für Papers mit eingefrorener Bausteinliste aus dem Prüfer befüllt, mit dem
  Hinweis „Existenzprüfung, keine Nutzungsprüfung“. Ohne Ausführungsabdeckung wird `ja` nie als
  „Methode belegt ausgeführt“ formuliert.
- **Scheitern:** Achse 2 bleibt „nicht geprüft“; eine weitere Eichung braucht wieder neue Codes.

## Grenzen (offen)
- Selbst geschriebene Varianten können von echten Agenten-Codes abweichen. Ergänzung später: echte Codes aus
  Läufen, von Hand eingeordnet.
- Die Labels stammen von der Autorin/dem Autor der Codes (Claude Code) und sind noch nicht vom Projektteam gegengelesen.

## Kombinationsregel mit der späteren Ausführungsabdeckung (vor dem Lauf festgelegt)
Die Abdeckung ist deterministisch (Unit-Tests, keine statistische Eichung) und darf ein Prüferurteil nur
**herabstufen**, nie heraufsetzen. Damit bleibt diese Eichung gültig.
- Prüfer `ja`, aber zitierte Kernzeilen im ersten vollständigen Seed-Lauf nicht ausgeführt → `nein`
  (`teilweise`, wenn nur ein Teil der tragenden Bausteine betroffen ist).
- Prüfer `teilweise` oder `nein` → unverändert.
- Abdeckung nicht verfügbar → Prüferurteil unverändert, Hinweis im Zeugnis.
- Gemessen wird im ersten vollständigen Seed-Lauf, nicht im Smoke-Lauf (Epochen-, Early-Stopping- und
  Auswertungszweige laufen dort nie).
