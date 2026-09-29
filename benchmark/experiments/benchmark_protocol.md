# Benchmark-Protokoll: mit und ohne Tavily, mit Placebo-Claims

Vorab festgelegt und committet **vor dem ersten Benchmark-Lauf**. Nichts wird nach Sicht der Ergebnisse geändert. Prod-Modelle (Token Factory), Ergebnisse dürfen ins README.

## Problem, das dieses Protokoll löst
Bei allen vier Papers ist das richtige Urteil „reproduziert“. Ein Agent, der immer „reproduziert“ sagt, käme auf 100 %. Deshalb (a) ein Referenzurteil aus einer menschlichen Referenzlösung statt aus einer nicht vorhandenen Vergleichsstudie und
(b) je Paper ein **Placebo-Claim** mit absichtlich falschem Zielwert, dessen richtiges Urteil „nicht reproduziert“ ist. Placebos sind **konstruierte Kontrollen, keine echten Papers**; sie ersetzen den echten negativen Fall (Dacrema, Entscheidung 12. Oktober) nicht.

## Ground Truth: das Referenzurteil
Das richtige Urteil eines Claims ist das Urteil, das die **menschliche Referenzlösung im selben Harness** erzielt (Achse 1, Mittel über die Seeds des Papers gegen Zielwert und Toleranz, Regel ≤ Toleranz reproduziert, ≤ 3 × Toleranz teilweise, sonst nicht reproduziert). Referenzmessungen liegen vor dem Lauf fest:

| Paper | Referenz (Mittel) | Seeds | Echter Claim: Ziel ± Tol. | Referenzurteil | Placebo: Ziel | Referenzurteil Placebo |
|---|---|---|---|---|---|---|
| Kipf & Welling, GCN | 0,8136 (`results/cora_references.txt`) | 10 | 0,815 ± 0,01 | reproduziert | 0,88 | nicht reproduziert (Abstand 0,066 > 0,03) |
| Wu, SGC | 0,8081 (`results/cora_references.txt`) | 10 | 0,81 ± 0,01 | reproduziert | 0,88 | nicht reproduziert (0,072) |
| Zeng, DLinear (MSE, Horizont 96) | 0,3809 (`results/etth1_harness_reference.txt`) | 5 | 0,375 ± 0,01 | reproduziert (Abstand 0,0059) | 0,300 | nicht reproduziert (0,081) |
| Shchur, GCN (zufällige Splits) | 0,8107 (`results/lcc_harness_reference_30x1.txt`) | 30 | 0,815 ± 0,01 | reproduziert (0,0043) | 0,88 | nicht reproduziert (0,069) |

Referenzlösungen: `tests/reference/*_reference.py`. Sie sind Selbsttests der Pipeline, keine Agentenergebnisse.

## Aufbau
- Pro Paper und Arm **3 frische Generierungen** (LLM-Cache aus), je Generierung ein Lauf mit **beiden Claims** (echt und Placebo, je eigener Plan, Code und Versuchen). 4 Papers × 2 Arme × 3 = 24 Läufe = 24 Urteile echt + 24 Placebo.
- **Arm A (ohne Tavily):** keine Rollen 2 und 4. **Arm B (mit Tavily):** Rolle 2 (Zielwert belegt, Spaltenprüfung) und Rolle 4 (Setup-Details aus Paper und, wenn verlinkt, Repo). Rollen 3, 5, 6, 7 sind nicht in den Loop eingebunden und **nicht Teil dieses Benchmarks**.
- **Gleiche Zielwerte in beiden Armen:** Arm A nutzt die Werte aus `papers.yaml`, Arm B den mit Tavily belegten Wert. Beide sind gleich (Prüfung im Lauf: `target_matches_yaml`); weicht ein belegter Wert ab, wird das gemeldet und der Lauf zählt trotzdem, mit Vermerk.
- Seeds je Paper wie konfiguriert (GCN 10, SGC 10, DLinear 5, Shchur 30 Splits × 1 Initialisierung). DLinear nur der MSE-Claim. Modus „Nachbau“, gleicher Harness, gleiche Isolation, Formprüfung und Graph-Ablation aktiv. `src/` bleibt während des Benchmarks unverändert.

## Zählregeln
- **Übereinstimmung:** Achse-1-Urteil des Agenten **gleich** dem Referenzurteil (dreistufig: reproduziert / teilweise / nicht reproduziert). „Nicht reproduzierbar (Setup)“, „Agenten-Fehler vermutet“ und „abgebrochen“ zählen als **Ausfall**, nie als Übereinstimmung, und werden getrennt berichtet.
- Berichtet wird **getrennt für echte Claims und Placebos** und je Arm: Übereinstimmungen (von 12), Ausfälle, sowie die Urteilsverteilung. Zusätzlich (sekundär, milder): echter Claim „nah“ = reproduziert oder teilweise.
- **Achse 2 (Methodentreue) ist überall „nicht geprüft“** (Eichungen gescheitert); jedes Urteil ist „vorläufig“. Die Graph-Ablation wird dort berichtet, wo sie anwendbar ist (GCN, SGC, Shchur): Anzahl „Graph nicht genutzt“ / „nicht messbar“.
- Keine Signifikanzaussagen: 12 gegen 12 Urteile sind Zählwerte mit Vorbehalt, keine Tests.

## Berichtsvorlage (Zielsatz für das README)
„Bei echten Claims stimmten mit Tavily X von 12 Urteilen mit dem Referenzurteil überein, ohne Tavily Y von 12. Bei Placebo-Claims erkannte der Agent mit Tavily P, ohne Tavily Q von 12 als nicht reproduziert. Ausfälle: … Alle Urteile vorläufig, Methodentreue nicht geprüft.“

## Grenzen (vorab)
- Nur vier Papers, alle mit Referenzurteil „reproduziert“; Placebos sind konstruiert. Kein echter negativer Fall.
- Das Referenzurteil stammt aus **unseren** Referenzlösungen im **eigenen** Harness; es ist nicht unabhängig von unseren Protokollentscheidungen.
- Der Agent sieht den (Placebo-)Zielwert im Prompt und könnte darauf hintrainieren wollen; der Harness verhindert Fälschung, nicht Feinabstimmung auf Validierungsdaten.
- Arm B unterscheidet sich außer durch Tavily auch durch Promptlänge.
- Achse 2 fehlt: ein „reproduziert“ heißt „Zielwert erreicht“, nicht „Methode reproduziert“ (Beispiel Ridge-Regression bei DLinear).
- Rolle 3 (Einwände) ist in diesem Benchmark nicht enthalten.
