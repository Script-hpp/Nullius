# v5 Schritt A: Änderungen am Auswertungsskript (Nachbesserungsrunde 1)

Vorab-Festlegung (`benchmark/experiments/v5_step_a.md`) gilt weiter: Soll-Werte, Soll-Punkte und die drei Ziele sind **unverändert**. Geändert wurden nur
Erkennungsmuster und Auswertungslogik in `scripts/measure_setup_fidelity.py` und die `patterns`/`misleading`-Muster in `v5_step_a_soll.yaml`.
Grund: Befund der Diagnose (Auftrag 29) — mehrere "falsch"-Wertungen waren Scorer-Artefakte, keine Fehler der Extraktion.
Beide Labels (basis, v1) werden mit demselben (neuen) Skript ausgewertet; alte und neue Wertung stehen unten.

## Liste der Änderungen

| # | Änderung | Betrifft | Begründung (Befund) |
|---|---|---|---|
| 1 | Wertformen werden vor dem Muster kanonisiert (`canonical_value`): `5e-4` = `5 · 10 −4` = `5·10⁻⁴` = `5 × 10^-4` = `0.0005` (Dezimalform wird angehängt; Superscript-Ziffern, Unicode-Minus, `\times`/`\cdot`) | alle Punkte mit Zahlenwert | Kipf weight_decay 3x "falsch", weil "5 · 10 −4" das Muster `5e-?0?4` verfehlte (Basis hatte 2x die Form `5e-4`) |
| 2 | Schalterwerte bei Normalisierungs-Themen: `false/no/none/off` → "not normalised", `true/yes/on` → "normalised" | feature_norm (alle Einheiten) | Shchur: `normalize_features: false` (Wert "False") wurde vom Muster `not normali` nicht erkannt |
| 3 | `feature_norm.wrong_unless: graph\|procedure\|dataset_graph\|hyperparameter\|selection`: die falsch-Regel (u. a. `standardi[sz]`) gilt nicht für Werte, die erkennbar Graph-Standardisierung oder das Trainingsprotokoll beschreiben | feature_norm aller 6 Einheiten | "standardized training and hyperparameter selection procedure" und `dataset_graph.standardize()` sind keine Merkmals-Standardisierung |
| 4 | Shchur feature_norm `wrong` ergänzt um `normalised` (ohne vorangehendes not/un/non-) | shchur_pitfalls, shchur_gat | damit `normalize_features: true` für Shchur (Soll: NICHT normalisiert) als falsch zählt, nachdem Schalterwerte erkannt werden |
| 5 | Kipf `epochs.quote_value`: `200 epochs/iterations` im **Zitat** einer Angabe mit anderem Thema (z. B. `optimizer`-Zeile) zählt als korrekt | kipf epochs | "We train all models for a maximum of 200 epochs" stand nur im Zitat der optimizer-Angabe, Thema passte nicht auf `epoch` |
| 6 | Kipf `epochs.wrong_unless: no early stopping\|without early stopping\|depth\|layers`: "400 epochs (no early stopping)" (Anhang-Experiment zur Modelltiefe) ist nur dann falsch, wenn es nicht erkennbar ein anderes Protokoll beschreibt; ohne korrekte Angabe bleibt es "unklar" (Zweifelsfall) | kipf epochs | Reviewer-Vorgabe: falsch nur, wenn als gültig für das Hauptexperiment geführt |
| 7 | Rohantworten je LLM-Aufruf und Verwurfsgründe je Chunk werden bei `--label` gespeichert: `runs/v5_step_a/<label>/raw/<einheit>_<wdh>.json` (`raw_trace`, Verwurfsgründe deterministisch mit `verify_raw` nachgerechnet) | Messlauf | Diagnose: ohne Rohantworten war nicht entscheidbar, ob das Modell `[]` liefert oder die Prüfung verwirft (DGI) |

Keine Änderung an: Soll-Liste, Zielschwellen, Ziel-3-Regeln (`misleading`), Zweifelsfall-Behandlung (unklar zählt nie als korrekt).

## Wirkung auf die vorhandenen Messungen (alt → neu, Punkt × Wiederholung)

Alt = Skript und Muster von `origin/main` vor dieser Änderung. Es ändern sich nur Wertungen aus den Mustern 1 bis 6; die Extraktionen selbst sind unverändert.

    basis kipf_gcn_2017          weight_decay    Wdh 1: falsch -> korrekt
    basis kipf_gcn_2017          epochs          Wdh 2: falsch -> korrekt
    basis shchur_gat_2018        feature_norm    Wdh 0: falsch -> unklar
    basis shchur_gat_2018        feature_norm    Wdh 1: unklar -> korrekt
    basis shchur_pitfalls_2018   feature_norm    Wdh 0: falsch -> korrekt
    v1    kipf_gcn_2017          weight_decay    Wdh 0: falsch -> korrekt
    v1    kipf_gcn_2017          weight_decay    Wdh 2: falsch -> korrekt
    v1    kipf_gcn_2017          epochs          Wdh 0: falsch -> korrekt
    v1    kipf_gcn_2017          epochs          Wdh 1: falsch -> korrekt
    v1    shchur_gat_2018        feature_norm    Wdh 1: unklar -> korrekt
    v1    shchur_gat_2018        feature_norm    Wdh 2: unklar -> korrekt
    v1    shchur_pitfalls_2018   feature_norm    Wdh 0: falsch -> unklar
    v1    shchur_pitfalls_2018   feature_norm    Wdh 1: falsch -> korrekt
    v1    shchur_pitfalls_2018   feature_norm    Wdh 2: falsch -> unklar
    {('basis', 'falsch', 'korrekt'): 3, ('basis', 'falsch', 'unklar'): 1, ('basis', 'unklar', 'korrekt'): 1, ('v1', 'falsch', 'korrekt'): 5, ('v1', 'unklar', 'korrekt'): 2, ('v1', 'falsch', 'unklar'): 2}
