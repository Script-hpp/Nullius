# Ergebnis: Eichung des Methodentreue-Prüfers v3 — Regel NICHT erfüllt (8/11), endgültig

Vorab-Regel: `../calibration_fidelity_v3.md` (vor dem Lauf committet). Rohausgabe: `calibration_fidelity_v3.txt`.
Lauf: Token Factory, Rolle `extract` (Super), 3× je Code, Cache aus, Kosten 0,107 $ netto, `src/` unverändert.

| Code | Gold | Urteile (3×) | Mehrheit |
|---|---|---|---|
| dl3_module_constant_kernel | ja | teilweise ×3 | falsch |
| dl3_default_kernel_effective | ja | teilweise ×3 | falsch |
| dl3_cumsum_moving_average | ja | teilweise ×3 | falsch |
| dl3_default_overridden_positional | teilweise | teilweise ×3 | ok |
| dl3_default_overridden_keyword | teilweise | teilweise ×3 | ok |
| dl3_trend_only | teilweise | teilweise ×3 | ok |
| dl3_tiny_transformer | nein | nein, nicht geprüft, nein | ok |
| dl3_gru | nein | nein ×3 | ok |
| gcn3_sparse_tensor_faithful | ja | ja ×3 | ok |
| gcn3_no_selfloops_symmetric | teilweise | teilweise teilweise nein | ok |
| gcn3_three_layers | teilweise | teilweise ×3 | ok |
| *dead_code* dl3_decomposition_overwritten | nein | ja ja teilweise | nur berichtet (Mehrheit ja) |
| *dead_code* gcn3_propagation_result_discarded | nein | teilweise ×3 | nur berichtet |

**Regelmenge: 8/11 richtig (Schwelle 9/11), 0 ja↔nein-Vertauschungen, 3/13 Codes mit uneinigen Urteilen. Regel nicht erfüllt.**

## Festgelegte Konsequenz (vorab, endgültig)
Achse 2 (Methodentreue durch den Prüfer) bleibt für den Benchmark „nicht geprüft“, offen im README (Eichung v1, v2, v3 gescheitert).
Keine weitere Eichung vor der Abgabe. Für GCN bleibt die Graph-Ablation (`ablation_gcn_validation.md`) als deterministische, nur herabstufende Prüfung.

## Verlauf über die drei Eichungen (Regelmenge)
v1: 5/10, 2 ja↔nein-Vertauschungen. v2: 7/10, 0. v3: 8/11, 0. Kein falscher Code wurde in der Regelmenge je als `ja` gewertet;
alle Fehler in v2 und v3 sind vorsichtige Abwertungen treuer Codes. Der tote Code `dl3_decomposition_overwritten` wurde dagegen
mehrheitlich als `ja` gewertet: der Existenzprüfer sieht ausgeführten, aber überschriebenen Code nicht (bekannte Grenze; nicht in der Regel).

## Beobachtungen (post hoc, keine Regeländerung)
- Bei den drei treuen DLinear-Codes wird je mindestens ein Baustein als „Code-Zitat steht nicht im Code (verworfen)“ gewertet
  (Wiederholung 1 gedruckt: `dual_linear_layers`, `seasonal_trend_decomposition`, `moving_average_kernel_25`). Mögliche Ursache
  (nicht geprüft): Zitate über mehrere nicht zusammenhängende Zeilen oder mit Auslassungen, die die Zitatprüfung ablehnt.
- Bei `dl3_default_kernel_effective` wird auch der Kernel-Baustein verworfen, obwohl der neue Prompt die Zeile mit dem Wert verlangt.
- Die AST-Prüfung für überschriebene Standardwerte hat nicht falsch ausgelöst (beide Override-Codes `teilweise`), sie ist aber durch die Prüferurteile
  nicht separat belegt (beide wären auch ohne sie `teilweise` geworden, falls der Prüfer sie schon abwertet).
