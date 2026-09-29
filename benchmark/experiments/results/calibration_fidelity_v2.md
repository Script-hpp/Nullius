# Ergebnis: Eichung des Methodentreue-Prüfers v2 — Regel NICHT erfüllt (7/10)

Vorab-Regel: `../calibration_fidelity_v2.md` (Commits `d1dcb9d`, Kombinationsregel `HEAD~` davor festgelegt). Rohausgabe: `calibration_fidelity_v2.txt`.
Lauf: Token Factory, Rolle `extract` (Super), 3× je Code, Cache aus, Kosten 0,091 $ netto, `src/` unverändert.

| Code | Gold | Urteile (3×) | Mehrheit |
|---|---|---|---|
| dlinear_new_faithful_replicate | ja | teilweise teilweise ja | falsch |
| dlinear_new_faithful_conv_kernel | ja | teilweise ×3 | falsch |
| dlinear_new_kernel_13 | teilweise | teilweise ×3 | ok |
| dlinear_new_shared_linear | teilweise | teilweise ×3 | ok |
| dlinear_new_mlp_no_decomposition | nein | nein ×3 | ok |
| dlinear_new_nlinear | nein | nein ×3 | ok |
| gcn_new_faithful_dense | ja | ja ×3 | ok |
| gcn_new_faithful_scatter | ja | ja nein teilweise | falsch (kein Mehrheitsurteil) |
| gcn_new_row_normalised | teilweise | teilweise ×3 | ok |
| gcn_new_one_layer | teilweise | teilweise ×3 | ok |
| *dead_code* dlinear_new_decomposition_discarded | nein | teilweise ×3 | nur berichtet |
| *dead_code* gcn_new_adjacency_stored_unused | nein | teilweise ×3 | nur berichtet |

**Regelmenge: 7/10 richtig, 0 ja↔nein-Vertauschungen, 2/12 Codes mit uneinigen Urteilen. Regel (≥ 8/10, keine Vertauschung) nicht erfüllt.**
Gegenüber v1 (5/10, 2 Vertauschungen): keine Vertauschung mehr, aber weiter unter der Schwelle.

## Festgelegte Konsequenz
Achse 2 bleibt „nicht geprüft“, Gesamtaussage „vorläufig“. Eine weitere Eichung braucht wieder **neue** Codes (v2 ist verbraucht).

## Beobachtungen (post hoc, keine Regeländerung)
- **Keine Fehlbewertung nach unten bei den falschen, sondern zu vorsichtig bei den treuen:** Alle drei Fehler der Regelmenge
  sind treue Codes (Gold ja), die als `teilweise` oder uneinig bewertet wurden. Kein falscher Code wurde als treu (`ja`) gewertet.
- Bausteinausgabe (nur Wiederholung 1 gedruckt): Bei `faithful_replicate` wird der Baustein `moving_average_kernel_25` als
  „Code-Zitat steht nicht im Code (verworfen)“ gewertet; bei `faithful_conv_kernel` derselbe Ausfall bei `seasonal_trend_decomposition`.
  Die Ursache (Zitat weicht ab, Wert steht nicht im zitierten Ausschnitt, oder Zitat über mehrere Stellen) ist nicht geprüft.
- Die zwei toten Codes werden wie erwartet als `teilweise` bewertet, nicht als `nein`: Der Existenzprüfer kann sie nicht erkennen.
  Sie bleiben der Testfall für die Ausführungsabdeckung.
- `gcn_new_faithful_scatter` (Gold ja) liefert ja/nein/teilweise: instabil, das ist der einzige Code mit Ja-Nein-Streuung.
