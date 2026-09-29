# Ergebnis: Eichung des Methodentreue-Prüfers v1 — Regel NICHT erfüllt

Vorab-Regel: `../calibration_fidelity.md` (Commit `066f4c2`, vor dem Lauf). Rohausgabe: `calibration_fidelity_v1.txt`.
Lauf: Token Factory, Rolle `extract` (Super), 3× je Code, Cache aus, Kosten 0,078 $ netto.

| Code | Gold | Urteile (3×) | Mehrheit |
|---|---|---|---|
| dlinear_A7_ridge | nein | nein nein nein | ok |
| dlinear_A6_flat_linear | nein | nein nein nein | ok |
| dlinear_A2_faithful | ja | teilweise ×3 | falsch |
| dlinear_B5_faithful | ja | teilweise ×3 | falsch |
| dlinear_variant_kernel_5 | teilweise | teilweise ×3 | ok |
| gcn_cached_faithful | ja | nein ja nein | **ja↔nein** |
| gcn_sparse_selfloop | ja | nein ×3 | **ja↔nein** |
| gcn_edge_selfloop | ja | ja ×3 | ok |
| gcn_variant_no_selfloops | teilweise | teilweise teilweise nein | ok |
| gcn_variant_adjacency_unused (schwer) | nein | teilweise ×3 | falsch |

**5/10 richtig, 2 ja↔nein-Vertauschungen, 2/10 Codes mit uneinigen Urteilen. Regel (≥ 8/10, keine Vertauschung) nicht erfüllt.**

## Festgelegte Konsequenz
Achse 2 (Methodentreue) bleibt „nicht geprüft“, Gesamtaussage „vorläufig“. Eine neue Eichung braucht
**neue** Codes; diese 10 sind verbraucht.

## Post hoc (nicht vorab festgelegt, nur Hinweise für die Verbesserung)
- Bei beiden GCN-Fehlvertauschungen liefert der Prüfer für alle 7 Bausteine „keine Antwort“
  (Rohausgabe): vermutlich unlesbares oder abgeschnittenes JSON bei `max_tokens=8000` (Modell denkt in der Antwort),
  also ein technischer Ausfall, kein Urteil über den Code. Nicht geprüft, ob es daran liegt.
- Bei DLinear werden Code-Zitate verworfen („steht nicht im Code“). Möglich: Zitat weicht in Whitespace/Sonderzeichen ab.
  Nicht geprüft.
- Der schwere Fall (Adjazenz gebaut, nie multipliziert) wird als „teilweise“ statt „nein“ gewertet.
