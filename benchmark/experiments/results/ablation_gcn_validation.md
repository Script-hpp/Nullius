# Ergebnis: Validierung der GCN-Graph-Ablation — 9/10, Regel formal NICHT erfüllt

Vorab-Regel: `../ablation_gcn.md` (Commit `0577f49`, vor dem Lauf). Rohausgabe: `ablation_gcn_validation.txt`. Kein LLM, Seed 0, je 2 Läufe pro Code.

| Code | mit Graph | ohne Kanten | Einordnung (erwartet) |
|---|---|---|---|
| v1 gcn_cached_faithful | 0,804 | 0,512 | Graph genutzt ✔ |
| v1 gcn_sparse_selfloop | 0,802 | 0,499 | Graph genutzt ✔ |
| v1 gcn_edge_selfloop | 0,788 | 0,477 | Graph genutzt ✔ |
| v1 gcn_variant_no_selfloops | 0,789 | n/a (Harness: nur eine Klasse) | **nicht messbar** (erwartet: genutzt) ✘ |
| v1 gcn_variant_adjacency_unused | 0,512 | 0,512 | Graph nicht genutzt ✔ |
| v2 gcn_new_faithful_dense | 0,820 | 0,568 | Graph genutzt ✔ |
| v2 gcn_new_faithful_scatter | 0,815 | 0,556 | Graph genutzt ✔ |
| v2 gcn_new_row_normalised | 0,830 | 0,568 | Graph genutzt ✔ |
| v2 gcn_new_one_layer | 0,717 | 0,568 | Graph genutzt ✔ |
| v2 gcn_new_adjacency_stored_unused | 0,568 | 0,568 | Graph nicht genutzt ✔ |

**9/10 wie erwartet. Regel (alle 10, kein `nicht messbar`) nicht erfüllt.** Die Schwellen bleiben unverändert.

## Beobachtungen (post hoc, keine Regeländerung)
- Beide toten Fälle (`adjacency_unused`, `stored_unused`), die der Existenzprüfer und die Zeilenabdeckung nicht erkennen, werden korrekt als
  „Graph nicht genutzt“ erkannt, ohne Fehlalarm bei den 8 graphnutzenden Codes.
- Die eine Abweichung: `no_selfloops` hat ohne Kanten eine Nulladjazenz (keine Selbstschleifen), gibt konstante Ausgaben und der Harness weist
  „nur eine Vorhersage-Klasse“ zurück. Das ist eher ein Zeichen **für** Graphabhängigkeit, wurde aber vorab als `nicht messbar` definiert.
- Folge im Betrieb: `nicht messbar` lässt Achse 2 unverändert (vorsichtige Richtung). Eine Regelerweiterung („ungültig wegen einer Klasse ⇒ genutzt“)
  würde neue Codes zur Validierung brauchen und ist nicht umgesetzt.
- Grenze: Fixture-Codes sind selbst geschrieben; geprüft wird nur, ob der Graph benutzt wird, nicht die Details der Propagationsregel.
