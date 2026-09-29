# Graph-Ablation für GCN (Nutzungsprüfung ohne Sprachmodell)

Vorab festgelegt und committet **vor** dem ersten Lauf. Nichts wird nach Sicht der Ergebnisse geändert.

## Idee
Der Agenten-Code läuft ein zweites Mal, mit demselben Seed und denselben Daten, aber `/data/cora_public.npz` hat
**keine Kanten** (`edges` leer, Form `(0, 2)`). Nutzt der Code den Graphen, fällt die Genauigkeit stark; bleibt
sie gleich, wird der Graph nie benutzt, egal wie der Code aussieht. Deterministisch, kein Eingriff in den Code.
Harness und Testlabels bleiben unverändert (Labels lädt der Harness frisch vom Original).

## Entscheidungsregel (Genauigkeit `acc_full` mit Graph, `acc_ablated` ohne Kanten, Seed 0)
- Abfall `acc_full − acc_ablated` **≥ 0,10** → `Graph genutzt`
- Abfall **≤ 0,02** → `Graph nicht genutzt`
- dazwischen → `unklar`
- Lauf ohne Kanten stürzt ab oder ist ungültig → `nicht messbar` (kein Beleg für Nutzung)
Im Zeugnis: `Graph nicht genutzt` setzt Achse 2 (Methode) auf `nein` (deterministischer Beleg, nur herabstufend).
`Graph genutzt` belegt nur diesen einen Aspekt und setzt Achse 2 **nicht** auf `ja`.

## Validierung an den 10 GCN-Fixture-Codes (kein LLM, Seed 0, je 2 Läufe)
Erwartung aus der Konstruktion:
- `Graph genutzt`: v1 `gcn_cached_faithful`, `gcn_sparse_selfloop`, `gcn_edge_selfloop`, `gcn_variant_no_selfloops`;
  v2 `gcn_new_faithful_dense`, `gcn_new_faithful_scatter`, `gcn_new_row_normalised`, `gcn_new_one_layer`.
- `Graph nicht genutzt`: v1 `gcn_variant_adjacency_unused`, v2 `gcn_new_adjacency_stored_unused`.

**Erfolg:** alle 10 werden wie erwartet eingeordnet (kein `unklar`, kein `nicht messbar`, keine Verwechslung).
**Scheitern:** Abweichungen werden berichtet, die Schwellen werden **nicht** angepasst; eine Änderung braucht neue Codes.

## Grenzen
- Bei DLinear nicht anwendbar (kein Graph). Die Zerlegung würde eine andere Ablation brauchen; dort erreicht laut Paper
  schon ein einfaches Linear-Modell denselben Wert, die Methodentreue ändert das Zahlenurteil kaum.
- Prüft Nutzung des Graphen, nicht die Details der Propagationsregel (Normalisierung, Selbstschleifen, Schichtzahl).
- Die Fixture-Codes sind selbst geschrieben und können von echten Agenten-Codes abweichen.
