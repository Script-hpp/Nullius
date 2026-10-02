# Auftrag 26 (worker-2): v5 Schritt B, zwei deterministische Prüfungen

Grundlage: `benchmark/experiments/failure_analysis_v4.md`, Vorschläge V4 und V5. Kein LLM, keine echten Aufrufe, Tests mit Fakes. Branch `v5-step-b` von `main`, Arbeit nur im Worktree `../research-a2`. Meldung an den Reviewer (neue Terminal-ID nach Neustart): `term_119aeb49-f2cb-4724-a838-c3fb945da750`.

Beide Prüfungen laufen nur im Produktpfad (`graph_repair=True`), ändern nie das Urteil (Achse 1) direkt und bringen keine Test- oder Validierungszahl in einen Prompt.

## B1 (V4): Validierungslabels im Training

Befund: Die A1-Auswahl (beste Validierungs-Metrik unseres Harness) bevorzugt Code, der auf den Validierungslabels trainiert (v4: SGC A Gen 1, GAT B Gen 1).

Regel, wie die Graph-Ablation (zweiter Lauf, kein LLM):
- Für jeden lauffähigen Kandidaten in `_run_candidates` (Seed 0): zweiter Lauf im frischen Branch, in dem die Labels der Validierungsknoten/-fenster im Agenten-Datensatz maskiert sind (Cora, Cora-LCC: `labels[val_idx] = -1` in der npz wie bei Testknoten; ETTh1: prüfen, ob es dort getrennte Validierungsziele gibt, die der Agent zum Training nutzen könnte; wenn nicht sinnvoll maskierbar, B1 für ETTh1 auslassen und begründen).
- Vergleich der **Validierungs**-Metrik des Harness (eigene Labels) mit und ohne Maske. Abfall ≥ 0,10 (Accuracy) = "Validierungslabels im Training genutzt"; ≤ 0,02 = "nicht genutzt"; dazwischen "unklar"; Absturz/ungültig = "nicht messbar". Schwellen wie `classify_ablation`, als Konstanten.
- Achtung: Early Stopping/Modellauswahl auf Validierungs-Accuracy ist erlaubt und üblich und braucht die Labels. Ein Code, der Validierungslabels nur zur Modellauswahl nutzt, fällt mit Maske ebenfalls (kann nicht mehr auswählen) oder stürzt ab. Deshalb im Bericht sauber trennen: Die Regel erkennt "Validierungs-Metrik hängt stark an den Validierungslabels". Vorschlag zur Unterscheidung: zusätzlich statische Suche im Code, ob `val_idx` in einer Loss-Berechnung oder in der Indexmenge des Trainings vorkommt (`harness/code_check.py`-Stil, eigene Funktion, nur Hinweis). Schreib im Plan, wie du die beiden Fälle trennst, bevor du baust; wenn es nicht sauber trennbar ist, sag das.
- Folge bei "genutzt": Kandidat scheidet aus der A1-Auswahl aus (bleibt als Versuch im Dossier, Feld `val_label_check`), Ereignis `val_label_check`. Sind alle Kandidaten betroffen: Auswahl wie bisher, Warnung im Dossier.
- Kosten: ein zusätzlicher Seed-0-Lauf je Kandidat (kein LLM).

## B2 (V5): Kanten in beide Richtungen

Befund: Shchur-GAT B Gen 2 nutzte Kanten nur in einer Richtung (0,706).

- Deterministische Prüfung für Graph-Tasks, kein LLM: zweiter Lauf, in dem der Agenten-Datensatz nur eine Richtung jeder Kante enthält (`edges` enthält heute beide Richtungen; halbiere auf u < v) gegen den normalen Lauf. Ein Code, der Kanten selbst symmetrisiert, liefert (fast) dasselbe; ein Code, der nur die gegebene Richtung nutzt, fällt. Das prüft aber die Robustheit gegen gerichtete Eingabe, nicht den Fehler "nutzt nur eine Richtung der vollständigen Liste". Prüfe zuerst am Fall Shchur-GAT B Gen 2 (`runs/shchur_gat_2018/…`, Code im Lauf), welcher Test den Fehler wirklich erkennt (Vorschlag des Berichts: kleiner gerichteter Testgraph im Smoke-Lauf, Änderung an Knoten j muss die Ausgabe von Nachbar i in beiden Richtungen ändern). Wähle die Variante, die den echten Fall erkennt und die Referenzlösungen (`tests/reference/`) nicht fälschlich meldet; belege beides lokal (Referenzen laufen lokal auf CPU).
- Folge bei Treffer: Reparaturrunde mit festem Hinweis ohne Zahlen ("the graph must be used in both directions of every edge"), teilt das Budget der Graph-Reparaturen; Ereignis `edge_direction_check`.

## Abnahme

- Plan zuerst (kurz, einzeilig an den Reviewer), besonders zur Trennung in B1 und zur Testvariante in B2.
- Tests mit Fakes, Suite (`--deselect tests/test_harness_baselines.py --deselect tests/test_nebius_sandbox.py`), lokale Belege mit den Referenzlösungen (kein Fehlalarm) und mit den echten v4-Codes der drei Fälle (SGC A Gen 1, GAT B Gen 1, Shchur-GAT B Gen 2; Agenten-Code nie direkt auf dem Host ausführen: nur in Docker (`LocalSandbox`) ohne Netz; Referenzlösungen dürfen lokal laufen).
- Keine Zahl aus Test oder Validierung in einem Prompt (Test dafür).
