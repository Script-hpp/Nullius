# GAT: Demo-Lauf mit Live-Tavily-Aufrufen (kein Benchmark-Protokoll, keine Vorab-Regel)

Lauf `runs/velickovic_gat_2018/20260929-155442`, Prod-Modelle, `TAVILY_MODE=live` (Standard). Export in die Evidence-Ansicht: `web/public/data/velickovic_gat_2018/`.

## Ergebnis
- **Urteil „teilweise“:** Mittel 0,8036 (Std 0,0088) gegen Zielwert 0,83 ± 0,01 (Abstand 0,0264 ≤ 3 × Toleranz). Alle 10 Seeds liegen bei 0,790 bis 0,815, kein Seed erreicht 0,82.
- Achse 2 nicht geprüft. Graph-Ablation: „Graph genutzt“ (0,808 mit Graph, 0,553 ohne Kanten). 10 Versuche, 0 Reparaturen, Kosten 0,041 $ netto.
- **Live-Aufrufe belegt** (Dossier `sources`, Ereignisse `source` und `tavily_call`, Evidence-Ansicht): Paper-Text live um 13:54:42 UTC (Rolle 2, Extract, SHA-256 `f7f86c91c2`, 43 997 Zeichen), README des Repos live um 13:55:00 UTC (Rolle 4, Extract). Zielwert 0,83 aus dem GAT-Paper selbst belegt („GAT (ours) 83,0 ± 0,7 %“, Spaltenprüfung ok); Gegenprobe gegen den Wert in `papers.yaml` stimmt (`target_check`).

## Beobachtungen (post hoc)
- **Ein Fehler beim ersten Anlauf:** Rolle 2 scheiterte, weil die Ergebnistabelle hinter der 14 000-Zeichen-Grenze des Ausschnitts lag (der Datensatzname „Cora“ steht im ganzen Paper). Behoben (Fenster mit Methodenname und vielen Dezimalzahlen zuerst, `_windows(prefer=...)`).
- **Rolle 4 (24 Details, 17 verworfen):** 13 aus dem Paper, 11 aus den Repo-Dateien `execute_cora.py` und `execute_cora_sparse.py`. Die 11 Repo-Details sind **allgemeine Ablaufdetails** (Early-Stopping-Bedingung, Checkpoint speichern und wiederherstellen, Dropout beim Test 0,0, Batch-Schleifen), **keine Hyperparameter-Werte**. Damit ist es nicht die Konstellation von DLinear (verstreute, widersprüchliche Werte), sondern ein Lückenproblem.
- **Lücke:** Im Paper-Text stehen Lernrate 0,005, Dropout p = 0,6 (auf Eingaben und Attention-Koeffizienten) und Geduld 100 Epochen; unter den 13 behaltenen Paper-Details fehlen sie (L2 0,0005, 8 Köpfe × 8 Merkmale, Skip-Verbindungen sind dabei). Der Agenten-Code nutzt Dropout 0,6 trotzdem (Quelle nicht geprüft, vermutlich Vorwissen). Ob die Lücke die 2,6 Punkte erklärt, ist offen.
- Hinweis für Tavily v2 (Entwicklung nur auf der Entwicklungsmenge, GAT ist Prüffall): Rolle 4 verwirft viele Angaben (17 von 41) und verfehlt Kernhyperparameter des Papers; die Wiedergabe der Zitate scheitert vermutlich an der PDF-Textform. Das ist eine Vermutung, nicht untersucht.
