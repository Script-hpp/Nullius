# Experiment: Hilft Rolle 4, wenn die Angaben gebündelt an einer Stelle stehen? — GCN auf Cora (Shchur et al. 2018)

Vorab festgelegt, bevor ein Experimentlauf gestartet wurde. Nichts wird nach Sicht der Ergebnisse geändert.
Läuft auf Token Factory (Prod-Modelle); Ergebnisse dürfen im README stehen. Auswertung mit **Median** (Lehre aus `role4_dlinear.md`).

## Hintergrund
Rolle 4 v1 half bei DLinear nicht (`results/role4_dlinear.md`, Primärregel klar verfehlt). Dort standen Fenster, Lernrate, Epochen und
Schedule verstreut in Repo-Dateien. Bei Shchur stehen die Trainingsangaben gesammelt im Paper-Anhang C (Tabelle 4: Hidden 64, lr 0,01,
Dropout 0,8, L2 0,001; Early Stopping auf Gesamt-Validierungsloss, Geduld 50, beste Gewichte zurücksetzen).

## Hypothese (Erklärungsversuch, kein Beleg)
„Rolle 4 v1 scheiterte bei DLinear, weil die Information verstreut war. Liegt sie an einer Stelle, hilft sie.“ Der Versuch testet nur den
zweiten Satz für dieses Paper. DLinear und Shchur unterscheiden sich in vielem (Domäne, Modell, Protokoll); ein Erfolg bestätigt die Erklärung
nicht, ein Misserfolg widerlegt sie nicht.

## Aufbau
- Ein Claim `shchur_gcn_cora_acc`: GCN, Cora (größte Komponente), zufällige 20/30/Rest-Splits, Accuracy, Ziel 0,815 (Tabelle 1, Rolle 2), Toleranz 0,01.
- 30 Läufe pro Generierung (10 Splits × 3 Initialisierungen, dieselben 10 Splits in jeder Generierung), Urteil auf dem Mittelwert (`shchur_task.md`).
- Zwei Arme, je **6 frische** Generierungen (LLM-Cache aus), abwechselnd (A0, B0, A1, B1, …).
  - Arm A: ohne Rolle 4 (`setup_details=False`).
  - Arm B: Rolle 4 **nur aus dem Paper-Text** (`paper_only=True`: kein README, kein Repo, kein Sandbox-Klon).
- Beide Arme mit Tavily Rolle 2 (Zielwert); Rollen `reasoning`=Ultra, `code`=Super, `extract`=Super; Modus „Nachbau“, gleicher Harness, gleiche Isolation.
- `src/` und Prompts bleiben während des Experiments unverändert.
- Pro Generierung protokolliert (`runs/experiment_role4_shchur.jsonl`): Urteil, Gesamtaussage, Accuracy (Mittel, Std), Kosten, Zahl belegter Setup-Details,
  Graph-Ablation, und ob die Anhang-C-Werte im Code vorkommen (Regex-Heuristik: 64, 0.8, 0.01, 50, 1e-3).

## Zählregeln
- „Reproduziert“ = |Mittel − 0,815| ≤ 0,01. Generierung ohne gültigen Lauf oder mit „Agenten-Fehler vermutet“ zählt als nicht reproduziert und wird separat gezählt.
- Abweichung je Generierung = |Mittel − 0,815|; Median über die gültigen Generierungen je Arm.

## Erfolgsregel
Erfolg, wenn **beides** gilt:
1. Der Median der Abweichung ist in Arm B mindestens 0,005 (0,5 Prozentpunkte) kleiner als in Arm A.
2. Arm B hat mindestens 2 „reproduziert“ mehr als Arm A.

**Deckeneffekt (vorab):** Hat Arm A bereits mindestens 5 von 6 „reproduziert“, ist die Hypothese mit diesem Aufbau nicht testbar; das gilt nicht als Misserfolg
von Rolle 4, sondern als „nicht entscheidbar“.
Sonst: nicht bestätigt, Ergebnis wird mit allen Zahlen berichtet.

## Einschränkungen (vorab)
- 6 Generierungen pro Arm, ein Paper, ein Claim; Modelle nicht deterministisch. Bei 6 gegen 6 sind kleine Unterschiede nicht belastbar.
- Die menschliche Referenz mit den Werten aus Anhang C liegt bei 0,8065 (Abweichung 0,0085, innerhalb der Toleranz, aber knapp; `shchur_task.md`).
  Der Spielraum ist klein: schon kleine Abweichungen vom Protokoll führen zu „teilweise“.
- Ohne Rolle 4 kennt das Modell die Standard-GCN-Werte (Hidden 16, Dropout 0,5, L2 5e-4, 200 Epochen) womöglich aus dem Gedächtnis; die Unterschiede zu Anhang C
  können klein sein, dann ist der Versuch wenig empfindlich.
- Die zehn Splits sind in allen Generierungen gleich; ein systematischer Abstand des Mittels zum Paperwert durch diese Stichprobe wirkt auf beide Arme gleich.
- Arm-Unterschiede können auch vom Umfang der Prompts kommen, nicht nur vom Inhalt.
- Anhang C steht im extrahierten Paper-Text (Fixture, v2 PDF); Rolle 4 findet ihn nur, wenn die abschnittsweise Extraktion ihn trifft.
