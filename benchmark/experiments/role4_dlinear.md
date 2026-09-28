# Experiment: Bringt Rolle 4 (Setup-Details aus Paper + Repo) etwas? — DLinear auf ETTh1

Vorab festgelegt, bevor ein Experimentlauf gestartet wurde. Nichts wird nach Sicht der Ergebnisse
geändert. Läuft auf Token Factory (Prod-Modelle), Ergebnisse dürfen im README stehen.

## Hintergrund
Beim GCN-Experiment (`role4_gcn.md`, OpenRouter Free) gab es kaum etwas zu retten. Bei DLinear
stehen Eingabefenster, Lernrate, Epochen und Lernraten-Schedule nicht im Paper, nur im Repo
(`cure-lab/LTSF-Linear`). Probeläufe (nicht Teil des Experiments) mit Rolle 4: MSE 0,398
(Ziel 0,375), Code übernahm 336/0,005/10 Epochen aus den Skripten, aber nicht den Schedule.
Meine Referenz-DLinear: 0,428 mit konstanter, 0,381 mit pro Epoche halbierter Lernrate.

## Hypothese
Mit Rolle 4 landen mehr frische Generierungen nahe am Paperwert als ohne, und die erreichten
MSE-Werte sind besser.

## Aufbau
- Ein Claim: DLinear, ETTh1, Horizont 96, MSE, Ziel 0,375 (Table 2, arXiv v3, Rolle 2), Toleranz 0,01.
- Zwei Arme, je 8 **frische** Generierungen (LLM-Cache aus, neuer Plan/Code je Generierung),
  je 5 Seeds mit demselben Code, Urteil auf dem Mittelwert. Arme abwechselnd (A0, B0, A1, B1, …).
  - Arm A: ohne Rolle 4 (`--no-setup-details`).
  - Arm B: mit Rolle 4 (Paper, README, Repo-Dateien, Commit `0c11366`).
- Modus „Nachbau“, gleicher Harness, gleiche Isolation, Rollen `reasoning`=Ultra, `code`=Super.
- Rolle-4-Prompts (`setup_details.py`, `repo_source.py`) bleiben wie im Commit vor Start.
- Pro Generierung protokolliert: Urteil, MSE, Kosten, ob der Code einen Lernraten-Abfall enthält
  (Regex-Heuristik: `0.5 **`, `gamma=0.5`, `ExponentialLR`, `StepLR`, `lr_lambda`), ob 336 als
  Fenster genutzt wird.

## Zählregeln
- „Nah“ = Urteil reproduziert oder teilweise (MSE ≤ 0,405, also ≤ 3× Toleranz).
- Generierung ohne lauffähigen Code (nicht reproduzierbar (Setup)) zählt als nicht nah und
  wird separat gezählt.

## Erfolgsregel
Erfolg, wenn **beides** gilt:
1. Arm B hat mindestens 3 „nah“ mehr als Arm A.
2. Der mittlere MSE der gültigen Generierungen ist in Arm B mindestens 0,02 niedriger als in Arm A.

Sonst: nicht bestätigt, Ergebnis wird trotzdem mit allen Zahlen berichtet.

## Einschränkungen (vorab)
- 8 Generierungen pro Arm, ein Paper, ein Horizont; Modelle nicht deterministisch.
- Zielwert-Spalte (DLinear) ist nicht deterministisch belegt (Warnung im Zeugnis).
- Test-Eingaben der Fenster überlappen; ein absichtlich betrügender Code könnte spätere Fenster
  nutzen. Die Codeprüfung deckt das nicht ab.
- Arm-Unterschiede können auch vom Umfang der Prompts (Länge) kommen, nicht nur vom Inhalt.
