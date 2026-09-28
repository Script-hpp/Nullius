# Experiment: Bringt Tavily-Rolle 4 (Setup-Details) etwas? — GCN auf Cora

Vorab festgelegt, bevor ein Rolle-4-Lauf gestartet wurde. Nichts davon wird nach Sicht der
Ergebnisse geändert. Ausgang ist ein Befund aus den ersten Läufen: Code aus einer frischen
Generierung (Lauf 4) lieferte 0,7973 („teilweise“), der gecachte Code (Lauf 3) 0,8124
(„reproduziert“). Der frische Code ließ die zeilenweise Feature-Normalisierung weg.

## Hypothese
Wenn der Agent die im Paper belegten Setup-Details bekommt (Rolle 4), erreichen mehr frische
Generierungen „reproduziert“ als ohne.

## Aufbau
- Zwei Arme, je 5 **frische** Generierungen (LLM-Cache aus, jeweils neuer Plan und neuer Code).
  - Arm A: ohne Rolle 4 (`--no-setup-details`).
  - Arm B: mit Rolle 4.
- Pro Generierung 5 Seeds, Urteil auf dem Mittelwert, Toleranz ±0,01, Zielwert 0,815
  (aus Tabelle 2, arXiv v4, per Rolle 2 belegt).
- Gleiches Modell (OpenRouter Free, Prototyp), gleicher unabhängiger Harness, gleiche Isolation.
- Rolle 4 ist allgemein formuliert (fragt nach *allen* Angaben zum Versuchsaufbau, nennt keine
  Normalisierung). Der Prompt in `setup_details.py` wird für das Experiment nicht angepasst.
- Zusätzlich gemessen: enthält der generierte Code eine Feature-Normalisierung (Regex-Heuristik)?

## Erfolgsregel
Erfolg, wenn **beides** gilt:
1. Arm B: mindestens 4 von 5 Generierungen „reproduziert“.
2. Arm B hat mindestens 2 „reproduziert“ mehr als Arm A.

Sonst: Hypothese nicht bestätigt. Das Ergebnis wird trotzdem berichtet, mit beiden Zahlen.

## Einschränkungen (vorab)
- Prototyp-Modell, Ergebnisse nicht als Benchmark-Zahlen veröffentlichen; Wiederholung auf
  Token Factory nötig.
- 5 Generierungen pro Arm sind eine kleine Stichprobe.
- Nur ein Paper. Die Verallgemeinerung wird an DLinear geprüft.
