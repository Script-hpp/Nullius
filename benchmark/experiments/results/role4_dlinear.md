# Ergebnis: Rolle 4 bei DLinear/ETTh1 (v1)

Vorab festgelegt in `../role4_dlinear.md` (Commit `468b0be`). Rohdaten: `role4_dlinear.jsonl`.
Token Factory (Prod-Modelle), 8 frische Generierungen pro Arm, je 5 Seeds mit demselben Code,
Ziel MSE 0,375 (Table 2, arXiv v3), Toleranz 0,01. `src/` blieb während des Experiments unverändert.

## Primärregel: nicht erfüllt
| Kriterium | Gefordert | Gemessen |
|---|---|---|
| B hat mindestens 3 „nah“ mehr als A | +3 | **−5** (A 6 von 8, B 1 von 8) |
| Mittlerer MSE gültiger Generierungen in B um mindestens 0,02 niedriger | −0,02 | **−0,009** (B 0,5635, A 0,5724) |

Die Hypothese „Rolle 4 in dieser Form hilft“ ist nicht bestätigt, das erste Kriterium klar verfehlt.

## Zahlen je Arm
| | nah | reproduziert | Totalausfall | Agenten-Fehler (MSE > 1,1099) | Median MSE (gültig) | Kosten/Generierung |
|---|---|---|---|---|---|---|
| A ohne Rolle 4 | 6 | 2 | 0 | 1 | 0,396 | 0,010 $ |
| B mit Rolle 4 | 1 | 1 | 3 | 1 | 0,431 | 0,076 $ |

Exakter Test auf „nah“ 6/8 gegen 1/8: p = 0,041 (unkorrigiert, kleine Stichprobe, post hoc, nur
als Zusatzinformation).

## Explorativ (vorab definiert, vor den letzten Ergebnissen)
Median statt Mittelwert; ohne „Agenten-Fehler“ (MSE > 1,1099, schlechter als die Nullvorhersage):
A n=7 Median 0,396; B n=4 Median 0,419.

## Post hoc (nicht vorab festgelegt, als solche zu lesen)
- **Lernrate:** Alle B-Codes nutzen 0,005 (Repo-Skripte), alle A-Codes 0,001. Ablation ohne LLM
  (5 Seeds, Median): A-Code 0,394 → 0,494 bei Lernrate 0,005; B-Code 0,438 → 0,385 bei 0,001.
  Im Repo gehört 0,005 zu einem Schedule (`lradj type1`, Halbierung pro Epoche). B-Generierung 5 mit
  Abfall erreichte 0,371, bester Wert des Experiments (n = 1).
- Die Information war im Prompt vollständig, aber über vier Dateien verteilt und mit Widersprüchen
  (Default 0,0001 in `run_longExp.py`, 0,005 in den Skripten, Schedules `type1` bis `type6`).
- **Totalausfälle in B (3 von 8):** Tensorform-Fehler, nur `csv`/`numpy`/`os`/`torch` importiert, keine
  Repo-Bezüge, Code nicht abgeschnitten. Der Kontext ist in B 8000–10 000 Tokens gegen ~1500 in A.
  Länge und Inhalt sind nicht getrennt.
- **Methodentreue (Regex, nicht von Hand geprüft):** 5 von 15 Codes haben keine Zerlegung, davon
  A 7 („reproduziert“, 0,3765, std exakt 0) als geschlossene Ridge-Regression, kein DLinear.
  „Reproduziert“ heißt hier „Zielwert erreicht“, nicht „Methode reproduziert“.

## Einschränkungen
8 Generierungen pro Arm, ein Paper, ein Horizont; Modelle nicht deterministisch. Test-Eingaben
überlappen. Arm B war im Prompt ~6× länger. Die Zielwert-Spaltenzuordnung (DLinear = drittes
Paar) wurde nachträglich über die Kopfzeile der Tabelle belegt; die Warnung im Zeugnis war falsch.

## Was daraus folgt
Rolle 4 v2 (eigene Vorab-Festlegung, beide Arme neu mit derselben Pipeline): wirksame Konfiguration
berechnen, Bündel nur vollständig übergeben, Formprüfung vor dem Training, Methodentreue-Prüfung.
