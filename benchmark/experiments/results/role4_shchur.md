# Ergebnis: Rolle 4 (nur Paper-Anhang C) bei Shchur/Cora

Vorab festgelegt in `../role4_shchur.md` (Commit vor dem Start, Konfiguration 30 Splits × 1 Initialisierung, `ec36d79`). Rohdaten: `role4_shchur.jsonl`.
Token Factory (Prod-Modelle), je 6 frische Generierungen pro Arm, je 30 Läufe (30 Splits) pro Generierung, Ziel 0,815 (Tabelle 1, Rolle 2), Toleranz 0,01. `src/` blieb unverändert.
(Nebenbei liefen zeitweise zwei weitere Sandbox-Jobs ohne LLM; das verlängerte einzelne Läufe, ändert keine Werte.)

## Primärregel: nicht erfüllt
| Kriterium | Gefordert | Gemessen |
|---|---|---|
| Median der Abweichung in B mindestens 0,005 kleiner als in A | ≥ 0,005 | **0,0046** (A 0,0153 gegen B 0,0107, jeweils Median über die gültigen Generierungen: A 6, B 5) |
| B hat mindestens 2 „reproduziert“ mehr als A | ≥ +2 | **+1** (A 0, B 1) |

Deckeneffekt (Arm A mit ≥ 5 von 6 „reproduziert“) lag nicht vor (A: 0 von 6). Das Ergebnis ist also **nicht bestätigt**, aber Kriterium 1 wird nur um 0,0004 verfehlt.

## Zahlen je Generierung
| Gen. | A: Urteil, Accuracy | B: Urteil, Accuracy |
|---|---|---|
| 0 | teilweise 0,7980 | teilweise 0,8029 |
| 1 | teilweise 0,8017 | nicht reproduzierbar (Setup) |
| 2 | teilweise 0,8016 | teilweise 0,8047 |
| 3 | nicht reproduziert 0,7838 | teilweise 0,8019 |
| 4 | teilweise 0,8014 | **reproduziert** 0,8054 |
| 5 | teilweise 0,7920 | teilweise 0,8043 |

| | Median Abweichung | Mittel Accuracy (gültig) | reproduziert | Totalausfall | Kosten gesamt (netto) |
|---|---|---|---|---|---|
| A ohne Rolle 4 | 0,0153 | 0,7964 | 0 | 0 | 0,057 $ |
| B mit Rolle 4 (nur Paper) | 0,0107 | 0,8038 | 1 | 1 | 0,120 $ |

Alle Generierungen: Gesamtaussage „vorläufig“ (Achse 2 nicht geprüft, Eichung gescheitert). Graph-Ablation: 11 von 12 „Graph genutzt“, B2 „nicht messbar“ (ohne Kanten nur eine Vorhersageklasse, Harness lehnt ab).

## Beobachtungen (post hoc, nicht vorab festgelegt)
- **Rolle 4 übertrug die Anhang-C-Werte zuverlässig.** Anteil der Codes, in denen die Werte vorkommen (Regex-Heuristik, grob): Dropout 0,8: B 6/6, A 0/6; L2 1e-3: B 6/6, A 0/6; Geduld 50: B 5/6, A 0/6; Hidden 64: B 6/6, A 3/6. Der Mechanismus funktioniert; die Belege pro Generierung: 8 bis 21 Setup-Details.
- **Streuung:** Die gültigen B-Generierungen liegen eng beieinander (0,8019 bis 0,8054), die A-Generierungen weiter (0,7838 bis 0,8017). Kein A-Wert erreicht 0,805.
- **Abstand zur Referenz:** Die menschliche Referenz mit denselben Anhang-C-Werten liegt auf denselben 30 Splits bei 0,8107. B-Codes erreichen 0,8038 (Mittel), also etwa 0,7 Punkte weniger; ein weiteres Protokolldetail (nicht untersucht) fehlt den Agenten-Codes.
- **Totalausfall B1:** „IDs unvollständig, fehlend: 2115“, also 20 von 2135 Testknoten. Der Code hat offenbar die Smoke-Vorgabe (20 Knoten) auch im Vollauf umgesetzt; nicht geprüft.
- **Kosten:** Arm B kostet etwa das Doppelte von A (0,020 $ gegen 0,010 $ je Generierung), insgesamt 0,177 $ netto für alle zwölf.

## Einschränkungen
6 Generierungen pro Arm, ein Paper, ein Claim; Modelle nicht deterministisch; bei 6 gegen 6 sind kleine Unterschiede nicht belastbar. Arm B hatte einen längeren Prompt. Die Splits sind für beide Arme gleich (gepaart). Die Regex-Zählung der Anhang-C-Werte ist eine Heuristik.

## Einordnung
Die Richtung stimmt (B näher, enger, mit den Paperwerten im Code), die vorab verlangte Größe des Unterschieds nicht. Vorsichtige Lesart für das README: „Rolle 4 setzte die gebündelt im Anhang stehenden Werte zuverlässig um und verbesserte das Ergebnis leicht (Median 0,5 Punkte näher am Zielwert), erfüllte aber unsere vorab festgelegte Schwelle nicht.“ Das bestätigt die Erklärung „gebündelte Information hilft“ nicht (Regel verfehlt), widerlegt sie aber auch nicht (Richtung positiv, Stichprobe klein).
