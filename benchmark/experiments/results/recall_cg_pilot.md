# Code-grounded Pilot: Ergebnis

Stand 2026-10-08. Vorab-Festlegung mit zwei Addenda: `benchmark/experiments/recall_cg_pilot.md` (Commits `22c9df2`, `ba8d570`, `480382f`). Rohdaten, Pläne, Code, Reparaturen, Steckbriefe samt rohem Ausschnitt: `recall_cg_pilot_samples/`.

**Einordnung: In-Sample-Pilot, keine Benchmark-Zahl.** Entwicklungseinheiten wurden vier Mal von Hand analysiert, `code_grounded` ist auf genau diese Fehlerbilder gebaut. Nichts davon gehört ins README oder Video.

## Entscheidung nach der vorab festgelegten Regel: **bestanden**

| Bedingung | Wert | Schwelle | Ergebnis |
|---|---|---|---|
| 1a `cg` auf NLinear, SGC, GAT | **3 von 6** | ≥ 3 | erfüllt |
| 1b Abstand zu `heute` | `heute` 0 von 6, also +3 | ≥ +2 | erfüllt |
| 2 Kipf: `cg` ≥ `heute` − 1 | `cg` 0, `heute` 0 | ≥ −1 | erfüllt, **aber ohne Aussage** (siehe unten) |

Pilot vollständig (16 von 16 Proben gemessen). Treffer = Klasse „reproduziert“ gleich Referenzurteil.

## Treffer je Einheit und Arm (2 Proben, T 0,2 / 0,8)

| Einheit | `heute` | `cg` |
|---|---|---|
| Zeng NLinear (5 Seeds) | 0 von 2 (teilweise 0,3856 / 0,3881) | **2 von 2** (reproduziert 0,3774 / 0,3722) |
| Wu SGC (10 Seeds) | 0 von 2 (Setup-Ausfall; Agenten-Fehler vermutet 0,477) | **1 von 2** (reproduziert 0,8054; zweite Probe Setup-Ausfall) |
| Veličković GAT (10 Seeds) | 0 von 2 (Setup-Ausfall) | 0 von 2 (Setup-Ausfall) |
| **Summe Entwicklungseinheiten** | **0 von 6** | **3 von 6** |
| Kipf GCN (10 Seeds, Nicht-Verschlechterung) | 0 von 2 (teilweise 0,8002; Setup-Ausfall) | 0 von 2 (beide Setup-Ausfall) |

Referenzurteil jeweils „reproduziert“ (Referenzmittel NLinear 0,3739, SGC 0,8081, GAT 0,8280, GCN 0,8136).

## Welche Details ankamen (Fakt-Checks, Heuristik)

- **NLinear:** `cg` in 2 von 2 Proben alle fünf Fakten (Fenster 336, lr 0,005, Halbierung je Epoche, letzten Wert abziehen, keine Zerlegung). `heute` in 2 von 2: nur Fenster 336 und keine Zerlegung; **lr 0,005, Halbierung und Letzter-Wert-Abzug fehlten**. Das ist der Mechanismus aus der Hypothese: der rohe Ausschnitt (Modelldatei, `adjust_learning_rate`, Startskript) plus erzwungenes CONFIG bringt die Repo-Details in den Code. Eine `cg`-Probe (T 0,8) bestand erst nach der Reparatur (CONFIG-Befund „lr/patience nie gelesen“).
- **SGC:** Zeilen-Normalisierung (L1) in `cg` 2 von 2 Proben, in `heute` 0 von 2. Die getunte Weight-Decay-Zahl steht nur in einer Pickle-Datei und erreichte keinen Arm (nicht prüfbar).
- **GAT:** Zeilen-Normalisierung in `cg` 2 von 2, `heute` 0 von 2, aber **keine** der vier GAT-Proben lief bis zum Urteil (siehe Ausfälle).
- Overrides (S2): in keiner Probe stand am Ende ein Override im Code; die Wirkung kam über die gezielte Reparatur.

## Ausfälle (wichtig für die Lesart)

9 von 16 Proben enden als „nicht reproduzierbar (Setup)“ (kein Urteil über die Methode), jeweils nach der einen erlaubten Reparatur: SGC 2, GAT 4, Kipf 3. Letzte Fehler (aus den Proben-Notizen): überwiegend Fehler am **Ein-/Ausgabevertrag** bzw. an der Smoke-Teilmenge (`IndexError: index 1708 is out of bounds … size 1000`, `numpy.ndarray … has no attribute 'cpu'`, `NameError`, falsche Zeilenzahl, `SparseTensor`-Import), bei GAT dreimal `Training > 600s`. Das sind die Fehlerklassen, die der **Minimal-I/O-Helfer (Hebel 4)** adressieren würde; er ist nicht gebaut. Der Pilot erlaubt nur **eine** Reparatur je Probe (Produkt: bis zu vier je Seed plus Kandidaten); er unterschätzt beide Arme.

- **Kipf (Nicht-Verschlechterung) ist nicht aussagekräftig:** beide Arme treffen 0 von 2, weil 3 von 4 Proben am I/O-/Smoke-Vertrag scheiterten und die vierte (`heute`, T 0,2) 0,8002 „teilweise“ erreichte. Die Bedingung 2 ist formal erfüllt (0 ≥ 0 − 1), belegt aber nicht, dass `cg` Kipf nicht verschlechtert. In Benchmark v4 Arm B traf Kipf 3 von 3; hier gibt es dazu keinen Vergleich.
- **GAT:** `cg` hat die Details, aber alle vier Proben (beide Arme) scheitern an Code-Fehlern oder am Zeitlimit. Nach Regel kein Beleg für oder gegen `cg` bei GAT.

## Vorfälle und Abweichungen (dokumentiert in den Addenda)

1. **Tageslimit:** OpenRouter Free (50 Anfragen/Tag) nach 29 gezählten Anfragen erreicht. Generierung (4 S1, 8 Pläne, 16 Code-Proben) lag vollständig vor, Messung und Reparaturen standen aus. Entscheidung der Nutzerin/des Nutzers: Rest auf Token Factory, **Super für alle Rollen, kein Ultra**, Kostengrenze 0,30 $ brutto (Addendum 1). Alle Reparaturen kommen von Super auf Token Factory, in beiden Armen gleich.
2. **Messartefakt (eigener Fehler, korrigiert):** Im ersten Messlauf liefen vier Docker-Container ohne Kernzuteilung parallel und überbuchten alle 16 Kerne (ein Seed allein 20 s, vier parallel alle über 600 s). Das verfälschte NLinear und SGC. Korrektur: `--cpuset-cpus` mit 4 Kernen je Container (Addendum 2), alle 16 Proben neu gemessen, die alten Ergebnisse liegen unter `recall_cg_pilot_samples/superseded_run1/` und zählen nicht. Auf die Entscheidung hat das einen Einfluss: Im ersten (verfälschten) Lauf stand NLinear `cg` bei 0 von 2.
3. **Reparaturen:** 11 Aufrufe auf Token Factory (7 vor, 4 nach der Korrektur; die ersten 7 blieben im Cache und wurden wiederverwendet). Aufrufzähler gesamt 40 von 45 (inkl. 1 abgewiesene Anfrage).
4. Kein Prototyp-Modus-Verstoß: nur öffentliche Papers und Repos im Prompt; Agenten-Code lief ausschließlich in Docker, im Container ohne Netz (`docker network disconnect` während des Trainings, Netz nur für den Harness).

## Kosten

OpenRouter Free: 0 $ (29 Anfragen). Token Factory (Super): dieser letzte Prozess 0,023 $ netto laut Log, dazu die 7 früheren Reparaturen rund 0,04 $ netto (Schätzung; das Log wird von anderen Sitzungen mitbenutzt). **Insgesamt ≈ 0,06 bis 0,08 $ brutto**, unter der Grenze von 0,30 $. Kein Tavily, kein Nebius.

## Grenzen der Aussage

- 6 Versuche je Arm auf den Entwicklungseinheiten, zwei Proben je Einheit und Arm teilen einen Plan. „3 von 6 gegen 0 von 6“ ist ein Signal, kein Test. Zwei der drei Treffer sind dieselbe Einheit (NLinear), der dritte eine einzelne SGC-Probe.
- Die NLinear-Fehlschläge von `heute` sehen nicht zufällig aus: Die Fakt-Checks zeigen in beiden Proben genau die drei Repo-Details als fehlend. Das stützt den Mechanismus, ist aber ebenfalls In-Sample.
- Kein Held-out: keine neue Einheit wurde getestet. Ob `cg` auf neuen Papers hilft, beantwortet erst das volle Gate G0 (`docs/recall_analysis_3.md` Abschnitt 5 mit RLinear).
- Modus-Änderung (Nachbau aus Paper **und** Autorencode, unabhängig gemessen) bleibt eine Entscheidung der Nutzerin/des Nutzers für README, Video und Zeugnis.

## Empfehlung

Regel erfüllt: Der Pilot gibt grünes Licht für das volle Gate G0 (10 $ OpenRouter, neun Einheiten, drei Proben, mit RLinear als Held-out). Vorher sinnvoll und billig: den **I/O-Helfer (Hebel 4)** bauen, weil die Mehrzahl der Ausfälle im Pilot am Ausgabevertrag scheitert, und ihn in G0 in beide Arme geben; sonst misst G0 wieder vor allem Vertragsfehler. Außerdem für G0 die Reparaturzahl wie im Produkt wählen oder die Kipf-Bedingung mit mehr Proben prüfen, sonst bleibt die Nicht-Verschlechterung unbelegt.
