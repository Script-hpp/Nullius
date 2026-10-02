# Budgetplan Token Factory bis 15.12.2026

Stand 2026-10-02 abends. Alle Beträge brutto (netto × 1,19), aus `logs/costs.jsonl`. Maßgeblich bleibt die Nebius-Konsole; der letzte Abgleich war am 01.10. (Abweichung 0,4 %). **Vor dem Deployment erneut in der Konsole prüfen**, die Vergleichswerte in `scripts/cost_report.py` sind älter.

## Stand

| | $ |
|---|---|
| Kontostand zum Stichtag 2026-09-29 | 59,50 |
| Verbraucht laut Log | 28,38 |
| Rest | 31,12 |

Verbrauch je Tag: 29.09. 1,73 · 30.09. 0,32 · 01.10. 11,83 · 02.10. 14,51 (Benchmark v4, v5 Schritt A, Probelauf).

Zugesagt, aber noch nicht eingelöst: Builders Program, +25 $ Token Factory (verfällt 90 Tage nach Ausgabe). Der Plan unten kommt **ohne** dieses Guthaben aus.

## Plan bis zur Abgabe (30.10.)

| Posten | Obergrenze $ | Art |
|---|---|---|
| Probelauf v5 Schritt C, Rest (läuft aus, wird nicht wiederholt) | 0,50 | Schätzung |
| Nachlauf M3, Entwicklungsläufe an RLinear und DLinear | 1,00 | Schätzung, Reviewer stoppt bei 1 $ |
| Nachlauf M4, Prüfung an vier neuen Einheiten | 3,00 | fester Stopp laut `nachlauf_v1.md` |
| Vorzeige-Läufe für Video und Katalog (Kipf live, RLinear mit `drop_last`-Hinweis, ein Nachlauf, zwei bis drei Katalog-Nachläufe) | 2,00 | Obergrenze |
| Reserve für Unvorhergesehenes | 4,00 | |
| **Summe** | **10,50** | |

Bleibt für die Jury-Phase: rund **20 $**.

Nicht mehr geplant: Benchmark v5 und ein zweiter Probelauf (vom Nutzer gestrichen), weitere Messrunden von Schritt A. Jeder weitere Lauf mit Kosten über 1 $ braucht eine Entscheidung des Nutzers.

## Jury-Phase (30.10. bis 15.12., 47 Tage)

Die Demo muss erreichbar und kostenlos testbar bleiben. Juroren müssen nicht selbst testen; ein gesperrter Live-Lauf ist kein Ausfall, das Frontend zeigt dann die Wiedergabe.

**Einstellungen auf dem Server** (`deploy/.env` bzw. Coolify):

| Variable | Wert | Grund |
|---|---|---|
| `BUDGET_TOTAL_USD` | 20.00 | Anteil des Servers, **nicht** der ganze Kontostand (siehe „Zwei Logs") |
| `BUDGET_SINCE` | Zeitpunkt des Deployments (UTC) | Server zählt nur eigene Läufe ab dann |
| `MAX_DAILY_REPRO_JOBS` | 1 | schlimmster Fall 47 × 0,50 $ = 23,50 $; der Client stoppt vorher bei 95 % = 19,00 $ |
| `MAX_DAILY_JOBS` (Discovery) | 3 | Discovery braucht vor allem Tavily-Credits |
| `MAX_COST_PER_RUN_USD` | 0.50 | unverändert, bis die Messung unten vorliegt |

Erwartung statt schlimmstem Fall: Eine Schnellprüfung kostete vor `setup_v2` 0,03 bis 0,10 $ (`docs/deploy.md`). Mit `setup_v2` ist sie teurer; der Probelauf kam auf rund 0,39 $ je Benchmark-Lauf mit **zwei** Claims und vollen Seeds, das ist keine Messung einer Schnellprüfung. **Offen:** Kosten einer Schnellprüfung mit dem aktuellen Code messen (fällt beim Vorzeige-Lauf fürs Video ohnehin an) und hier eintragen. Liegt sie unter 0,25 $, kann `MAX_COST_PER_RUN_USD` auf 0.30 sinken.

Wird das Builders-Guthaben eingelöst: `MAX_DAILY_REPRO_JOBS` auf 3, Server-Anteil auf 35 $, beide Werte und `BUDGET_SINCE` neu eintragen.

## Zwei Logs, ein Konto

Server und lokaler Rechner nutzen dasselbe Nebius-Konto, führen aber getrennte Kosten-Logs. Die Budgetprüfung sieht nur das eigene Log. Deshalb bekommt jede Seite einen festen Anteil:

- Server: 20 $ ab Deployment (oben).
- Lokal: nach dem Deployment `BUDGET_TOTAL_USD` auf den Kontostand der Konsole **minus 20 $** setzen, `BUDGET_SINCE` auf denselben Zeitpunkt.

Token Factory belastet die Karte, wenn das Guthaben ins Minus geht. Die Summe beider Anteile darf deshalb nie über dem Kontostand der Konsole liegen.

## Regeln und Auslöser

- Echte Läufe startet nur der Reviewer; nach jedem Lauf stehen die Kosten aus dem Log im Bericht.
- Lokal 80 % erreicht (Warnung des Clients): keine Läufe mehr außer den Posten der Tabelle oben.
- Lokal 95 % erreicht: der Client bricht ab. Nicht das Budget erhöhen, sondern Konsole prüfen und neu planen.
- Server 80 %: `MAX_DAILY_REPRO_JOBS` bleibt 1, Hinweis an den Nutzer. Server 95 %: Live-Läufe gesperrt, Wiedergabe und Spiegel bleiben.
- Konsole prüfen: vor dem Deployment, am 30.10. nach der Abgabe, danach jeden Montag bis 15.12. Weicht das Log um mehr als 10 % ab: Log oder Preistabelle korrigieren, nicht das Budget.

## Bekannte Lücke

Der Probelauf von Schritt C hat seinen Stopp von 2 $ überschritten (3,08 $ bis zum Stand dieses Plans; drei Generierungen statt einer). Vor dem Start von M4 prüft der Reviewer mit `--dry-run` und einem Blick in den Code, dass der Stopp von 3 $ im Lauf wirklich greift, und überwacht den Lauf zusätzlich von außen.

## Tavily

Kontingent getrennt von Token Factory. Im Plan oben fallen Tavily-Aufrufe nur bei den Vorzeige-Läufen an (Standard `TAVILY_MODE=live`) und in der Jury-Phase bei Live-Läufen und Discovery. M1 bis M4 des Nachlaufs laufen ohne Tavily. Stand der Credits vor dem Deployment im Tavily-Dashboard prüfen und hier eintragen.
