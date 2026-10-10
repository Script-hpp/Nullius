# Budgetplan Token Factory bis 15.12.2026

> **Stand 2026-10-09:** Ultra (`nvidia/Nemotron-3-Ultra-550b-a55b`) wurde von Token Factory entfernt (404). Rolle `reasoning` läuft auf Super (`TF_MODEL_REASONING=nvidia/nemotron-3-super-120b-a12b`), Kosten je Plan und Urteil damit niedriger als unten mit Ultra gerechnet. Die Ultra-Zeile unten ist der Abgleich vom 02.10. und bleibt als Historie.

Stand 2026-10-02 abends. Alle Beträge brutto (netto × 1,19), aus `logs/costs.jsonl`. Maßgeblich bleibt die Nebius-Konsole; letzter Abgleich am 02.10. 19:43 UTC (siehe unten). **Vor dem Deployment erneut in der Konsole prüfen**; die Vergleichswerte in `scripts/cost_report.py` sind älter und dort noch nicht nachgetragen.

## Abgleich mit der Nebius-Konsole (2026-10-02, 19:43 UTC)

Vom Nutzer aus der Konsole abgelesen (Usage 02.09. bis 02.10.2026, „Last updated: 17:45 UTC"), daneben das Log bis 17:45 UTC, ohne Cache-Treffer:

| | Konsole | Log | Abweichung |
|---|---|---|---|
| Ultra, Eingabe / Ausgabe (Mio. Tokens) | 1,08 / 0,84 | 1,07 / 0,83 | Rundung |
| Lightning, Eingabe / Ausgabe | 0,10 / 0,08 | 0,10 / 0,08 | keine |
| Super, Eingabe / Ausgabe | 30,10 / 14,21 | 30,10 / 14,21 | keine |
| Summe netto | 25,43 $ | 25,41 $ | 0,1 % |
| MwSt. | 4,83 $ | 4,83 $ | |
| Summe brutto | 30,25 $ | 30,24 $ | 0,03 % |
| **Kontostand** | **30,74 $** | 31,09 $ (59,50 minus Verbrauch seit Stichtag) | 0,35 $ |

Lesart: Verbrauch und Preise im Log stimmen mit der Konsole überein (weit unter der Schwelle von 10 %). Der **Kontostand der Konsole liegt 0,35 $ unter dem Wert, den das Log ableitet**. Die Ursache ist nicht geklärt; möglich ist, dass der Kontostand 59,50 $ nicht genau zum Stichtag 29.09. 08:00 UTC abgelesen wurde. Nach 17:45 UTC kamen laut Log nur 0,04 $ dazu. **Maßgeblich ist die Konsole: für die Planung gilt 30,74 $.**

Stichtag neu gesetzt (vom Nutzer am 2026-10-02 in `.env` eingetragen, vom Reviewer geprüft): `BUDGET_TOTAL_USD=30.74`, `BUDGET_SINCE=2026-10-02T19:43Z`. Code und Konsole rechnen damit wieder gleich. Warnung bei 80 % verbraucht (24,59 $), automatischer Stopp bei 95 % (29,20 $ verbraucht, 1,54 $ Rest).

## Stand

| | $ |
|---|---|
| Kontostand zum Stichtag 2026-09-29 | 59,50 |
| Verbraucht laut Log | 28,38 |
| Rest laut Log (Stand des ersten Plans, vor dem Abgleich oben) | 31,12 |
| **Kontostand laut Konsole, 02.10. 19:43 UTC** | **30,74** |

Verbrauch je Tag: 29.09. 1,73 · 30.09. 0,32 · 01.10. 11,83 · 02.10. 14,51 (Benchmark v4, v5 Schritt A, Probelauf).

**Builders Program:** laut Nutzer (2026-10-02) längst eingelöst; das Guthaben steckt also schon im Kontostand. **Es kommt kein weiteres Guthaben hinzu.** Was auf dem Konto ist, muss bis 15.12. reichen; darüber hinaus zahlt die hinterlegte Karte. Verfall des Builders-Guthabens (90 Tage nach Ausgabe): Datum in der Konsole prüfen, es muss nach dem 15.12.2026 liegen.

## Plan bis zur Abgabe (30.10.)

| Posten | Obergrenze $ | Art |
|---|---|---|
| Probelauf v5 Schritt C, Rest (läuft aus, wird nicht wiederholt) | 0,50 | Schätzung |
| Nachlauf M3, Entwicklungsläufe an RLinear und DLinear | 1,00 | Schätzung, Reviewer stoppt bei 1 $ |
| Nachlauf M4, Prüfung an vier neuen Einheiten | 3,00 | fester Stopp laut `nachlauf_v1.md` |
| Vorzeige-Läufe für Video und Katalog (Kipf live, RLinear mit `drop_last`-Hinweis, ein Nachlauf, zwei bis drei Katalog-Nachläufe) | 2,00 | Obergrenze |
| Reserve für Unvorhergesehenes | 4,00 | |
| **Summe** | **10,50** | |

Bleibt für die Jury-Phase: rund **20 $** (30,74 $ Kontostand minus 10,50 $; die 0,35 $ Differenz zum Log gehen zulasten der Reserve).

Nicht mehr geplant: Benchmark v5 und ein zweiter Probelauf (vom Nutzer gestrichen), weitere Messrunden von Schritt A. Jeder weitere Lauf mit Kosten über 1 $ braucht eine Entscheidung des Nutzers.

## Jury-Phase (30.10. bis 15.12., 47 Tage)

**Update 2026-10-10:** The allocation table below is an older plan. The current deployment reference is [`docs/deploy.md`](deploy.md#limits-bis-1512) and `deploy/.env.example` ($12 server allocation, $0.50 run threshold). Verify the current console balance before deploying; historical balances are not current funds. [`Cost policy review`](cost_policy_review.md) documents why these thresholds are not guaranteed spending caps.

Die Demo muss erreichbar und kostenlos testbar bleiben. Juroren müssen nicht selbst testen; ein gesperrter Live-Lauf ist kein Ausfall, das Frontend zeigt dann die Wiedergabe.

**Einstellungen auf dem Server** (`deploy/.env` bzw. Coolify):

| Variable | Wert | Grund |
|---|---|---|
| `BUDGET_TOTAL_USD` | 20.00 | Anteil des Servers, **nicht** der ganze Kontostand (siehe „Zwei Logs") |
| `BUDGET_SINCE` | Zeitpunkt des Deployments (UTC) | Server zählt nur eigene Läufe ab dann |
| `MAX_DAILY_REPRO_JOBS` | 1 (ab 2026-12-01: 3) | Stand 2026-10-07: zwei Stufen und Rechnung in `docs/deploy.md`, „Limits bis 15.12.“ (gemessen 15,30 $ von 17,00 $ verfügbar); die Untergrenze `LIVE_MIN_BUDGET_USD=3` ist die harte Obergrenze |
| `MAX_DAILY_JOBS` (Discovery) | 1 | Discovery braucht vor allem Tavily-Credits (ca. 16 je Lauf) |
| `MAX_COST_PER_RUN_USD` | 0.75 | seit 2026-10-09 Standardmodus „Nachbau aus Paper und Autorencode“ (recall cg v1: Mittel 0,25 $, Maximum 0,46 $ brutto je Lauf); Planungswert 0,30 $ je Lauf, Folgen für die Tageslimits in `docs/deploy.md`, „Limits bis 15.12.“. Kontostand laut Log nach recall cg v1: 17,39 $ (Konsole nicht verifiziert) – der Server-Anteil von 20,00 $ ist damit nicht mehr gedeckt |

Erwartung statt schlimmstem Fall: Eine Schnellprüfung kostete vor `setup_v2` 0,03 bis 0,10 $ (`docs/deploy.md`). Mit `setup_v2` ist sie teurer; der Probelauf kam auf rund 0,39 $ je Benchmark-Lauf mit **zwei** Claims und vollen Seeds, das ist keine Messung einer Schnellprüfung. **Offen:** Kosten einer Schnellprüfung mit dem aktuellen Code messen (fällt beim Vorzeige-Lauf fürs Video ohnehin an) und hier eintragen. Liegt sie unter 0,25 $, kann `MAX_COST_PER_RUN_USD` auf 0.30 sinken.

Die Werte oben gelten ohne Aufstockung. Lädt der Nutzer das Konto auf, können `MAX_DAILY_REPRO_JOBS` und der Server-Anteil steigen; dann beide Werte und `BUDGET_SINCE` neu eintragen.

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

Kontingent getrennt von Token Factory. Stand laut Dashboard (vom Nutzer abgelesen, 2026-10-02 abends):

| | verbraucht | verfügbar |
|---|---|---|
| Monatsplan „Researcher" | 100 | 1 000 Credits je Monat |
| Add-on (Builders Program) | 0 | 3 125 Credits |
| „Pay as you go" | aus | Bei leerem Kontingent schlagen Aufrufe fehl, es entstehen keine Zusatzkosten. **So lassen.** |

Verfügbar bis 15.12.: 900 Credits im laufenden Monat, dazu der Monatsplan für November und Dezember (falls der Plan weiterläuft: je 1 000) und 3 125 Credits Add-on. Ob und wann das Add-on verfällt, im Dashboard prüfen.

Verbrauch je Lauf ist **nicht gemessen**. Bis zum Deployment einmal messen (Credits im Dashboard vor und nach einem Live-Lauf und einer Discovery) und hier eintragen; erst dann lässt sich sagen, wie viele Läufe pro Tag das Kontingent bis 15.12. trägt. Bis dahin gilt (Stand 2026-10-07): `MAX_DAILY_JOBS=1` (Discovery) und `MAX_DAILY_REPRO_JOBS=1`.

Im Plan oben fallen Tavily-Aufrufe nur bei den Vorzeige-Läufen an (Standard `TAVILY_MODE=live`) und in der Jury-Phase bei Live-Läufen und Discovery. M1 bis M4 des Nachlaufs laufen ohne Tavily. Regeln unverändert: Crawl immer mit `limit`, Research höchstens einmal je Paper, keine Tests mit echten Aufrufen.
