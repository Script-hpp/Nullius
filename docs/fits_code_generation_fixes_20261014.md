# FITS: Code-Erzeugung und Fehlertexte nach Live-Lauf 35758c32

Grundlage: autonomer Live-Lauf `35758c32-51ae-4c01-85c4-9922cde52ce3` (Deployment `be1a643`, 0/3 Seeds,
kein Paperurteil). Die beiden Hauptursachen dieses Laufs (gekappte Dateiliste, unklarer Datenhelfer-Vertrag)
behebt bereits `30c67b0`. Dieser Nachtrag behebt drei weitere, im selben Ereignisprotokoll belegte Ursachen.

## 1. Temperaturstaffel erzeugte unbrauchbaren Code

Kandidat i startete bei `0.2 + 0.3 i`, jede Wiederholung nach fehlendem Code-Block lag 0.3 hoeher.
Kandidat 2 lief damit bei 0.8 / 1.1 / 1.4. Im Lauf:

- sechs `code_missing`: Antworten brechen mitten im Skript ab, ohne `finish_reason length`
  (die Kosten der Aufrufe liegen im Bereich vollstaendiger Antworten, kein Ausgabelimit erkennbar);
- bei 1.1 ein `SyntaxError` in einem f-String, bei 1.4 Wortsalat in mehreren Sprachen;
- Kandidat 2 lieferte nie Code. Kandidat 0 bei 0.2 hatte in beiden FITS-Laeufen keinen Abbruch.

Fix: Code-Antworten sind auf 0.7 gedeckelt (Kandidaten 0.2 / 0.5 / 0.7), Wiederholungen steigen um 0.15
bis zur Grenze. Damit der LLM-Cache an der Grenze nicht dreimal dieselbe Antwort liefert, traegt der
Hinweis ab der zweiten Wiederholung die Versuchsnummer.

## 2. Ende langer Trainingsausgaben ging auf Nebius verloren

Das Sandbox-SDK kappt stdout und stderr bei 65535 Bytes und behaelt den Anfang. Traceback,
`WorkerFailure`-Zusammenfassung des Brokers und faulthandler-Stacks stehen am Ende. Smoke- und
Trainingslauf schreiben jetzt in ein frisches `mktemp -d`-Verzeichnis; danach gehen nur die letzten
30000 Bytes je Strom zurueck, Exit-Code unveraendert. Nicht-ASCII-Bytes werden ersetzt, weil das SDK
strikt UTF-8 dekodiert und ein Schnitt mitten im Zeichen sonst eine Ausnahme ausloest.

## 3. Websuche fuer Fehler im eigenen Datenhelfer

Fehler, die in `nullius_io.py` ausgeloest werden, betreffen die Nutzung unseres Helfers. Die
Tavily-Fehlersuche lieferte dafuer pycbc-, xarray- und langchain-Treffer. Sie entfaellt jetzt fuer
diese Fehler; der Helfervertrag steht ohnehin im Prompt.

## Nicht geaendert

Keine Hyperparameter, kein Zielwert, keine Methode und keine Auswahlregel. Der Formfehler 186 vs. 96
(Rekonstruktion plus Forecast) bleibt Sache des Agenten; mit `train_mode 1` ist das Ausschneiden der
letzten 96 Schritte laut `exp/exp_main_F.py` korrekt, und der Autorencode liegt nach `30c67b0` im Prompt.
Ob der autonome Nachbau jetzt gelingt, zeigt nur ein neuer Live-Lauf.

## Gegenprobe: woher kam 0.7036 im Lauf ee039f0f?

Der Autorenkern `models/FITS.py` (Commit `d040bb0`) lief lokal im selben kausalen Vertrag, netzlos,
mit genau den Werten des Live-Plans: L90, `cut_freq` 3, LR 0.001, Batch 32, 10 Epochen, Patience 3.
Seeds 0/1/2: 0.70405 / 0.70345 / 0.70321, Mittel **0.70357 +/- 0.00043**. Der Live-Lauf hatte
**0.70358 +/- 0.00061**. Mit dem veroeffentlichten Preset liefert derselbe Aufbau 0.37205.

Damit ist belegt: Harness und Vertrag messen richtig, der Agentencode von ee039f0f setzte FITS
funktional wie der Autorenkern um, und die gesamte Abweichung kommt aus dem falschen Setting
(welcher Einzelwert dominiert, ist nicht ablatiert; plausibel ist `cut_freq` 3: bei 90 Schritten liegt die Tagesperiode bei Frequenzindex 3.75 und wird
abgeschnitten). Das stuetzt die Setting-Fixes aus `be1a643`/`30c67b0`; es ist kein Nachweis eines
autonomen Erfolgs. Skript `scripts/review/fits_live_setting_probe.py`, Beleg
`docs/audits/fits_live_setting_probe_20261014.json`. Keine Provideraufrufe.

## Lokaler autonomer Lauf: FITS reproduziert

Gleiche Produktkonfiguration wie ein Live-Schnelltest (`api.make_orchestrator(quick=True)`,
`catalog.run_candidate`), gestartet von diesem Rechner: Token Factory, Nebius-Sandbox, Tavily live,
kein LLM-Cache. `src` identisch mit main `469fc10`. Skript `scripts/review/local_product_run.py`,
kuratierte Ereignisse `docs/audits/fits_local_autonomous_20261014.json`.

- Dateiliste 1410/1410, gelesen `models/FITS.py`, `scripts/FITS/etth1_best.sh`, `run_longExp_F.py`,
  `exp/exp_main_F.py`; kein `AD/`.
- Spec mit allen neun Werten aus dem Autorenrepo (LR 0.0005, 50 Epochen, Batch 128, Patience 20,
  L360, `cut_freq` 106, 7 Kanaele, H96, `train_mode` 1); Spec-Pruefung 9/9 fuer alle Kandidaten.
- Alle drei Kandidaten liefen nach je einer bzw. zwei Reparaturen; Auswahl nur ueber die verborgene
  Validierung (Kandidat 1).
- Seeds 0/1/2: 0.3719944 / 0.3720073 / 0.3720915, Mittel **0.3720310 +/- 0.0000527**, Ziel 0.372,
  Produkturteil `reproduziert`, Methode `nicht geprueft`, Gesamtaussage `vorlaeufig`.
- Dauer 1391 s (Server-Watchdog 2700 s), Modellkosten **0.2513 $ netto / 0.299 $ brutto**.

Damit funktioniert der autonome Nachbau fuer FITS lokal. Der Live-Nachweis auf dem Server steht aus.

### Restliche Reibung, allgemein behoben

- **Ausgabelaenge:** Jeder der drei Kandidaten scheiterte zuerst daran, dass FITS Rekonstruktion plus
  Forecast liefert (456 bzw. 186 Schritte); zweimal im Loss, einmal in der Rueckgabe des Predictors.
  Im Live-Lauf 35758c32 war genau das der letzte, nicht mehr reparierte Fehler. Der Forecast-Vertrag
  sagt jetzt allgemein: Loss und Rueckgabe decken genau die 96 Zukunftsschritte ab; ein Modell, das
  auch das Eingabefenster ausgibt, wird auf die letzten 96 Schritte geschnitten, ausser das Paper
  trainiert auch auf der Rekonstruktion. Das betrifft jede Backcast-/Rekonstruktionsarchitektur,
  nicht nur FITS, und legt keinen Hyperparameter fest.
- **Websuche:** Auch Fehler, die unser `forecast_worker.py`/`forecast_runner.py` ausloest
  (z. B. falsche Rueckgabeform), loesen keine Tavily-Suche mehr aus.

### Beobachtet, nicht behoben

Eine Code-Antwort brach erneut mitten im Skript ab, `finish_reason stop`, 4193 Completion-Tokens
(davon 1586 Reasoning) bei Limit 24000. Kein Ausgabelimit; der Wiederholungsversuch lieferte Code.
Ursache auf Provider-/Modellseite nicht weiter eingrenzbar.

## Live-Lauf 5fdc78fc nach Redeploy (Server, `469fc10`)

[Live-Verlauf](https://nullius.akca.dev/#/live/5fdc78fc-6b99-450a-a9d9-9fd599a9bea9). Listing-, Setting- und
Helfer-Fixes sind auf dem Server wirksam: gelesen `etth1_best.sh`, `run_longExp_F.py`, Spec 9/9 mit LR 0.0005
aus dem Startskript. Kandidat 0 lief **im ersten Versuch** mit Test-MSE 0.3719944, Kandidat 1 nach einer
statischen Checkpoint-Reparatur mit demselben Wert. Danach endete der Lauf **ohne Urteil**:
`BudgetExceeded: Live model budget floor reached`. Modellkosten des Jobs 0.182 $ netto.

Zwei Ursachen:

1. **Server-Budgetkonfiguration:** Die Untergrenze `LIVE_MIN_BUDGET_USD` (3 $) wird gegen
   `BUDGET_TOTAL_USD` minus geloggte Ausgaben seit `BUDGET_SINCE` geprueft. Auf dem Server war dieser Rest
   knapp ueber 3 $; die konservativen Reservierungen (max_tokens x Preis) haben ihn waehrend des Laufs
   unterschritten. Laut Nutzer liegen auf dem Konto rund 11 $. Abhilfe ist eine Konfigurationsaenderung in
   Coolify (aktueller Kontostand und Zeitpunkt eintragen), kein Code.
2. **Allgemeiner Codefehler:** Die Grenze traf die optionale Reparatur des Repo-Abgleichs (A2). Der Abgleich
   selbst fing `BudgetExceeded` schon ab, die nachfolgende Reparatur nicht. Damit beendete eine optionale
   Verbesserung einen Lauf mit validiertem Code, obwohl die restlichen Seeds kein Modell brauchen. Jetzt
   bleibt der validierte Code und wird weiter gemessen (`36d036b`, Regressionstest in `tests/test_repo_diff.py`).

## Zweites Paper lokal: SARAF (Allgemeinheit)

Zwei weitere allgemeine Blocker vor jedem Experiment gefunden und behoben, beide offline am echten
Autorenrepo `ShiqiaoZhou/SARAF@7b49aa8` reproduziert (`scripts/review/repo_closure_probe.py`):

- **Importhuelle:** Der Repo-Leser hoerte nach geschaetzten 120k Zeichen (pauschal 10k je Datei) auf; der
  Quellenwaechter verlangt aber die vollstaendige Importhuelle der Trainingsdateien (hier `utils/dtw.py`,
  7 kB, lazy in `utils/augmentation.py`). Der Leser zaehlt jetzt echte Groessen bis 240k, oberhalb der
  Waechtergrenze 160k. Zusaetzlich laeuft der deterministische Waechter direkt nach dem Lesen, vor der
  bezahlten Rolle-4-Extraktion (SARAF hatte 0.12 $ vor dem Stopp bezahlt). (`03e0e3b`)
- **Geraeteflags:** `run.py` setzt `args.gpu = args.device_ids[0]`; der Startparameter-Waechter meldete `gpu`
  als ungeloesten Modellparameter. Die vorhandene Infrastrukturliste (gpu, devices, use_gpu, seed, ...) gilt
  jetzt auch dort. Offline danach: L720, topm 20, 10 Epochen, Batch 8, LR 0.001 aus `scripts/ETTh_720.sh`. (`36d036b`)

### SARAF-Lauf c: zwei weitere allgemeine Ursachen

- **Ein Flag unter zwei Namen erzwungen:** Der Extraktor ordnete `--n_period` (Anzahl Perioden) dem
  Schluessel `layers` zu; das Zitat stand woertlich im Repo und wurde akzeptiert. Der Startparameter-Leser
  liefert `n_period` ohnehin. Beide Eintraege waren erzwungen, der Agent konnte nur einen Namen nutzen,
  und die Spec-Reparaturen drehten sich bis zum Jobbudget im Kreis (0.36 $ netto ohne Messung). Ein
  zitiertes Flag, das der Startaufruf schon unter eigenem Namen aufloest, gilt jetzt nicht zusaetzlich
  als anderer Schluessel (`f1764ff`).
- **Vollstaendiger Code mit Tag statt Zaun:** Zwei bezahlte, vollstaendige Skripte endeten mit
  `</|tool_call|>` bzw. `</output>` statt ``` ``` ```. Der tolerante Parser nahm das Tag mit, `ast.parse`
  scheiterte, die Antwort galt als „kein Code“. Abschliessende Tag-Zeilen werden jetzt entfernt (`f1764ff`).
  Die frueheren FITS-Abbrueche ohne Tag (mitten in einer Zeile, `finish_reason stop`) bleiben echte Abbrueche.

### SARAF-Lauf d: vollstaendig gemessen

Lokal, `src` = main `f1764ff`: Spec 12 Werte aus `scripts/ETTh_720.sh`/`run.py` (L720, topm 20,
10 Epochen, Batch 8, LR 0.001). Drei Kandidaten; Kandidat 0 scheiterte an vier Methoden-Codefehlern
(Retrieval-Formen), Kandidaten 1 und 2 liefen nach Reparaturen. Auswahl ueber verborgene Validierung:
Kandidat 2. Seeds 0/1/2: 0.38263 / 0.38445 / 0.38391, Mittel **0.38366 +/- 0.00094**, Ziel 0.371,
Produkturteil `teilweise`, Gesamtaussage `vorlaeufig`. Frueherer Live-Wert 0.4917; die menschliche
Referenz mit unveraendertem Autorenkern im selben Vertrag lag bei 0.3696. Keine Parameter nach Testwerten
gewaehlt. Kosten 0.460 $ netto / 0.548 $ brutto, 1453 s. Beleg `docs/audits/saraf_local_autonomous_20261014.json`.

Daraus ein weiterer allgemeiner Fix: Der Lauf brauchte mehr als das Live-Jobbudget von 0.50 $. Die
Erzeugung weiterer Kandidaten fiel an der Grenze schon auf validierte Kandidaten zurueck, eine Reparatur-
oder Spec-Runde eines spaeteren Kandidaten beendete aber den ganzen Lauf. Jetzt gilt dieselbe Regel auch
dort (`229255a`). Zusaetzlich nennt der Forecast-Vertrag die fehlenden Bibliotheken (pandas, tqdm,
scikit-learn), weil Kandidaten Autorencode mit `tqdm` kopierten.

## Stand und offene Punkte

| Paper | Ort | Ergebnis |
| --- | --- | --- |
| FITS | lokal autonom | reproduziert, 0.37203 +/- 0.00005 (Ziel 0.372) |
| FITS | live 5fdc78fc | Kandidat 0.37199, dann Abbruch an Server-Budgetuntergrenze, kein Urteil |
| SARAF | lokal autonom | teilweise, 0.38366 +/- 0.00094 (Ziel 0.371) |

- **Server-Konfiguration (Nutzer):** `BUDGET_TOTAL_USD`/`BUDGET_SINCE` in Coolify auf den aktuellen
  Kontostand setzen; sonst blockiert `LIVE_MIN_BUDGET_USD` jeden weiteren Live-Lauf. Ob `MAX_COST_PER_RUN_USD`
  fuer SARAF-grosse Repos (~0.55 $) angehoben werden soll, ist eine Budgetentscheidung.
- **Deployment:** Fixes ab `bd6af43` sind gepusht, aber nur per manuellem Redeploy aktiv; der GitHub-Webhook
  erreicht Coolify nicht (Payload-URL ist ein Tailnet-Name, letzte Zustellungen `EOF`).
- **Nicht behoben:** echte Abbrueche mitten in Code-Antworten mit `finish_reason stop` (Retry faengt sie);
  SARAF-Abstand 0.013 zum Ziel (Methodenebene, nicht weiter untersucht, kein Tuning).
- Gesamtkosten dieser Nacht: lokal 1.58 $ brutto (FITS 0.30, SARAF a-d 1.28), live 0.22 $ brutto.
