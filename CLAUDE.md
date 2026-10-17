# Paper-Reproduktions-Agent

**Jury-Pfad "Apollo" (2026-10-15):** Jury-Klicks muessen kontrolliert enden (Urteil oder erklaerter Stopp vor Kosten). Fixes: Redeploy uebernimmt kuratierte Urteile/Jury-Flags aus catalog.yaml ins Volume (`d5b19ea`, vorher blieb Server-Jury-Set alt); Discovery-Stream gepuffert mit `from=`/Reconnect, Tab-Schliessen bricht Suche nicht ab (`d847cce`); **Vorpruefung** nach jeder Discovery (`56e9809`): `Orchestrator.preflight` laeuft alle Schritte vor erster bezahlter Codegenerierung (Ziel/Tabelle, Repo, Setup, Steckbrief, Modus), Startknopf nur bei Bestehen im Autorencode-Modus, Lauf setzt exakt am gespeicherten Stand fort (`logs/preflight/*.pkl`, nur gleicher Claim/Config/Codehash). Alte gespeicherte Funde: `python -m repro_agent.preflight_saved --run` im Container. Video v10 bleibt final (zeigt TSMixer/SparseTSF/GLinear als erreicht). Lokale Volllaeufe nach `71afca2`: BANGS Urteil teilweise 0.819+/-0.007 (Ziel 0.842, 0.44 $ brutto, ~80 min wegen 600-s-Trainings); SST Budgetstopp ohne Seed, Ursache GPU-Kernel `mamba_ssm` im Autorenmodell -> Vorpruefung sperrt jetzt GPU-only-Bibliotheken ohne Fallback (`docs/audits/preflight_real_20261015.json`). Offen: Laufdauer grosser Graph-Methoden, Budget 0.50 $ fuer grosse Repos.

**FITS Deployment-Gegenprobe und Nachfix (2026-10-13):** `docs/fits_deployed_verification_20261013.md`. Echter autorisierter Job35758c32 nach be1a643: Ziel0.372/v3/tab:FITSett und L360 bestaetigt, aber0/3Seeds/keinPaperurteil. Modellkosten0.27836$ netto (~0.33125$ bei19%VAT); kein zweiterLivejob. Neue belegte Ursache: Nebius-stdout65535 kappt FITS-Dateiliste83540Bytes/1410Pfade vor models/FITS.py@82903 und Startskripten; vorige lokale Attrappe lieferte volle Liste. verified_listing ueber eigene NUL-Datei, Bytes/Anzahl/SHA, max1MB vor Transfer; auch ls-tree-Unterordner, Sonderzeichen und kaputteMetadaten fail-closed. setup_details.repo.n_tracked sichtbar. Zweiter belegter Helper-Promptmangel: forecast_data returnsdict, Agent benutzteTuple/split; genaueSignatur+Keys+Formen+ausgefuehrtesBeispiel nun im Agentenvertrag. OpusDelta8d85810 keinBlocker; 1577 BackendTests,162naheGates, neuesProduktionsimage/Health200. Darin cap65535 aufjedemstdout: volle1410Pfadliste, kanonischesModell/Startskript, ADraus, korrekte9FITS-Settings. Nachfix braucht Deployment und autonomenNeuversuch; erfolgreicher autonomerFITSLauf weiteroffen. Separate SDK-stderr-Kappung und LLM-Promptliste600 weiteroffen/imReview dokumentiert.

**FITS-Settings-Ursachenfix (2026-10-13, lokal):** `docs/fits_setup_binding_fixes_20261013.md`. Aktive LaTeX-Kommentare/Mehrfachheader ausgeschlossen, belegte Discovery360 direkt in Spec; Forecast-Dateiauswahl korrigiert AD->models/FITS.py + etth1_best.sh + run_longExp_F.py und echte Importkette. Begrenzte statische Shell/argparse/Arithmetik ergibt L360/batch128/LR0.0005/50Epochen/pat20/cut106/train_mode1; Autorenprogramme dabei nicht ausgefuehrt. Identitaetsschleifen fuer SparseTSF/SegRNN auf H96, multivariate FeaturesM, NLinear nicht DLinear; unbekannte Mutationen/gleichrangige Konflikte stoppen. Pflichtsettings bis finale Zulassung abgesichert. Primaere Worker-Ausnahme ueber eigenen FD statt Brokerfehler. Voller 13-Dateienpfad im Produktionsimage netzlos bestanden. Drei neue lokale Trainingslaeufe mit automatisch kompilierter CONFIG: 0.372051543/0.371997134/0.372089070, Mittel0.372045916 +/-0.000046225. **Fester menschlicher Referenzadapter, kein autonomer Nemotron-Reproduktionsnachweis.** Historischer schlechter Lauf unveraendert. Null neue Nullius-Provider-Aufrufe; separate Opus-Beratung ueber Orca. Schlussgates bestanden: 1567 Backend-Tests, 143 fokussierte Settings-/Repo-Tests, 32 Docker-Vertragstests; Produktionsbuild/Quellhash-Smoke/Health200. Opus-Schlusskontrolle kein Blocker. Bereit fuer Deployment; autonomer FITS-Neulauf weiterhin offen.

**Katalog-Regression lokal (2026-10-14):** `docs/audits/catalog_regression_local_20261014.json`. Kipf reproduziert 0.815+/-0.0085, GLinear reproduziert 0.3867+/-0.0020, FITS live reproduziert; TSMixer mit korrekten Paper-Settings >600 s CPU (frueherer Erfolg 9.10. vermutlich mit argparse-Defaults), SparseTSF Stopp (Autoren-Startskript startet ohne --model_type die MLP-Variante). Fixes: LaTeX-Hyperparameter-Tabellen, Spaltenleser Methoden-Zeilen, Binaerdateien nie 'gekuerzt', Timeout-Stopp 2/3, nio-Import, numpy 2. **Regel: nach groesseren Aenderungen erst Katalog lokal rechnen, dann deployen.** Jury-Set (`jury.ready`): nur FITS, Kipf GCN Cora, GLinear; uebrige per `jury_override` gesperrt (sperrt nur, gibt nie frei).

**Discovery/Zielpruefung Ursachenfixes (2026-10-14, Nachmittag):** bekannte Papers vor dem Lesen ueberspringen, kostenlose Tabellen-Vorpruefung, Limit 3 nur fuer bezahlte Lesungen (`ed48b85`); Quote-Fehler nie cachen + ein Retry (`9bef7a9`); JSON-Escape-Schaeden (`\t`->Tab, `\b`->Backspace) in Zitaten reparieren (`b6c4393`); Accuracy-Claims an eigene LaTeX-Tabelle binden, LaTeX in Dokumentreihenfolge (`a5d9ea5`, BANGS 84.23 tab:GCN statt Anhang 83.47). Neue Funde: BANGS, CLAR, Graph Ladling, SST. Details am Ende von `docs/fits_code_generation_fixes_20261014.md`.

**Unabhaengiger Live-Test RLinear (2026-10-14):** Discovery 1ec41249: GILT war Few-shot/In-Context (Fix `6618c29`, Discovery sperrt Shot-Tabellen). RLinear ba9fa94f: 0.397 +/- 0.013, Ziel 0.366, nicht reproduziert, 0.19 $ brutto. Ursachen allgemein: Laenge nur in Claim-Tabellen-Caption ("historical horizon is 336") nicht gebunden -> argparse 96 (Fix `8abc8da`); Signatur-Default `EarlyStopping(patience=7)` schlug uebergebenen `--patience` 10 (Fix `bb5f890`). Zielerreichung mit Fixes ungeprueft. Abschnitt in `docs/fits_code_generation_fixes_20261014.md`.

**FITS live autonom reproduziert (2026-10-14):** Job a514edeb nach Redeploy b4bb239: Spec 9/9, Seeds 0.371994/0.372007/0.372091, **0.37203 +/- 0.00005**, Ziel 0.372, Urteil reproduziert/vorlaeufig, 0.31 $ brutto. Beleg `docs/audits/fits_live_a514edeb_20261014.json`, Abschnitt in `docs/fits_code_generation_fixes_20261014.md`. Danach lokal: Schlusszeichen-`}` statt Codezaun behalten, Syntaxfehler im Produktpfad vor Sandbox neu schreiben (spart Sandbox, Tavily, Reparatur); nicht deployt.

**Autonome Laeufe FITS/SARAF, Ursachenfixes (2026-10-14):** `docs/fits_code_generation_fixes_20261014.md`. Lokaler autonomer Produktlauf (TF + Nebius, `scripts/review/local_product_run.py`): **FITS reproduziert 0.37203 +/- 0.00005** (Ziel 0.372, 0.30 $), **SARAF teilweise 0.38366 +/- 0.00094** (Ziel 0.371, frueher live 0.4917, 0.55 $). Live 5fdc78fc: FITS-Kandidat 0.37199, dann Abbruch an Server-Budgetuntergrenze (Coolify `BUDGET_TOTAL_USD`/`BUDGET_SINCE` aktualisieren). Allgemeine Fixes: Code-Temperatur <=0.7; Nebius-Ausgabe-Ende behalten; Forecast-Vertrag (96 Zukunftsschritte, keine pandas/tqdm); Importhuelle nach echter Groesse lesen + Quellenwaechter vor bezahlter Extraktion; Geraeteflags kein Modellparameter; ein Flag nie unter zwei Schluesseln; tag-geschlossene Codeantworten behalten; Budgetgrenze in optionalen Reparaturen/spaeteren Kandidaten behaelt validierten Code. Autorenkern mit Live-Setting ee039 = 0.70357 (Abweichung kam allein vom Setting). 1593 Backend-Tests. Webhook erreicht Coolify nicht (Tailnet-URL), Redeploy manuell. Tests lesen lokale `.env` mit: `.env` nur temporaer.

**FITS-Live-Gegenprobe nach Quellenfix (2026-10-13):** `docs/fits_live_setup_verification_20261013.md`, Belege `docs/audits/fits_live_setup_evidence_20261013.json`. Nutzerjob ee039f0f vollstaendig: Quelle frisch v3/9022-HASH, Target0.372/tab:FITSett ok; dennoch 3 Seeds Mittel0.7035825614 +/-0.0006090188, vorlaeufig/nicht reproduziert/Methode ungeprueft, Modellkosten0.3067$ brutto. Neue allgemein relevante Ursachen lokal bestaetigt, noch nicht gefixt: Spec erzwingt90 aus Mehrfachheader statt belegter360; Discovery-Quellenname wird durch verify_entries abgelehnt, gleiche360-Quote als paper akzeptiert; batch32 aus inline-LaTeX-Kommentar angenommen; Repo liest AD/FITS.py/AD-Solver, nicht Forecast models/FITS.py/etth1_best.sh/run_longExp_F.py. Code cut_freq3 statt Autoren106. Kandidat83f508ba9484 echter synthetischer netzloser Smoke -> Tensor186 vs96; Live-Reparatur sieht nachfolgenden Broker-BrokenPipe statt Primaerfehler. Dritter Reparaturcode schneidet96 korrekt aus, falsche Parameter bleiben. Keine Produktfixes/weiteren bezahlten Jobs in dieser Gegenpruefung. Naechster Fix muss Setting/Startpfad/Detail->Spec->CONFIG und primaere Ausnahme abdecken, kein Testscore-Tuning.

**Quellenverfuegbarkeit nach zweitem FITS-Live-Stopp (2026-10-13):** `docs/paper_source_availability_fix_20261013.md`. Offizielle zweite Downloadroute, separate persistente LaTeX-Quellen unter logs/paper_sources, exakte Paper-/Versions-/SHA-Pruefung inkl. Discovery-Hash. FITS v3/Hash/Haupttabelle gepinnt und primaere komprimierte Quellenreserve im Image; frischer Download hat Vorrang, Cache-Einsatz offen mit Abrufdatum und letztem Fehler. Kein LaTeX->Tavily-Ausweichen bei strikten Claims. 1524 Backend-Tests, Opus238/kein Blocker, Produktionsimage und echte netzlose Orchestrator-Zielpruefung: FITS0.372/tab:FITSett aus Reserve, 0 Modell-/Tavily-Aufrufe. Vergangener Serverfehler ohne damaligen HTTP-Status nicht genauer belegbar. Nur FITS mitgelieferte Reserve; autonomer neuer Vollrun und Nebius-Isolation weiter offen. Keine neuen Providerkosten.

**Katalog-/Produktionsfix nach FITS-Live-Stopp (2026-10-13):** `docs/catalog_source_binding_fixes_20261013.md`. Alle 17 Katalogeintraege + 5 oeffentliche Live-Kandidaten, 19 Quellen lesbar, 0 neue Provider-Aufrufe. Forecast-Targetgate bindet vorhandenes Zitat an Originaltabelle (Methode/Datensatz/Horizont/Metrik/Zelle), transponierte und verschachtelte Kopfzellen, mean/std und explizite Metrikzeilen. Missing LaTeX-Anker nur eindeutiger deterministischer Legacy-Ersatz, sonst Stopp ohne LLM; benoetigter LaTeX-Text kein Tavily-Formatwechsel. Max2 kostenlose Downloadversuche30s; ermittelte Version in echter URL. Neue Discovery behaelt Quellhash/Art/URL/Label. FITS-Laenge360 mit Beleg korrigiert; protokoll-/setupbelegte Laengen erhalten, keine unbelegte Discovery-Vorgabe. Acacia frozen und xCPD ohne TSMixer-Backbone/mit falscher Laenge vor Start gesperrt. Redeploy aktualisiert persistente Katalogdefinitionen ohne Outcome-/Indexverlust. 1508 Backend-Tests + fokussierte Schlussgates; Opus311 passed/keine Blocker; Produktionsimage+netzloser Health/Export/Eligibility-Smoke bestanden. Historische Werte/Urteile erhalten. Weiter offen: autonomer FITS-Vollrun und Nebius-Isolation. Markdown-/PDF-Altzitate teils weiterhin generische Extraktion; unbekannte Version nicht erfunden.

**Unabhaengige Paper-Gegenprobe FITS (2026-10-13):** `docs/fits_independent_verification_20261013.md`. Nutzer erlaubt Push/main und anderes Paper gegen SARAF-Spezialisierung. Fixes bis `4e1093d` auf origin/main gepusht. FITS-Kern gepinnt auf d040bb015b6299da26d879b90dd19c80fb72c160; offizielles etth1_best-Preset VOR Testwerten: L360, H96, cut106, H_order6, lr0.0005, effektiver Batch128 (CLI64 verdoppelt), train_mode1, max50, patience20, lradj type3/0.95. Seeds0/1/2, keine Nachjustierung/Paperwechsel. Alle erfolgreich, Test-MSE **0.3720459157 +/- 0.0000462254**, Ziel **0.372**, einzelne0.3720515430/0.3719971343/0.3720890698; 36.03/36.64/37.39s. Normaler neuer kausaler Vertrag, getrennte Bewertung, nur Val fuer Checkpoint; 6 Driver-Tests. Keine Produkt-/historischen Daten geaendert. Lokale menschliche Autorenkern-Referenz, kein autonomer KI-Nachbau/Nebius-Nachweis; gleiches Dataset ETTh1, Checkpoint-Val sichtbarer dichter Praefix statt voller Autoren-Val. Belege unter docs/audits/fits_independent_20261013.*.

**Lokaler SARAF-Volltest nach Freigabe (2026-10-13):** Folgeabschnitt in `docs/saraf_local_contract_verification_20261013.md`; `scripts/review/saraf_contract_full.py`, Ergebnisse `docs/audits/saraf_contract_full_20261013.json`. Menschlicher Adapter um unveraenderten Autorenkern, neue Kalender-/causal_stream_v2-Eingaben, zehn Epochen, originale Hyperparameter, nur sichtbare Val fuer Checkpoint, kein Testfeedback. Seeds 0/1/2 vollstaendig, MSE 0.3711257752 / 0.3685850588 / 0.3690020300; Mittel **0.3695709547 +/- 0.0013625588**, Ziel **0.371**, alle innerhalb +/-0.01. 92.55/110.45/110.92s; frischer Bewertungsbranch, echter Harness und verifizierte Offline-CSV. Fuenf Driver-/Batch-Tests. Keine Modell-/Tavily-/Nebius-/Produktionsauftraege, historische Daten unveraendert. **Beweisgrenze:** zeigt Zielerreichbarkeit des lokalen Verfahrens mit Autorenkern; keine autonome KI-Reproduktion/Nebius-Abnahme. Checkpoint-Val taeglich/93 statt Autoren dicht/voll; Seeds 0/1/2 statt Autoren 2021/0/42; Online-Retrieval veraendert Zufallsreihenfolge. Erste ungemessene Runde scheiterte am lokalen Adapter-Batchindex, dokumentiert und formkorrigiert ohne Hyperparametertuning.

**Lokale SARAF-Vertragsprobe (2026-10-13):** `docs/saraf_local_contract_verification_20261013.md`. Nutzer erlaubt lokale Arbeit. E24 belegt: bisherige ETT-Eingaben verlieren echte Kalendermerkmale, die Autoren-Retrieval mit Gewicht 0.5 nutzt (Zwischen-Aehnlichkeit ohne Kalender bis 0.6314 anders; kein Forecastscore). Kalenderfelder jetzt passend zu Train/sichtbarer Val und kausalen Historien; optionales `predictor(history, *, calendar)`, alte einargumentige Forecaster kompatibel, Legacy-Zahlen unveraendert. Neuer ETT-Vertrag `causal_stream_v2`; Auswahl bleibt withheld_validation_v1. Originaler SARAF-Kern im netzlosen Docker-Smoke bestanden (4.57s, ein Adam-Batch, je acht Val/Testfenster, keine MSE/kein Paperurteil). Stochastisches MMR muss pro bereits gesehenem Fenster gespeichert werden wie im Original; Wiederholungspruefung bleibt aktiv. 1462 Backend-Tests, 25 fokussierte Docker-/Vertragspruefungen, Docker-Build bestanden. Keine neuen Provider-/Tavily-/Nebius-/Produktionsauftraege. **Weiter offen:** Nebius-Preflight, autonomer Volltraining-Lauf und Ziel 0.371.

**Allgemeiner Audit und Ursachenfixes (2026-10-13):** `docs/general_reliability_audit_20261013.md` und `docs/general_reliability_fixes_20261013.md`. Opus bestaetigt E01-E23 unabhaengig (`docs/opus_general_audit_review_20261013.md`), seine Quellen-/Parameter-/Guardfixes integriert. Frischer unabhaengiger Bewertungsbranch, verborgene Auswahlmenge (20% Val), kausaler Forecast-Broker mit eigener UID und geschuetzten CSVs, finale Codehash-Zulassung/Spec-Score, Rohdatenidentitaet/Snapshot-Version, begrenzte Transfers, sequentielle Kandidaten bei Budgetknappheit, persistente Fehlerstreams und claim-konsistentes Replay. Neuer Produktvertrag `withheld_validation_v1`/`causal_stream_v1`; historische Benchmarks/Katalogdaten unveraendert. 1456 Backend-/153 Frontend-Tests, 19 netzlose Docker-Gegenproben, Produktions-/Docker-Build und Healthcheck bestanden; GCN-Referenz liefert alt/neu identische CSV (synthetische Score-Oracle, kein neuer Benchmarkwert). Opus-Abschlussreview `docs/opus_forecast_contract_final_review_20261013.md`: lokal kein Blocker. **Offen:** Nebius setpriv/chown/Capabilities vor ETT-Produktlauf live pruefen; neue ETT-Einzelinferenz kann 600s ueberschreiten; SARAF-Zielwert weiter nicht bewiesen. Keine neuen Provider-/Tavily-/Nebius-/Produktionslaeufe. Audit-Proben sind eingefrorene Gegenbelege des alten Codes, nicht heutige Abnahmetests. Lokale Umsetzung, kein Deploymentnachweis.

**Echte SARAF-Gegenprobe nach Deployment (2026-10-13):** `docs/saraf_live_verification_20261013.md`. Nutzer autorisierte Live-Test, Job `175724ef-f713-488f-b0cd-304eb880ea46`. Ziel 0.371 und voller Autoren-Kontext (134162 Zeichen/14 Dateien) bestaetigt, Auswahlpolitik validation_only. Trotzdem 0/3 Seeds gemessen: zwei Retrieval-Platzhalter erkannt, anderer Kandidat Dataset-Tupel 5 vs Inferenz 4. Reparatur danach durch Job-Geldbudget gestoppt; kein Paperurteil. Weitere offene Ursachen: interne Validierung gegen Nulltargets, Inferenz-Batch 32 als Trainingsparameter, Default-Stripping fuer unpassende Skriptaufrufe und Konfliktpruefung verwechselt Retrieval-Batch 1024 mit Training. Belege lokal `logs/verification/saraf-20261013-080931/`; keine zweite bezahlte Gegenprobe. Dieser Test bestaetigt Quellenfix/Schutzstopp, keinen Reproduktionserfolg.

**Codebase-Ursachenfixes (2026-10-13):** `docs/codebase_reliability_audit_20261013.md`. Folgearbeit zur lokalen SARAF-Diagnose: ganze Kernquellen und Abhaengigkeiten statt Dateipraefixe; frisches Live-Repo; generische Trainingsdateien korrekt zuordnen; erkennbare flache Checkpoints vor Sandbox stoppen. Standard-Auswahl/Reparatur ueber Validierung und Struktur, keine Testscore-Reparatur; Code ueber Seeds eingefroren, Teilmessungen ohne Urteil. Exakte Endversuche/Zweitmetriken/Export trotz wiederverwendeter Hashs, Task-/Seed-Smoke-Cache, endliche Metriken/gueltige IDs, CONFIG-Ueberschreibungen und Budgetkontext in Threads abgesichert. Bekannte numerische CONFIG-Konflikte stoppen auch nach erschoepften Reparaturen vor Sandbox-Kommandos. 1369 Backend-/151 Frontend-Tests bestanden; 14 Browser-Faelle, Produktions-/Docker-Build und Container-Healthcheck erfolgreich. Historische Daten unveraendert, keine neuen bezahlten Laeufe; Methodentreue und SARAF-Zielwert nicht allgemein bewiesen. Dieser Bericht dokumentiert die Umsetzung der vorher offenen Verfahrensfixes.

**SARAF lokal untersucht / Live-Messungen (2026-10-13):** `docs/saraf_local_audit_20261013.md`. Live-Finalwert 0.491668 +/- 0.003141, Ziel 0.371. Erste Version lokal fast exakt 0.826957; unabhaengiger bester Checkpoint allein 0.562586. Autoren-Prompt schnitt SARAF vor encoder ab (6000 vs Beginn 6116), Retrieval nur 3297 Zeichen ohne Kernmethoden; Trainingsdatei falsch als anderes Modell ausgeschlossen. Mechanismus-, Achsen-, Checkpoint- und Retrieval-Selbsttreffer belegt, Verfahrensfixes noch offen. Anzeige unterscheidet Kandidaten/Code und publiziert exakte Endmessungsauswahl; keine Zahlen/Urteile umgeschrieben. Lokale CPU-Diagnose, keine neuen bezahlten Auftraege.

**Discovery -> Ziel-Gegenpruefung (2026-10-13):** `docs/discovery_target_handoff_20261013.md`. SARAF 0.371 stoppte, weil zweite KI einen vereinfachten, nicht woertlichen LaTeX-Ausschnitt erfand. Gemeinsamer Tabellenparser prueft jetzt die frisch geladene Quelle vor numerischem LLM-Aufruf, inkl. MSE/MAE, Methode/Datensatz/Horizont/Spalte und Quellenhash. Strikter Zielvergleich bleibt: alte 0.366/veraenderte Quellen weiter gestoppt. Echte Discovery -> Orchestrator-Zielpruefung fuer SARAF/GLinear getestet; 335 Backend-/140 Frontend-Tests, Browser, kompletter Docker-Build. Kein bezahlter Lauf. Bestehender 0.371-Claim nach Deployment erneut pruefbar, Reproduktionserfolg bleibt offen.

**Discovery live bestaetigt (2026-10-13):** Server-Suche fand frisch SARAF 0.371 (`disc_2606_04135_f7b4e6db785fdb63`, cached false, runnable true), alter 0.366-Claim bleibt separat. Protokollwarnung durch doppelte LaTeX-Escapes: Reparatur nur bei komplettem Quellenmatch, betroffene Cache-Eintraege gezielt erneut lesen; keine Historie umschreiben. 163 fokussierte Tests. Details im Folgeabschnitt von `docs/discovery_reliability_20261013.md`. Aktueller Start nur durch Reproduktions-Tageslimit blockiert.

**Discovery-Zuverlaessigkeit (2026-10-13):** `docs/discovery_reliability_20261013.md`. Normale automatische Suche erhalten. Tavily-Treffer am Volltext statt nur Abstract belegen; explizit benoetigte vortrainierte Modelle frueh aussortieren. Eindeutige Haupttabellen liefern eigenen MSE/Horizont/Look-back direkt aus Quelle, unabhaengig von leerer/falscher KI-Zahl. Mehrzeilige LaTeX-Zellen und begrenzte Multirow-Datensatzzuordnung. Echter SARAF-0.371- und GLinear-0.382-Fund bis Speicherung/Eligibility offline belegt; Times2D-Vergleich nicht als neues Verfahren ausgeben. Cache v6; alte Ergebnisse und Kostenlimits unveraendert. 324 Backend-/140 Frontend-Tests, Browser Desktop/Mobil. Keine bezahlten Tests oder Garantie fuer beliebige Websuche/Reproduktion.

**Discovery-Horizont (2026-10-12):** `docs/discovery_horizon_20261012.md`. Drei generische Live-Rejects waren kein Beleg gegen Papers: fehlendes LLM-Feld/kurzes Zitat und echter falscher Horizont wurden gleich dargestellt und gecacht. Verifizierte LaTeX-Quellzeile liefert Horizont und vollstaendiges Zitat; unbestaetigte Extraktion nicht als Paper-Befund gecacht. Cache v5 liest v4 neu. Echter SARAF-Tabellenausschnitt: 0.371 akzeptiert, RAFT 0.366/falscher Horizont/Durchschnitt weiter gesperrt. Keine bezahlten Tests oder Erfolgsgarantie.

**SARAF-Schutzpruefungen (2026-10-12):** `docs/saraf_guardrails_20261012.md`. Live-Lauf waehlte RAFT-Spalte 0.366 statt SARAF 0.371 und generierte Retrieval-Platzhalter. LaTeX-Spaltenparser, verbindliche Ziel-Gegenpruefung vor Experiment, begrenzte Reparatur bei belegten Retrieval-Platzhaltern, Modell-/Layer-Prioritaet im Repo-Abgleich und ausdruecklich unvollstaendige Pruefung statt falscher Sicherheit. Discovery-Cache v4 liest alte Extraktionen neu; alte Claims/Laeufe unveraendert. UI Stopped ohne Paperurteil und mit gespeichertem Grund. Keine allgemeine Methodentreue-Zertifizierung, keine neuen bezahlten Laeufe; SARAF-Erfolg bleibt ungetestet.

**Plan-Darstellung (2026-10-12):** Live-/Replay-Plan rendert Ueberschriften, Listen, fette Labels und Inline-Code als React-Elemente statt roher Markdown-Zeichen. Sechs-Zeilen-Vorschau und Vollansicht erhalten; Desktop/Mobil, Umbruch und Expand/Collapse im Produktionsbuild geprueft. Keine Aenderung am Experiment.

**Discovery → Experiment (2026-10-12):** `docs/discovery_to_experiment_20261012.md`. Server speichert exakte Live-Claims in `logs/live_candidates.json`; Discover bietet Check this claim und stellt gespeicherte Funde nach Neuladen wieder her. Alter validierter Discovery-Cache wird einmalig ohne API-Aufrufe uebernommen. Eligibility-Pruefung vor Startknopf in Discover/Katalog, Cora ohne belegten Task bleibt gesperrt. Neueste vollstaendige Suche als `logs/discovery_replay.json`, Replay bevorzugt Server-Aufnahme; September-Fallback klar Demo. Bestehende Limits und unverifizierte Methode/Spalten bleiben. 147 Backend-, 137 Frontend-Tests; Browser Desktop/Mobil; keine bezahlten Tests. Nutzer deployt nach Push selbst.

**Discovery-Praesentation (2026-10-12):** `docs/discovery_presentation_20261012.md`. Suche zeigt ausgewaehlte Claims; alte Reproduktionsurteile in datierten Details, nur bei passendem Claim. Replay-Marken recorded, Live-Marken live. Gefuehrter naechster Schritt zu einem ausdruecklich separaten positiven Katalogbeispiel. Keine Messwerte oder Replay-Daten veraendert; 135 Frontend-Tests.

**Discovery-Startfix (2026-10-12):** `docs/discovery_failure_20261012.md`. Nutzer-Container meldete fehlenden `OPENROUTER_API_KEY`; Produktionsmodus in Coolify auf `LLM_PROVIDER=tokenfactory` setzen. SSE-Fehler bisher unsichtbar und als Finished/NaNs dargestellt; Fix mit explizitem Ende, Zeitstempeln und redigiertem Fehlerlog. 51 API- und 132 Frontend-Tests bestanden; zwei bestehende Discovery-Suchlimit-Tests erwarten 3 statt unveraenderter 5. Keine bezahlte Suche fuer die Reparatur.

**Jury-Fixes (2026-10-12):** Befunde `docs/jury_review_20261012.md`, Umsetzung und offene Aufgaben `docs/jury_fixes_20261012.md`. Ergebnisdiagramme, Seed-Tabelle, Replay und PDF verwenden die vom Orchestrator mit `chosen` markierten Endmessungen; andere Kandidaten bleiben in der Versuchshistorie. Der Nutzerbegriff REPRODUCED bleibt, mit sichtbarer numerischer Einordnung und ungepruefter Methode. Demo https://nullius.akca.dev/ erreichbar; lokale Aenderungen nicht automatisch deployed. Menschlicher Produkttest und neues Einreichungsvideo bleiben offen.

**UI/UX-Fixes (2026-10-12):** Erstnutzer-Durchgang und anschliessende Umsetzung `docs/ux_walkthrough_20261012.md`. Originalquelle auch bei fehlgeschlagener automatischer Extraktion, Setup-Referenzen mit Herkunft, direkte Code-/Messungsaktionen, eindeutige Film-/Log-Links, Katalogfilter/Scroll-Rueckweg und kompakte mobile Einleitungen. 129 Frontend-Tests; neues Deployment fuer diese Folgeaenderungen erforderlich. Keine neuen bezahlten Laeufe, menschlicher Test weiter offen.

**Jury-Verstaendlichkeit (Nutzer, 2026-10-10):** Hackathon-Jury ohne vorausgesetztes ML-/Statistik-Fachwissen behandeln; keine wissenschaftliche Fachjury voraussetzen. Hauptweg auf Website und im Video: Was behauptet das Paper? Was hat Nullius selbst gemessen? Wo ist der Beleg? Kurze erklaerte Begriffe, klare Navigation und konkrete Nutzungsentscheidung zuerst; wissenschaftliche Einordnung und Details erreichbar/aufklappbar halten. Framing soll belegte Erfolge zeigen und Grenzen passend zuordnen, ohne unbelegte Erfolgsquoten oder Garantien.

Nebius x NVIDIA Global AI Hackathon (Devpost). Track: Coding and Agentic Engineering, zusätzlich Best Use of Tavily.

> „Andere Agenten fixen Code, bis Tests grün werden. Bei uns ist Paper die Testsuite."

## Umgebungen: Prototyp vs. Prod

**Aktuell PROTOTYP-Phase.** Modellaufrufe laufen über kostenlose Nemotron-Endpunkte auf OpenRouter. Token Factory (25 $ Guthaben) nur für echte Benchmark-Läufe, Demo, Einreichung.

Einreichung muss auf Nebius Token Factory laufen und mindestens ein NVIDIA-Modell nutzen.

| | Prototyp (jetzt) | Prod (Benchmark, Demo, Einreichung) |
|---|---|---|
| LLM-Anbieter | OpenRouter, kostenlose NVIDIA-Endpunkte | Nebius Token Factory |
| Basis-URL | `https://openrouter.ai/api/v1` | `https://api.tokenfactory.nebius.com/v1/` |
| Kosten | 0 $ | aus 25 $ Guthaben |
| Ultra verfügbar | Nein, Rolle `reasoning` nutzt Super | Nein, seit 2026-10-09 von Token Factory entfernt (404); `reasoning` nutzt Super |
| Zuverlässigkeit | niedrig (ca. 91 % Uptime, ~36 Token/s, Rate Limits) | hoch |
| Sandbox | `nebius` (Beta genehmigt, kostenlos) | `nebius` (Pflicht) |
| Tavily | vorhandener API-Key | vorhandener API-Key |

### Regeln für Prototyp (OpenRouter Free)

- Keine vertraulichen Daten in Prompts. NVIDIA protokolliert Anfragen auf kostenlosen Endpunkten. Öffentliche Papers/Code unproblematisch, API-Keys, Pfade mit Nutzernamen, private Daten niemals in Prompts.
- Retries Pflicht: exponentielles Backoff, max 5 Versuche pro Aufruf. Rund jede zehnte Anfrage kann fehlschlagen.
- Keine Benchmark-Zahlen aus Prototyp veröffentlichen. Alle Zahlen in README und Video stammen aus Prod-Läufen.
- Wechsel zu Prod ausschließlich über `.env`. Im Code keine Anbieter-Sonderfälle außer im LLM-Client.

### Konfiguration (`.env`, nicht in Git)

```bash
# ---- Umgebung ----
LLM_PROVIDER=openrouter          # openrouter | tokenfactory
SANDBOX_BACKEND=nebius           # local | nebius (Beta genehmigt, kostenlos)

# ---- OpenRouter (Prototyp) ----
OPENROUTER_API_KEY=
OR_MODEL_FAST=nvidia/nemotron-3.5-lightning:free
OR_MODEL_CODE=nvidia/nemotron-3-super-120b-a12b:free
OR_MODEL_REASONING=nvidia/nemotron-3-super-120b-a12b:free   # Ultra wäre auch :free verfügbar (PRD-Annahme war falsch), bewusst Super gewählt

# ---- Token Factory (Prod) ----
NEBIUS_API_KEY=
NEBIUS_PROJECT_ID=
TF_MODEL_FAST=nvidia/Nemotron-3_5-Lightning        # geprueft via GET /v1/models, 2026-09-28
TF_MODEL_CODE=nvidia/nemotron-3-super-120b-a12b
TF_MODEL_REASONING=nvidia/nemotron-3-super-120b-a12b   # Ultra bis 2026-10-09, dann von Token Factory entfernt (404)

# ---- Tavily ----
TAVILY_API_KEY=

# ---- Kostenbremsen ----
MAX_COST_PER_RUN_USD=0.50        # brutto (inkl. MwSt.)
VAT_RATE=0.19                    # Log speichert netto, Limits und Budget gelten brutto
BUDGET_TOTAL_USD=29.75           # brutto, Kontostand zum Stichtag; ab 80 % Warnung, ab 95 % Abbruch
BUDGET_SINCE=2026-09-29T08:00Z   # Stichtag (UTC), nur Log-Eintraege ab dann zaehlen
MAX_ATTEMPTS_PER_CLAIM=4
LLM_CACHE=on                     # on | off
```

## Problem

Viele Ergebnisse in ML-Forschung nicht reproduzierbar. Wer Paper nachbauen will, verliert Tage an kaputtem Code, fehlenden Hyperparametern, Zahlen die nicht herauskommen. Ob Behauptung stimmt, weiß man meist erst, wenn jemand diese Mühe auf sich genommen hat.

Zielgruppe: Reviewer, Doktoranden, ML-Teams, die auf Paper aufbauen wollen.

## Lösung

Nutzer gibt ML-Paper (arXiv-Link) ein. Agent:

1. extrahiert 3–5 überprüfbare Behauptungen mit Zielwerten,
2. sammelt mit Tavily Code, Daten, Setup-Details und Einwände der Community,
3. rechnet Experimente in Sandbox-Branches nach,
4. prüft jeden Einwand mit eigenem Experiment,
5. liefert Reproduktions-Zeugnis mit Urteilen *reproduziert*, *teilweise* oder *nicht reproduziert*, jeweils mit Messwert, Streuung, Quellen, Branch-Baum als Beleg.

## Ziele und Nicht-Ziele

**Ziele**
- End-to-end-Lauf für mindestens 3 Demo-Papers
- Benchmark mit 8–10 Papers mit bekanntem Ergebnis, mit und ohne Tavily
- Nachvollziehbarkeit: jede Zahl und Aussage belegt

**Nicht-Ziele**
- Kein GPU-Training. Nur Claims, die in unter 10 Minuten auf CPU laufen
- Keine LLM-Papers, keine großen Modelle
- Keine PDF-Bildanalyse. Nutze HTML-Version auf arXiv (Nano Omni auf Token Factory nur als Dedicated-Modell verfügbar)
- Kein Nutzer-Login, keine Mehrbenutzerfähigkeit

## Architektur

```
App (Python, Orchestrator)
 ├── LLM-Client ────▶ OpenRouter (Prototyp) | Token Factory (Prod)
 ├── Sandbox-Adapter ─▶ Local/Docker (Fallback) | Nebius Sandboxes
 ├── Tavily-Client ──▶ Search, Extract, Crawl, Research
 ├── Harness ────────▶ läuft im Sandbox-Branch, selbst geschrieben
 └── Report + UI
```

Grundprinzip: Nemotron ist Gehirn, Sandbox ist Labor, Tavily ist Rechercheur, Orchestrator ist Laborleiter. Nemotron führt nie selbst Code aus, Sandbox denkt nie.

### Modell-Routing

| Rolle | Aufgabe | Prototyp | Prod |
|---|---|---|---|
| `fast` | Log-Parsing, Triage, Ergebnis auswerten | Lightning (free) | Nano oder Lightning |
| `code` | Experiment-Code schreiben und reparieren | Super (free) | Super |
| `reasoning` | Forschungsplan, Hypothesen, Endurteil | Super (free) | Super, nur mit verdichteter Forschungsakte (Ultra bis 2026-10-09; Benchmark v2–v4 liefen mit Ultra für `reasoning`, recall cg v1 und G0/G1 mit Super überall) |

### Sandbox-Adapter

Eine Schnittstelle, zwei Implementierungen:

```python
class Sandbox(Protocol):
    def base_snapshot(self, image: str, setup_cmds: list[str]) -> SnapshotId: ...
    def branch(self, snapshot: SnapshotId) -> Branch: ...
    def run(self, branch: Branch, cmd: str, timeout_s: int) -> RunResult: ...
    def put_file(self, branch: Branch, path: str, content: bytes) -> None: ...
    def get_file(self, branch: Branch, path: str) -> bytes: ...
```

- `NebiusSandbox`: `contree-sdk`, echtes Branching. Beta genehmigt, kostenlos, jetzt Standard-Backend, auch im Prototyp. Doku: `docs.tokenfactory.nebius.com/sandboxes/sdk/python_sdk/branching`
- `LocalSandbox`: Docker-Container, Checkpoints per `docker commit`. Nur noch Fallback, falls Nebius Rate-Limits oder Ausfälle zeigt. KI-Code nie direkt auf Host ausführen.

Rate-Limits bei Nebius-Sandbox noch prüfen (offener Punkt).

## Der Loop (pro Claim)

```
[Tavily: Einwände sammeln] ──▶ Claim/Einwand
                                   │
                                   ▼
         ┌──────────────▶ Hypothese formulieren (reasoning)
         │                         │
         │  [Tavily: Setup-Details]▶ Code schreiben (code) ◀──────────┐
         │                         │                                 │
         │                         ▼                                 │
         │                 Branch vom Snapshot (Sandbox)             │
         │                         │                                 │
         │                         ▼                                 │
         │                 Unabhängig prüfen (Harness)               │
         │                         │                                 │
         │                         ▼                                 │
         │                 Ergebnis auswerten (fast)                 │
         │           ┌─────────────┼──────────────┐                  │
         │     Abweichung        stabil         Fehler               │
         │           ▼             │              ▼                  │
         └─[Tavily: Abweichung     │     [Tavily: Fehler lösen] ─────┘
            klären]                ▼       + Rollback zum Snapshot
              [Tavily: Quellen] ▶ Urteil mit Beleg
```

- **Fehler** (Crash, fehlende Library, toter Link): Rollback, Tavily sucht Fehlertext, neuer Code. Max `MAX_ATTEMPTS_PER_CLAIM` Versuche, danach Urteil „mit unserem Setup nicht reproduzierbar" mit dokumentierten Versuchen.
- **Abweichung** (sauber gemessen, andere Zahl): Befund, kein Fehler. Tavily sucht Erklärungen, jede Erklärung wird neue Hypothese mit eigenem Branch.
- **Stabil**: Messung liegt über alle Seeds in Toleranz → Urteil.
- Claims und Einwände laufen parallel, jeder in eigenem Branch.

## Tavily-Rollen

Jeder Aufruf muss Frage beantworten, die Agent sonst nicht beantworten könnte.

| # | Rolle | Endpunkt | Wann | Ergebnis |
|---|---|---|---|---|
| 1 | Material finden | Search (Domains: github.com, arxiv.org, openreview.net) | Start | Repo, Datensatz, Projektseite |
| 2 | Claims lesen | Extract (`extract_depth="advanced"`) auf arXiv-HTML | Start | Tabellen mit Zielwerten |
| 3 | Einwände sammeln | Search + Crawl (`chunks_per_source`, immer `limit` setzen) | vor Experimenten | Issues, Gutachten, Kritik-Papers → neue Einwände |
| 4 | Setup-Details holen | Extract auf README, Configs, Paper-Anhang | vor „Code schreiben" | Hyperparameter, Datenaufteilung |
| 5 | Fehler lösen | Search mit Fehlertext | bei Fehler | Lösungshinweis für `code` |
| 6 | Abweichung klären | Search: neuere arXiv-Version, Errata, Issues | bei Abweichung | neue Hypothese |
| 7 | Quellen belegen | Extract; Research max 1× pro Paper | beim Urteil | zitierfähige Quellen im Bericht |

Kernidee für Tavily-Preis (ursprünglich): Tavily-Funde werden zu Experimenten. Beispiel: Issue meldet Seed-Abhängigkeit → 5 Branches mit verschiedenen Seeds.

**Stand Tavily (2026-10-04):** Gebaut und im Produktlauf live: Rolle 1 (Repo-Suche, Discovery neuer Papers auf unterstützten Datensätzen), 2 (Zielwert mit Zitat- und Spaltenprüfung), 4 (Setup-Details mit Zitat), 5 (Fehlertext-Suche, seit v3), 3/7 und 6 als **reine Berichtsfunktion** „What others found“ / „Possible explanations“ (`community.py`, nach dem Urteil, ändert nie Plan, Code oder Urteil). **Nicht gebaut:** Funde als eigene Branches (Rolle 3 Stufe 2, Rolle 6 als Hypothese). **Entscheidung: kein weiterer Ausbau vor der Abgabe.** Alle vorab festgelegten Versuche, mit Tavily die Trefferquote messbar zu heben, sind gescheitert oder blieben unter der Schwelle: Rolle 3 Stufe 1 (`results/role3_dlinear_stage1.md`, kein Protokoll-Einwand gefunden), Rolle 4 DLinear und Shchur (`results/role4_*.md`, Shchur um 0,0004 verfehlt), v5 Schritt A und C (`results/v5_step_a.md`, `v5_step_c.md`), Benchmark v2–v4 ohne messbaren Unterschied der Arme (v4: 2 vs. 1 von 8; Median-Abstand post hoc 0,017 vs. 0,029). Tavily wird deshalb als Produkt-Fähigkeit dargestellt (Discovery, Belege mit Hash, „Worte vs. Messung“), nicht als Genauigkeitsgewinn.

> **Standard ist live** (`TAVILY_MODE=live`): ein normaler Lauf ruft Tavily zur Laufzeit auf (Paper-Text, README), gespeichert mit Modus, Zeitstempel und Hash (`Dossier.sources`, `events.jsonl`, `paper_text.json`). Fixtures nur mit `--offline` oder `TAVILY_MODE=fixture` (Tests, vorab festgelegte Experimente, Benchmark: `tavily_mode="fixture"` in den Skripten). Der Tavily-Preis verlangt einen echten Aufruf zur Laufzeit.

> **Tavily-Key sparsam benutzen.** Free/Trial-Kontingent begrenzt (Credits pro Monat), kein unlimitiertes Guthaben wie bei OpenRouter Free. Jeder Aufruf muss Frage beantworten, die sonst nicht beantwortbar ist (siehe Tabelle oben) — kein Testen/Debuggen mit echten Tavily-Calls, wenn Mock/Fixture reicht. Crawl immer mit `limit`, Research max 1× pro Paper (siehe Rolle 7). Bei Unsicherheit: erst Dossier/Cache prüfen, ob Frage schon beantwortet wurde, bevor neuer Call.

## Modus und Rolle 4 (Stand Meilenstein 2)

**Modi (Stand 2026-10-09).** Der Agent schreibt in beiden Nachbau-Modi eigenen Code; das Original-Repo liegt nie in einem Agenten-Branch, es wird nur in einem eigenen Lese-Snapshot des Orchestrators geklont (Commit im Dossier), Autorencode wird nie ausgeführt, die Zahl misst unser Harness.
- **„Nachbau aus Paper und Autorencode“ (`nachbau_autorencode`, `code_grounded=True`, `io_helper=True`, inkl. Fixes F1–F4/F8): Standard im Produkt** (Entscheidung des Nutzers 2026-10-09 auf Grundlage `benchmark/experiments/results/recall_cg_v1.md`: Entwicklung 7/8 In-Sample, RLinear Held-out 0/1, Placebos 0/9; v4 nur Paper: 2/8; die Zahlen nie verrechnen). Der rohe Repo-Ausschnitt geht in Plan-, Code- und Reparatur-Prompt, Zahlen erzwingt der Steckbrief (S2).
- **„Nachbau aus Paper“ (`nachbau`):** nur belegte Angaben aus Paper und Repo-Zitaten. Das ist die Konfiguration von Benchmark v2–v4 und der **Rückfall**, wenn kein Repo gefunden wird, der Klon scheitert oder der Ausschnitt leer bleibt (`Orchestrator._resolve_mode`, Ereignis `mode` mit `fallback_reason`, Warnung „Autorencode nicht verfügbar“). `Dossier.mode` trägt den tatsächlichen Modus (nie den angeforderten), Zeugnis (`report.mode_line`), `index.json`/`catalog.json` und die Web-App (`ModeNote`, „Rebuild from paper and authors' code“ / „Rebuild from paper“) zeigen ihn.
- **Wo der Standard gilt:** nur im Produktpfad über `loop.product_kwargs()` (`api.make_orchestrator`, CLI `repro_agent.loop`, `scripts/curate_catalog.py`). Der Orchestrator-Default bleibt `code_grounded=False`, damit Benchmark- und Experiment-Skripte (`run_benchmark.py`, `experiment_*`, `recall_cg_v1.py` mit eigener Konfiguration) ihre vorab festgelegte Konfiguration nicht stillschweigend ändern. Schalter zurück auf Paper-Nachbau: API-Feld `author_code=false`, CLI `--no-author-code` (CLI: dann die bisherige schlichte Konfiguration), `curate_catalog.py --no-author-code`.
- Der Modus „Nachbau“ ohne Zusatz wie bis Meilenstein 2 (nur Paper und belegte Angaben) bleibt der Referenzmodus der Benchmarks; im README stehen v4-Zahlen ausdrücklich als Paper-Modus. „Nachlauf“ (Original-Code ausführen) prüft etwas anderes; er ist als Experiment gebaut (`nachlauf*.py`, `scripts/run_nachlauf.py`), trägt aber nur an den zwei Entwicklungs-Einheiten (RLinear, DLinear) und scheiterte an vier neuen Einheiten zweimal mit 0 von 4 (`results/nachlauf_v1.md`). Nicht im Produkt, nicht in der API, nicht bewerben; kein dritter Versuch. Modus steht im Dossier und im Zeugnis.

**Rolle 4 (Setup-Details):** Quellen sind Paper-Text, README (Tavily Extract) und vom LLM gewählte Repo-Dateien (Sandbox-`git clone`). Extraktion abschnittsweise mit Rolle `extract` (= Modell von `code`, Super; Plan und Endurteil bis 2026-10-09 Ultra, seitdem ebenfalls Super). Jede Angabe braucht ein wörtliches Zitat, das im jeweiligen Dokument vorkommt, sonst wird sie verworfen. Jede Angabe trägt ihre Quelle.

**Rolle 2 Grenze:** Der Check belegt, dass Zitat und Zahl im Text stehen, nicht die Spalten-/Zeilenzuordnung. Fehlt der Methodenname im Zitat, steht im Zeugnis eine Warnung.

**Fehlertext für Reparatur:** Ende von stderr (1500 Zeichen), keine LLM-Zusammenfassung. `fast` schreibt seinen Denkprozess in die Antwort und liefert abgeschnitten keinen Fehler. Modelle, die im Antworttext „denken“, brauchen bei Extraktionen viel `max_tokens` und Zerlegung in kleine Abschnitte.

**Kosten (Token Factory, GCN mit Rolle 2 und 4, ein Paper): ca. 0,09 $** (vor Umstellung von Rolle 4 auf Super, bei Ultra). Der Kostenzähler liegt im LLM-Client und zählt alle Aufrufe.

## Unabhängiger Prüf-Harness

Grundsatz: Agent liefert Vorhersagen, keine Zahlen.

Vertrag: Agent schreibt `/output/predictions.csv` mit Spalten `id,prediction`. Nichts sonst zählt.

Ablauf im Branch, nach Agenten-Lauf:
1. Harness frisch aus unserem Repo in Branch kopieren (Agent kann ihn vorher nicht verändern)
2. Testlabels selbst aus Original-Datensatz laden, nie aus Dateien des Agenten
3. Plausibilität: IDs vollständig und eindeutig, nicht nur eine Klasse
4. Metrik mit festem, getestetem Code berechnen
5. Mit Toleranz vergleichen; 3–5 Seeds für Stabilität

Toleranz: Streuung aus Paper, sonst feste Regel (z. B. ±1 Prozentpunkt bei Genauigkeit). Wird vor Lauf festgelegt und im Bericht genannt.

Das Urteil beruht auf dem **Mittelwert** über alle Seeds, nicht auf einzelnen Seeds. Die Streuung (std) wird immer berichtet. Einzelne Seeds außerhalb der Toleranz sind normal. Seed-Anzahl steht pro Paper in `benchmark/papers.yaml` (`seeds`, Default 5).

Leak-Schutz (Schichten, im Dossier unter `isolation` festgehalten und im Zeugnis genannt):
1. Trainingslauf des Agenten-Codes ohne Netz (`unshare -n`). Das SDK bietet keine Netz-Option, `resolv.conf` im Snapshot bleibt nicht erhalten.
2. Statische Codeprüfung (`harness/code_check.py`) auf Netz, Download, Prozessstart, dynamischen Code. Treffer = Versuch `ungueltig`.
3. Testlabels nie im Agenten-Datensatz, Harness lädt sie frisch.

| Abstand | Urteil |
|---|---|
| ≤ Toleranz | reproduziert |
| ≤ 3 × Toleranz | teilweise |
| darüber | nicht reproduziert |
| Harness-Check fehlgeschlagen | ungültig |

**Zweidimensionales Urteil.** Achse 1 (Zahl): reproduziert / teilweise / nicht reproduziert / Agenten-Fehler vermutet / nicht reproduzierbar (Setup), vom Prüfer nie verändert. Achse 2 (Methodentreue, `Verdict.methode`): ja / teilweise / nein / nicht geprüft. Gesamtaussage (`dossier.gesamtaussage`): reproduziert + ja = „Paper reproduziert“; reproduziert + nein/teilweise = „Zielwert erreicht mit anderem Modell“; nicht reproduziert + ja = „nicht reproduziert“; Achse 2 nicht geprüft = „vorläufig“. Die gebaute Gesamtaussage steht im Zeugnis; der **Benchmark v2 zählt nicht die Gesamtaussage, sondern Achse 1** (Mehrheitsregel, siehe Abschnitt „Benchmark“). **Stand:** Eichung des Methodentreue-Prüfers v1 gescheitert (5/10, 2× ja↔nein, `benchmark/experiments/results/calibration_fidelity_v1.md`), Achse 2 daher überall „nicht geprüft“. Eichungen v2 (7/10) und v3 (8/11, Schwelle 9/11) ebenfalls gescheitert (`results/calibration_fidelity_v2.md`, `_v3.md`). **Endgültig, vorab festgelegt:** Achse 2 bleibt für den Benchmark „nicht geprüft“, im README offen benennen (Eichungen v1–v3 mit Ergebnissen); keine weitere Eichung vor der Abgabe. Für GCN gilt zusätzlich die Graph-Ablation (Graph-Modelle allgemein, siehe unten).

**Formprüfung und Ausführungsabdeckung (ohne LLM).** Vor dem Training läuft der Agenten-Code mit `SMOKE=1` (Cora 20 Knoten/1 Epoche, ETTh1 8 Fenster/1 Batch); Form gegen Task-Vorgabe (`harness/smoke.py`), Fehlermeldung geht an die Reparaturschleife. Im ersten vollständigen Seed-Lauf (Seed 0) wird die Zeilenabdeckung gemessen (`coverage` im Basis-Snapshot, `Attempt.coverage`). Kombinationsregel Prüfer × Abdeckung (`fidelity.apply_coverage`): Abdeckung darf nur herabstufen (Prüfer `ja` + zitierte Zeilen nie ausgeführt → `teilweise`/`nein`), `teilweise`/`nein` bleiben, ohne Abdeckung bleibt das Urteil mit Hinweis. **Grenze:** Zeilenabdeckung erkennt nur nie ausgeführten Code. Ausgeführter, aber verworfener Code (Zerlegung berechnet, Ergebnis ungenutzt; Adjazenz gebaut, nie multipliziert) gilt als abgedeckt; das braucht einen Identitäts-/Ablationstest (offen). Der Prüfer ist bisher nicht in den Loop eingebunden (Eichung gescheitert). **Graph-Ablation (GCN, kein LLM):** Der Code läuft ein zweites Mal auf Cora ohne Kanten (`ablation_gcn.md`); Abfall ≥ 0,10 = „Graph genutzt“, ≤ 0,02 = „Graph nicht genutzt“ (setzt Achse 2 auf `nein`), sonst unklar, Absturz/ungültig = nicht messbar (Achse 2 unverändert). Validierung 9/10 (`results/ablation_gcn_validation.md`); erkennt die toten Fälle, die Prüfer und Abdeckung nicht sehen.

Benötigte Metriken: Accuracy, MSE/MAE, Recall@K/NDCG. Bei Recsys exaktes Auswertungsprotokoll des Papers umsetzen (Anzahl Ranking-Kandidaten usw.).

## Datenmodell (Forschungsakte)

Forschungsakte ist Gedächtnis des Agenten. `reasoning` bekommt nur verdichtete Fassung, nie alle Logs.

```python
@dataclass
class Claim:
    id: str
    text: str                  # "GCN erreicht 81,5 % auf Cora"
    metric: str                # "accuracy"
    target: float
    tolerance: float
    source: str                # "Tabelle 2, arXiv v1"
    origin: Literal["paper", "einwand"]
    evidence_urls: list[str]   # Tavily-Quellen

@dataclass
class Attempt:
    claim_id: str
    branch_id: str
    parent_snapshot: str
    code_hash: str
    seed: int
    status: Literal["fehler", "abweichung", "stabil", "ungueltig", "timeout"]
    measured: float | None
    error_summary: str | None  # verdichtet von `fast`
    tavily_queries: list[str]
    cost_usd: float

@dataclass
class Verdict:
    claim_id: str
    urteil: Literal["reproduziert", "teilweise", "nicht reproduziert", "nicht reproduzierbar (Setup)"]
    measured_mean: float
    measured_std: float
    explanation: str
    sources: list[str]
```

Gespeichert als JSON pro Paper-Lauf unter `runs/<paper_id>/<timestamp>/`.

## Budget und Kostenkontrolle

Verfügbar: Token Factory laut Konsole 30,74 $ (Stand 2026-10-02, Builders-Guthaben bereits enthalten, kein weiteres zugesagt), Tavily Monatsplan 1 000 Credits plus 3 125 Add-on (Builders, verfällt 90 Tage nach Ausgabe). Planung: `docs/budget_plan.md`.

> Token Factory belastet hinterlegte Karte automatisch, wenn Guthaben ins Minus geht. Überschreitungen zahlen wir selbst.

- Prototyp läuft auf OpenRouter Free. Token Factory erst ab Meilenstein 3.
- LLM-Cache: identische Anfragen aus Datei lesen (`LLM_CACHE=on`)
- Kostenbremse: Abbruch bei `MAX_COST_PER_RUN_USD`, geprüft gegen **Brutto** (netto × (1 + `VAT_RATE`))
- **MwSt.:** Nebius rechnet netto + 19 % ab. `logs/costs.jsonl` speichert netto (`cost_usd`), Limits und Budget gelten brutto, `scripts/cost_report.py` zeigt beides.
- **Globales Budget:** `BUDGET_TOTAL_USD` (brutto). Vor jedem API-Aufruf werden die bisherigen Bruttokosten aus dem Log summiert: ab 80 % Warnung, ab 95 % `BudgetExceeded` (kein Aufruf). `BUDGET_TOTAL_USD` ist der Kontostand zu einem Stichtag (`BUDGET_SINCE`), der Code zählt nur Log-Einträge ab dann, damit Konsole und Code übereinstimmen. Beim Aufladen beide Werte neu eintragen (aktuellen Kontostand, aktuellen Zeitpunkt). Ohne `BUDGET_SINCE` zählt das ganze Log (sichere Richtung, zu vorsichtig).
- **Das Log ist eine Schätzung, maßgeblich ist die Konsole.** Bei Abweichung > 10 % (`scripts/cost_report.py`, Konsolenwerte dort eintragen) Log/Preistabelle korrigieren, nicht das Budget erhöhen. Stand 2026-09-29: Abweichung ≤ 1,3 % (Rundung der Konsole).
- **Abrechnungs-/Auth-Fehler** (HTTP 401/402/403, oder Text mit billing/suspended/insufficient/payment/api key, auch im 200-Body): `BillingBlocked`, kein Retry, kein Reparaturversuch, alle weiteren Aufrufe des Clients scheitern sofort, CLI bricht mit „Nebius-Abrechnung blockiert – Konsole prüfen“ ab. Grund: Konto war nach Trial-Ende „Billing: Suspended“. Geprüft werden nur Statuscode und strukturiertes `error`-Feld der API, nie der vom Modell erzeugte Inhalt (sonst blockiert eine Antwort mit „api_key“ den Lauf). Echte Fehlerantwort bei nächster Sperrung als Fixture speichern. Nur im LLM-Client.
- Log-Felder pro Aufruf: `provider`, `base_url`, `model`, `prompt_tokens`, `completion_tokens` (aus `usage` der API, nie geschätzt; Reasoning-Tokens sind enthalten, zusätzlich `reasoning_tokens`), `cache_hit`, `finish_reason`, `status` (`ok`/`bad_response`/`cached`), `cost_usd` netto. Leere, aber bezahlte Antworten werden mitgeloggt.
- Versuchs-Limit: `MAX_ATTEMPTS_PER_CLAIM=4`
- Kosten-Log: jeder Aufruf mit Modell, Tokens, geschätzten Kosten in `logs/costs.jsonl`
- Plan und Endurteil: bis 2026-10-09 Ultra, seit der Entfernung durch Token Factory Super (kein Code setzt Ultra voraus)
- Tavily Crawl immer mit `limit`, Research max 1× pro Paper
- Tavily-Key ist scharf (echtes Kontingent) — auch beim Entwickeln/Testen sparsam einsetzen, nicht in Schleifen oder Smoke-Tests verballern
- Usage-Seite in TF-Konsole wöchentlich prüfen

## Benchmark

**Stand: Benchmark v4 ist das maßgebliche Ergebnis** (Vorab-Festlegungen `benchmark/experiments/benchmark_v2.md`, `_v3.md`, `_v4.md`; Ergebnisse `results/benchmark_v2.md`, `_v3.md`, `_v4.md`; alle Läufe fertig, `src/` nicht mehr eingefroren). Kein v5 (Bedingungen in `results/v5_step_c.md` verfehlt). Die ursprüngliche Planung (8–10 Papers, Vergleich mit menschlichen Reproduktionsstudien) gilt nicht mehr. Aufbau unten wie in v2, v3 und v4 gleich.
- **6 Papers, 8 Paper-Claim-Einheiten** (Kipf, Wu/SGC, Zeng DLinear und NLinear, Shchur GCN und GAT, Veličković GAT, DGI), jede Einheit **mit und ohne Tavily** (beide Arme mit gespeicherten Tavily-Ergebnissen, `tavily_mode="fixture"`), je 3 frische Generierungen, dazu je Lauf ein **Placebo-Claim** (bewusst falscher Zielwert) als Negativkontrolle. 48 Läufe, Store `runs/benchmark_v2.jsonl`, Skripte `scripts/run_benchmark.py --v2` und `scripts/eval_benchmark_v2.py`.
- **Produktkonfiguration** (`graph_repair=True`) in beiden Armen.
- **Kennzahl = Zahl-Urteil (Achse 1), Mehrheit je Paper-Claim-Einheit über 3 Generierungen** (mindestens 2 gleiche Klassen), Übereinstimmung mit dem Referenzurteil; Rohzählung sekundär. Das ersetzt die frühere Regel „nur Paper reproduziert/nicht reproduziert“ (siehe „Zweidimensionales Urteil“). Die Kennzahl misst „Zielwert laut Harness erreicht“, nicht „Methode reproduziert“; im README steht immer „method fidelity not checked“.
- **Referenzurteile** sind überwiegend (R) = eigene Referenzlösung im eigenen Harness (`tests/reference/`), nur Shchur-GCN und Shchur-GAT haben zusätzlich (H) = unabhängige Neumessung im Kritik-Paper.
- **Dacrema nicht gebaut** (MovieLens-1M: NeuMF reproduziert, kein Negativfall); Shwartz-Ziv zugeschnitten und verworfen (`benchmark/experiments/shwartz_ziv_scoping.md`). Es gibt keinen echten Negativfall, die Placebos ersetzen ihn nicht.
- **v1 ist eingefroren** (`benchmark/experiments/benchmark_protocol.md`, vier Papers, Produktkonfiguration aus); Verlauf in `docs/research_log.md`.

**Ergebnis v4 (im README, Erzählung „hohe Präzision“):** 0 von 48 Placebos „reproduziert“ (auch in v2 und v3 je 0 von 48); 7 von 7 „reproduziert“-Urteilen stimmen mit der Referenz (post hoc); Primärkennzahl: mit Tavily 2 von 8 Einheiten (Kipf-GCN, Shchur-GCN) stimmen mit dem Referenzurteil, ohne Tavily 1 von 8 (DLinear). Wortlaut: „matched the reference verdict“, nicht „target reached“; immer „method fidelity not checked“. Ältere Versionen: `docs/benchmark_history.md`.

**Task Shchur (Cora, zufällige Splits, gebaut):** GCN-Claim 81,5 % (Tabelle 1, belegt per Tavily Extract + Spaltenprüfung), größte Komponente, 20/30/Rest, 30 Splits × 1 Initialisierung statt 100 × 20 (Details, Toleranz und Harness-Validierung: `benchmark/experiments/shchur_task.md`; menschliche Referenz 81,1 ± 1,2 %). Der Task prüft nur die GCN-Zahl, nicht die Rangfolge der Modelle. Abgeschnittene LLM-Antworten (`finish_reason: length`) werden nie gecacht, Wiederholungen der Extraktion nutzen andere Temperatur und doppelte Grenze (sonst kam die abgeschnittene Antwort aus dem Cache).

Kandidaten (Zielwerte direkt aus Papers übernehmen, nicht aus Gedächtnis):

| Paper | Erwartung | Datensatz |
|---|---|---|
| Kipf & Welling 2017, GCN | reproduziert | Cora |
| Shchur et al. 2018, Pitfalls of GNN Evaluation | Ranking hängt vom Split ab | Cora, Citeseer |
| Zeng et al. 2023, Are Transformers Effective for TS Forecasting? (DLinear) | reproduziert | ETTh1 |
| Ferrari Dacrema et al. 2019, Are We Really Making Much Progress? | Baselines schlagen neuronale Modelle | MovieLens |
| Shwartz-Ziv & Armon 2021, Tabular Data: DL Is Not All You Need | XGBoost vorn | kleine Tabellen-Datensätze |

Stand der Kandidaten: Kipf, Shchur (GCN und GAT) und Zeng (DLinear, NLinear) sind im Benchmark (v2–v4); dazu Wu/SGC, Veličković GAT und DGI. **Dacrema und Shwartz-Ziv sind nicht gebaut** (Gründe im Abschnitt „Benchmark“). Die Spalte „Erwartung“ ist die ursprüngliche Planung, maßgeblich sind die Referenzurteile in `benchmark/experiments/benchmark_v2.md`.

## Demo und Einreichung

**Jury-Framing (Nutzerentscheidung 2026-10-10):** Produktnutzen und konkrete Erfolge zuerst zeigen; Einschraenkungen klar zuordnen, ohne aus einem einzelnen fehlgeschlagenen Holdout eine allgemeine Erfolgsquote abzuleiten. recall cg v1 bleibt 7/8 In-Sample, RLinear 0/1 Held-out, Placebos 0/9. Davon getrennt: Katalog-Laeufe 2026-10-09 TSMixer 0,362416 gegen 0,361 und SparseTSF 0,365314 gegen 0,359, je 3 Seeds, Zielwert erreicht. Dossiers und Events am 2026-10-10 geprueft; ausserhalb der acht Entwicklungseinheiten, post hoc, Kandidaten teilweise selbst ausgewaehlt, kein neuer Benchmark und keine allgemeine Erfolgsquote. SL-GCN nicht reproduziert, HaKAN Setup-Fehler; Acacia laut Nutzer nicht machbar, Katalog weiterhin ungeprueft (kein abgeschlossenes Urteil). README zeigt die neuen Erfolge und Grenzen. Video vorhanden; spaeter um 7/8 mit Autorencode samt In-Sample-Hinweis und getrenntem Held-out ergaenzen. Devpost-Entwurf stammt vom Projektanfang und wird vor Abgabe aktualisiert.

**Kostenpruefung 2026-10-10:** `docs/cost_policy_review.md` beschreibt die tatsaechlichen Grenzen. Dokumentierte 11,40 $ sind keine garantierte Obergrenze: direkte LLM-Aufrufe umgehen die Orchestrator-Laufbremse, Retries pruefen das globale Budget nicht erneut, der Watchdog beendet Worker nicht. Keine bezahlten Prueflaeufe; Kontostand und Serverkonfiguration nicht live verifiziert. Fuer Deployment-Werte gilt der neuere Plan in `docs/deploy.md` und `deploy/.env.example`, nicht die aeltere 20-$-Serverplanung.

**Aktuelle Einreichungsarbeit (2026-10-10, CEO):** README mit SVG-Wortlogo (hell/dunkel), aktuellen Produktbildern, GLinear und getrennten Evaluationszahlen ueberarbeitet; Detailauswertung in `docs/evaluation_notes.md`. Devpost-Entwuerfe auf denselben Stand gebracht. Diese Praesentationsarbeit und die vorherigen zehn lokalen Agenten-Commits sind auf main gepusht; Repo bleibt privat. Geplante Demo-Adresse laut Nutzer `https://nullius.akca.dev`, noch nicht veroeffentlicht. **Nutzerentscheidung:** CEO-Videoschnitt verworfen, saemtliche neuen Videoquellen und Skriptaenderungen zurueckgenommen. Originalfilm v8 im Reviewer-Workspace `video/out/film.mp4` (2:38, 4 Oct) unveraendert erhalten; Nutzer uebernimmt Videoaenderungen spaeter selbst. Keine weitere automatische Videoarbeit. Produkt-Test gemaess `docs/product_test_20261010.md` bleibt ausstehend, danach Live-Demo-Schritte und Autorenansprache entscheiden. TimeBase danach SOFTS bleiben Kandidaten; kein Experiment laeuft.

**Video (Stand 2026-10-04): Film v8 fertig gerendert**, 2:38, ohne Sprecher, nur synthetische Musik und Einblendungen. Quelle und Render: Worktree `reviewer`, Branch `video-stage`, `video/` (Ausgabe `video/out/film.mp4`, Regie `docs/video_director.md`); gerendert aus der App auf `main` `ceb3ea3`. Zuständig: Reviewer-Session. `docs/video_script.md` enthaelt den aktuellen Revisionsbrief und das historische Skript; Nutzer uebernimmt den Videoschnitt selbst. Ablauf: Hook (Nature-Umfrage 70 %) → ein Claim („GCN achieves 81.5 % on Cora“) → Discovery (aufgezeichneter Live-Lauf, beschriftet) → Katalog → Live-Seite (Kipf-Katalog-Lauf vom 4. Okt als Replay, beschriftet) → „Read. Write. Run. Measure.“ (Tavily, Nemotron auf Token Factory mit Ultra/Super/Lightning, Nebius Sandboxes, Harness) → 10 Seeds, „Reproduced“ → Placebo 88 % „Not reproduced“ → Beleg (10 min, 0,14 $) → Evidence, What-if, Branch-Baum → „48 of 48 caught“ → ehrliche Grenze (2 von 8, nur Graph/Zeitreihe, CPU) → Namen statt Logos. Offen: Upload auf YouTube (öffentlich), Link ins Devpost-Formular. Jede neue UI-Änderung auf `main` heißt: neu rendern lassen, bevor hochgeladen wird.

Evidence-Ansicht in UI: links, was Tavily gefunden hat, rechts, was Sandbox gemessen hat.

Pflicht bei Einreichung:
- [ ] Läuft auf Token Factory, nutzt Nemotron
- [ ] Öffentliches Repo mit Open-Source-Lizenz
- [ ] Demo-URL
- [ ] Demo-Video (max. 3 Min.)
- **Video-Entscheidung 2026-10-10:** Separate Variante 2 (2:52.5, Python-generierte Musik, keine Stimme) liegt im benachbarten Ordner `nullius-video-review-20261010/video/out/film_v8_updated_REVIEW.mp4`. Nutzer findet sie besser und schneidet selbst weiter. Original v8 und seine Quellen im Worktree `reviewer` unveraendert lassen; Variante 2 nicht als Ersatz mergen oder hochladen.
- **Coolify-Check 2026-10-10:** Nutzer deployt selbst. Web-/Docker-Build, Browsernavigation und API-/Kosten-Tests bestanden; PDF-Druckknopf fuer die Produktions-CSP repariert. Erstes Deployment mit `deploy/coolify-preview.env.example` (beide Tages-Startlimits 0). Live-Kostenobergrenzen bleiben unbewiesen; Befunde und Anleitung: `docs/deploy_readiness_20261010.md`. Keine bezahlten Laeufe zur Deploy-Pruefung starten.
- [ ] Feedback zu Nebius Token Factory, AI Cloud und NVIDIA-Tools/Modellen (Pflichtbestandteil der Einreichung laut Devpost; Preis: 10 × 100 $; Entwurf `docs/nebius_feedback.md`). Abgabe: 30.10.2026, 10:00 PDT
- [ ] README: Architektur, Benchmark-Ergebnisse, Hinweis auf verwandte Arbeiten (PaperBench, CORE-Bench) und unser Unterschied

Urteile vorsichtig formulieren: „mit unserem Setup nicht reproduziert", immer mit Versuchen.

## Design Philosophy

Das UI-Design orientiert sich an modernen, minimalistischen SaaS-Plattformen (wie dem Tavily-Dashboard) und vermeidet klassische "AI-Klischees" (wie leuchtende Neon-Farbverläufe oder überladene Chat-Interfaces).

*   **Ästhetik & Layout:** Clean UI, großzügiger Whitespace (Leerraum), klare Kanten und subtile Schatten für Tiefe.
*   **Theme:** Dark Mode und Light Mode werden nativ unterstützt. Eine weiche Hintergrundfarbe (z. B. Off-White im Light Mode) sorgt für ein professionelles Erscheinungsbild.
*   **Typografie:** Hochwertige, serifenlose Schriften (wie Inter oder SF Pro). Der Text muss flüssig und leicht lesbar sein; Hierarchien werden durch Schriftstärke und Graustufen statt greller Farben unterschieden.
*   **Informationsarchitektur:** Die UI darf nicht mit Informationen überfüllt wirken. Daten (wie Metrics oder Logs) werden gut strukturiert in Cards, Tabellen oder ausklappbaren Modulen präsentiert, um kognitive Überlastung zu vermeiden.
*   **Logo & Bildsprache:** Das Leitmotiv "Nullius in verba" ("auf niemandes Wort hin") muss spürbar sein. Es geht um den visuellen Kontrast: "Worte vs. Messung" und "Behauptung vs. Wiederholung" – nicht um generische Wissenschafts-Symbole (Atome, Reagenzgläser). Die Ästhetik entspricht einem präzisen, sauberen AI-Benchmark, das Papers prüft (z. B. Haken, Checkmarks, klare Linien).
*   **Ikonografie & Fokus:** Keine Emojis verwenden, sondern ausschließlich konsistente, professionelle Icons (z. B. Lucide Icons). Das gesamte Erlebnis muss sich nach einem polierten "Produkt" anfühlen und nicht nach einem rohen "Debugging"-Interface für Entwickler.

# Nullius: Produkt- und Entwicklungsstrategie für die Endphase

Dieses Dokument fasst die strategische Ausrichtung für die verbleibende Zeit des Hackathons zusammen. Der Fokus verlagert sich von der reinen Verbesserung der Evaluierungsgenauigkeit hin zu einem vollständigen, überzeugenden Produkterlebnis für die Jury und zukünftige Nutzer.

## 1. Die Produktvision & Der Rahmen
Ein Viertel der Hackathon-Bewertung basiert auf einem "vollständigen Produkterlebnis" und ein weiteres auf dem echten Nutzen für die Zielgruppe. 

**Die ehrliche Grenze des Produkts:** 
*Nullius prüft Papers, die auf bereits unterstützten Benchmarks (z. B. Cora, ETTh1) evaluieren.* 
Es ist glaubwürdig und realistisch zu zeigen, dass für neue Datensätze einmalig Tasks/Harnesses gebaut werden müssen, aber neue Papers auf bestehenden Datensätzen out-of-the-box evaluiert werden können.

## 2. Der Nutzer-Workflow (Die "Nullius-Experience")
Aus Sicht einer Nutzerin (z. B. einer Doktorandin) oder der Jury soll folgender Ablauf erlebbar sein:
1. **Paper-Auswahl/Eingabe:** Auswahl aus einem kuratierten Katalog (siehe unten) oder Eingabe eines arXiv-Links.
2. **Claims auswählen:** Nullius schlägt überprüfbare Behauptungen automatisch vor (basierend auf Rolle 2), die Nutzerin wählt aus.
3. **Zuschauen (Live-Lauf):** Der Auswertungslauf ist live sichtbar (Tavily-Funde, Code-Generierung, Branches, Messungen).
4. **Zeugnis erhalten:** Ein klares Urteil (Evidence-Ansicht) mit Belegen und Einschränkungen als teilbare Seite.
5. **Verlauf & Benchmarks:** Eine Übersichtsseite mit früheren Prüfungen und globalen Benchmarks.

## 3. Die "Live-Katalog"-Idee (Der Jury-Pitch)
Anstatt nur einen arXiv-Link einzufügen, bieten wir der Jury eine kuratierte Liste topaktueller Papers, die sie live testen können.
* **Tavily als Kurator:** Tavily sucht selbstständig nach neuen Papers, die auf unseren unterstützten Benchmarks evaluieren (z. B. neue GNNs auf Cora). Rolle 2 prüft, ob die Claims extrahierbar sind. 
* **Vorteil:** Dies ist ein hervorragender und eigenständiger Use-Case für Tavily, der große Chancen auf den Tavily-Sonderpreis hat.

**Wichtige Schutzmaßnahmen für die Jury-Präsentation:**
1. **Kuration:** Jedes Paper im Katalog muss vorher einmal komplett durchlaufen worden sein. Nur Papers, die (auch bei negativem Ergebnis) nicht abstürzen, kommen in die Auswahl.
2. **Wartezeit & Fast-Mode:** Da vollständige Läufe (wie Shchur oder GAT) lange dauern, bieten wir für die Live-Demo einen **"Schnellprüfungs-Modus"** (mit weniger Seeds/Epochen) an. Das vorab berechnete, vollständige Ergebnis wird daneben angezeigt.
3. **Fallback-Stabilität:** Sandbox-Umgebungen können instabil sein. Schlägt ein Live-Lauf fehl, schaltet das Frontend auf eine "Wiedergabe" (Playback) des Vorab-Laufs um, versehen mit einem ehrlichen Hinweis auf temporäre Instabilität.
4. **Kosten- & Missbrauchsschutz:**
    * ~~Zugangscode für die Jury.~~ Ersetzt (Entscheidung des Nutzers, 2026-10-05) durch die offene Demo mit Budget- und Tavily-Untergrenze, Tageslimits und Limit je IP (siehe „Live-Modus“).
    * Warteschlange (Queue) im Backend (nur ein Lauf gleichzeitig).
    * Tägliches Kostenlimit/Budget auf Server-Seite.

## 4. Architektur-Setup (Deployment auf dem Server)
Um dieses Produkterlebnis bereitzustellen, wird die Architektur wie folgt aufgebaut:

* **Backend (FastAPI):**
    * Verwaltet eine Warteschlange (Queue) und einen Worker, der den bestehenden Nullius-Loop ausführt.
    * API-Keys bleiben strikt serverseitig und sicher.
    * Beinhaltet Kostenkontrolle und Zugangsschutz.
* **Live-Übertragung (Server-Sent Events - SSE):**
    * Das bereits bestehende Ereignisprotokoll des Agenten wird via SSE an das Frontend gestreamt, um den Live-Lauf greifbar zu machen.
* **Frontend (React App):**
    * **Seite 1: Katalog & Startseite** (Unterstützte Datensätze, kuratierte Papers, Eingabefeld).
    * **Seite 2: Live-Lauf** (Fortschrittsanzeige, Log-Stream).
    * **Seite 3: Zeugnis / Evidence** (Das Endresultat, die bestehende Evidence-Ansicht).

## 5. Nächste Schritte (Roadmap für die nächsten 1-1,5 Wochen)

**Stand 2026-10-04:** Schritte 1–5 erledigt (Backend, Live-Streaming, Katalog-Pipeline, Design, Katalog mit Referenz-Papers). Offen nur Schritt 6 (Deployment, Upload, Abgabe), siehe „Offene Punkte“.
1. **Tavily v2 (vorerst) stoppen:** Die bisherigen Verbesserungen (Zitat-Fix) sind super und bleiben. Weitere tiefe Optimierungen werden gestoppt, um Zeit für das Produkt zu haben.
2. **Frontend & Backend Bootstrap:** FastAPI-Gerüst und React-App aufsetzen.
3. **Katalog-Generierung:** Das Skript schreiben, in dem Tavily nach neuen Papers sucht und diese in eine `catalog.yaml` oder Datenbank ablegt.
4. **Live-Streaming implementieren:** SSE-Endpunkt für den bestehenden Loop bauen.
5. **Design & Branding:** Einheitliches Logo und Layout (Startseite -> Live -> Zeugnis) umsetzen.
6. **Abschluss:** README aktualisieren, Einreichungsvideo drehen und Projekt abgeben.


## Hackathon-Regeln (Official Rules, geprüft)

### Fristen
- Abgabe: 30.10.2026, 10:00 PT = **18:00 Uhr deutscher Zeit** (nach EU-Zeitumstellung am 25.10.).
  Entwurf auf Devpost früh anlegen, bis zur Frist änderbar. Danach keine Änderungen mehr.
- Bewertung: 01.–15.12.2026. Gewinner um den 11.01.2027.
- **Demo muss bis mindestens 15.12.2026 erreichbar und kostenlos testbar sein.**
  Server bis dahin betreiben, Spiegel auf GitHub Pages als Absicherung.

### Pflichtbestandteile der Einreichung
- Track: Coding and Agentic Engineering (Agenten, die Code in Token Factory Sandboxes
  schreiben, ausführen und testen).
- Projekt macht Laufzeit-Aufrufe an die Token-Factory-Inferenz-API und nutzt
  mindestens ein NVIDIA-Open-Source-Modell (Nemotron).
- Demo-URL (bei Zugangsschutz: Zugangsdaten in den Testhinweisen).
- Textbeschreibung der Funktionen.
- Öffentliches Repo (GitHub/GitLab/Bitbucket) mit allem Code, Assets und Anleitung.
- Open-Source-Lizenzdatei (MIT), muss auf GitHub im "About"-Bereich erkannt werden.
- README mit Einrichtung und Anleitung zum Ausführen. Ausdrücklich hervorheben:
  - wie Nemotron genutzt wird,
  - wo Token Factory den Workflow beschleunigt hat,
  - welche weiteren Nebius-Dienste genutzt werden (Sandboxes).
- Video unter 3 Minuten, öffentlich auf YouTube, zeigt das Projekt in Funktion.
  Keine geschützte Musik, keine fremden Marken ohne Erlaubnis.
- Alles auf Englisch (oder mit englischer Übersetzung).
- **Feedback zu Token Factory, AI Cloud und NVIDIA-Tools/Modellen ist Pflichtfeld**
  im Einreichungsformular. Entwurf: docs/nebius_feedback.md.

### Bewertung
- Stufe 1 (bestanden/nicht): passt zum Track, nutzt die geforderten APIs/SDKs.
- Stufe 2, vier gleich gewichtete Kriterien:
  - Technological Implementation: Qualität, Nutzung von Token Factory und Nemotron.
  - Design: **vollständiges Produkterlebnis, nicht nur technischer Prototyp**
    → Dashboard ist Bewertungsbestandteil, kein Extra.
  - Potential Impact: konkreter Fall für echtes Problem und echte Zielgruppe,
    belegt durch das Gezeigte → Zielgruppe im README klar benennen.
  - Quality of the Idea: kreative, nicht naheliegende Nutzung, echtes Verständnis
    des Problemfelds.
- Gleichstand: entschieden in Reihenfolge der Kriterien, zuerst Technological Implementation.
- Juroren müssen nicht selbst testen und können nur nach Text, Bildern und Video urteilen
  → README und Video tragen die Bewertung.

### Preise
- Kombinierbar: ein Gesamt- ODER Track-Preis plus ein Bonus-Preis.
- **Best Use of Tavily (3.000 $): verlangt einen funktionierenden Tavily-API-Aufruf
  zur Laufzeit als Teil der Lösung.**
  → Normaler Lauf ruft Tavily live auf. Fixtures nur über ausdrücklichen Schalter
    (Tests, vorab festgelegte Experimente). Live-Aufrufe im Ereignisprotokoll,
    in der Evidence-Ansicht, im README und im Video sichtbar machen.
- Most Valuable Feedback: 10 × 100 $ plus NVIDIA-Swag. Bewertet nach Vollständigkeit,
  Umsetzbarkeit und Wirkung des Feedbacks.
- City Winner Awards: nur für Teilnehmer, die ein IRL-Event besucht haben
  → für uns nicht erreichbar.

### Sonstiges
- Drittanbieter-SDKs, APIs und Daten nur im Rahmen ihrer Lizenzen nutzen
  (Datensätze und Paper-Texte beachten).
- Bei Widersprüchen zwischen Hackathon-Seite und Official Rules gelten die Official Rules.

## Risiken

| Risiko | Gegenmaßnahme |
|---|---|
| Nebius-Sandbox hat Rate-Limits | prüfen, sonst `LocalSandbox` als Fallback |
| Fremde Repos laufen nicht (Versionen, tote Links) | einfache Papers zuerst, Tavily-Fehlersuche, Versuchs-Limit |
| Budget leer | Prototyp auf OpenRouter Free, Cache, Kostenbremse, Budgetplan (`docs/budget_plan.md`) |
| Free-Endpunkt instabil | Retries mit Backoff; kritische Läufe nur auf Prod |
| Agent schönt Zahlen | unabhängiger Harness, Agent liefert nur Vorhersagen |
| Falsches „nicht reproduziert" | vorsichtige Formulierung, Toleranz vorab, Versuche offenlegen |

## Evidence-Ansicht (`web/`)

Statische React-App (Vite, Tailwind, Recharts, React Flow, Framer Motion), keine Backend-Pflicht, keine Keys. Grundlage: `runs/<paper>/<ts>/events.jsonl` (Ereignisprotokoll des Loops, `events.py`) und `dossier.json`. Export ausgewählter Läufe mit `scripts/export_site_data.py` nach `web/public/data`; nur Prod-Läufe veröffentlichen. Ältere Läufe ohne `events.jsonl` werden aus dem Dossier abgespielt. Keys nie ins Frontend.

**Live-Modus (gebaut):** FastAPI `src/repro_agent/api.py`, Start `uvicorn repro_agent.api:app`; Auslieferung von API und Web-App in einem Container über `deploy/serve.py` (`Dockerfile`, `docs/deploy.md`). Die Demo ist **offen, ohne Zugangscode**; die Sperre für Live-Läufe (`src/repro_agent/live_gate.py`, eingebunden in `api.py`) ersetzt ihn. Ein Lauf gleichzeitig (Semaphor im Speicher, genau ein uvicorn-Worker). Sperrgründe (HTTP 503 bzw. 429, Antwort nennt nur den Grund, nie Kontostände): `budget_floor` (`BUDGET_TOTAL_USD` minus Bruttokosten seit `BUDGET_SINCE` unter `LIVE_MIN_BUDGET_USD`, Default 3,0), `tavily_floor` (`TAVILY_CREDITS_TOTAL` minus Credits seit `TAVILY_CREDITS_SINCE` unter `LIVE_MIN_TAVILY_CREDITS`, Default 300; Zähler `src/repro_agent/tavily_usage.py` → `logs/tavily_usage.jsonl`, Schätzung, im Zweifel zu hoch), `daily_limit` (`MAX_DAILY_JOBS` Discovery, Default 5, und `MAX_DAILY_REPRO_JOBS` Reproduktion, Default 3), `busy`. Dazu Limit je IP und Tag (`MAX_DAILY_REPRO_PER_IP`, `MAX_DAILY_DISCOVER_PER_IP`, Default je 1); `TRUST_PROXY=1` nur hinter einem Reverse-Proxy (letzter `X-Forwarded-For`-Eintrag). `GET /api/status` meldet `live`, `reason`, `reset_at` und `kinds`; bei Sperre zeigt das Frontend die Wiedergabe. **Grenze:** Tageszähler und IP-Limits liegen im Speicher, ein Neustart setzt sie zurück; die Untergrenzen bleiben wirksam, weil sie aus den Logs unter `logs/` (Volume!) rechnen.
- `POST /api/discover`, Stream `GET /api/discover/{job_id}/stream`: Tavily-Discovery auf unterstützten Datensätzen.
- `POST /api/reproduce` (Katalogeintrag, Default Schnellprüfung mit `QUICK_SEEDS = 3`), Stream `GET /api/reproduce/{job_id}/stream?from=<n>` (SSE, Reconnect liefert Events erneut), `GET /api/jobs/current`. Der Orchestrator läuft mit `graph_repair=True` und **ohne LLM-Cache**; die API **schreibt nie `benchmark/catalog.yaml`**. Der fertige Lauf wird nach `REPRO_EXPORT_DIR` exportiert (`site_export.export_live_run`) und erscheint in der Evidence-Ansicht.
- Web: Live-Lauf `#/live/<job>` (Phasen-Timeline `web/src/components/run/`), Evidence `#/run/<id>` mit Replay, Katalog `#/catalog`, Discover `#/discover`. Die Steuerung erscheint nur, wenn die API erreichbar ist; sonst Replay des letzten Laufs. Statischer Spiegel: `.github/workflows/pages.yml`.
- Katalog: `benchmark/catalog.yaml` (gepflegt von `scripts/curate_catalog.py`, Logik `src/repro_agent/catalog.py`), Jury-Filter `jury_ready` (Gesamtaussagen „nicht bewertbar (Methode nicht umgesetzt)“, „Agenten-Fehler vermutet“, „nicht reproduzierbar (Setup)“ sind nicht jury-tauglich). Feld `origin`: `discovery` (Default) oder `reference` (Benchmark-Papers, UI-Badge „Reference paper“ mit ihrer v4-Zahl, berechnet aus `benchmark.json`). Stand 2026-10-04, jury-tauglich: Kipf GCN (reference, reproduziert 0,8153 ± 0,0106, 10 Seeds, 0,074 $), Shchur GCN (reference, teilweise 0,8049 ± 0,0158, 30 Splits, 0,113 $), SL-GCN (discovery, nicht reproduziert, vorläufig).
- Overview gruppiert Läufe je Paper-Claim (neuester Lauf als Karte, ältere eingeklappt); Badge bleibt „REPRODUCED“ mit Zusatz „target value reached; method fidelity not checked“ (Entscheidung des Nutzers); Live-Lauf zeigt geschätzte Dauer aus Event-Zeitstempeln.

## Offene Punkte

- [x] Sandbox-Beta-Freigabe abwarten (genehmigt, kostenlos)
- [x] Rate-Limits von Nebius-Sandbox prüfen (2026-09-28: 20 parallele Branches ohne Fehler; höher nicht getestet, Semaphor im Orchestrator trotzdem sinnvoll)
- [x] Exakte Modell-IDs auf OpenRouter und Token Factory eintragen (TF geprüft 2026-09-28: Lightning 0,06/0,24 $, Super 0,30/0,90 $, Ultra 1/3 $ pro Mio. Tokens)
- [x] Builders-Program-Codes eingelöst (Token Factory und Tavily; Stand und Planung in `docs/budget_plan.md`)
- [ ] Preise pro Modell in TF-Konsole notieren, Kosten-Log kalibrieren
- [x] Benchmark-Liste festgelegt: 6 Papers, 8 Paper-Claim-Einheiten statt 8–10 Papers (`benchmark/experiments/benchmark_v2.md`); v2, v3, v4 gelaufen und ausgewertet
- [x] Live-Modus (FastAPI + SSE) gebaut (`src/repro_agent/api.py`)
- [x] Hosting-Optionen verglichen, Empfehlung Coolify (`docs/deploy.md`); **Deployment und Demo-URL liegen beim Nutzer**
- [x] Benchmark auswerten, Zahlen ins README (v4, präzisionsorientiert; Devpost-Texte auf v4)
- [x] Katalog: Referenz-Papers Kipf und Shchur mit echten Läufen (2026-10-04)
- [x] Demo-Video gerendert (Film v8, siehe „Demo und Einreichung“)
- [x] Live-Läufe überschrieben `benchmark/fixtures/` (gefunden und behoben 2026-10-04, Merge `fix-live-fixtures`): `TAVILY_MODE=live` schreibt nie mehr nach `benchmark/fixtures/`, Tests arbeiten auf einer Kopie mit Wächter. Bekannte Grenze: Repo-Suche, Repo-Dateien und Zielwert-Extraktion **lesen** im Live-Modus weiter vorhandene Fixtures (kein neuer Tavily-/LLM-Aufruf, wenn eine Fixture existiert)
- [x] Offene Demo ohne Zugangscode (Merge `open-demo-lock`, 2026-10-07): Budget- und Tavily-Untergrenze, Limit je IP, `/api/status`, Tavily-Credit-Zähler; Nutzungsfälle im README (Merge `impact-usecases`)
- [ ] Datensätze v2 (`datasets-v2`) in Arbeit (Stand 2026-10-07)
- [ ] **Film neu rendern:** Film v8 zeigt in Szene 3 das Zugangscode-Feld und passt nicht mehr zur offenen Demo; neues Skript `docs/video_script.md`, Regie `docs/video_director.md` anpassen (Reviewer-Session)
- [ ] **Limits in zwei Stufen** (Nutzer setzt die Variablen in Coolify und deployt neu; Begründung und Rechnung `docs/deploy.md`, „Limits bis 15.12.“): bis 2026-11-30 `MAX_DAILY_REPRO_JOBS=1`, `MAX_DAILY_JOBS=1`; **ab 2026-12-01** (Bewertung 01.–15.12.) `MAX_DAILY_REPRO_JOBS=3`. Nicht während eines laufenden Jobs deployen
- [ ] Nutzer: Devpost-Formular füllen (Text `docs/devpost_submission.md`, Stand 17.10.; Galeriebilder `docs/img/devpost/`), Feedback-Feld aus `docs/nebius_feedback.md`. Erledigt: Deployment, Demo-URL, Repo öffentlich, Video öffentlich auf YouTube (17.10.)

## Projektstruktur (tatsächlich)

```
.
├── CLAUDE.md, README.md, LICENSE (MIT), Dockerfile, pyproject.toml, uv.lock
├── src/repro_agent/
│   ├── loop.py            # Orchestrator (Rollen 2, 4, 5, Plan, Code, Branches, Urteil)
│   ├── community.py       # Rollen 3/7 und 6 als Berichtsfunktion nach dem Urteil
│   ├── nachlauf*.py       # Experiment „Nachlauf“, nicht im Produkt
│   ├── llm.py, costs.py, config.py   # LLM-Client (Retry, Cache, Kosten, Budget), .env
│   ├── tavily_tools.py, claims_source.py, repo_source.py, setup_details.py, table_check.py
│   ├── tasks.py           # Cora, Cora-LCC, ETTh1
│   ├── harness/           # unabhängiger Prüf-Harness, Baselines, Ablation, Formprüfung, Code-Check
│   ├── sandbox/           # base.py, nebius.py (contree-sdk), local.py (Docker-Fallback)
│   ├── discovery.py, catalog.py, site_export.py
│   ├── api.py             # FastAPI, SSE, Warteschlange
│   └── dossier.py, events.py, fidelity.py, objections.py, report.py
├── benchmark/             # papers.yaml, catalog.yaml, fixtures/, experiments/ (Vorab-Festlegungen, results/)
├── scripts/               # run_benchmark.py, eval_benchmark_v2.py, curate_catalog.py, export_site_data.py, Validierungen
├── web/                   # React-App (Vite), public/data = exportierte Läufe
├── deploy/                # serve.py (API und Web in einem Prozess), Caddyfile, docker-compose.yml
├── tests/                 # Fakes, tests/reference/ = Referenzlösungen
├── docs/                  # Pläne, Deployment, Forschungslog, Feedback, Video und Devpost
├── runs/                  # nicht in Git (Läufe, benchmark_v2.jsonl)
└── logs/                  # nicht in Git (costs.jsonl)
```
