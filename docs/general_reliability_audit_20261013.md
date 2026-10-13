# Allgemeine Zuverlaessigkeitspruefung — nur Untersuchung

Stand: **2026-10-13**, Basis **`e945e01c2e02ddf60635bf44b31698f859bb9cf9`**. Auftrag: allgemeine Fehler und problematische Verfahrensluecken untersuchen, belegen und dokumentieren; **keine Fixes anfangen**.

Das wichtigste Ergebnis: Eine formal korrekte Zahl aus einem frisch kopierten Harness reicht derzeit nicht fuer einen belastbaren Reproduktionsnachweis. Die groessten Luecken betreffen die Integritaet der Ausfuehrungsumgebung, Zukunftsinformation in Zeitreihen-Eingaben und unabhaengige Validierung. Daneben koennen Quellen- und Parameterpruefungen korrekten Code verschlechtern oder falsche Einstellungen akzeptieren.

**29 Gegenproben wurden erfolgreich reproduziert:** 25 Python-Proben, 3 reine Frontend-Ereignisproben und 1 isolierte Docker-Probe. Daraus ergeben sich **23 Befundgruppen/Schutzluecken und 4 gesonderte Verfahrensrisiken**. Erfolgreich bedeutet hier: Der problematische Mechanismus ist belegt, nicht dass die Anwendung bestanden hat.

**Keine Produkt-, Test-, Benchmark-, Deployment- oder Konfigurationsdatei wurde geaendert.** Hinzugekommen sind dieses Dokument und die Untersuchungsskripte/-ergebnisse unter `docs/audits/`. Keine Modell-, Tavily-, Nebius- oder Produktionsauftraege wurden gestartet. Netzwerkverbindungen waren in den Python-Proben gesperrt; der Docker-Container hatte kein Netzwerk und nur einen schreibgeschuetzten Mount des Audit-Verzeichnisses.

## Umfang und Beweisstaerke

Untersucht wurden Quellen/Discovery-zu-Claim-Uebergabe, Repository/Steckbrief/Parameterzuordnung, Code- und Reparaturpfad, Kandidaten/Seeds, Graph- und Zeitreihen-Harness, Validierung/Ablation, Kostensteuerung, API-Lebenszyklus und Ergebnis-/Replay-Darstellung. Schwerpunkte folgen den Fehlern des echten SARAF-Laufs, die Gegenproben verwenden aber ueberwiegend andere, synthetische Verfahren.

Die Pruefung ist pfadbezogen, keine formale Verifikation jeder Zeile. Die Belege unterscheiden:

- **Direkter Funktionsbeleg:** Der unveraenderte Produktcode liefert fuer konkrete Eingaben ein problematisches Ergebnis.
- **Kontrollierter Pipelinebeleg:** Original-Orchestrierung mit lokalen Fake-Clients/-Sandboxen und gezielt gesetzten Messungen oder Fehlerzustaenden. Belegt Entscheidungen und Fehlerfortpflanzung, keine Erfolgsquote realer Modelle.
- **Isolierter Ausfuehrungsbeleg:** Tatsaechliches Verhalten im Task-Image, ohne Produktionszugriff.
- **Verfahrensrisiko:** Ein nachvollziehbarer moeglicher Schaden, aber kein belegter aktueller Produktionsvorfall beziehungsweise eine bewusst gewaehlte Politik mit Nachteilen.

**Es gibt keinen Nachweis, dass bisherige Live-Laeufe absichtlich Scores manipuliert oder Zukunftsinformation aus fremden Fenstern genutzt haben.** Die Gegenproben zeigen, dass die derzeitigen Schutzschichten diese Faelle nicht ausreichend ausschliessen.

## Reproduzierbare Belege

| Artefakt | Inhalt / Aufruf |
| --- | --- |
| [Python-Proben](audits/general_error_probes_20261013.py) | `uv run python docs/audits/general_error_probes_20261013.py` — 25 assertionsgestuetzte Gegenproben, Netz gesperrt; SHA-256-Vergleich der geschuetzten Repo-Dateien vor/nach Ausfuehrung. |
| [Python-Ergebnisse](audits/general_error_probes_20261013.json) | Basis-Commit, konkrete Werte, Kandidatenentscheidungen und `product_files_unchanged=true`. |
| [Frontend-Proben](audits/general_ui_probes_20261013.mjs) | `node docs/audits/general_ui_probes_20261013.mjs` — 3 Gegenproben des unveraenderten Render-Modells, keine HTTP-/Browser-Aufrufe. |
| [Frontend-Ergebnisse](audits/general_ui_probes_20261013.json) | Stopp-Zustand, gemischte Claims und ein bereits gespeicherter echter Placebo-Fall. |
| [Harness-Umgebungsprobe](audits/harness_environment_probe_20261013.py) | Ausschliesslich im Wegwerf-Container ausfuehren, Kommando im Skript. Aendert nur dessen Python-Laufzeit; kein Host-Python. |
| [Harness-Ergebnis](audits/harness_environment_probe_20261013.json) | Tatsaechlicher Score 8.0, vom Last-Line-Parser gelesener Score 0.371; statischer Code-Guard meldet keinen Treffer. |
| [Gespeicherte Live-Gegenprobe](saraf_live_verification_20261013.md) | Bereits vorhandene echte SARAF-Ereignisse und erzeugter Code. Wurde fuer diesen Audit nicht erneut ausgefuehrt. |

Die Docker-Probe verwendete das bereits vorhandene Task-Image `python:3.11-slim`, Digest `sha256:0dd364ba7e10242f07755449e3a3d0e35f9efd987952737b90def6709ab0c5ce`, Python 3.11.17, Default-Benutzer root. Container danach entfernt; das Basisimage bleibt unveraendert.

## Prioritaet 0 — Integritaet der Messung

### E01 — Frischer Harness-Code schuetzt keine veraenderbare Python-Laufzeit

**Belegt im Docker-Fallback.** Agentencode und spaeterer Harness benutzen denselben beschreibbaren Branch und dieselbe Python-Installation. Eine kleine Datei in `site-packages/sitecustomize.py` kann spaetere Evaluationsprozesse beeinflussen. Der Code-Guard akzeptiert die Schreiboperation. Die Probe erstellt den eigentlichen Harness erst danach: Er gibt korrekt MSE 8.0 aus; ein Startup-Hook haengt beim Prozessende eine zweite, gueltige JSON-Zeile mit MSE 0.371 an. Der Orchestrator liest die letzte Zeile.

**Quelle:** `sandbox/local.py:16,36`, `tasks.py:353`, `loop.py:848-865`, `harness/code_check.py:25`. **Beleg:** `harness_environment_probe_20261013.json`.

**Auswirkung:** Numerische Ergebnisse koennen trotz unveraenderter, frisch kopierter Harness-Quelldateien unzuverlaessig sein. Netzabschaltung schuetzt diesen Dateisystem-/Interpreterpfad nicht. Die Probe manipuliert keine echten Resultate. **Nebius-UID/Mount-Rechte sind hier nicht live getestet**; dessen Adapter setzt selbst keine getrennte schreibgeschuetzte Harness-Laufzeit auf. Eine allgemeine Aussage ueber die Integritaet aller Backends waere deshalb unbelegt.

**Spaetere Abnahmebedingung:** Eine vom Agenten vorgenommene Aenderung seiner Laufzeit darf den vertrauenswuerdigen Evaluationsprozess oder dessen Ergebnis nicht beeinflussen. Zusaetzliche agentenkontrollierte Ausgaben duerfen nicht als Harness-Ergebnis gelten.

### E02 — Ueberlappende Forecast-Testeingaben verraten Zukunftswerte

`prepare_public` uebergibt alle Testfenster gleichzeitig. Fuer Fenster `i`, Forecast-Schritt `h`, liegt dessen Zielwert im letzten Eingangswert von Fenster `i+h+1`, solange dieses Fenster existiert. Die Textanweisung verbietet die Nutzung anderer Fenster, der Standardlauf erzwingt sie nicht durch die bestehende Unabhaengigkeitsprobe.

**Quelle:** `harness/ett_data.py:97,119`, `tasks.py:266`, `loop.py:771`; Unabhaengigkeitspruefung separat in `nachlauf.py`. **Probe:** `future_window_leak`.

**Ergebnis:** Synthetische Daten: 93.24 % der Zielwerte aus Eingaben rekonstruierbar, Rundungsfehler maximal 1.05e-7, ohne irgendein Ziel-/Label-Feld. Mit den ETTh1-Dimensionen sind rechnerisch **98.26 %** der Zielwerte auf diese Weise vorhanden. Das ist eine Berechnung der Verfuegbarkeit, kein gemessener Live-Exploit oder neuer Benchmarkscore.

**Auswirkung:** Ein nicht kausaler Forecaster kann gute Scores erzielen. Die Kopie echter Testlabels allein zu verbieten reicht nicht. **Abnahme:** Jede Vorhersage muss unabhaengig von zukuenftigen/anderen Testfenstern bleiben, mit wirksamer Kontrolle vor einem belastbaren Urteil.

## Prioritaet 1 — Validierung und Kontrollversuche

### E03 — Direkt kopierte Graph-Val-Labels koennen die Auswahl gewinnen

Der Code `pred[d['val_idx']] = d['labels'][d['val_idx']]` wird statisch als `selection` statt `training` eingestuft. Die dynamische Val-Permutation belegt Label-Nutzung (Val-Score 0.95 -> 0.25), trotzdem ist `excluded=false`. Der manipulierte Kandidat gewinnt gegen den unabhaengigen Kandidaten mit 0.80. Die Standardpolitik `allow_test_feedback=False` war aktiviert.

**Quelle:** `loop.py:739-752,1420`, `harness/val_use_check.py:246`. **Probe:** `copied_graph_validation_labels_are_selected` (kontrollierte Pipeline mit Fakes).

**Auswirkung:** Val-Auswahl ist nicht unabhaengig, obwohl der Trainings-/Testscore-Feedback-Fix aktiv ist. Der Test-Harness selbst muss dadurch noch keinen falschen Score liefern. **Abnahme:** Early Stopping darf Val nutzen; direkt aus Val-Labels gebaute Ausgaben duerfen nicht als unabhaengige Kandidatenguete gelten.

### E04 — Forecast-Val-Nutzung im Training wird nicht kontrolliert

ETT-Tasks haben kein `mask_val_cmd`. `_run_candidates` ruft den Label-Nutzungscheck fuer sie nicht auf. Eine gemeldete perfekte Val-Metrik kann die Auswahl gewinnen, ohne dass ihre unabhaengige Entstehung geprueft wird. Die Probe injiziert einen Kandidaten mit Val-MSE 0.0 und bestaetigt Auswahl ohne Label-Check; sie behauptet keine echte Modellleistung.

**Quelle:** `tasks.py:353-390`, `loop.py:1408-1420`. **Probe:** `unchecked_forecast_validation`. **Abnahme:** Daten, die zur Auswahl dienen, duerfen bei Forecasting ebenso wenig zum Modelltraining oder zum direkten Erzeugen der Val-Ausgaben benutzt werden.

### E05 — Fehlgeschlagene Val-Kontrollen lassen Kandidaten durch

Zwei belegte Faelle: Die Permutation der Val-Labels endet mit Exit 1, wird ignoriert, und unveraenderte Daten liefern den irrefuehrenden Befund `Validierungslabels nicht genutzt`. Oder die Kontrollausfuehrung crasht, obwohl die statische Analyse Training auf Val erkennt; dann lautet der Befund `nicht messbar`, aber ebenfalls `excluded=false`.

**Quelle:** `loop.py:812-815,739-752`. **Probe:** `validation_control_failures`. **Auswirkung:** Ein fehlgeschlagener Schutztest wird praktisch zu einer Zulassung. **Abnahme:** Der erfolgreiche Kontroll-Eingriff und dessen auswertbare Ausfuehrung muessen belegt sein; fehlende Evidenz darf nicht als bestandene Unabhaengigkeit gelten.

### E06 — Fehlgeschlagene Graph-Ablation erzeugt ein falsches Methodenurteil

Der Befehl zum Entfernen der Kanten kann Exit 1 liefern. Der Orchestrator trainiert trotzdem weiter auf dem unveraenderten Graph. Gleiche Accuracy wird als `Graph nicht genutzt` interpretiert; Detail ist `None`.

**Quelle:** `loop.py:809-812`, `loop.py:ablation_check`. **Probe:** `failed_ablation`. **Auswirkung:** Ein Verfahren mit Graphnutzung kann faelschlich herabgestuft werden. **Abnahme:** Erfolgreich geaenderte Eingabedaten bestaetigen, bevor eine kausale Schlussfolgerung aus dem Kontrollscore gezogen wird.

## Prioritaet 1 — Parameter und Quellen

### E07 — Quellenrang gewinnt ohne Pruefung des Experimentzwecks

Die reproduzierte Extraktion erzwingt Batch 32 aus "inference speed with batch size 32", obwohl das Training Default 8 hat. `verify_entries` prueft das Zitat und die Zahl, nicht deren Trainingsrolle. Ebenso kann ein nur im globalen Parser deklariertes, vom konkreten Modell gar nicht verwendetes Dropout erzwungen werden. Ein originalgetreues Modell ohne Dropout erhaelt dann `unused` und eine Reparaturforderung.

**Quelle:** `spec.py:412,461,542`, `spec_check.py:119`. **Proben:** `training_vs_inference_batch`, `inactive_model_setting_enforced`. Beide haben eine konkrete Entsprechung im gespeicherten SARAF-Lauf. **Abnahme:** Parameter muessen zum ausgefuehrten Modell, Trainingsvorgang, Datensatz und Horizont gehoeren; ein echter, aber anders gemeinter Zahlenbeleg genuegt nicht.

### E08 — Ein fremder Startaufruf entfernt den richtigen Default

`drop_run_script_defaults` sammelt Flags aus allen Aufrufen eines Shellskripts. Ein Override nur fuer Datensatz Other/Horizont 192 entfernt dadurch den Default fuer Wanted/Horizont 96, obwohl dessen Aufruf das Flag nicht setzt. Reproduziert mit `time_aware_weight`: Default 0.5 verschwindet wegen eines fremden Overrides 0.3.

**Quelle:** `spec.py:507`. **Probe:** `unrelated_default_stripping`. **Abnahme:** Nur Overrides des passenden Aufrufs duerfen dessen wirksame Defaults ersetzen.

### E09 — Unterschiedliche Aufrufe lassen den passenden Rohparameter verschwinden

`raw_items` sieht verschiedene Lernraten in einem Skript und verwirft den Schluessel fuer die gesamte Datei. Wanted/96 hat eindeutig 0.01; Other/192 hat 0.02. Der passende Rohwert fehlt trotzdem. Damit haengt die korrekte Uebernahme wieder von der KI-Interpretation des gesamten Skripts ab.

**Quelle:** `spec.py:357`. **Probe:** `multi_command_values`. **Abnahme:** Werte je passendem Aufruf auswerten; wirkliche Mehrdeutigkeit dieses Aufrufs von mehreren verschiedenen Experimenten unterscheiden.

### E10 — Gleich benannte Keywordargumente werden als dieselbe Einstellung behandelt

Die aktuelle Pruefung meldet Retrieval-Vorberechnungsbatch 1024 als Konflikt mit Trainingsbatch 32. Sie meldet ausserdem `F.normalize(x, p=2)` als falsches Dropout 2 statt 0.1. Der Aufrufer und die Bedeutung des Parameters werden nicht ausgewertet.

**Quelle:** `spec_check.py:77`. **Probe:** `unrelated_keyword_conflicts`. **Auswirkung:** Unnoetige Reparaturen, Budgetverbrauch und moegliche Aenderung des richtigen Verfahrens. **Abnahme:** Trainingsbatch, Retrievalbatch und Normalisierungsordnung muessen ihre eigenen Rollen behalten.

### E11 — CONFIG kann widersprechen und trotzdem volle Konformitaet erhalten

Alias (`cfg=CONFIG; cfg['lr']=1.0`), komplettes Neusetzen des Dictionary oder eine Zuweisung in `if True` entziehen sich `_config_mutations`. In allen drei Faellen wird 0.001 verlangt, tatsaechlich 1.0 an den synthetischen Optimizer uebergeben und Score **1/1** gemeldet; kein Konflikt, kein erforderlicher Override.

**Quelle:** `spec_check.py:46,91,119`, `loop.py:780-787`. **Probe:** `silent_config_overwrites`. **Abnahme:** Konformitaet darf sich nicht allein auf den ersten Dictionaryblock und vorhandene Lesezugriffe stuetzen.

### E12 — Eine fachfremde Paperzahl entscheidet einen Parameterkonflikt

`resolve` bekommt fuer `corroborate` nur ein globales Zahlenset. Ein Paper-Horizont 32 kann damit zwei gleichrangige Batch-Konfigurationen 16/32 zugunsten von Batch 32 entscheiden und den Konflikt als aufgeloest/erzwingbar ausgeben. Die Probe belegt diese Entscheidung fuer `corroborate={32}`.

**Quelle:** `spec.py:paper_numbers,461`. **Probe:** `paper_number_collision`. **Abnahme:** Bestaetigung muss zum gleichen Parameter gehoeren, nicht nur zur gleichen Zahl irgendwo im Paper.

### E13 — Konformitaetsscore kann nach einer Reparatur veraltet sein

`_run_candidates` speichert den Spec-Score vor der Qualitaetsreparatur. Aendert diese den Code, wird der Score nicht erneut berechnet. Kontrollierter Fall: Ein Kandidat startet konform mit 1/1, sein reparierter Code hat 0/1, gewinnt aber weiterhin vor einem wirklich konformen Kandidaten.

**Quelle:** `loop.py:1398-1410,1433-1437`. **Probe:** `stale_spec_score`. **Abnahme:** Jede fuer Auswahl verwendete Bewertung muss zur exakten finalen Code-Version gehoeren, nicht zu deren Vorganger.

### E14 — Graph-CSV ueberschreibt doppelte IDs still

`verify.load_predictions` erstellt unmittelbar ein Dictionary. CSV mit `0,9`, danach `0,0`, danach `1,1` wird zu zwei sauberen IDs und besteht die Plausibilitaetspruefung. Die erste Vorhersage verschwindet. Der ETT-Parser und der Unabhaengigkeitsparser erkennen Duplikate bereits; der Graphpfad nicht.

**Quelle:** `harness/verify.py:29-39`; Nutzer `planetoid_eval.py`, `planetoid_val_eval.py`, `cora_lcc_eval.py`. **Probe:** `duplicate_classification_ids`. **Abnahme:** Doppelte IDs vor der Umwandlung in ein Dictionary erkennen, fuer Test und Val.

### E15 — Trainingsabhaengigkeiten bleiben teilweise unvollstaendig

`author_excerpt` verfolgt Modell-Imports transitiv, bei Trainings-/Protokolldateien aber nur die erste Helferebene. `train.py -> utils.schedule -> utils.decay` laesst `utils.decay` aus, obwohl dessen Datei vorhanden ist. Ein bekannter, aber gar nicht gelesener erster Trainingshelfer stoppt ebenfalls nicht: Die Aufloesung nutzt dort nur die gelesenen Dateien statt des vollstaendigen Listings.

**Quelle:** `source_context.py:44-53`. **Probe:** `protocol_dependency_omission`. **Auswirkung:** Scheduler, Stopping oder Preprocessing koennen erneut unvollstaendig im Prompt landen. **Abnahme:** Fuer benoetigte Trainingsabhaengigkeiten dieselbe Vollstaendigkeitslogik wie fuer den Modellkern nachweisen.

### E16 — Expliziter Git-Ref eines Paperlinks geht verloren

Links auf `tree/v1/model` und `tree/main/model` ergeben denselben Klonplan: Repositorywurzel plus Unterordner. Der Branch/Tag wird nicht uebernommen; gelesen wird Default-HEAD. Ein Paper kann dadurch mit einer anderen Version seines offiziellen Codes verglichen werden. Der protokollierte Commit zeigt zwar, was gelesen wurde, erfuellt aber den angeforderten Ref nicht.

**Quelle:** `repo_source.py:parse_github_url,309,467-472`. **Probe:** `requested_repo_ref_lost`. **Abnahme:** Angeforderte Revision erhalten oder ihre Nichterfuellbarkeit explizit melden.

## Prioritaet 1/2 — Grenzen der Vorpruefungen

### E17 — Nichttriviale Platzhalter passieren nach kleiner Umformulierung

`result=torch.zeros_like(x); return result` wird nicht als konstante Rueckgabe erkannt. Ein anders benannter `fetch_context` wird nicht mit dem Referenz-`retrieve` verglichen. Beide synthetischen Retrieval-Platzhalter erzeugen keine Meldung. Der bestehende Guard beansprucht selbst keine allgemeine Methodenzertifizierung; diese Gegenproben zeigen die praktisch relevante Restluecke.

**Quelle:** `method_guard.py:56`. **Probe:** `placeholder_blind_spots`. **Abnahme:** Die bekannten Varianten muessen erkannt werden oder das Ergebnis muss ausdruecklich unvollstaendig bleiben; ein leerer Befund ist kein positiver Methodennachweis.

### E18 — Der Checkpoint-Guard deckt nur einige Schreib-/Ladeformen ab

Ein flacher Checkpoint bleibt unentdeckt bei `load_state_dict(state_dict=best)`, einem Optimizer namens `trainer` und bei `self.best` als Checkpoint. Das sind gewoehnliche Python-/PyTorch-Schreibweisen; der bekannte Aliasfehler kann damit wieder auftreten. Die bereits vorhandene Deepcopy-Korrektur und Tests fuer die erkannten Formen bleiben richtig.

**Quelle:** `harness/training_check.py:10`. **Probe:** `checkpoint_blind_spots`. **Abnahme:** Die benoetigten Formen gezielt pruefen; Grenzen nicht als vollstaendige Garantie darstellen.

### E19 — Harmloser Text wird als verbotene Operation gewertet

Ein Kommentar mit Paper-URL und die rein lokale Funktion `fetch_neighbors_from_memory` werden wegen Regex-Woertern zurueckgewiesen, obwohl keiner der beiden Faelle Netzwerkzugriff oder Download ausfuehrt. Der Netzwerk-Namespace bleibt eine separate Schutzschicht.

**Quelle:** `harness/code_check.py:25`. **Probe:** `comments_rejected_as_operations`. **Auswirkung:** Unnoetige Abbrueche/Reparaturen bei legitimen Nachbauten. **Abnahme:** Codeoperation und beschreibenden Text auseinanderhalten, ohne den eigentlichen Netzschutz zu schwaechen.

### E20 — Ein Kandidat ueberspringt den Graph-Val-Labelcheck

Bei `n_candidates=1` nutzt `run_claim` einen anderen Pfad. Der Label-Check wird nicht aufgerufen, selbst wenn die statische Analyse Training auf Val erkennen wuerde; ein Urteil entsteht. **Betrifft die interne Einzelkandidaten-Konfiguration; der aktuelle oeffentliche API-Standard erzeugt drei Kandidaten.**

**Quelle:** `loop.py:1221-1242` gegen `1408-1412`. **Probe:** `single_candidate_skips_label_check`. **Abnahme:** Spar-/Einzelkandidatenmodus darf verpflichtende Datenkontrollen nicht abschalten.

## Prioritaet 2 — Ergebnisdarstellung und Beobachtbarkeit

### E21 — Gestoppter Job wird im Seed-Bereich als laufend beschrieben

`candidateSelectionPending` ist auch bei `running=false` wahr, falls vor Abbruch Kandidaten angekuendigt, aber nie ausgewaehlt wurden. `RunTimeline` zeigt daraufhin weiter "are being tested in parallel". Der globale STOPPED-Status ist korrekt.

**Quelle:** `web/src/components/run/phases.ts:151`, `RunTimeline.tsx:307`. **Probe:** `stopped_candidate_status`; ausserdem im echten SARAF-Stopp beobachtet. **Abnahme:** Alle untergeordneten Hinweise muessen denselben terminalen Jobzustand wiedergeben.

### E22 — Replay/Graph mischt mehrere Claims eines Laufs

`buildRunModel` nimmt Claim/Target/Metrik vom ersten Claim, finale Auswahl und Urteil aber vom letzten. Ein synthetischer Zweiclaim-Lauf zeigt Accuracy-Ziel 0.83 zusammen mit Forecast-MSE 0.4. **Auch bestaetigt an vorhandenen Daten:** `kipf_gcn_2017--20261002-110746` hat fuer den Paperclaim `reproduziert`, fuer den Placebo `nicht reproduziert`. Der Renderer zeigt Paper-Ziel 0.815 zusammen mit dem negativen Placebo-Urteil. `RunView` uebergibt dem Replay/StepGraph alle Events; der separate Kurzfilm filtert Claims bereits korrekt.

**Quelle:** `web/src/components/run/phases.ts:140-151,448,527`, `RunView.tsx:220-221`. **Proben:** `multi_claim_event_mix`, `existing_saved_placebo_verdict_mix`. **Abnahme:** Metadaten, Versuche, Auswahl und Urteil muessen denselben Claim betreffen.

### E23 — Der angezeigte Modellaufruf-Zaehler ist unvollstaendig

Direkte Setup-Extraktion ruft `llm.complete` auf, ohne die `llm_call`-Events des Orchestrator-Wrappers zu erzeugen. Die Gegenprobe zaehlt zwei wirkliche Fake-Client-Aufrufe und null Live-LLM-Events. Das Frontend zaehlt nur diese Events. Die aufsummierten Kosten koennen trotzdem die direkten Aufrufe enthalten; Aufrufzahl und Kostenverlauf sind somit unterschiedlich vollstaendig.

**Quelle:** `setup_details.py:113,207`, `loop.py:475-489`, `web/src/components/run/phases.ts:590`. **Probe:** `extraction_calls_missing_from_event_count`. **Abnahme:** Entweder alle Anfragen messen oder den Zaehler entsprechend begrenzt benennen. Aus der Probe folgt keine genaue Anzahl bisheriger kostenpflichtiger Aufrufe.

## Gesonderte Verfahrensrisiken — keine gleichgesetzten Fehlerbeweise

| ID | Nachweis / Risiko | Einordnung und spaetere Abnahme |
| --- | --- | --- |
| **R01: Daten-/Umgebungsversion** | `data_version_drift`: Vorbereitung und Evaluation koennen gleich grosse, aber verschiedene Datenversionen verwenden; synthetische Target-MSE zwischen Versionen 14.4647. ETT/Planetoid werden fuer Scoring erneut von beweglichen GitHub-URLs geladen, Public-NPZ traegt keine Datenidentitaet. `tasks.py` installiert ausserdem Bibliotheken ohne Version-Pins. | Keine tatsaechliche Aenderung des heutigen ETT-/Planetoid-Datensatzes behauptet. Vorbereitungs- und Scoringdaten sowie relevante Laufzeitversionen eindeutig zuordnen; die Bibliotheksvariante ist Quellenreview, keine ausgefuehrte Versionsvergleichsmessung. |
| **R02: Budget vor frueher Codepruefung** | `budget_aborts_before_valid_candidate_trial`: Kandidat 0 ist bereits erzeugt, Budgetfehler bei Generierung von Kandidat 1 stoppt alles, bevor Kandidat 0 geprueft wird. Der Ablauf generiert alle Kandidaten vor deren Pruefung/Reparatur. | Die Kostenbremse darf weiterhin konservativ stoppen. Problem ist die Verteilung begrenzter Mittel und fehlende fruehe Rueckmeldung; keine Rechtfertigung fuer das Umgehen des Limits. Abnahme: begrenztes Budget sinnvoll einplanen und bereits vorhandene valide Evidenz eindeutig bewahren. |
| **R03: Link auf fehlgeschlagenen Live-Job** | `aborted_live_job_link_is_volatile`: Nach Registry-Pruning liefert der Link 404; Redeploy verliert die Registry ebenfalls. Fehlgeschlagene Checks werden ueber diese API nicht als dauerhaft oeffentliche Run-Artefakte exportiert. | Lokale Events/Dossiers sind nicht generell verloren: `run` speichert das Dossier im `finally`. Die Einschraenkung betrifft oeffentliche Nachvollziehbarkeit. Abnahme: Ein geteilter Fehlerbeleg bleibt nach Neustart nachvollziehbar oder die begrenzte Verfuegbarkeit wird klar benannt. |
| **R04: Verdachtsheuristik ueber fremde Metrik** | `unclaimed_metric_can_override_matched_claim`: Ziel-MSE 0.37 exakt getroffen; MAE 0.6 ist schlechter als Baseline 0.5, waehrend deren MSE 0.5 schlechter ist. Ergebnis wird `Agenten-Fehler vermutet`. Die kleine Zahlenprobe zeigt einen normalen MSE/MAE-Zielkonflikt. | Dies ist eine ausdrueckliche Verdachtsheuristik, kein Beweis eines Agentenfehlers. Sie kann die numerische Claim-Erfuellung ueberdecken. Abnahme: Claim-Metrik, andere Diagnosemetriken und nachgewiesene Implementierungsfehler getrennt ausweisen. |

## Gespeicherter Live-Beleg zum Datenvertrag

Der vorherige SARAF-Check bleibt ein zusaetzlicher, nicht erneut ausgefuehrter Beleg: `StaticInputDataset` liefert fuenf Werte, die Inferenz erwartet vier. Derselbe Dataset-Typ erzeugt Dummy-Nulltargets, die spaeter als echte Validierungsziele fuer Checkpoint-Auswahl verwendet werden. Das sind Fehler des erzeugten Programms, keine Behauptung, dass der Harness seine eigenen Targets falsch berechnet.

Die allgemeine Luecke liegt davor: Der Forecast-Helfer stellt `train`, `val` und `test_inputs` bereit, aber keine verbindlichen Dataset-/Loader- beziehungsweise Val-Fenster/Target-Objekte. Jeder Nachbau muss diesen Teil neu erzeugen. Ein formal korrektes CSV am Ende belegt noch nicht die richtige interne Checkpoint-Auswahl. Diese Familie sollte in einer spaeteren Fixphase mit einem einfachen, deterministischen Datenvertragsfall abgenommen werden, bevor weitere grosse Paper als Erfolgsnachweis dienen.

## Bewusst nicht als neuer Befund gefuehrt

- Ein Test-Harness mit Exit ungleich 0 oder Timeout wird bereits zurueckgewiesen (`loop.py:853`). Die Val-Variante besitzt ebenfalls die neue Pruefung. Kein Fehlerbeleg fuer still akzeptierte nichtendliche Endmetriken.
- Standard-Testscore-gesteuerte Reparaturen sind durch `allow_test_feedback=False` abgeschaltet. Die hier gezeigten Val-/Umgebungsluecken sind andere Mechanismen.
- Die Default-Auswahl stoppt bereits, wenn alle dynamisch ausgeschlossenen Kandidaten auf Val trainierten oder kein verwertbarer Val-Score vorliegt. Der Befund betrifft Faelle, die dieser Ausschlusslogik entgehen.
- Teilweise Seed-Messungen geben im Standardpfad kein Paperurteil aus. Die genaue Endmessungsauswahl ist fuer die bereits getesteten Einclaim-Faelle repariert; E22 betrifft Claim-Zuordnung.
- Dossier/Events werden bei einem spaeten Abbruch lokal gespeichert. R03 bedeutet keinen allgemeinen Verlust aller lokalen Belege.
- Ein reproduzierter MSE/MAE-Wert zertifiziert selbst bei sicherer Messung keine allgemeine Methodentreue. Mehrere Guards dokumentieren ihre begrenzte Erkennung bereits richtig.

## Verifikation dieses Untersuchungsstands

Die Python-Gegenproben vergleichen vor/nach Ausfuehrung die Hashes aller getrackten Dateien unter `src/`, `web/src/`, `tests/`, `benchmark/`, `deploy/` sowie des Dockerfiles: **unveraendert**. Ihre Ergebnisse bestaetigen 25 Szenarien. Die Frontend-Proben bestaetigen drei Szenarien. Die isolierte Umgebungsprobe bestaetigt ihren eigenen Integritaetsfall.

Zusaetzlich bestanden **100 bestehende Tests** aus Spec/Raw/Config, Val-Label/Validation-only, Source-Integrity, Method-/Checkpoint-Guard und finaler Messungszuordnung. Diese gruene Suite widerspricht den Befunden nicht: Die neuen Gegenproben decken zusaetzliche Eingaben oder Entscheidungen ab, die die bestehenden Tests nicht pruefen beziehungsweise bewusst als Heuristik zulassen. Die zuvor dokumentierte komplette Suite wurde ohne Produktaenderung nicht nochmals vollstaendig wiederholt.

Die Repro-Skripte sind bewusst **kein neuer gruener Produkt-Abnahmetest**: Ihre Assertions erwarten das gegenwaertige problematische Verhalten. Nach einer spaeter autorisierten Reparatur muessen die betreffenden Gegenbeispiele als Tests der gewuenschten Invariante umgeschrieben werden. Erst dann bestaetigt ein gruener Test die Reparatur.

## Reihenfolge fuer eine spaetere Fixphase

1. Vertrauenswuerdige Messung und kausale Forecast-Eingaben (E01/E02).
2. Echte, unabhaengige Validierung und belegte Kontrollversuche (E03-E06, Datenvertrag, E20).
3. Ausgefuehrtes Experiment und exakte Code-Version korrekt zuordnen (E07-E16).
4. Bekannte Guard-Varianten sowie Kosten-/Aufrufbeobachtung verbessern (E17-E19, E23, R02).
5. Claim-korrekte, terminale und dauerhaft nachvollziehbare Ergebnisdarstellung (E21/E22, R03); Diagnosepolitik R04 gesondert bewerten.

Diese Reihenfolge ist eine Bewertung mit Abnahmebedingungen, **keine begonnene Implementierung**. Kein SARAF-Zielwert wurde optimiert, keine Schutzpruefung abgeschwaecht und kein historisches Ergebnis umgeschrieben.
