# Allgemeine Zuverlaessigkeitsfixes und Abnahme

Stand: **2026-10-13**. Ausgangsbasis `e945e01c2e02ddf60635bf44b31698f859bb9cf9`.
Auftrag: nach dem Untersuchungsteil die bestaetigten Ursachen mit Regressionstests beheben.

## Ergebnis und Beweisgrenze

Die Ausgangsbefunde stehen im [Erstaudit](general_reliability_audit_20261013.md). Der vom Nutzer beauftragte
Opus-Analyst hat alle E01-E23 unabhaengig nachvollzogen, Reichweiten korrigiert und N01-N03 ergaenzt:
[Gegenpruefung](opus_general_audit_review_20261013.md). Seine Quellen-/Parameterfixes und Tests wurden
integriert. Seine [Review der ersten Umsetzung](opus_ceo_change_review_20261013.md) dokumentiert den
damaligen Zwischenstand; die nachfolgenden Aenderungen sind unten getrennt beschrieben.
Die [abschliessende Forecast-Review](opus_forecast_contract_final_review_20261013.md) bestaetigt
mit 31 bestandenen Tests, dass die Nachbesserungen lokal integrierbar sind.

Die Abnahme verwendet lokale Tests, simulierte Provider und echte Container ohne Netzwerk fuer die
Ausfuehrung von generiertem Code. Keine neuen Modell-, Tavily-, Nebius- oder Produktionsauftraege.
Die gespeicherten historischen Scores, Katalogeintraege und Benchmarkdaten bleiben unveraendert.
Der neue Produktvertrag ist **kein Nachweis, dass SARAF jetzt den Paper-Zielwert erreicht**.

## Umsetzung je Befund

| Befund | Ursachenfix | Nachweis |
| --- | --- | --- |
| E01 | Test- und Val-Harness in frischem Branch desselben vertrauenswuerdigen Setup-Snapshots; nur begrenzte CSV-Artefakte uebertragen. Genau ein Harness-JSON akzeptieren. | `test_trusted_evaluation`: echte Docker-Gegenprobe, Agent-Startup-Hook veraendert weder Test- noch Val-Wert. |
| E02 | ETT-Produktcode laeuft als unprivilegierter Worker. Vollstaendige Testhistorien und verborgene Auswahlziele bleiben beim vertrauenswuerdigen Broker. Eine Vorhersage wird vor der naechsten Historie festgeschrieben. | `test_selection_contract`: keine Testhistorien beim Fit, private NPZ nicht lesbar, echte Repeat-Vorhersagen. |
| E03/E04 | Letzte 20 % der bisherigen Validierungsdaten fuer Kandidatenauswahl zurueckhalten. Graph-Labels dort -1; ETT-Fit sieht nur Train und den sichtbaren Val-Praefix. Direktes Kopieren von Labels/Val-Fit statisch erkennen. | `test_selection_contract`, `test_final_admission`: kopierte Labels koennen Auswahl nicht gewinnen. |
| E05/E06 | Fehler/Timeout bei Kontrollvorbereitung machen den Versuch ungueltig. Nicht messbare Maskenkontrolle schliesst Kandidaten im Produkt aus; keine bestandene Ablation erfinden. | `test_trusted_evaluation`, `test_final_admission`, bestehende Ablations-/Val-Tests. |
| E07-E10 | Trainingsparameter an Modell, Datensatz, Horizont und Aufrufzweck binden; Inferenz-/Retrieval-Batch und inaktive Parserdefaults nicht als Trainingsvorgaben erzwingen. | Opus-Regressionstests in `test_spec`, `test_spec_check`, Quellenkontexttests. |
| E11/E12 | Weitere CONFIG-Mutationen erkennen; Paper-Belege fuer Konfliktaufloesung an Parameternamen statt globalem Zahlenvorkommen binden. | Opus-Tests zu Aliases, Reassignment, AugAssign, setitem und passenden Belegen. |
| E13/N03 | Spec-Score fuer finalen Codehash berechnen. Jede Ersatzversion und der Endhash vor den restlichen Seeds erneut zur Validierung zulassen. | `test_final_admission`: Crash-/Qualitaetsreparatur verliert stale Spec-Score; Repo-Ersatz mit Val-Training bleibt ausgeschlossen. |
| E14 | Doppelte Graph-IDs und malformed CSV-Zeilen vor Dictionary-Konvertierung ablehnen. | `test_trusted_evaluation` plus bestehende Harness-Tests. |
| E15 | Transitive Modell-/Trainingsabhaengigkeiten lesen; fehlende erforderliche Quelldateien sichtbar stoppen. | Opus-Quellenkontext-/Repo-Tests. |
| E16 | Jeden ausdruecklichen Ref einschliesslich main/master auschecken. Nur HEAD bezeichnet Default; fehlender Ref erzeugt Setupfehler ohne stillen zweiten Klon. | Repo-Default master gegen explizites main, Discovery-Linktests. |
| E17/E18 | Bekannte Retrieval-Platzhalter und flache Checkpoint-Muster breiter erkennen. | Opus-Methoden-/Checkpoint-Regressionstests. Heuristiken bleiben begrenzt. |
| E19 | Kommentare/Docstrings nicht als ausgefuehrte Netzwerkoperation behandeln; bekannte lokale Dataset-Aufrufe korrekt behandeln. | Opus-Code-Guard-Tests mit positiven und negativen Kontrollen. |
| E20 | Einzelkandidat ebenso vor weiteren Seeds zulassen. | `test_final_admission::test_single_candidate_is_checked_before_any_other_seed`. |
| E21/E22 | Gestoppte Kandidatenauswahl nicht als laufend darstellen. Replay, Timeline und StepGraph auf denselben Claim begrenzen. | Frontendtests fuer terminale Auswahl und mehrteilige Claim-Ereignisse. |
| E23 | Vollstaendige LLM-Antworten zentral im Client zaehlen, direkte `complete`-Aufrufe eingeschlossen; `_ask` nicht doppelt zaehlen. | `test_observability_and_archives`, Frontendlabel `Nemotron responses`. |
| R01 | SHA-256 der Rohdaten bei Vorbereitung festhalten und bei unabhaengiger Bewertung abgleichen. Produkt-Abhaengigkeiten pinnen; Snapshot-Schluessel enthaelt Setup und Auswahlvertrag. | `test_data_identity_and_candidate_budget`: veraenderte Rohdaten abgelehnt. |
| R02 | Im Produkt Kandidat erzeugen und validieren, bevor der naechste erzeugt wird. Bei Generierungsbudgetende nur bereits unabhaengig validierte Kandidaten nutzen. Billing-/Cancellation-Abbruch bleibt Abbruch. | `test_data_identity_and_candidate_budget`: Reihenfolge, validierter Rueckfall, kein Rueckfall ohne gueltigen Kandidaten. |
| R03 | Sanitisierte terminale Reproduktionsereignisse atomar unter `logs/reproduction_jobs/<UUID>.json.gz` speichern; Stream nach Registry-Pruning/Neustart wieder laden. | `test_observability_and_archives`: gespeicherter Fehler erreichbar, defekte Archive/Pfade abgelehnt. |
| R04 | Baseline-Fehlerheuristik fuer das Paper-Zahlurteil nur auf dessen Claim-Metrik anwenden. Sekundaermetriken weiter zeigen. | MSE trifft trotz legitimer MAE-Abweichung: `test_final_admission`; angepasste bestehende Fault-Tests. |
| N01/N02 | LocalSandbox legt Zielverzeichnisse an und trennt vor Agentenstart das Docker-Netz, ohne SYS_ADMIN fuer unshare zu brauchen. Eigene Branches ueber API-/CLI-finally entfernen. | Echte netzlose Docker-Pipeline; CLI-Ressourcen auch bei fehlgeschlagenem Lauf geschlossen. |

## Nachbesserungen aus der Opus-Review

- Alte Snapshots ohne Rohdatenmanifest werden durch einen neuen Setup-/Vertragsschluessel nicht still wiederverwendet.
  Fehlende Datenidentitaet im neuen Produktvertrag stoppt als Infrastrukturproblem.
- Der Forecast-Broker wiederholt acht zufaellig ausgewaehlte bisherige Historien. Abweichung ueber 1e-4 macht den Versuch
  ungueltig. Dies erkennt zustandsabhaengige Inferenz in der Gegenprobe; es zertifiziert nicht beliebigen Modellcode.
- `/output` gehoert ausschliesslich dem Broker (0700). Der Worker kann aufgezeichnete Vorhersagen weder ueberschreiben
  noch durch andere Dateien ersetzen, nachdem spaetere Historien bekannt werden. Checkpoints sind unter
  `/tmp/forecast-agent` moeglich. Docker-Gegenprobe verlangt PermissionError beim CSV-Schreibversuch.
- Repeat-/Zero-Baselines werden auf denselben verborgenen ETT-Val-Fenstern gerechnet, nicht abgeschaltet.
- Zeugnis nennt Auswahlvertrag und Stichprobengroesse: ETTh1 **21 von 117 Val-Fenstern**, Cora **100 von 500 Val-Knoten**.
- Vorhersageuebertragungen einschliesslich Smoke sind begrenzt. Docker nutzt Archive-Metadaten vor dem Lesen;
  Nebius begrenzt komprimierte Groesse und Dekompression. Mock-Gegenproben pruefen uebergrosse Metadaten und stark
  komprimierte uebergrosse Inhalte ohne Providerzugriff.
- Eine menschlich geschriebene GCN-Referenz wurde im alten und neuen Orchestrierungspfad in getrennten netzlosen
  Containern ausgefuehrt: **exakt gleiche Vorhersage-CSV und gleicher Wert derselben synthetischen Scoring-Oracle**.
  Das prueft den Transport-/Laufzeitwechsel, erzeugt aber keinen neuen Cora-Benchmarkscore.

## Neuer Produktvertrag und Betrieb

`product_kwargs()` aktiviert `independent_selection=True`. Historische Experiment-/Benchmarkaufrufe mit direktem
`Orchestrator(...)` behalten ihren Datenvertrag; saubere Bewertungsbranches und strenges JSON gelten auch dort.
Das Zeugnis neuer Produktlaeufe nennt `withheld_validation_v1`, fuer ETT zusaetzlich `causal_stream_v1`.

ETT-Code implementiert jetzt `train_forecaster(data, seed, smoke) -> predictor(history)`. Er schreibt keine
Vorhersage-CSV. `nullius_io.forecast_data(length)` liefert benannte Eingaben und **echte, passend verschobene Ziele**
fuer Train/sichtbare Val. Originale Train-/Testgrenzen und die Testmetrik bleiben erhalten; die interne
Auswahlstichprobe und der Inferenzvertrag aendern sich. Alte Katalogwerte sind deshalb kein Beweis fuer den neuen Pfad.

**Nebius-Preflight offen:** ETT benoetigt `setpriv`, chown und das wirksame Abgeben aller Capabilities. Lokal im
Task-Image belegt, im Nebius-Backend hier nicht live ausgefuehrt. `_prepare_selection_contract` prueft die
UID-Abgabe und den fehlenden Lesezugriff auf private Daten vor Agentenstart und stoppt sonst als
Infrastrukturfehler. Bevor ein ETT-Produktlauf als einsatzbereit gilt, muss dieser Preflight in der realen
Nebius-Umgebung funktionieren. Ohne diese Rechte braucht es dort einen kompatiblen Isolationsmechanismus;
die Schutzschicht darf nicht abgeschaltet werden, um einen Score zu erhalten.

`logs/` weiter als persistentes Volume behalten. Darin liegen neben Budget-/Tageszaehlern nun terminale
Jobarchive. Keine Umstellung vorhandener Limits noetig. Dieses Dokument beschreibt lokale Aenderungen,
keinen Deployment- oder neuen Live-Reproduktionsnachweis.

## Grenzen

Statische Guards sind Musterpruefungen, keine allgemeine Methodentreue-Zertifizierung. Dynamische CONFIG-
Aenderungen innerhalb von Trainingsloops bleiben unter anderem wegen erlaubter Schedules nicht allgemein
verboten. Graph-Agenten bleiben privilegiert im eigenen, von der Bewertung getrennten Branch; dort duerfen
keine verborgenen Labels/Rohdateien verbleiben. Die kleiner gewordene Auswahlmenge erhoeht Unsicherheit;
es wurde weder auf dem Testscore optimiert noch ein Paper-Erfolg zugesagt. Sichtbare Val fuer Training
bleibt im Forecast-Prompt untersagt, obwohl sie strukturell von der verborgenen Auswahl getrennt ist.

Die Audit-Proben unter `docs/audits/` sind eingefrorene Nachweise des **alten** Verhaltens. Sie sollten
nicht als heutige Abnahmetests ausgefuehrt oder ihre JSON-Ergebnisse ueberschrieben werden.

Fensterweise Inferenz ersetzt Batching: fuer ETTh1 2910 Aufrufe einschliesslich Val und Wiederholungsproben.
Opus mass lokal etwa 7.5 Sekunden reine JSON-Pipe-Kosten; schwere Retrieval-Modelle koennen mit Training
und Einzelinferenz trotzdem das 600-Sekunden-Limit erreichen. Ein Timeout ist dann keine Widerlegung
des Paperclaims. ETT-Abdeckungsmessung ist unter dem Broker deaktiviert. Direkte native stdout-Ausgaben
koennen das JSON-Zeilenprotokoll stoeren und stoppen den Versuch statt einen Score durchzulassen.
Diese Grenzen sind keine offenen Belege fuer eine falsche Messung, bleiben aber fuer Live-Tests relevant.

## Abnahme

| Pruefung | Ergebnis |
| --- | --- |
| Fokussierte Backend-Suite fuer Quellen, Parameter, Tasks und Admission | 258 passed, 1 skipped |
| `NULLIUS_DOCKER_TESTS=1 uv run pytest -q tests/test_selection_contract.py tests/test_trusted_evaluation.py tests/test_reference_evaluation_regression.py` | 19 passed; netzlose Container, Referenzvergleich eingeschlossen |
| Abschliessende komplette Backend-Suite ohne Provider-Integration | 1456 passed, 8 skipped, 8 deselected, 5 xfailed; 307.85 Sekunden |
| `cd web && npm test` | 153 passed |
| `cd web && npm run build` | bestanden; bestehender Hinweis zur Chunk-Groesse |
| `docker build -t nullius:reliability-20261013 .` | bestanden |
| Produktionscontainer ohne Netzwerk: `/api/health` und `/` intern ueber HTTP | jeweils 200; Container danach entfernt |
| `git diff --check`; keine Aenderungen an `benchmark/`, `web/public/`, `deploy/` | bestanden |

Docker-Gegenproben benoetigen die bereits vorhandenen lokalen Images `python:3.11-slim`,
`repro-snapshot:gate-etth1` und `repro-snapshot:gate-cora`. Ohne die explizite Variable
`NULLIUS_DOCKER_TESTS=1` werden diese Containerpruefungen uebersprungen. Die Gesamtsuite verwendet
`uv run pytest -q -m 'not integration' --ignore=tests/test_harness_baselines.py`;
Provider-Integrationstests sind ausdruecklich keine bestandene Live-Abnahme.

Der abschliessende Gesamtlauf fand auf eingefrorenem Quellstand statt; anschliessend wurden die
SHA-256-Hashes aller 365 Python-/TypeScript-Quell- und Testdateien unveraendert bestaetigt.
Bestehende Warnungen: Starlette/httpx-TestClient-Deprecation und Vite-Chunk-Groesse.
Ein vorheriger Gesamtlauf hatte eine veraltete Task-Attrappe im Katalogtest; diese bildet nun
die neuen Vertragsfelder ab. Ein weiterer Quelltextlesetest scheiterte an einer waehrend
des Laufs geaenderten Kommentar-Zeilenposition. Der eingefrorene Abschlusslauf ist fehlerfrei.
