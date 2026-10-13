# FITS-Quellenabruf nach Produktionsstopp

## Beobachtung und Beweisgrenze

Der Produktionsauftrag `2d8719b4-f197-4824-a0d0-b312747d44e6` stoppte nach
ungefaehr 7,28 Sekunden vor Modellaufruf und Experiment: Die benoetigte
LaTeX-Quelle war nicht lesbar. Der alte Reader gab bei Downloadfehlern nur
leeren Text zurueck. Welcher HTTP-Status oder Verbindungsfehler damals
vorlag, ist deshalb nicht aus dem Ereignisprotokoll belegbar.

Kostenfreie lokale Abrufe der offiziellen FITS-Endpunkte waren erfolgreich.
Die erneut gelesene v3-Quelle hat denselben Text-Hash wie der erste
Produktionslauf:
`9022b12b628fc1807f0ef11bfd7d1266f25b8575c1418afeb7bb90c251a00d64`.
Dieser Befund zeigt einen Quellenzugriffsfehler, keine neue Aussage ueber
die Reproduzierbarkeit des Papers.

## Umsetzung

- Hoechstens zwei Versuche am Export-Endpunkt und ein Versuch am zweiten
  offiziellen arXiv-Endpunkt, jeweils 30 Sekunden Timeout. Permanente
  HTTP-4xx werden am selben Host nicht wiederholt; HTML-Fehlerseiten gelten
  nicht als LaTeX. Die tatsaechliche Download-URL wird gespeichert.
- Erfolgreiche Quellen werden atomar unter `logs/paper_sources/` gespeichert.
  Das bestehende persistente `/app/logs`-Volume erhaelt sie ueber Deployments.
  Live-Aufrufe schreiben keine Benchmark-Fixtures um.
- Bei Downloadausfall sind gespeicherte Quellen nur mit passender Paper-ID,
  Version, Quellenart und nachgerechneter SHA-256 zulaessig. Ein vorhandener
  Discovery-Hash muss ebenfalls passen. Eine unbekannte aktuelle Version
  wird offengelegt; gespeicherte Versionsnummern werden numerisch sortiert.
  Frisch lesbare Quellen haben Vorrang vor dem Cache.
- FITS ist auf v3 samt Quellenhash und Haupttabelle `tab:FITSett` gebunden.
  Das Produktionsimage enthaelt eine komprimierte, unveraenderte primaere
  Quellenreserve, erneut abgerufen am `2026-10-13T16:29:12+00:00` von
  `https://export.arxiv.org/src/2307.03756v3`. Attribution und CC-BY-4.0-Lizenz
  stehen in `benchmark/source_reserves/README.md`. Sie enthaelt weder
  Modellantworten noch Experimentmessungen.
- Cache-Nutzung erscheint als Warnung und `from_cache` im Quellenereignis,
  mit urspruenglichem Abrufdatum, Version und letztem Downloadfehler. Ohne
  passende Quelle bleiben strikte LaTeX-Claims vor Experiment und bezahltem
  Formatwechsel gesperrt; Stoppmeldung und Ereignis nennen jetzt den
  HTTP-Status oder die Fehlerklasse.

Zielwert 0,372, Tabellen-, Methoden- und Protokollpruefungen bleiben aktiv.
Die gespeicherte Quelle ersetzt ausschliesslich den ausgefallenen Download.

## Validierung

- Fokussierte Quellen-/Katalog-/API-Pruefungen: **265 passed**.
- Backend ohne Integration und die gesonderte Harness-Baseline-Suite:
  **1524 passed, 9 skipped, 8 deselected, 5 xfailed**. Befehl:
  `uv run pytest -q -m 'not integration' --ignore=tests/test_harness_baselines.py --tb=short`.
- Unabhaengige Opus-Schlusspruefung: **238 passed**, kein Blocker.
  Bericht: `docs/opus_fits_target_binding_catalog_review_20261013.md`,
  Nachtrag 3; Proben unter `docs/audits/opus_source_cache_probe_final_*`.
- Vollstaendiges Produktionsimage `nullius:source-cache-20261013` gebaut.
  Echter `Orchestrator.resolve_targets` darin mit `--network none` und
  verbotenem Modellaufruf ausgefuehrt: alle drei echten Downloads scheitern
  mit `ConnectError`; Reserve wird verwendet, v3/Hash bestaetigt,
  **0,372 aus `tab:FITSett`** erneut belegt, Cache-Warnung und Quellenereignis
  vorhanden, **0 Modell-/Tavily-Aufrufe**.
- Regressionen sperren falsche ID/Version/Quellenart, geaenderten Cache-Text
  auch mit neu berechnetem Hash, fehlerhafte Dateinamen und kaputtes JSON.
  Ein frischer veraenderter Tabellenwert wird weiter vor Experiment blockiert.

**Neue Providerkosten: 0 Dollar.** Kein neuer Produktionsauftrag gestartet.

## Verbleibende Grenzen und Deployment

Nur FITS besitzt eine mitgelieferte Quellenreserve. Andere Papers profitieren
von gespeicherten erfolgreichen Abrufen; ohne passende gespeicherte Quelle
und lesbaren Download bleiben sie gesperrt. Der konkrete vergangene
Serverfehler kann nicht nachtraeglich rekonstruiert werden.

Dieser Nachweis betrifft Quellenverfuegbarkeit und Zielpruefung, keinen
autonomen Trainingslauf und keine Nebius-Isolationsabnahme. Nach Deployment
kann der Nutzer FITS einmal neu starten. Der bereits gestoppte Auftrag
bleibt als historischer Fehlversuch erhalten.
