# Deploy-Bereitschaft — 10. Oktober 2026

**Ergebnis: bereit fuer Coolify mit vorhandenen Ergebnissen und gesperrten Live-Laeufen. Die Kostenbremsen wurden nach diesem Erstcheck repariert und offline geprueft; vor Live-Freigabe Kontostaende und persistente Speicher abstimmen.** Das Deployment selbst uebernimmt der Nutzer. Kein bezahltes Experiment wurde fuer diese Pruefung gestartet.

## Behobene Produktfehler

- Katalog: Titel und freie Kartenflaeche oeffnen die veroeffentlichte Evidenz. Replay-, Code-, arXiv- und Live-Knoepfe behalten eigene Ziele; Eintraege ohne veroeffentlichten Lauf bekommen keinen erfundenen Evidenzlink.
- Startseite: das gemeinsame Beispiel und der 30-Sekunden-Knopf zeigen TSMixer `disc_2303_06053--20261009-201230`, mit aufgezeichneten Tavily-Aufrufen.
- „The full lesson“ zeigt auf `docs/evaluation_notes.md#our-own-small-bitter-lesson`. Der alte README-Anker existiert wieder und verweist auf diesen Abschnitt.
- PDF: der Druckknopf war unter der Produktions-CSP blockiert. Ein Ereignis-Listener ersetzt den Inline-Handler; die CSP bleibt unveraendert streng.

## Erstes Deployment in Coolify

1. Repository `Script-hpp/Nullius`, Branch `main`, Build Pack **Dockerfile**, Base Directory `/`, Dockerfile `/Dockerfile`, Port `8000`. Privates Repo ueber die GitHub-App bzw. einen Deploy Key anbinden.
2. Domain `https://nullius.akca.dev`, sofern dieser Hostname verwendet wird; DNS auf den Coolify-Host setzen. Diese Adresse ist durch diese Pruefung noch nicht als live bestaetigt.
3. Die Werte aus [coolify-preview.env.example](../deploy/coolify-preview.env.example) als **Runtime Environment Variables** setzen. Keine Build-Variablen. `MAX_DAILY_REPRO_JOBS=0` und `MAX_DAILY_JOBS=0` sperren beide bezahlten Start-Endpunkte. Ohne Budget/Stichtag sperrt Token Factory zusaetzlich mit `budget_floor`. Fuer diese Variante sind keine API-Schluessel erforderlich.
4. Benannte persistente Volumes: `nullius-data` nach `/data`, `nullius-logs` nach `/app/logs`, `nullius-runs` nach `/app/runs`. Nur **eine Instanz/ein Worker**. Bind-Mounts muessen fuer UID 10001 beschreibbar sein. Volumes beim Redeploy behalten.
5. Healthcheck GET `/api/health`, Port `8000`, Code `200`. Auto-Deploy aus; bewusst von Hand deployen. Die ausfuehrliche [Coolify-Anleitung](deploy.md#schritt-für-schritt-coolify) beschreibt Proxy und Healthcheck.
6. Nach Deployment Startseite, Katalog-Karte, TSMixer-Replay und PDF-Druckknopf pruefen. `/api/status` soll `live: false` melden. Kein Live-Lauf zum Testen erforderlich.

Ein leeres Datenvolume wird mit den Exporten aus dem Image befuellt. Ein bereits vorhandenes `index.json` wird beim Redeploy absichtlich nicht ersetzt: fuer spaeter hinzugefuegte Repo-Exporte ist dann ein gesondertes Zusammenfuehren der Daten noetig. Nicht das Volume loeschen; darin koennen neue Live-Ergebnisse liegen.

## Kostenpruefung

Vorhanden und durch Offline-Tests geprueft: Budget-/Tavily-Startschwellen, Tages- und IP-Limits, drei Seeds statt voller Laeufe, geteilte Job-Sperre, persistente Startzaehler, globale Budgetpruefung vor logischen Modellaufrufen sowie Buchung bezahlter leerer Antworten.

**Nachtraeglicher Fix:** Alle Modellrollen und Retries teilen jetzt ein API-Jobbudget; vor jedem Versuch werden Kosten reserviert und Gesamtbudget, Tagesbudget und Reserve geprueft. Discovery bekommt ein eigenes Modelllimit (Standard 0,25 $), Reproduktion bleibt bei 0,50 $. Unbekannte Modellpreise blockieren Anfragen. Tavily reserviert Credits vor jedem SDK-Versuch. Der Watchdog sperrt weitere Anbieteraufrufe und haelt den Jobplatz bis zum Worker-Ende. Kostenbuchungen werden vor Freigabe einer Reservierung dauerhaft geschrieben; unklare Anfragen behalten ihre Reservierung ueber Neustarts. Sandbox-/Hostingkosten und externe Kontonutzung bleiben ausserhalb der Inferenzschaetzung.

Beide Startlimits fuer das erste Preview-Deployment bei **0** lassen. Vor Live-Freigabe den [aktuellen Kostenreview](cost_policy_review.md#production-fix--10-oct-after-review) beachten und die echten Kontostaende/Reservierungen abstimmen. Die historischen 12-Dollar-/Tavily-Werte aus Beispieldateien sind kein verifizierter aktueller Kontostand.

## Validierung

- Produktions-Web-Build und Docker-Build erfolgreich; Container laeuft als UID 10001 und ohne `.env` im Image.
- 108 Frontend-Tests bestanden; 91 gezielte Backend-/Kosten-/API-/Deployment-Tests bestanden (76 plus 15 API-Schutztests). Modell-/Tavily-Aufrufe sind in den Tests ersetzt.
- Browserpruefung: TSMixer-Ziel und Film, Lesson-Link, Katalog-Titel und Kartenflaeche, eigenstaendige Replay-/arXiv-Ziele; keine Browserfehler und keine API-Starts.
- Produktionscontainer: Web, Daten, Healthcheck und CSP erreichbar; `live: false`; beide Start-Endpunkte liefern `503 budget_floor`; keine Job-/Kostenlogs angelegt.
- PDF-Druck unter echter Produktions-CSP vor dem Fix reproduziert und nach dem Fix erneut geprueft.

Provider-Rechnungen, reale Coolify-Proxy-/DNS-Konfiguration und Verfuegbarkeit nach dem oeffentlichen Deployment sind damit nicht verifiziert. Video-Original und separate Variante 2 werden von diesen Aenderungen nicht beruehrt.
