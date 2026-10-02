# Deployment: Demo-URL bis mindestens 15.12.2026

Official Rules: Demo muss bis 15.12.2026 erreichbar und kostenlos testbar sein (Bewertung 01.–15.12.). Zwei Schichten: **Live-Server** (Container mit API und Web-App) und **statischer Spiegel** auf GitHub Pages als Absicherung (nur Exporte, Replay, keine Live-Steuerung). Stand der Preise: Nebius-Preisseite am 2026-10-01 abgerufen; Werte mit **[zu prüfen]** vor der Buchung gegenlesen.

## Was der Dienst braucht
- **Ein Prozess, ein Job gleichzeitig:** Jobs, Warteschlange und Semaphor liegen im Speicher der API (`api.py`). Genau **ein** uvicorn-Worker, kein Autoscaling auf mehrere Instanzen.
- **Lange Läufe:** Reproduktionsläufe bis etwa 20 Minuten, Schnellprüfung (3 Seeds) kürzer. Der Live-Stream (SSE, `/api/reproduce/{id}/stream`) bleibt so lange offen (Keepalive alle 15 s). Proxy ohne Pufferung und mit Lese-Timeout über 20 Minuten.
- **Persistenz:** Exporte neuer Läufe (`REPRO_EXPORT_DIR`, im Container `/data`: `index.json`, `catalog.json`, Laufordner) und `logs/costs.jsonl` (Budgetzähler: `BUDGET_TOTAL_USD` zählt aus diesem Log; ein leeres Log nach Neustart ließe das Budget zurückspringen). Beides als Volume.
- **Ausgehend erreichbar:** Token Factory (`api.tokenfactory.nebius.com`), Tavily, Nebius Sandboxes (`contree-sdk`), arXiv/GitHub für Quellen. Keine eingehenden Verbindungen außer 80/443.
- **Schlüssel nur als Umgebungsvariablen** (`deploy/.env`, nie im Image und nicht in Git). Zugang für Live-Läufe nur mit `ACCESS_CODE`; die Web-App ohne Code ist lesend (Replay/Evidence).
- **Klein:** Die Berechnung läuft in der Nebius-Sandbox, nicht im Container. 2 vCPU und 4 bis 8 GB RAM genügen; Plattenbedarf gering (Image unter 1 GB, Exporte einige MB).

## Optionen im Vergleich

| Option | Passt zu den Anforderungen | Kosten bis 15.12. (grob) | Nachteile |
|---|---|---|---|
| **Coolify** (eigener Server des Nutzers, Traefik-Proxy) | Dockerfile-Build aus dem Repo, persistente Volumes, Domain mit automatischem TLS, Healthcheck, Logs und Neustart in der Oberfläche; ein Container, kein Autoscaling | **0 $ zusätzlich** (Server und Domain vorhanden) | Nicht Nebius, zahlt also nicht auf „weitere Nebius-Dienste“ im README ein (Sandboxes und Token Factory genügen den Pflichtangaben); Verfügbarkeit hängt vom eigenen Server ab; Proxy-Einstellungen für SSE prüfen |
| **Nebius Compute (VM)**, CPU-VM mit Docker (Alternative) | Volle Kontrolle: ein Prozess, Volumes, kein Request-Timeout, SSE ohne Limit. Zählt als **weiterer Nebius-Dienst** fürs README (Pflicht: „welche weiteren Nebius-Dienste genutzt werden“) | Intel Ice Lake ab 0,06 $/h (ab 1.10.2026, vorher 0,05 $/h) **[zu prüfen: kleinster Preset 2 vCPU/8 GB]**: rund 76 Tage × 24 h ≈ **110 $**; AMD Genoa ab 0,13 $/h ≈ 240 $. Block-Volume 0,053 $/GB/Monat (30 GB ≈ 4 $), öffentliche IP und Traffic laut Preisseite kostenlos. Bezahlung aus Guthaben/Karte: **Account klärt der Reviewer mit dem Nutzer** | Selbst betreiben (Updates, TLS, Neustart); dauerhaft laufende Kosten; Builders-Credits gelten vermutlich nur für Token Factory **[zu prüfen]** |
| **Nebius Serverless** (AI Cloud, Endpoints/Jobs) | Container-Start ohne VM-Pflege | **[zu prüfen]**, nicht recherchiert | Skalierung auf null und Request-Limits passen schlecht zu 20-Minuten-SSE und In-Memory-Jobs; Volumes/Persistenz unklar. Nicht empfohlen ohne Test |
| **Kleine VM eines anderen Anbieters** (z. B. Hetzner CX22, etwa 4 bis 5 €/Monat **[zu prüfen]**) | Gleiches Setup wie Nebius-VM (Docker + Caddy), am günstigsten (≈ 10 bis 15 € bis 15.12.) | ≈ 12 € | Zahlt **nicht** auf „Nebius-Dienste“ im README ein; zusätzlicher Anbieter-Account |
| **Fly.io / Render o. ä.** (PaaS) | Einfaches Deployment, TLS inklusive; Fly mit Volume und `auto_stop_machines = false` möglich | Fly: kleine Maschine mit Volume grob 5 bis 10 $/Monat **[zu prüfen]**; Render: Free-Tarife schlafen ein (nicht geeignet), bezahlt ab etwa 7 $/Monat **[zu prüfen]** | Schlafende Instanzen brechen Jobs ab; Proxy-Timeouts der Plattform prüfen; wieder nicht Nebius |
| **Nur GitHub Pages** | Kostenlos, stabil | 0 | Keine Live-Läufe; erfüllt „kostenlos testbar“ nur als Replay. Bleibt als Spiegel |

## Empfehlung
1. **Live-Server auf dem vorhandenen Coolify-Host des Nutzers** (Dockerfile-Build aus dem Repo, eigene Subdomain): keine Zusatzkosten, Schlüssel bleiben in Coolify-Umgebungsvariablen, Anleitung im nächsten Abschnitt.
2. **GitHub-Pages-Spiegel** parallel (Workflow `.github/workflows/pages.yml`), Link im README als Rückfall.
3. Rückfall bei Ausfall des Servers oder des Budgets: Spiegel bleibt erreichbar, das Frontend spielt die Läufe als Replay ab (Hinweis sichtbar).
4. **Alternative: Nebius-Compute-VM**, nur wenn Nebius-Guthaben für Compute gilt (Account klärt der Reviewer mit dem Nutzer) und die Nennung als weiterer Nebius-Dienst im README gewünscht ist; Schritte im Abschnitt „Nebius-VM“ weiter unten. Andere kleine VMs funktionieren gleich.

## Schritt für Schritt: Coolify

Voraussetzungen: Coolify-Instanz mit Zugriff auf das GitHub-Repo (GitHub-App oder Deploy Key), Subdomain des Nutzers (hier `nullius.example.com`), der Server erlaubt ausgehende Verbindungen (Token Factory, Tavily, Nebius Sandboxes, arXiv, GitHub). Coolify baut aus einem frischen Git-Checkout: lokale, nicht committete Dateien gelangen nicht ins Image (beim Bauen auf dem eigenen Rechner wäre das anders).

1. **DNS:** A-Record `nullius.example.com` auf die öffentliche IP des Coolify-Servers (oder CNAME auf dessen Hostnamen). Liegt die Domain bei Cloudflare: Wolke **grau** (nur DNS), nicht proxied. Der Cloudflare-Proxy schneidet stille Verbindungen nach etwa 100 s ab und kann SSE puffern; mit Keepalive alle 15 s klappt es meist, ist aber ein unnötiges Risiko.
2. **Ressource anlegen:** Project > Environment > **+ New Resource > Public/Private Repository (GitHub)** > Repo `Script-hpp/Nullius`, Branch `main`. **Build Pack: Dockerfile**, Base Directory `/`, Dockerfile Location `/Dockerfile`.
3. **Ports und Domain:** *Ports Exposes* = `8000`. Unter *Domains* `https://nullius.example.com` eintragen. Coolify legt Traefik-Labels an und holt das Let's-Encrypt-Zertifikat, sobald DNS stimmt (Ports 80 und 443 des Servers offen).
4. **Umgebungsvariablen** (Configuration > Environment Variables; Namen wie in `deploy/.env.example`, **Werte nur hier**): `LLM_PROVIDER=tokenfactory`, `SANDBOX_BACKEND=nebius`, `TAVILY_MODE=live`, `NEBIUS_API_KEY`, `NEBIUS_PROJECT_ID`, `TF_MODEL_FAST`, `TF_MODEL_CODE`, `TF_MODEL_REASONING`, `TAVILY_API_KEY`, `ACCESS_CODE`, `MAX_DAILY_JOBS`, `MAX_DAILY_REPRO_JOBS`, `MAX_COST_PER_RUN_USD`, `VAT_RATE`, `BUDGET_TOTAL_USD`, `BUDGET_SINCE`, `MAX_ATTEMPTS_PER_CLAIM`, `REPRO_EXPORT_DIR=/data` (steht schon im Image). **„Build Variable“ nicht ankreuzen**: Schlüssel dürfen nie in den Build. `SITE_ADDRESS` und `CADDYFILE` werden nicht gebraucht (nur für `docker-compose.yml` mit Caddy).
5. **Persistente Speicher** (Configuration > Persistent Storage > Add, Typ *Volume*): `nullius-data` → `/data`, `nullius-logs` → `/app/logs`, `nullius-runs` → `/app/runs`. Benannte Volumes übernehmen beim ersten Start Inhalt und Besitzer aus dem Image (Nutzer `app`, UID 10001); bei Bind-Mounts auf dem Host müsste `chown 10001` gesetzt werden. `/app/logs` enthält `costs.jsonl`, den Budgetzähler: nie löschen, sonst springt das Budget zurück.
6. **Healthcheck** (Configuration > Healthcheck, aktivieren): Methode GET, Schema http, Host `localhost`, Port `8000`, Pfad `/api/jobs/current`, Rückgabecode `200`, Intervall 30 s, Timeout 5 s, Retries 3, Start-Zeitraum 20 s. (Das Dockerfile hat einen eigenen `HEALTHCHECK` mit demselben Pfad; Coolify wertet seine Einstellung aus.) **Nur eine Instanz** betreiben (keine Replikate): Jobs und Semaphor liegen im Speicher der API.
7. **Auto-Deploy ausschalten** (Configuration > Advanced: *Auto Deploy* aus, Webhooks nicht einrichten). Ein Redeploy beendet den laufenden Container und damit einen laufenden Live-Lauf; während der Bewertung (01.–15.12.) nur von Hand und nicht während eines Laufs deployen. Mit Coolify-Rolling-Update wartet der alte Container auf den Healthcheck des neuen, ein aktiver Job geht trotzdem verloren (Speicher).
8. **Speicher begrenzen** (Advanced > Resource Limits): RAM 1 bis 2 GB, CPU 1 bis 2 (die Berechnung läuft in der Nebius-Sandbox, der Container ist leicht).
9. **Deploy** starten und das Build-Log beobachten (Node-Stufe baut `web/`, Python-Stufe installiert mit `uv sync --frozen`). Danach ist der Status *Running (healthy)*.
10. **Proxy: SSE ohne Puffer, lange Timeouts.** Traefik puffert Antworten nicht und gibt Streams unverändert weiter; Standardwerte von Coolifys Traefik passen meist: keine Lese-/Schreib-Timeouts auf dem HTTP-Entrypoint (`readTimeout`/`writeTimeout` 0), `idleTimeout` 180 s betrifft nur ruhende Verbindungen, der Stream schickt alle 15 s einen Keepalive **[zu prüfen an der echten Instanz]**. Zu beachten:
    - Keine Kompression oder Puffer-Middleware für `/api/*` hinzufügen (Coolify-Option „gzip/compress“ an der Ressource aus lassen, oder `text/event-stream` ausnehmen).
    - Steht ein weiterer Proxy davor (Cloudflare, Nginx): dort Pufferung aus (`proxy_buffering off`, `X-Accel-Buffering` kommt von der App nicht), Lese-Timeout mindestens 30 min.
    - Falls ein Timeout den Stream vor 20 Minuten trennt: Traefik-Entrypoint-Timeouts in den Coolify-Proxy-Einstellungen (Servers > Proxy > Konfiguration) anheben (`--entrypoints.https.transport.respondingTimeouts.idleTimeout=1800s`), danach Proxy neu starten.
11. **Test von außen:** `curl -s https://nullius.example.com/api/jobs/current` liefert `{"busy":false}`. Seite im Browser öffnen (Overview, Catalog). Dann als **Schnellprüfung** einen Live-Lauf mit dem Zugangscode starten und die Stream-Anzeige bis zum Ende beobachten (kostet wenige Cent). Probe für den Proxy, ohne Kosten: `curl -N -m 60 https://nullius.example.com/api/reproduce/<job>/stream` während eines Laufs zeigt die Ereignisse zeitnah, nicht erst am Ende.
12. **Eintragen:** Demo-URL `https://nullius.example.com` und Spiegel-Link in Devpost-Testhinweise und README; `ACCESS_CODE` **nur** im Devpost-Testhinweisfeld. Spiegel aktivieren: GitHub > Settings > Pages > Source „GitHub Actions“, Workflow „Pages mirror“ einmal manuell starten (`https://script-hpp.github.io/Nullius/`).

## Alternative: Nebius-VM mit Docker und Caddy

Voraussetzungen: Nebius-Projekt mit Compute-Zugang, `nebius` CLI angemeldet (`nebius profile create`), SSH-Schlüssel, Repo-Stand auf `main` (sauberer Checkout: das Image enthält sonst lokale, nicht committete Dateien unter `web/public/data`).

1. **VM anlegen** (CPU, Ubuntu 24.04; Plattform und Preset vor dem Anlegen mit `nebius compute platform list` nachsehen **[zu prüfen]**, Subnetz-ID mit `nebius vpc subnet list`):
   ```bash
   export USER_DATA=$(jq -Rrs '.' <<EOF
   #cloud-config
   users:
     - name: nullius
       sudo: ALL=(ALL) NOPASSWD:ALL
       shell: /bin/bash
       ssh_authorized_keys:
         - $(cat ~/.ssh/id_ed25519.pub)
   EOF
   )
   nebius compute instance create \
     --name nullius-demo \
     --resources-platform <platform> --resources-preset <preset> \
     --boot-disk-managed-disk-name nullius-boot \
     --boot-disk-managed-disk-type network_ssd \
     --boot-disk-managed-disk-size-gibibytes 30 \
     --boot-disk-managed-disk-source-image-family-image-family ubuntu24.04-driverless \
     --boot-disk-attach-mode READ_WRITE \
     --cloud-init-user-data "$USER_DATA" \
     --network-interfaces '[{"name": "eth0", "ip_address": {}, "public_ip_address": {}, "subnet_id": "<subnet_ID>"}]'
   ```
   Die öffentliche IP ist dynamisch: nach dem Anlegen notieren (`nebius compute instance get ...`). Für eine stabile Adresse eine statische Adresse reservieren **[zu prüfen]**, sonst ändert sich der Hostname nach einem Neuanlegen.
2. **Firewall/Security Group:** eingehend nur TCP 22 (besser nur die eigene IP), 80 und 443. Port 8000 nicht öffnen (der Container ist nur intern über Caddy erreichbar).
3. **Docker installieren** (per SSH):
   ```bash
   ssh nullius@<IP>
   curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $USER && exit   # neu einloggen
   ```
4. **Repo holen und Konfiguration anlegen:**
   ```bash
   git clone https://github.com/Script-hpp/Nullius.git && cd Nullius
   cp deploy/.env.example deploy/.env && chmod 600 deploy/.env
   nano deploy/.env     # alle Werte eintragen (Schlüssel, ACCESS_CODE, Limits, BUDGET_TOTAL_USD und BUDGET_SINCE aus der Konsole)
   ```
5. **Hostname und TLS wählen** (keine eigene Domain):
   - **Variante A, empfohlen: sslip.io-Hostname** mit Let's Encrypt. Die IP `203.0.113.7` wird zu `203-0-113-7.sslip.io` (löst ohne Einrichtung auf die IP auf; nip.io funktioniert gleich: `203.0.113.7.nip.io`). In `deploy/.env` setzen: `SITE_ADDRESS=203-0-113-7.sslip.io`, `CADDYFILE=Caddyfile`. Ports 80 und 443 müssen aus dem Internet erreichbar sein (HTTP-Challenge). **Risiko:** sslip.io und nip.io sind gemeinsame Domains; Let's-Encrypt-Limits pro registrierter Domain können die Ausstellung zeitweise blockieren, dann hilft Wiederholen nach einigen Stunden (Caddy versucht ZeroSSL als Zweitaussteller). Zertifikat früh ausstellen lassen, nicht am letzten Tag.
   - **Variante B: nur IP, Selbstzertifikat.** `CADDYFILE=Caddyfile.selfsigned` (kein `SITE_ADDRESS` nötig). Aufruf `https://<IP>/`, der Browser warnt einmalig (Zertifikat `tls internal`). Für die Jury schlechter (Warnseite), nur als Notlösung.
6. **Starten:**
   ```bash
   docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build
   docker compose -f deploy/docker-compose.yml ps        # app "healthy", caddy "running"
   curl -s https://203-0-113-7.sslip.io/api/jobs/current # {"busy":false}
   ```
7. **Testen:** Seite im Browser öffnen (Overview, Catalog). Live-Lauf mit dem Zugangscode als **Schnellprüfung** starten und beobachten (kostet wenige Cent). `docker compose ... logs -f app` zeigt Start und Fehler.
8. **Automatischer Neustart nach VM-Reboot:** `restart: unless-stopped` ist gesetzt; Docker startet beim Booten (`sudo systemctl enable docker`, Standard).
9. **Spiegel aktivieren:** GitHub > Settings > Pages > Source „GitHub Actions“, danach Workflow „Pages mirror“ manuell starten (Actions > Run workflow). Die Adresse lautet `https://script-hpp.github.io/Nullius/`.
10. **Eintragen:** Demo-URL (VM-Hostname) und Spiegel-Link in Devpost-Testhinweise und README; Zugangsdaten (ACCESS_CODE) **nur** im Devpost-Testhinweisfeld, nicht im Repo.

## Kosten grob (bis 15.12.2026, etwa 76 Tage ab 1.10.)
- **Coolify auf dem eigenen Server: 0 $ zusätzlich** (Server und Domain vorhanden). Die folgenden VM-Zahlen gelten nur für die Nebius-Alternative.
- Nebius-VM Ice Lake ab 0,06 $/h: ≈ 110 $ **[zu prüfen]**; Volume 30 GB ≈ 4 $; IP und Traffic laut Preisseite 0 $. Summe ≈ **115 $**, bei AMD Genoa ≈ 245 $. Die VM erst kurz vor der Abgabe (etwa 20.10.) anlegen spart rund 20 Tage (≈ 30 $); bis dahin reicht der lokale Betrieb.
- Live-Läufe: Schnellprüfung etwa 0,03 bis 0,10 $ je Lauf (Katalog-Läufe: SL-GCN 0,025 bis 0,028 $, MoHETS 0,093 $ brutto). `MAX_DAILY_REPRO_JOBS=3` begrenzt das auf rund 0,30 $ pro Tag, also bis 15.12. höchstens etwa 25 $, zusätzlich gedeckelt durch `BUDGET_TOTAL_USD` (Warnung 80 %, Stopp 95 %). **Stand 2026-10-02:** Diese Zahlen stammen aus der Zeit vor `setup_v2`; maßgeblich ist `docs/budget_plan.md` (`MAX_DAILY_REPRO_JOBS=1`, Server-Anteil 20 $ als `BUDGET_TOTAL_USD`, Kosten einer Schnellprüfung mit aktuellem Code noch zu messen).
- Anderer Anbieter statt Nebius-VM: ≈ 12 €.

## Betrieb bis 15.12.
- **Neustart (Coolify):** Configuration > *Restart*; Logs im Reiter *Logs*, Shell im Reiter *Terminal*. **Neustart (VM mit Compose):** `docker compose -f deploy/docker-compose.yml --env-file deploy/.env restart app`. Laufende Jobs gehen verloren (im Speicher); das Frontend zeigt dann den Vorab-Lauf als Replay.
- **Update:** Coolify: *Deploy* von Hand (Volumes bleiben). VM: `git pull && docker compose ... up -d --build`. Volumes (`data`, `logs`, `runs`) bleiben. Ein neuer Stand der Exporte im Repo überschreibt das Volume **nicht** (Seed nur bei leerem Volume); neue Katalog-/Demo-Daten per `docker cp` oder Volume leeren (Hinweis in `deploy/serve.py`).
- **Logs:** `docker compose ... logs --since 1h app`, Kosten in `/app/logs/costs.jsonl` (`docker compose exec app python scripts/cost_report.py` gibt es im Image nicht, `scripts/` ist ausgeschlossen: Log per `docker compose cp app:/app/logs/costs.jsonl .` holen und lokal mit `scripts/cost_report.py` auswerten).
- **Budget überwachen:** Usage-Seite der Token-Factory-Konsole wöchentlich, bei Abweichung über 10 % Log/Preistabelle korrigieren (CLAUDE.md). Bei Annäherung `MAX_DAILY_REPRO_JOBS` senken oder `ACCESS_CODE` ändern (Neustart nötig), um Läufe zu sperren. Der Replay und der Spiegel bleiben unabhängig davon nutzbar.
- **Zertifikat:** Traefik (Coolify) beziehungsweise Caddy (VM) erneuert automatisch; `caddy_data`-Volume nicht löschen. Bei Wechsel der IP ändert sich der sslip.io-Hostname: `SITE_ADDRESS` anpassen und Demo-URL in Devpost aktualisieren (nach der Abgabefrist nicht mehr änderbar!). Deshalb statische IP reservieren.
- **Ausfall der VM:** Spiegel-Link in den Testhinweisen nennen; Frontend-Replay deckt die Bewertung ab.
- **Nach dem 15.12.:** VM löschen (`nebius compute instance delete`), Token-Factory-Schlüssel rotieren.

## Lokaler Test (ohne echte Schlüssel)
```bash
docker build -t nullius-test .
docker run -d --name nullius-t -p 8011:8000 -e ACCESS_CODE=fake -e NEBIUS_API_KEY=fake -e NEBIUS_PROJECT_ID=fake -e TAVILY_API_KEY=fake -v nullius-t-data:/data nullius-test
curl -s localhost:8011/api/jobs/current        # {"busy":false}
```
Am 2026-10-01 geprüft: Build erfolgreich, Container `healthy`, `/api/jobs/current` antwortet, Seite und `/data/index.json` werden ausgeliefert, falscher Zugangscode ergibt 403, Volume wird beim ersten Start befüllt. Kein Live-Lauf, keine echten Aufrufe. Der Spiegel wurde unter dem Unterpfad `/Nullius/` lokal geprüft (Assets und Daten laden, keine Live-Steuerung sichtbar; einziger 404 ist `events.jsonl` eines Altlaufs ohne Ereignisprotokoll, das Frontend spielt ihn aus dem Dossier ab).

## Offen
- Coolify: Menünamen und Traefik-Standardwerte stammen aus dem Gedächtnis und sind an der echten Instanz zu prüfen (Domain-/Port-Feld, Healthcheck-Reiter, Timeouts, Kompression). Nicht getestet, da keine Coolify-Instanz verfügbar ist; der Container selbst ist lokal mit Docker getestet (siehe oben).
- Plattform- und Preset-Namen der Nebius-CPU-VM, statische IP, Preis des kleinsten Presets: vor dem Anlegen prüfen.
- Nebius Serverless nicht getestet.
- Ob Builders-/Trial-Guthaben für Compute gilt: Reviewer mit dem Nutzer klären.
- `api.py` bleibt unverändert; `src/` ist eingefroren. Nicht nötig, aber hilfreich für später: `/api/health` (heute dient `/api/jobs/current` als Healthcheck).
