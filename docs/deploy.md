# Deployment: Demo-URL bis mindestens 15.12.2026

Official Rules: Demo muss bis 15.12.2026 erreichbar und kostenlos testbar sein (Bewertung 01.–15.12.). Zwei Schichten: **Live-Server** (Container mit API und Web-App) und **statischer Spiegel** auf GitHub Pages als Absicherung (nur Exporte, Replay, keine Live-Steuerung). Stand der Preise: Nebius-Preisseite am 2026-10-01 abgerufen; Werte mit **[zu prüfen]** vor der Buchung gegenlesen.

## Was der Dienst braucht
- **Ein Prozess, ein Job gleichzeitig:** Jobs, Warteschlange und Semaphor liegen im Speicher der API (`api.py`). Genau **ein** uvicorn-Worker, kein Autoscaling auf mehrere Instanzen.
- **Lange Läufe:** Reproduktionsläufe bis etwa 20 Minuten, Schnellprüfung (3 Seeds) kürzer. Der Live-Stream (SSE, `/api/reproduce/{id}/stream`) bleibt so lange offen (Keepalive alle 15 s). Proxy ohne Pufferung und mit Lese-Timeout über 20 Minuten.
- **Persistenz:** Exporte neuer Läufe (`REPRO_EXPORT_DIR`, im Container `/data`: `index.json`, `catalog.json`, Laufordner) und das Verzeichnis `logs/` mit `costs.jsonl` (Budgetzähler: `BUDGET_TOTAL_USD` zählt aus diesem Log) und `tavily_usage.jsonl` (Tavily-Credit-Zähler der Live-Sperre). Ein leeres `logs/` nach Neustart oder Redeploy ließe beide Untergrenzen auf „voll“ zurückspringen. `logs/` und `/data` als Volume (Coolify: `nullius-logs` → `/app/logs`, Compose: `logs:/app/logs`; das ganze Verzeichnis, nicht einzelne Dateien).
- **Ausgehend erreichbar:** Token Factory (`api.tokenfactory.nebius.com`), Tavily, Nebius Sandboxes (`contree-sdk`), arXiv/GitHub für Quellen. Keine eingehenden Verbindungen außer 80/443.
- **Schlüssel nur als Umgebungsvariablen** (`deploy/.env`, nie im Image und nicht in Git). Die Demo ist offen (kein Zugangscode); Live-Läufe sperrt die API selbst, wenn Budget oder Tavily-Credits unter die Schwelle fallen (`LIVE_MIN_BUDGET_USD`, `LIVE_MIN_TAVILY_CREDITS`), dazu Tageslimits und ein Limit je IP. Replay und Evidence bleiben immer nutzbar.
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
4. **Umgebungsvariablen** (Configuration > Environment Variables; Namen wie in `deploy/.env.example`, **Werte nur hier**): `LLM_PROVIDER=tokenfactory`, `SANDBOX_BACKEND=nebius`, `TAVILY_MODE=live`, `NEBIUS_API_KEY`, `NEBIUS_PROJECT_ID`, `TF_MODEL_FAST`, `TF_MODEL_CODE`, `TF_MODEL_REASONING`, `TAVILY_API_KEY`, `MAX_DAILY_JOBS`, `MAX_DAILY_REPRO_JOBS`, `MAX_DAILY_REPRO_PER_IP`, `MAX_DAILY_DISCOVER_PER_IP`, `LIVE_MIN_BUDGET_USD`, `LIVE_MIN_TAVILY_CREDITS`, `TAVILY_CREDITS_TOTAL`, `TAVILY_CREDITS_SINCE`, `TRUST_PROXY=1` (hinter Coolify/Caddy, sonst zählt jeder Besucher als dieselbe Proxy-IP), `MAX_COST_PER_RUN_USD`, `VAT_RATE`, `BUDGET_TOTAL_USD`, `BUDGET_SINCE`, `MAX_ATTEMPTS_PER_CLAIM`, `REPRO_EXPORT_DIR=/data` (steht schon im Image). **„Build Variable“ nicht ankreuzen**: Schlüssel dürfen nie in den Build. `SITE_ADDRESS` und `CADDYFILE` werden nicht gebraucht (nur für `docker-compose.yml` mit Caddy).
5. **Persistente Speicher** (Configuration > Persistent Storage > Add, Typ *Volume*): `nullius-data` → `/data`, `nullius-logs` → `/app/logs`, `nullius-runs` → `/app/runs`. Benannte Volumes übernehmen beim ersten Start Inhalt und Besitzer aus dem Image (Nutzer `app`, UID 10001); bei Bind-Mounts auf dem Host müsste `chown 10001` gesetzt werden. `/app/logs` enthält `costs.jsonl` (Budgetzähler) und `tavily_usage.jsonl` (Tavily-Credit-Zähler): nie löschen, sonst springen `LIVE_MIN_BUDGET_USD` und `LIVE_MIN_TAVILY_CREDITS` auf „voll“ zurück. Vor dem ersten Redeploy prüfen: `docker exec <container> ls /app/logs` zeigt beide Dateien auch nach dem Neustart.
6. **Healthcheck** (Configuration > Healthcheck, aktivieren): Methode GET, Schema http, Host `localhost`, Port `8000`, Pfad `/api/jobs/current`, Rückgabecode `200`, Intervall 30 s, Timeout 5 s, Retries 3, Start-Zeitraum 20 s. (Das Dockerfile hat einen eigenen `HEALTHCHECK` mit demselben Pfad; Coolify wertet seine Einstellung aus.) **Nur eine Instanz** betreiben (keine Replikate): Jobs und Semaphor liegen im Speicher der API.
7. **Auto-Deploy ausschalten** (Configuration > Advanced: *Auto Deploy* aus, Webhooks nicht einrichten). Ein Redeploy beendet den laufenden Container und damit einen laufenden Live-Lauf; während der Bewertung (01.–15.12.) nur von Hand und nicht während eines Laufs deployen. Mit Coolify-Rolling-Update wartet der alte Container auf den Healthcheck des neuen, ein aktiver Job geht trotzdem verloren (Speicher).
8. **Speicher begrenzen** (Advanced > Resource Limits): RAM 1 bis 2 GB, CPU 1 bis 2 (die Berechnung läuft in der Nebius-Sandbox, der Container ist leicht).
9. **Deploy** starten und das Build-Log beobachten (Node-Stufe baut `web/`, Python-Stufe installiert mit `uv sync --frozen`). Danach ist der Status *Running (healthy)*.
10. **Proxy: SSE ohne Puffer, lange Timeouts.** Traefik puffert Antworten nicht und gibt Streams unverändert weiter; Standardwerte von Coolifys Traefik passen meist: keine Lese-/Schreib-Timeouts auf dem HTTP-Entrypoint (`readTimeout`/`writeTimeout` 0), `idleTimeout` 180 s betrifft nur ruhende Verbindungen, der Stream schickt alle 15 s einen Keepalive **[zu prüfen an der echten Instanz]**. Zu beachten:
    - Keine Kompression oder Puffer-Middleware für `/api/*` hinzufügen (Coolify-Option „gzip/compress“ an der Ressource aus lassen, oder `text/event-stream` ausnehmen).
    - Steht ein weiterer Proxy davor (Cloudflare, Nginx): dort Pufferung aus (`proxy_buffering off`, `X-Accel-Buffering` kommt von der App nicht), Lese-Timeout mindestens 30 min.
    - Falls ein Timeout den Stream vor 20 Minuten trennt: Traefik-Entrypoint-Timeouts in den Coolify-Proxy-Einstellungen (Servers > Proxy > Konfiguration) anheben (`--entrypoints.https.transport.respondingTimeouts.idleTimeout=1800s`), danach Proxy neu starten.
11. **Test von außen:** `curl -s https://nullius.example.com/api/jobs/current` liefert `{"busy":false}`. Seite im Browser öffnen (Overview, Catalog). Dann als **Schnellprüfung** einen Live-Lauf (ohne Code, ein Klick) starten und die Stream-Anzeige bis zum Ende beobachten (kostet wenige Cent). Probe für den Proxy, ohne Kosten: `curl -N -m 60 https://nullius.example.com/api/reproduce/<job>/stream` während eines Laufs zeigt die Ereignisse zeitnah, nicht erst am Ende.
12. **Eintragen:** Demo-URL `https://nullius.example.com` und Spiegel-Link in Devpost-Testhinweise und README; die Testhinweise nennen keinen Code (die Demo ist offen). Spiegel aktivieren: GitHub > Settings > Pages > Source „GitHub Actions“, Workflow „Pages mirror“ einmal manuell starten (`https://script-hpp.github.io/Nullius/`).

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
   nano deploy/.env     # alle Werte eintragen (Schlüssel, Limits, BUDGET_TOTAL_USD und BUDGET_SINCE aus der Konsole, TAVILY_CREDITS_TOTAL und TAVILY_CREDITS_SINCE aus dem Tavily-Dashboard)
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
7. **Testen:** Seite im Browser öffnen (Overview, Catalog). Live-Lauf als **Schnellprüfung** starten und beobachten (kostet wenige Cent). `docker compose ... logs -f app` zeigt Start und Fehler.
8. **Automatischer Neustart nach VM-Reboot:** `restart: unless-stopped` ist gesetzt; Docker startet beim Booten (`sudo systemctl enable docker`, Standard).
9. **Spiegel aktivieren:** GitHub > Settings > Pages > Source „GitHub Actions“, danach Workflow „Pages mirror“ manuell starten (Actions > Run workflow). Die Adresse lautet `https://script-hpp.github.io/Nullius/`.
10. **Eintragen:** Demo-URL (VM-Hostname) und Spiegel-Link in Devpost-Testhinweise und README; kein Zugangscode nötig.

## Kosten grob (bis 15.12.2026, etwa 76 Tage ab 1.10.)
- **Coolify auf dem eigenen Server: 0 $ zusätzlich** (Server und Domain vorhanden). Die folgenden VM-Zahlen gelten nur für die Nebius-Alternative.
- Nebius-VM Ice Lake ab 0,06 $/h: ≈ 110 $ **[zu prüfen]**; Volume 30 GB ≈ 4 $; IP und Traffic laut Preisseite 0 $. Summe ≈ **115 $**, bei AMD Genoa ≈ 245 $. Die VM erst kurz vor der Abgabe (etwa 20.10.) anlegen spart rund 20 Tage (≈ 30 $); bis dahin reicht der lokale Betrieb.
- Live-Läufe: Rechnung und Begründung der Limits im Abschnitt „Limits bis 15.12.“ (Basis: gemessener Live-Lauf Kipf, 0,139 $ brutto; Obergrenze je Lauf `MAX_COST_PER_RUN_USD=0.50`). Zahlen älterer Schnellprüfungen (0,03 bis 0,10 $, vor `setup_v2`) gelten nicht mehr.
- Anderer Anbieter statt Nebius-VM: ≈ 12 €.

## Limits bis 15.12.

Empfohlene Werte für `deploy/.env` bzw. Coolify, zwei Stufen (der Nutzer setzt die Variablen und deployt neu; nicht während eines laufenden Jobs):

| Variable | bis 2026-11-30 | ab 2026-12-01 | Grund |
|---|---|---|---|
| `MAX_DAILY_REPRO_JOBS` | 1 | 3 | Bis Ende November testet kaum jemand; ab 1.12. beginnt die Bewertung (01.–15.12.), dann sollen mehrere Juroren am selben Tag live prüfen können |
| `MAX_COST_PER_RUN_USD` | 0.75 | 0.75 | Standardmodus mit Autorencode (siehe unten); die Untergrenze `LIVE_MIN_BUDGET_USD=3` bleibt ein Vielfaches davon |
| `MAX_DAILY_JOBS` (Discovery) | 1 | 1 | Discovery ist billig in Token (ca. 0,02 $), kostet aber ca. 16 Tavily-Credits je Lauf (`docs/next_step_catalog_pipeline.md`, Zeile 83) |
| `MAX_DAILY_REPRO_PER_IP` / `MAX_DAILY_DISCOVER_PER_IP` | 1 / 1 | 1 / 1 | Ein Besucher kann nicht das ganze Tageslimit verbrauchen |
| `LIVE_MIN_BUDGET_USD` | 3 | 3 | Reserve, damit das Konto durch Live-Läufe nie ins Minus läuft (Karte würde belastet); ein Lauf kostet höchstens 0,50 $ |
| `LIVE_MIN_TAVILY_CREDITS` | 300 | 300 | Reserve für die Vorzeige-Läufe und Discovery bei der Abgabe |
| `BUDGET_TOTAL_USD` | 20.00 | 20.00 | Server-Anteil, `docs/budget_plan.md` („Zwei Logs, ein Konto") |
| `TRUST_PROXY` | 1 | 1 | hinter Coolify/Caddy, sonst zählt jeder Besucher als dieselbe Proxy-IP |

**Standardmodus seit 2026-10-09 („Nachbau aus Paper und Autorencode“, `loop.product_kwargs`):** die Prompts enthalten den rohen Repo-Ausschnitt, ein Lauf ist teurer als die gemessenen 0,139 $ des Paper-Modus. Gemessen (recall cg v1, `benchmark/experiments/results/recall_cg_v1.md`, 22 Läufe, Tavily aus Fixtures, kein Community-Check): Mittel 0,25 $ brutto, Maximum 0,46 $ brutto je Lauf (Generation 0 mit zwei Claims). Für die Planung gelten 0,30 $ brutto je Live-Lauf (Spanne 0,20 bis 0,35; die Schnellprüfung im Produkt ist **noch nicht gemessen**, das geschieht beim ersten echten Live-Lauf mit OK des Nutzers). Daher `MAX_COST_PER_RUN_USD=0.75` auf dem Server (wie in recall cg v1; Code-Standard 0.50 gilt weiter für Benchmarks). Fällt das Autoren-Repo weg, läuft der Lauf als Paper-Nachbau (günstiger, 0,14 $) weiter.

**Schlimmster Fall von heute (2026-10-07) bis 2026-12-15:** 70 Tage (7.10. bis 15.12. einschließlich) = 55 Tage Stufe 1 (25 im Oktober, 30 im November) plus 15 Tage Stufe 2. Verfügbar: 20,00 $ Server-Anteil minus 3,00 $ Untergrenze = **17,00 $**.

| Szenario | Reproduktion | Discovery | Summe | gegen 17,00 $ |
|---|---|---|---|---|
| Gemessen: 0,139 $ je Lauf (Kipf, `web/public/data/kipf_gcn_2017/20261002-110746`, `docs/devpost_submission.md`) | (55 × 1 + 15 × 3) = 100 Läufe × 0,139 = 13,90 $ | 70 × 0,02 = 1,40 $ | **15,30 $** | reicht, 1,70 $ Luft |
| **Standardmodus Autorencode, Planungswert 0,30 $ je Lauf** (Maximum gemessen 0,46 $) | 100 Läufe × 0,30 = 30,00 $ | 1,40 $ | **31,40 $** | **übersteigt 17,00 $** (siehe unten) |
| Theoretisch: Obergrenze 0,50 $ je Lauf (Paper-Modus; mit Autorencode 0,75 $ = 75,00 $) | 100 × 0,50 = 50,00 $ | 1,40 $ | 51,40 $ | übersteigt 17,00 $ |

**Folge für die Limits (Annahme gebrochen, Entscheidung beim Nutzer):** Mit 0,30 $ je Lauf trägt der Server-Anteil von 17,00 $ nutzbar (20,00 $ minus Untergrenze 3,00 $) nach Abzug von 1,40 $ Discovery rund 52 Läufe, nicht 100. Dazu kommt, dass der Konsolen-Kontostand nach recall cg v1 laut Log nur noch **17,39 $** beträgt (Stand 2026-10-09, Konsole nicht verifiziert), der geplante Server-Anteil von 20,00 $ also gar nicht mehr gedeckt ist. Vorschlag, solange kein Guthaben nachkommt: Stufe 1 (bis 2026-11-30) `MAX_DAILY_REPRO_JOBS=1` lassen, Stufe 2 (Bewertung 01.–15.12.) auf 2 statt 3 setzen; die Rechnung 55 × 1 + 15 × 2 = 85 Läufe × 0,30 = 25,50 $ bleibt auch dann über dem Anteil, die Untergrenze `LIVE_MIN_BUDGET_USD` pausiert Live-Läufe also vor dem 15.12. und die Wiedergabe übernimmt. Um das Bewertungsfenster zu schützen, den Server-Anteil nach Messung des echten Live-Laufs neu festlegen (Anteil = Läufe der Stufe 2 × Messwert + 3 $ Untergrenze) und `LIVE_MIN_BUDGET_USD` in Stufe 1 entsprechend höher setzen (Reserve für Stufe 2), statt `MAX_DAILY_REPRO_JOBS` allein zu verlassen.

Der theoretische Fall tritt nur ein, wenn jeder Lauf die Kostenbremse ausreizt. Dann **ist die Untergrenze die harte Obergrenze**, nicht das Tageslimit: Live-Läufe pausieren (HTTP 503, `GET /api/status` meldet `live: false`, das Frontend zeigt die Wiedergabe), sobald das Rest-Budget unter 3,00 $ fällt. Weil nur ein Lauf gleichzeitig läuft und die Prüfung vor dem Start geschieht, kann das Budget die Untergrenze um höchstens einen Lauf (0,50 $) unterschreiten. Das Konto bleibt also bei etwa 2,50 $ Rest des Server-Anteils; die Summe der Anteile von Server und lokalem Rechner darf nie über dem Konsolen-Kontostand liegen (`docs/budget_plan.md`).

Tavily: Discovery 70 × 16 = 1 120 Credits (`docs/next_step_catalog_pipeline.md`, Zeile 83). Der Verbrauch einer Reproduktion ist **nicht gemessen** (`docs/budget_plan.md`, „Tavily"); die Untergrenze `LIVE_MIN_TAVILY_CREDITS` schützt, bis ein Wert vorliegt. Verfügbarkeit laut `docs/budget_plan.md`: rund 6 025 Credits bis 15.12. (900 laufender Monat, 2 × 1 000 Monatsplan, 3 125 Add-on; Verfall des Add-ons im Dashboard prüfen).

**Bekannte Grenze:** Tageszähler (`MAX_DAILY_*`) und Limits je IP liegen im Speicher der API. Ein Neustart oder Redeploy setzt sie auf null zurück, der Tag kann dann mehr Läufe sehen als das Limit. Die Untergrenzen bleiben wirksam, weil sie aus `logs/costs.jsonl` und `logs/tavily_usage.jsonl` rechnen; das gilt nur, solange `logs/` ein persistentes Volume ist. Deshalb: Redeploys selten und von Hand (Schritt 7 oben), `logs/` nie leeren.

**Termine:** am 2026-12-01 `MAX_DAILY_REPRO_JOBS=3` setzen und neu deployen; wöchentlich Konsole und Tavily-Dashboard prüfen (`docs/budget_plan.md`, „Regeln und Auslöser").

## Betrieb bis 15.12.
- **Neustart (Coolify):** Configuration > *Restart*; Logs im Reiter *Logs*, Shell im Reiter *Terminal*. **Neustart (VM mit Compose):** `docker compose -f deploy/docker-compose.yml --env-file deploy/.env restart app`. Laufende Jobs gehen verloren (im Speicher); das Frontend zeigt dann den Vorab-Lauf als Replay.
- **Update:** Coolify: *Deploy* von Hand (Volumes bleiben). VM: `git pull && docker compose ... up -d --build`. Volumes (`data`, `logs`, `runs`) bleiben. Ein neuer Stand der Exporte im Repo überschreibt das Volume **nicht** (Seed nur bei leerem Volume); neue Katalog-/Demo-Daten per `docker cp` oder Volume leeren (Hinweis in `deploy/serve.py`).
- **Logs:** `docker compose ... logs --since 1h app`, Kosten in `/app/logs/costs.jsonl` (`docker compose exec app python scripts/cost_report.py` gibt es im Image nicht, `scripts/` ist ausgeschlossen: Log per `docker compose cp app:/app/logs/costs.jsonl .` holen und lokal mit `scripts/cost_report.py` auswerten).
- **Budget überwachen:** Usage-Seite der Token-Factory-Konsole wöchentlich, bei Abweichung über 10 % Log/Preistabelle korrigieren (CLAUDE.md). Die API sperrt Live-Läufe selbst (HTTP 503, `GET /api/status` meldet `live: false`), sobald das Rest-Budget unter `LIVE_MIN_BUDGET_USD` oder die Rest-Credits unter `LIVE_MIN_TAVILY_CREDITS` fallen; zusätzlich `MAX_DAILY_REPRO_JOBS` senken, wenn nötig (Werte und Termine: „Limits bis 15.12.“). `logs/tavily_usage.jsonl` zählt Tavily-Credits je Aufruf (Schätzung); gegen das Tavily-Dashboard prüfen und `TAVILY_CREDITS_TOTAL`/`_SINCE` nachziehen. Das Log liegt im Container unter `logs/`: für einen Neustart ohne Verlust ein Volume dafür einbinden, sonst startet die Zählung bei null. Der Replay und der Spiegel bleiben unabhängig davon nutzbar.
- **Zertifikat:** Traefik (Coolify) beziehungsweise Caddy (VM) erneuert automatisch; `caddy_data`-Volume nicht löschen. Bei Wechsel der IP ändert sich der sslip.io-Hostname: `SITE_ADDRESS` anpassen und Demo-URL in Devpost aktualisieren (nach der Abgabefrist nicht mehr änderbar!). Deshalb statische IP reservieren.
- **Ausfall der VM:** Spiegel-Link in den Testhinweisen nennen; Frontend-Replay deckt die Bewertung ab.
- **Nach dem 15.12.:** VM löschen (`nebius compute instance delete`), Token-Factory-Schlüssel rotieren.

## Lokaler Test (ohne echte Schlüssel)
```bash
docker build -t nullius-test .
docker run -d --name nullius-t -p 8011:8000 -e NEBIUS_API_KEY=fake -e NEBIUS_PROJECT_ID=fake -e TAVILY_API_KEY=fake -v nullius-t-data:/data nullius-test
curl -s localhost:8011/api/jobs/current        # {"busy":false}
```
Am 2026-10-01 geprüft: Build erfolgreich, Container `healthy`, `/api/jobs/current` antwortet, Seite und `/data/index.json` werden ausgeliefert, `/api/status` antwortet, Volume wird beim ersten Start befüllt. Kein Live-Lauf, keine echten Aufrufe. Der Spiegel wurde unter dem Unterpfad `/Nullius/` lokal geprüft (Assets und Daten laden, keine Live-Steuerung sichtbar; einziger 404 ist `events.jsonl` eines Altlaufs ohne Ereignisprotokoll, das Frontend spielt ihn aus dem Dossier ab).

## Offen
- Coolify: Menünamen und Traefik-Standardwerte stammen aus dem Gedächtnis und sind an der echten Instanz zu prüfen (Domain-/Port-Feld, Healthcheck-Reiter, Timeouts, Kompression). Nicht getestet, da keine Coolify-Instanz verfügbar ist; der Container selbst ist lokal mit Docker getestet (siehe oben).
- Plattform- und Preset-Namen der Nebius-CPU-VM, statische IP, Preis des kleinsten Presets: vor dem Anlegen prüfen.
- Nebius Serverless nicht getestet.
- Ob Builders-/Trial-Guthaben für Compute gilt: Reviewer mit dem Nutzer klären.
- `api.py` bleibt unverändert; `src/` ist eingefroren. Nicht nötig, aber hilfreich für später: `/api/health` (heute dient `/api/jobs/current` als Healthcheck).
