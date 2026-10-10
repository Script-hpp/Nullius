# Deployment: Demo-URL bis mindestens 15.12.2026

**Aktueller Deploy-Check (10.10.):** [Pruefergebnis und Coolify-Konfiguration](deploy_readiness_20261010.md). Fuer das erste oeffentliche Deployment `deploy/coolify-preview.env.example` verwenden: Katalog, Evidenz, Replays und PDF funktionieren; kostenpflichtige Starts sind gesperrt. Modellversuche und Tavily-Aufrufe reservieren jetzt vorab ihr Budget; Grenzen der Schaetzung und Betriebsanforderungen stehen im [Kostenreview](cost_policy_review.md).

Official Rules: Demo muss bis 15.12.2026 erreichbar und kostenlos testbar sein (Bewertung 01.–15.12.). Zwei Schichten: **Live-Server** (Container mit API und Web-App) und **statischer Spiegel** auf GitHub Pages als Absicherung (nur Exporte, Replay, keine Live-Steuerung). Stand der Preise: Nebius-Preisseite am 2026-10-01 abgerufen; Werte mit **[zu prüfen]** vor der Buchung gegenlesen.

## Was der Dienst braucht
- **Ein Prozess, ein Job gleichzeitig:** Jobs, Warteschlange und Semaphor liegen im Speicher der API (`api.py`). Genau **ein** uvicorn-Worker, kein Autoscaling auf mehrere Instanzen.
- **Lange Läufe:** Reproduktionsläufe bis etwa 20 Minuten, Schnellprüfung (3 Seeds) kürzer. Der Live-Stream (SSE, `/api/reproduce/{id}/stream`) bleibt so lange offen (Keepalive alle 15 s). Proxy ohne Pufferung und mit Lese-Timeout über 20 Minuten.
- **Persistenz:** Exporte neuer Läufe (`REPRO_EXPORT_DIR`, im Container `/data`: `index.json`, `catalog.json`, Laufordner) und das Verzeichnis `logs/` mit `costs.jsonl` (Budgetzähler: `BUDGET_TOTAL_USD` zählt aus diesem Log) `tavily_usage.jsonl` (Tavily-Credit-Zähler der Live-Sperre) und `live_jobs.jsonl` (Start-Log der Live-Jobs mit IP-Hash: stellt Tageslimits und Limits je IP nach einem Neustart wieder her). Ein leeres `logs/` nach Neustart oder Redeploy ließe Untergrenzen und Tageslimits auf „voll“ zurückspringen. `logs/` und `/data` als Volume (Coolify: `nullius-logs` → `/app/logs`, Compose: `logs:/app/logs`; das ganze Verzeichnis, nicht einzelne Dateien).
- **Ausgehend erreichbar:** Token Factory (`api.tokenfactory.nebius.com`), Tavily, Nebius Sandboxes (`contree-sdk`), arXiv/GitHub für Quellen. Keine eingehenden Verbindungen außer 80/443.
- **Schlüssel nur als Umgebungsvariablen** (`deploy/.env`, nie im Image und nicht in Git). Die Demo ist offen (kein Zugangscode); Live-Läufe sperrt die API selbst, wenn Budget oder Tavily-Credits unter die Schwelle fallen (`LIVE_MIN_BUDGET_USD`, `LIVE_MIN_TAVILY_CREDITS`), dazu Tageslimits, ein Tagesbudget in $ (`MAX_DAILY_COST_USD`) und ein Limit je IP. Ohne `BUDGET_TOTAL_USD` und `BUDGET_SINCE` bleibt Live bei `LLM_PROVIDER=tokenfactory` gesperrt (fail closed). Replay und Evidence bleiben immer nutzbar.
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
4. **Umgebungsvariablen** (Configuration > Environment Variables; Namen wie in `deploy/.env.example`, **Werte nur hier**): `LLM_PROVIDER=tokenfactory`, `SANDBOX_BACKEND=nebius`, `TAVILY_MODE=live`, `NEBIUS_API_KEY`, `NEBIUS_PROJECT_ID`, `TF_MODEL_FAST`, `TF_MODEL_CODE`, `TF_MODEL_REASONING`, `TAVILY_API_KEY`, `MAX_DAILY_JOBS`, `MAX_DAILY_REPRO_JOBS`, `MAX_DAILY_REPRO_PER_IP`, `MAX_DAILY_DISCOVER_PER_IP`, `MAX_DAILY_COST_USD`, `ALLOW_FULL_RUNS=0`, `LIVE_MIN_BUDGET_USD`, `LIVE_MIN_TAVILY_CREDITS`, `TAVILY_CREDITS_TOTAL`, `TAVILY_CREDITS_SINCE`, `TRUST_PROXY=1` (hinter Coolify/Caddy, sonst zählt jeder Besucher als dieselbe Proxy-IP), `MAX_COST_PER_RUN_USD`, `VAT_RATE`, `BUDGET_TOTAL_USD`, `BUDGET_SINCE`, `MAX_ATTEMPTS_PER_CLAIM`, `REPRO_EXPORT_DIR=/data` (steht schon im Image). Optional (Defaults genügen): `RATE_LIMIT_PER_MIN`, `RATE_LIMIT_POST_PER_MIN`, `MAX_BODY_BYTES`, `MAX_SSE_STREAMS`, `MAX_SSE_PER_IP`. Die Werte für Stufe 1 und 2 stehen in `deploy/.env.example`. **Schlüssel (`NEBIUS_API_KEY`, `NEBIUS_PROJECT_ID`, `TAVILY_API_KEY`) nur hier eintragen, als „Secret“ markieren; „Build Variable“ nicht ankreuzen**: Schlüssel dürfen nie in den Build. `SITE_ADDRESS` und `CADDYFILE` werden nicht gebraucht (nur für `docker-compose.yml` mit Caddy).
5. **Persistente Speicher** (Configuration > Persistent Storage > Add, Typ *Volume*): `nullius-data` → `/data`, `nullius-logs` → `/app/logs`, `nullius-runs` → `/app/runs`. Benannte Volumes übernehmen beim ersten Start Inhalt und Besitzer aus dem Image (Nutzer `app`, UID 10001); bei Bind-Mounts auf dem Host müsste `chown 10001` gesetzt werden. `/app/logs` enthält `costs.jsonl` (Budgetzähler und Tagesbudget), `tavily_usage.jsonl` (Tavily-Credit-Zähler), `live_jobs.jsonl` (Tageslimits) und `discovery_cache.json`: nie löschen, sonst springen `LIVE_MIN_BUDGET_USD`, `LIVE_MIN_TAVILY_CREDITS` und die Tageslimits auf „voll“ zurück. Vor dem ersten Redeploy prüfen: `docker exec <container> ls /app/logs` zeigt die Dateien auch nach dem Neustart.
6. **Healthcheck** (Configuration > Healthcheck, aktivieren): Methode GET, Schema http, Host `localhost`, Port `8000`, Pfad `/api/health`, Rückgabecode `200`, Intervall 30 s, Timeout 5 s, Retries 3, Start-Zeitraum 20 s. (Das Dockerfile hat einen eigenen `HEALTHCHECK` mit demselben Pfad; Coolify wertet seine Einstellung aus. `/api/health` liest nichts von der Platte und zeigt keinen Job-Zustand.) **Nur eine Instanz** betreiben (keine Replikate): Jobs und Semaphor liegen im Speicher der API.
7. **Auto-Deploy ausschalten** (Configuration > Advanced: *Auto Deploy* aus, Webhooks nicht einrichten). Ein Redeploy beendet den laufenden Container und damit einen laufenden Live-Lauf; während der Bewertung (01.–15.12.) nur von Hand und nicht während eines Laufs deployen. Mit Coolify-Rolling-Update wartet der alte Container auf den Healthcheck des neuen, ein aktiver Job geht trotzdem verloren (Speicher).
8. **Speicher begrenzen** (Advanced > Resource Limits): RAM 1 bis 2 GB, CPU 1 bis 2 (die Berechnung läuft in der Nebius-Sandbox, der Container ist leicht).
9. **Deploy** starten und das Build-Log beobachten (Node-Stufe baut `web/`, Python-Stufe installiert mit `uv sync --frozen`). Danach ist der Status *Running (healthy)*.
10. **Proxy: SSE ohne Puffer, lange Timeouts.** Die App setzt `X-Accel-Buffering: no` und `Cache-Control: no-store` auf den Streams selbst. Traefik puffert Antworten nicht und gibt Streams unverändert weiter; Standardwerte von Coolifys Traefik passen meist: keine Lese-/Schreib-Timeouts auf dem HTTP-Entrypoint (`readTimeout`/`writeTimeout` 0), `idleTimeout` 180 s betrifft nur ruhende Verbindungen, der Stream schickt alle 15 s einen Keepalive **[zu prüfen an der echten Instanz]**. Zu beachten:
    - Keine Kompression oder Puffer-Middleware für `/api/*` hinzufügen (Coolify-Option „gzip/compress“ an der Ressource aus lassen, oder `text/event-stream` ausnehmen).
    - Steht ein weiterer Proxy davor (Cloudflare, Nginx): dort Pufferung aus (`proxy_buffering off`, `X-Accel-Buffering` kommt von der App nicht), Lese-Timeout mindestens 30 min.
    - Falls ein Timeout den Stream vor 20 Minuten trennt: Traefik-Entrypoint-Timeouts in den Coolify-Proxy-Einstellungen (Servers > Proxy > Konfiguration) anheben (`--entrypoints.https.transport.respondingTimeouts.idleTimeout=1800s`), danach Proxy neu starten.
11. **Test von außen:** `curl -s https://nullius.example.com/api/jobs/current` liefert `{"busy":false}`; `curl -si https://nullius.example.com/api/health` zeigt `content-security-policy`, `x-content-type-options: nosniff`, `x-frame-options: DENY` und `cache-control: no-store`; `curl -so /dev/null -w '%{http_code}' https://nullius.example.com/docs` liefert 404. Seite im Browser öffnen (Overview, Catalog). Dann als **Schnellprüfung** einen Live-Lauf (ohne Code, ein Klick) starten und die Stream-Anzeige bis zum Ende beobachten (kostet wenige Cent). Probe für den Proxy, ohne Kosten: `curl -N -m 60 https://nullius.example.com/api/reproduce/<job>/stream` während eines Laufs zeigt die Ereignisse zeitnah, nicht erst am Ende.
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

**Pruefung 2026-10-10:** Die Zahlen unten sind Planungswerte, keine bewiesenen harten Obergrenzen. Ein laufender Modellaufruf kann eine Schwelle ueberschreiten; direkte Extraktionsaufrufe umgehen die Orchestrator-Laufbremse, und der Watchdog beendet den Worker nicht. Der Kontostand vom 9.10. ist vor Deployment neu zu verifizieren. Befunde und naechste Schritte: [`cost_policy_review.md`](cost_policy_review.md).

Stand 2026-10-09. Konsolen-Kontostand **17,02 $** (11:11 UTC); der Anteil des Servers ist **12,00 $ brutto**, rund 5 $ bleiben für den lokalen Rechner (Rest-Benchmarks, Video-Aufnahmen). Die Anteile von Server und Rechner dürfen zusammen nie über dem Konsolen-Kontostand liegen (`docs/budget_plan.md`, „Zwei Logs, ein Konto“); der lokale `BUDGET_TOTAL_USD` ist danach auf den Rechner-Anteil zu setzen. Werte in `deploy/.env.example`; der Nutzer setzt sie in Coolify und deployt neu (nicht während eines laufenden Jobs).

Mehrere Schutzschichten mit unterschiedlichen Pruefzeitpunkten:

| Schicht | Wert | Wirkung |
|---|---|---|
| Kostenbremse je Job | Reproduktion `MAX_COST_PER_RUN_USD=0.50`, Discovery `MAX_COST_PER_DISCOVERY_USD=0.25` | Jeder Modellversuch reserviert seinen geschaetzten Maximalpreis; alle Rollen, Extraktionen und Retries teilen das Joblimit |
| Schnellprüfung erzwungen | `ALLOW_FULL_RUNS=0` | Ein Client kann keinen teuren Vollauf (10 Seeds) anfordern |
| Tagesbudget | `MAX_DAILY_COST_USD` | Kosten des Tages plus offene Reservierungen plus naechster Modellversuch duerfen die Grenze nicht ueberschreiten; auch innerhalb eines Jobs |
| Tageslimits | `MAX_DAILY_REPRO_JOBS`, `MAX_DAILY_JOBS`, je IP 1 | Aus `live_jobs.jsonl` nach Neustart wiederhergestellt |
| Untergrenzen | `LIVE_MIN_BUDGET_USD`, `LIVE_MIN_TAVILY_CREDITS` | Vor jedem Modell-/Tavily-Versuch geprueft, einschliesslich der naechsten Reservierung; offene Anfragen zaehlen mit |
| Globale Bremse im LLM-Client | 95 % von `BUDGET_TOTAL_USD` | Kein Modellaufruf mehr, auch mitten im Lauf (`BudgetExceeded`) |
| Ein Job gleichzeitig, Wanduhr | 45 min Reproduktion, 15 min Discovery | Timeout beendet den Stream und sperrt neue Anbieteraufrufe des Jobs. Der Platz bleibt belegt, bis der Worker wirklich endet; kein spaeterer Erfolgsexport |

Zwei Stufen:

| Variable | Stufe 1: Deployment bis 2026-11-30 | Stufe 2: ab 2026-12-01 (Bewertung 01.–15.12.) | Grund |
|---|---|---|---|
| `BUDGET_TOTAL_USD` | 12.00 | 12.00 (nicht ändern, `BUDGET_SINCE` auch nicht) | Server-Anteil |
| `LIVE_MIN_BUDGET_USD` | **6.0** | **1.0** | Stufe 1 hält 6 $ als Reserve für das Bewertungsfenster zurück; Stufe 2 gibt sie frei und lässt eine Reserve von zwei Läufen (2 × 0,50 $) |
| `MAX_DAILY_REPRO_JOBS` | 1 | 2 | Bis Ende November testet kaum jemand; in der Bewertung sollen mehrere Juroren am Tag live prüfen können |
| `MAX_DAILY_COST_USD` | 0.75 | 1.25 | Mehr als ein Lauf plus Discovery pro Tag in Stufe 1 gibt es nie; Stufe 2 zwei Läufe (2 × 0,50 $) plus Discovery |
| `MAX_DAILY_JOBS` (Discovery) | 1 | 1 | Token billig (ca. 0,02 $), aber ca. 16 Tavily-Credits je Lauf |
| `MAX_DAILY_REPRO_PER_IP` / `MAX_DAILY_DISCOVER_PER_IP` | 1 / 1 | 1 / 1 | Ein Besucher kann nicht das ganze Tageslimit verbrauchen |
| `MAX_COST_PER_RUN_USD` | 0.50 | 0.50 | Siehe Messwerte; ein Lauf mit zwei Claims und Autorencode kann an die Grenze kommen (Maximum gemessen 0,46 $) und endet dann mit Teilergebnis statt mit mehr Kosten |
| `LIVE_MIN_TAVILY_CREDITS` | 300 | 300 | Reserve für Vorzeige-Läufe |
| `TRUST_PROXY` | 1 | 1 | Hinter Coolify/Caddy, sonst zählt jeder Besucher als dieselbe Proxy-IP |

**Worst-Case-Kosten bis 2026-12-15 gegen 17,02 $** (alles brutto, Server-Anteil 12,00 $):

| Fall | Rechnung | Summe |
|---|---|---|
| Server-Planungsgrenze (ohne Ueberschreitungen laufender Aufrufe) | min(95 % × 12,00 $ = 11,40 $; 12,00 $ − 1,00 $ Untergrenze Stufe 2 + 0,50 $ letzter Lauf = 11,50 $) | **11,40 $** als Planungsziel; verbleibender Puffer ist ohne Reservierungen nicht garantiert |
| Stufe 1, jeder Tag ausgereizt | höchstens 1 Lauf × 0,50 $ + Discovery 0,02 $ = 0,52 $ je Tag; die Untergrenze 6,0 $ stoppt nach 12 − 6 + 0,5 = **6,50 $** (in etwa 12 Tagen), danach Wiedergabe bis 30.11. | 6,50 $ |
| Stufe 2, jeder Tag ausgereizt | höchstens 2 × 0,50 $ + 0,02 $ = 1,02 $ je Tag × 15 Tage = 15,30 $ ohne Untergrenze; die Untergrenze 1,0 $ begrenzt auf den Rest der Stufe 1 minus 1,00 $ plus 0,50 $, also bei ausgeschöpfter Stufe 1 auf 5,00 $ | 5,00 $ (Stufe 1 + 2 zusammen ≤ 11,40 $) |
| Erwartet (Jury) | 20 bis 30 Schnellprüfungen × 0,14 bis 0,25 $ + wenige Discovery-Läufe | 3 bis 7 $ |
| Erwartet, Stufe 2 allein, 2 Läufe an allen 15 Tagen × 0,14 $ (gemessene Schnellprüfung) | 30 × 0,14 $ + 15 × 0,02 $ | 4,50 $, passt in die Reserve |

Planungsziel: 11,40 $ Serververbrauch. Vor jedem Modellversuch werden bekannte Tokenpreise, ein konservativer Texteingabe-Zuschlag und ein explizites Ausgabetokenlimit reserviert. Diese Schaetzung ist keine Garantie fuer Anbieterrechnungen; Sandbox-, Hosting- und externe Kontonutzung sind nicht im Modellbudget enthalten. Die Stufe-1-Reserve soll das Bewertungsfenster schuetzen (4,50 $ Nutzbudget in Stufe 2 entsprechen rund 9 Laeufen zu 0,50 $ oder rund 30 Schnellpruefungen zu 0,14 $). Vor Deployment Kontostand und Kostenpolitik pruefen; gemessene Kosten bestehender Laeufe fuer die Planung nutzen.

**Reservierungen persistent halten:** Neben den bisherigen Logs liegen `costs.reservations.json`, `tavily_usage.reservations.json` und `*.spend.lock` im selben `logs/`-Volume. Unsichere Modellanfragen (Netzwerkfehler, fehlende Usage) behalten ihre Reservierung ohne automatischen Ablauf, auch nach Neustart. Erst nach Abgleich mit dem Anbieter, bei gestopptem Server und ohne laufende Clients, einen belegten Eintrag korrigieren; nicht pauschal das Log oder die Reservierungsdateien loeschen. Schreibfehler sperren bezahlte Aufrufe. Genau einen API-Prozess betreiben; die Dateisperre koordiniert Ausgaben, Jobplaetze und Tageszaehler bleiben pro API-Prozess.

**Restrisiko:** Wer jeden Tag kurz nach Mitternacht (Reset) startet, belegt das einzige Tageslimit der Stufe 1 und hält andere ab; die Wiedergabe bleibt nutzbar. In Stufe 2 braucht ein Angreifer zwei IPs. Ein Zugangscode nur für Live-Läufe wurde bewusst nicht gebaut (Entscheidung des Nutzers vom 2026-10-05: offene Demo).

Tavily: Discovery 1/Tag × 16 Credits bis 15.12. ≈ 1 100 Credits; Reproduktion nicht gemessen (`docs/budget_plan.md`); die Untergrenze `LIVE_MIN_TAVILY_CREDITS` schützt. Verfügbarkeit laut `docs/budget_plan.md`: rund 6 025 Credits bis 15.12.

**Verhalten bei Erreichen einer Grenze:** `GET /api/status` meldet `live: false` mit Grund (`budget_floor`, `tavily_floor`, `daily_limit`, `busy`) und `reset_at` bei Tageslimits; `POST /api/reproduce` und `/api/discover` antworten 503 oder 429 mit demselben Grund (nie mit Kontoständen). Das Frontend zeigt „Live checks are paused (…). Here is the recorded run of this check.“ und spielt den Vorab-Lauf ab (`web/src/livestatus.ts`). Fällt der Server aus, bleibt der Pages-Spiegel.

**Termine:** am 2026-12-01 Stufe 2 setzen (`LIVE_MIN_BUDGET_USD=1.0`, `MAX_DAILY_REPRO_JOBS=2`, `MAX_DAILY_COST_USD=1.25`) und von Hand neu deployen; wöchentlich Konsole und Tavily-Dashboard prüfen (`docs/budget_plan.md`, „Regeln und Auslöser“).

## Sicherheit des öffentlichen Pfads

Angriffsfläche: nur diese Endpunkte, alle ohne Anmeldung. Keine Eingabe führt fremden Code aus; Experimente laufen in der Nebius-Sandbox (Netzsperre `unshare -n`, `harness/code_check.py`, Testlabels nie beim Agenten).

| Endpunkt | Eingabe | Schutz |
|---|---|---|
| `POST /api/reproduce` | `catalog_id` (nur Einträge aus `benchmark/catalog.yaml`, sonst 404, max. 120 Zeichen), `quick`, `author_code` | Sperre vor jedem Start, `quick` serverseitig erzwungen, kein freier arXiv-Link, kein Paper-Text vom Besucher |
| `POST /api/discover` | `datasets` (nur unterstützte Datensätze, höchstens 8, Duplikate entfernt) | Dieselbe Sperre; Suchergebnisse aus dem Netz sind nicht vertrauenswürdig: sie landen nur als Kandidaten im Stream, schreiben nie `catalog.yaml`, Code aus Autoren-Repos wird nie ausgeführt |
| `GET …/stream` | Job-ID (UUID4) | höchstens 30 Streams insgesamt, 4 je IP, 60 min je Stream; Reconnect mit `from=` |
| `GET /api/status`, `/api/jobs/current`, `/api/health` | keine | Rate-Limit 240/min je IP; keine Kontostände |

Weitere Schichten: Rate-Limit je IP (240/min GET, 10/min POST), Body-Limit 16 KB (App) und 64 KB (Caddy), kein CORS (gleiche Herkunft), `/docs`, `/redoc`, `/openapi.json` aus, Security-Header (CSP ohne `unsafe-inline` für Skripte, `nosniff`, `frame-ancestors 'none'`, `no-referrer`), Container als Nutzer `app` (UID 10001) ohne Capabilities (Compose). Ereignisse, Fehlermeldungen und die exportierten Dateien eines Live-Laufs laufen durch `web_guard.redact_text` (Schlüssel aus der Umgebung, bekannte Schlüsselmuster, `/home/<x>`-Pfade). Keine Admin- oder Debug-Endpunkte. IP-Adressen stehen nur als Hash in `live_jobs.jsonl`.

**Bekannte Grenzen:** Bei `TRUST_PROXY=1` muss der Port 8000 aus dem Internet unerreichbar sein (Coolify/Compose: nur der Proxy); sonst lässt sich die IP per `X-Forwarded-For` fälschen und das Limit je IP umgehen (die globalen Grenzen bleiben). Der Job-ID-Link zum laufenden Job steht in `/api/jobs/current` (Zuschauen ist gewollt, Steuern nicht möglich). Die CSP ist im Test (Header, Hash des Theme-Skripts) geprüft, aber nicht in einem echten Browser: nach dem ersten Deployment die Konsole auf „Refused to …“ prüfen.

## Backup

Wichtig sind `/app/logs` (Budget- und Limitzähler) und `/data` (Exporte der Live-Läufe); beides klein. Vor jedem Redeploy und wöchentlich:
```bash
# Volume-Namen mit "docker volume ls" nachsehen (Coolify stellt eine UUID voran)
docker run --rm -v <logs-volume>:/src:ro -v "$PWD":/backup alpine tar czf /backup/nullius-logs-$(date +%F).tgz -C /src .
docker run --rm -v <data-volume>:/src:ro -v "$PWD":/backup alpine tar czf /backup/nullius-data-$(date +%F).tgz -C /src .
```
Wiederherstellen: leeres Volume anlegen, `tar xzf` hineinentpacken, Container starten. Fehlt `costs.jsonl` nach einem Unfall, `BUDGET_TOTAL_USD` auf den aktuellen Konsolen-Stand setzen und `BUDGET_SINCE` auf jetzt (sonst zählt die Sperre zu großzügig). Coolify bietet dafür auch geplante Backups für Volumes. Die Läufe selbst liegen zusätzlich im Repo (`web/public/data`) und im Pages-Spiegel.

**Pages-Spiegel als Absicherung:** `.github/workflows/pages.yml` veröffentlicht `web/dist` mit den Exporten unter `https://script-hpp.github.io/Nullius/` (nur Replay, keine Live-Steuerung). Den Workflow nach jeder Änderung der Exporte neu starten und den Link in den Devpost-Testhinweisen als Rückfall nennen.

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
docker run -d --name nullius-t -p 8011:8000 -e LLM_PROVIDER=tokenfactory -e BUDGET_TOTAL_USD=12 -e BUDGET_SINCE=2026-10-09T11:11Z -e TRUST_PROXY=1 -e NEBIUS_API_KEY=fake -e NEBIUS_PROJECT_ID=fake -e TAVILY_API_KEY=fake -v nullius-t-data:/data nullius-test
curl -s localhost:8011/api/health              # {"ok":true}
```
Am 2026-10-09 erneut geprüft (nach den Schutzmaßnahmen): Build erfolgreich, Container `healthy` (Healthcheck `/api/health`), läuft als UID 10001, im Image keine `.env`-Dateien, `/docs` und `/openapi.json` 404, Security-Header und CSP mit Hash des Theme-Skripts auf der Seite, 20-KB-Body wird mit 413 abgelehnt; Caddyfile und Caddyfile.selfsigned mit `caddy validate` gültig. Am 2026-10-01: `/api/jobs/current` antwortet, Seite und `/data/index.json` werden ausgeliefert, `/api/status` antwortet, Volume wird beim ersten Start befüllt. Kein Live-Lauf, keine echten Aufrufe. Der Spiegel wurde unter dem Unterpfad `/Nullius/` lokal geprüft (Assets und Daten laden, keine Live-Steuerung sichtbar; einziger 404 ist `events.jsonl` eines Altlaufs ohne Ereignisprotokoll, das Frontend spielt ihn aus dem Dossier ab).

## Offen
- Coolify: Menünamen und Traefik-Standardwerte stammen aus dem Gedächtnis und sind an der echten Instanz zu prüfen (Domain-/Port-Feld, Healthcheck-Reiter, Timeouts, Kompression). Nicht getestet, da keine Coolify-Instanz verfügbar ist; der Container selbst ist lokal mit Docker getestet (siehe oben).
- Plattform- und Preset-Namen der Nebius-CPU-VM, statische IP, Preis des kleinsten Presets: vor dem Anlegen prüfen.
- Nebius Serverless nicht getestet.
- Ob Builders-/Trial-Guthaben für Compute gilt: Reviewer mit dem Nutzer klären.
- Browser-Test der CSP an der echten Instanz (siehe „Sicherheit des öffentlichen Pfads“).
