# Paper-Reproduktions-Agent

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
| Ultra verfügbar | Nein, Rolle `reasoning` nutzt Super | Ja |
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
TF_MODEL_REASONING=nvidia/Nemotron-3-Ultra-550b-a55b

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
| `reasoning` | Forschungsplan, Hypothesen, Endurteil | Super (free) | Ultra, nur mit verdichteter Forschungsakte |

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

Kernidee für Tavily-Preis: Tavily-Funde werden zu Experimenten. Beispiel: Issue meldet Seed-Abhängigkeit → 5 Branches mit verschiedenen Seeds.

> **Tavily-Key sparsam benutzen.** Free/Trial-Kontingent begrenzt (Credits pro Monat), kein unlimitiertes Guthaben wie bei OpenRouter Free. Jeder Aufruf muss Frage beantworten, die sonst nicht beantwortbar ist (siehe Tabelle oben) — kein Testen/Debuggen mit echten Tavily-Calls, wenn Mock/Fixture reicht. Crawl immer mit `limit`, Research max 1× pro Paper (siehe Rolle 7). Bei Unsicherheit: erst Dossier/Cache prüfen, ob Frage schon beantwortet wurde, bevor neuer Call.

## Modus und Rolle 4 (Stand Meilenstein 2)

**Modus „Nachbau“** (einziger gebauter Modus): Der Agent schreibt eigenen Code und übernimmt nur belegte Angaben aus Paper und Repo. Das Original-Repo liegt nie in einem Agenten-Branch, es wird nur in einem eigenen Lese-Snapshot des Orchestrators geklont (Commit im Dossier). „Nachlauf“ (Original-Code ausführen) prüft etwas anderes und ist nicht gebaut. Modus steht im Dossier und im Zeugnis.

**Rolle 4 (Setup-Details):** Quellen sind Paper-Text, README (Tavily Extract) und vom LLM gewählte Repo-Dateien (Sandbox-`git clone`). Extraktion abschnittsweise mit Rolle `extract` (= Modell von `code`, Super; Ultra nur für Plan und Endurteil). Jede Angabe braucht ein wörtliches Zitat, das im jeweiligen Dokument vorkommt, sonst wird sie verworfen. Jede Angabe trägt ihre Quelle.

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

**Zweidimensionales Urteil.** Achse 1 (Zahl): reproduziert / teilweise / nicht reproduziert / Agenten-Fehler vermutet / nicht reproduzierbar (Setup), vom Prüfer nie verändert. Achse 2 (Methodentreue, `Verdict.methode`): ja / teilweise / nein / nicht geprüft. Gesamtaussage (`dossier.gesamtaussage`): reproduziert + ja = „Paper reproduziert“; reproduziert + nein/teilweise = „Zielwert erreicht mit anderem Modell“; nicht reproduziert + ja = „nicht reproduziert“; Achse 2 nicht geprüft = „vorläufig“. Benchmark zählt nur „Paper reproduziert“ und „nicht reproduziert“. **Stand:** Eichung des Methodentreue-Prüfers v1 gescheitert (5/10, 2× ja↔nein, `benchmark/experiments/results/calibration_fidelity_v1.md`), Achse 2 daher überall „nicht geprüft“. Eichungen v2 (7/10) und v3 (8/11, Schwelle 9/11) ebenfalls gescheitert (`results/calibration_fidelity_v2.md`, `_v3.md`). **Endgültig, vorab festgelegt:** Achse 2 bleibt für den Benchmark „nicht geprüft“, im README offen benennen (Eichungen v1–v3 mit Ergebnissen); keine weitere Eichung vor der Abgabe. Für GCN gilt zusätzlich die Graph-Ablation.

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

Verfügbar: 25 $ Token Factory, Tavily-Key. Builders Program beantragt (+25 $ TF, +25 $ Tavily, verfallen 90 Tage nach Ausgabe).

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
- Ultra nur für Plan und Endurteil
- Tavily Crawl immer mit `limit`, Research max 1× pro Paper
- Tavily-Key ist scharf (echtes Kontingent) — auch beim Entwickeln/Testen sparsam einsetzen, nicht in Schleifen oder Smoke-Tests verballern
- Usage-Seite in TF-Konsole wöchentlich prüfen

## Benchmark

8–10 Papers mit bekanntem Reproduktionsergebnis (ML Reproducibility Challenge, Kritik-Papers). Jedes Paper läuft zweimal: mit und ohne Tavily.

Kennzahl: Übereinstimmung der Urteile mit menschlichen Reproduktionsstudien. Ziel-Satz fürs README: „Mit Tavily stimmten X von 10 Urteilen überein, ohne Tavily Y."

**Task Shchur (Cora, zufällige Splits, gebaut):** GCN-Claim 81,5 % (Tabelle 1, belegt per Tavily Extract + Spaltenprüfung), größte Komponente, 20/30/Rest, 30 Splits × 1 Initialisierung statt 100 × 20 (Details, Toleranz und Harness-Validierung: `benchmark/experiments/shchur_task.md`; menschliche Referenz 81,1 ± 1,2 %). Der Task prüft nur die GCN-Zahl, nicht die Rangfolge der Modelle. Abgeschnittene LLM-Antworten (`finish_reason: length`) werden nie gecacht, Wiederholungen der Extraktion nutzen andere Temperatur und doppelte Grenze (sonst kam die abgeschnittene Antwort aus dem Cache).

Kandidaten (Zielwerte direkt aus Papers übernehmen, nicht aus Gedächtnis):

| Paper | Erwartung | Datensatz |
|---|---|---|
| Kipf & Welling 2017, GCN | reproduziert | Cora |
| Shchur et al. 2018, Pitfalls of GNN Evaluation | Ranking hängt vom Split ab | Cora, Citeseer |
| Zeng et al. 2023, Are Transformers Effective for TS Forecasting? (DLinear) | reproduziert | ETTh1 |
| Ferrari Dacrema et al. 2019, Are We Really Making Much Progress? | Baselines schlagen neuronale Modelle | MovieLens |
| Shwartz-Ziv & Armon 2021, Tabular Data: DL Is Not All You Need | XGBoost vorn | kleine Tabellen-Datensätze |

## Demo und Einreichung

Drei Akte im 3-Minuten-Video:
1. GCN auf Cora → grün
2. DLinear schlägt Transformer → Überraschung hält
3. Recsys-Paper → fällt durch, mit Tavily-Evidenz und Branch-Baum

Evidence-Ansicht in UI: links, was Tavily gefunden hat, rechts, was Sandbox gemessen hat.

Pflicht bei Einreichung:
- [ ] Läuft auf Token Factory, nutzt Nemotron
- [ ] Öffentliches Repo mit Open-Source-Lizenz
- [ ] Demo-URL
- [ ] Demo-Video (max. 3 Min.)
- [ ] README: Architektur, Benchmark-Ergebnisse, Hinweis auf verwandte Arbeiten (PaperBench, CORE-Bench) und unser Unterschied

Urteile vorsichtig formulieren: „mit unserem Setup nicht reproduziert", immer mit Versuchen.

## Risiken

| Risiko | Gegenmaßnahme |
|---|---|
| Nebius-Sandbox hat Rate-Limits | prüfen, sonst `LocalSandbox` als Fallback |
| Fremde Repos laufen nicht (Versionen, tote Links) | einfache Papers zuerst, Tavily-Fehlersuche, Versuchs-Limit |
| Budget leer | Prototyp auf OpenRouter Free, Cache, Kostenbremse, Builders-Credits |
| Free-Endpunkt instabil | Retries mit Backoff; kritische Läufe nur auf Prod |
| Agent schönt Zahlen | unabhängiger Harness, Agent liefert nur Vorhersagen |
| Falsches „nicht reproduziert" | vorsichtige Formulierung, Toleranz vorab, Versuche offenlegen |

## Evidence-Ansicht (`web/`)

Statische React-App (Vite, Tailwind, Recharts, React Flow, Framer Motion), keine Backend-Pflicht, keine Keys. Grundlage: `runs/<paper>/<ts>/events.jsonl` (Ereignisprotokoll des Loops, `events.py`) und `dossier.json`. Export ausgewählter Läufe mit `scripts/export_site_data.py` nach `web/public/data`; nur Prod-Läufe veröffentlichen. Ältere Läufe ohne `events.jsonl` werden aus dem Dossier abgespielt. Live-Modus (FastAPI + SSE) ist nicht gebaut; wenn überhaupt, nur für ein festes Beispiel mit Zugangscode und Kostenlimit, Keys nie ins Frontend.

## Offene Punkte

- [x] Sandbox-Beta-Freigabe abwarten (genehmigt, kostenlos)
- [x] Rate-Limits von Nebius-Sandbox prüfen (2026-09-28: 20 parallele Branches ohne Fehler; höher nicht getestet, Semaphor im Orchestrator trotzdem sinnvoll)
- [x] Exakte Modell-IDs auf OpenRouter und Token Factory eintragen (TF geprüft 2026-09-28: Lightning 0,06/0,24 $, Super 0,30/0,90 $, Ultra 1/3 $ pro Mio. Tokens)
- [ ] Builders-Program-Codes einlösen
- [ ] Preise pro Modell in TF-Konsole notieren, Kosten-Log kalibrieren
- [ ] Benchmark-Liste auf 8–10 Papers mit belegten Ergebnissen erweitern
- [ ] Hosting für Demo-URL wählen (Nebius Serverless empfohlen, nicht Pflicht)

## Projektstruktur (Vorschlag)

```
repro-agent/
├── .env.example
├── CLAUDE.md
├── src/
│   ├── config.py          # liest .env, wählt Anbieter
│   ├── llm.py             # OpenRouter | Token Factory, Retry, Cache, Kosten-Log
│   ├── sandbox/
│   │   ├── base.py        # Protocol
│   │   ├── local.py       # Docker-Fallback
│   │   └── nebius.py      # contree-sdk
│   ├── tavily_tools.py    # die 7 Rollen
│   ├── harness/
│   │   ├── verify.py
│   │   └── metrics.py
│   ├── loop.py            # Orchestrator
│   ├── dossier.py         # Forschungsakte
│   └── report.py
├── benchmark/papers.yaml
├── runs/                  # nicht in Git
└── logs/                  # nicht in Git
```
