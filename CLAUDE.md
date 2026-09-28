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
TF_MODEL_FAST=TODO               # Nano oder 3.5 Lightning, ID via GET /v1/models prüfen
TF_MODEL_CODE=TODO               # Super
TF_MODEL_REASONING=TODO          # Ultra

# ---- Tavily ----
TAVILY_API_KEY=

# ---- Kostenbremsen ----
MAX_COST_PER_RUN_USD=0.50
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

| Abstand | Urteil |
|---|---|
| ≤ Toleranz | reproduziert |
| ≤ 3 × Toleranz | teilweise |
| darüber | nicht reproduziert |
| Harness-Check fehlgeschlagen | ungültig |

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
    status: Literal["fehler", "abweichung", "stabil", "ungueltig"]
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
- Kostenbremse: Abbruch bei `MAX_COST_PER_RUN_USD`
- Versuchs-Limit: `MAX_ATTEMPTS_PER_CLAIM=4`
- Kosten-Log: jeder Aufruf mit Modell, Tokens, geschätzten Kosten in `logs/costs.jsonl`
- Ultra nur für Plan und Endurteil
- Tavily Crawl immer mit `limit`, Research max 1× pro Paper
- Tavily-Key ist scharf (echtes Kontingent) — auch beim Entwickeln/Testen sparsam einsetzen, nicht in Schleifen oder Smoke-Tests verballern
- Usage-Seite in TF-Konsole wöchentlich prüfen

## Benchmark

8–10 Papers mit bekanntem Reproduktionsergebnis (ML Reproducibility Challenge, Kritik-Papers). Jedes Paper läuft zweimal: mit und ohne Tavily.

Kennzahl: Übereinstimmung der Urteile mit menschlichen Reproduktionsstudien. Ziel-Satz fürs README: „Mit Tavily stimmten X von 10 Urteilen überein, ohne Tavily Y."

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

## Offene Punkte

- [x] Sandbox-Beta-Freigabe abwarten (genehmigt, kostenlos)
- [ ] Rate-Limits von Nebius-Sandbox prüfen
- [ ] Exakte Modell-IDs auf OpenRouter und Token Factory (`GET /v1/models`) eintragen
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
