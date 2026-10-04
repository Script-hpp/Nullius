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

Kernidee für Tavily-Preis (ursprünglich): Tavily-Funde werden zu Experimenten. Beispiel: Issue meldet Seed-Abhängigkeit → 5 Branches mit verschiedenen Seeds.

**Stand Tavily (2026-10-04):** Gebaut und im Produktlauf live: Rolle 1 (Repo-Suche, Discovery neuer Papers auf unterstützten Datensätzen), 2 (Zielwert mit Zitat- und Spaltenprüfung), 4 (Setup-Details mit Zitat), 5 (Fehlertext-Suche, seit v3), 3/7 und 6 als **reine Berichtsfunktion** „What others found“ / „Possible explanations“ (`community.py`, nach dem Urteil, ändert nie Plan, Code oder Urteil). **Nicht gebaut:** Funde als eigene Branches (Rolle 3 Stufe 2, Rolle 6 als Hypothese). **Entscheidung: kein weiterer Ausbau vor der Abgabe.** Alle vorab festgelegten Versuche, mit Tavily die Trefferquote messbar zu heben, sind gescheitert oder blieben unter der Schwelle: Rolle 3 Stufe 1 (`results/role3_dlinear_stage1.md`, kein Protokoll-Einwand gefunden), Rolle 4 DLinear und Shchur (`results/role4_*.md`, Shchur um 0,0004 verfehlt), v5 Schritt A und C (`results/v5_step_a.md`, `v5_step_c.md`), Benchmark v2–v4 ohne messbaren Unterschied der Arme (v4: 2 vs. 1 von 8; Median-Abstand post hoc 0,017 vs. 0,029). Tavily wird deshalb als Produkt-Fähigkeit dargestellt (Discovery, Belege mit Hash, „Worte vs. Messung“), nicht als Genauigkeitsgewinn.

> **Standard ist live** (`TAVILY_MODE=live`): ein normaler Lauf ruft Tavily zur Laufzeit auf (Paper-Text, README), gespeichert mit Modus, Zeitstempel und Hash (`Dossier.sources`, `events.jsonl`, `paper_text.json`). Fixtures nur mit `--offline` oder `TAVILY_MODE=fixture` (Tests, vorab festgelegte Experimente, Benchmark: `tavily_mode="fixture"` in den Skripten). Der Tavily-Preis verlangt einen echten Aufruf zur Laufzeit.

> **Tavily-Key sparsam benutzen.** Free/Trial-Kontingent begrenzt (Credits pro Monat), kein unlimitiertes Guthaben wie bei OpenRouter Free. Jeder Aufruf muss Frage beantworten, die sonst nicht beantwortbar ist (siehe Tabelle oben) — kein Testen/Debuggen mit echten Tavily-Calls, wenn Mock/Fixture reicht. Crawl immer mit `limit`, Research max 1× pro Paper (siehe Rolle 7). Bei Unsicherheit: erst Dossier/Cache prüfen, ob Frage schon beantwortet wurde, bevor neuer Call.

## Modus und Rolle 4 (Stand Meilenstein 2)

**Modus „Nachbau“** (einziger gebauter Modus): Der Agent schreibt eigenen Code und übernimmt nur belegte Angaben aus Paper und Repo. Das Original-Repo liegt nie in einem Agenten-Branch, es wird nur in einem eigenen Lese-Snapshot des Orchestrators geklont (Commit im Dossier). „Nachlauf“ (Original-Code ausführen) prüft etwas anderes; er ist als Experiment gebaut (`nachlauf*.py`, `scripts/run_nachlauf.py`), trägt aber nur an den zwei Entwicklungs-Einheiten (RLinear, DLinear) und scheiterte an vier neuen Einheiten zweimal mit 0 von 4 (`results/nachlauf_v1.md`). Nicht im Produkt, nicht in der API, nicht bewerben; kein dritter Versuch. Modus steht im Dossier und im Zeugnis.

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
- Ultra nur für Plan und Endurteil
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

**Video (Stand 2026-10-04): Film v8 fertig gerendert**, 2:38, ohne Sprecher, nur synthetische Musik und Einblendungen. Quelle und Render: Worktree `reviewer`, Branch `video-stage`, `video/` (Ausgabe `video/out/film.mp4`, Regie `docs/video_director.md`); gerendert aus der App auf `main` `ceb3ea3`. Zuständig: Reviewer-Session. `docs/video_script.md` ist nur noch historisch. Ablauf: Hook (Nature-Umfrage 70 %) → ein Claim („GCN achieves 81.5 % on Cora“) → Discovery (aufgezeichneter Live-Lauf, beschriftet) → Katalog → Live-Seite (Kipf-Katalog-Lauf vom 4. Okt als Replay, beschriftet) → „Read. Write. Run. Measure.“ (Tavily, Nemotron auf Token Factory mit Ultra/Super/Lightning, Nebius Sandboxes, Harness) → 10 Seeds, „Reproduced“ → Placebo 88 % „Not reproduced“ → Beleg (10 min, 0,14 $) → Evidence, What-if, Branch-Baum → „48 of 48 caught“ → ehrliche Grenze (2 von 8, nur Graph/Zeitreihe, CPU) → Namen statt Logos. Offen: Upload auf YouTube (öffentlich), Link ins Devpost-Formular. Jede neue UI-Änderung auf `main` heißt: neu rendern lassen, bevor hochgeladen wird.

Evidence-Ansicht in UI: links, was Tavily gefunden hat, rechts, was Sandbox gemessen hat.

Pflicht bei Einreichung:
- [ ] Läuft auf Token Factory, nutzt Nemotron
- [ ] Öffentliches Repo mit Open-Source-Lizenz
- [ ] Demo-URL
- [ ] Demo-Video (max. 3 Min.)
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
    * Zugangscode für die Jury.
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

**Live-Modus (gebaut):** FastAPI `src/repro_agent/api.py`, Start `uvicorn repro_agent.api:app`; Auslieferung von API und Web-App in einem Container über `deploy/serve.py` (`Dockerfile`, `docs/deploy.md`). Das Starten von Jobs braucht `ACCESS_CODE`, ein Lauf gleichzeitig (Semaphor im Speicher, genau ein uvicorn-Worker), Tageslimits `MAX_DAILY_JOBS` (Discovery, Default 5) und `MAX_DAILY_REPRO_JOBS` (Reproduktion, Default 3).
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
- [x] Hosting-Optionen verglichen, Empfehlung Coolify (`docs/deploy.md`); **Deployment und Demo-URL stehen aus**
- [x] Benchmark auswerten, Zahlen ins README (v4, präzisionsorientiert; Devpost-Texte auf v4)
- [x] Katalog: Referenz-Papers Kipf und Shchur mit echten Läufen (2026-10-04)
- [x] Demo-Video gerendert (Film v8, siehe „Demo und Einreichung“)
- [x] Live-Läufe überschrieben `benchmark/fixtures/` (gefunden und behoben 2026-10-04, Merge `fix-live-fixtures`): `TAVILY_MODE=live` schreibt nie mehr nach `benchmark/fixtures/`, Tests arbeiten auf einer Kopie mit Wächter. Bekannte Grenze: Repo-Suche, Repo-Dateien und Zielwert-Extraktion **lesen** im Live-Modus weiter vorhandene Fixtures (kein neuer Tavily-/LLM-Aufruf, wenn eine Fixture existiert)
- [ ] Deployment und Demo-URL (Nutzer), Repo öffentlich (Nutzer)
- [ ] Video auf YouTube hochladen, Devpost-Formular füllen (Entwurf `docs/devpost_submission.md`; Demo-URL, Video-Link, Zugangscode noch `TBD`), Feedback-Feld aus `docs/nebius_feedback.md`

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
