# Nächster Schritt: Discovery-Kandidaten durch den Reproduktions-Loop (Katalog-Pipeline)

Stand: 2026-09-30, `main` bei `15a065a` (danach nur UI-Erklärtexte). Verfasst von der Reviewer-Session, die Discovery und UI begleitet hat.

## Worum es geht

Die Live-Discovery (`src/repro_agent/discovery.py`, UI `#discover`) findet mit Tavily neue Papers auf unseren Benchmarks und liefert **Kandidaten**: ein Claim mit Methode, Zahl, wörtlichem Zitat, Protokoll-Status und evtl. Code-Link. Diese Kandidaten sind **nur gelesen, nicht nachgerechnet** (Status `ungeprueft`, UI-Label "UNVERIFIED CANDIDATE").

Laut Produktstrategie (CLAUDE.md, Abschnitt 3 "Live-Katalog") darf ein Paper erst in den Jury-Katalog, wenn es **einmal komplett durch den echten Reproduktions-Loop** gelaufen ist (auch bei negativem Ergebnis, aber ohne Absturz). Dieser Schritt fehlt: Kandidat -> Loop -> Urteil -> Status `geprueft` -> Katalog.

## Was es schon gibt

| Baustein | Ort | Stand |
|---|---|---|
| Discovery | `src/repro_agent/discovery.py` | Tavily Search (Papers, Code, Community) + Extract mit Frage, LLM-Claim (Rolle `extract`, Super), Code-Prüfungen, Cache `logs/discovery_cache.json` (`cache_version` 2) |
| Discovery-API | `src/repro_agent/api.py` | `POST /api/discover` (Zugangscode aus `ACCESS_CODE`, Whitelist aus `TASKS`, 1 Job gleichzeitig, `MAX_DAILY_JOBS`), `GET /api/discover/{id}/stream` (SSE) |
| Discovery-UI | `web/src/components/Discovery.tsx`, `web/src/discovery.ts` | Live + Replay (`web/public/data/discovery/replay.jsonl`, Token-Factory-Aufnahme) |
| Reproduktions-Loop | `src/repro_agent/loop.py` (`Orchestrator.run(paper_id)`) | liest Paper und Claims aus `benchmark/papers.yaml` (`load_claims`), Rollen 2/4, Code schreiben, Sandbox, Harness, Urteil, `runs/<paper>/<ts>/` mit `dossier.json`, `events.jsonl`, `zeugnis.md` |
| Tasks | `src/repro_agent/tasks.py` | `Cora` (Planetoid), `Cora-LCC` (Shchur-Splits), `ETTh1` (Horizont 96, Eingabelänge frei bis 720) |
| Evidence-Ansicht | `web/` + `scripts/export_site_data.py` | zeigt Loop-Läufe aus `runs/` |

Kandidaten im Cache haben u. a.: `title`, `arxiv_id`, `dataset`, `claim` {`method`, `metric` (`accuracy`/`mse`/`mae`), `target`, `horizon` (nur ETTh1), `quote`, `protocol` (`passt`/`unklar`), `matched_task`, `protocol_quote`, `column_check` (`ok`/`unchecked`)}, `github_url`, `source_url`, `community`, `status`.

Beispiele aus echten Läufen (Token Factory): SL-GCN 82.92 % auf Cora-LCC (2511.08094, Protokoll per Zitat, Spalte geprüft, Code `kevvzhang/StuartLandauGNN`); MoHETS MSE 0.350 auf ETTh1 @96 (2601.21866); HaKAN MSE 0.369 auf ETTh1 @96 (2601.18837); Acacia frozen 58.17 % auf Cora (2609.30894).

## Was zu bauen ist

### 1. Kandidat -> Loop-Eingabe (Adapter)

Der Loop kennt nur `benchmark/papers.yaml`. Zwei Wege, Empfehlung **(b)**:

- (a) Kandidaten nach `papers.yaml` schreiben. Nachteil: vermischt kuratierten Benchmark (Vorab-Festlegung, Experiment-Disziplin) mit automatisch gefundenen Papers.
- (b) Neue Datei `benchmark/catalog.yaml` (oder JSON) nur für Discovery-Kandidaten, und `load_claims` bzw. `Orchestrator.run` bekommt eine zweite Quelle (z. B. `run(paper_id, source="catalog")` oder ein eigener Loader). `papers.yaml` bleibt unberührt.

Umrechnung pro Kandidat:
- `id`: z. B. `disc_<arxiv_id>` (Punkt durch `_`), `title`, `arxiv_url` = `https://arxiv.org/abs/<arxiv_id>`.
- `dataset` = **`claim.matched_task`**, nicht `dataset` der Discovery (sonst läuft ein Shchur-Paper auf Planetoid). Bei `protocol: unklar` ist `matched_task` leer: siehe Entscheidung E1.
- `claims[0]`: `metric` wie im Claim; **`target` normalisieren**: Loop und Harness rechnen Accuracy als Anteil (`0.815`), Discovery speichert Prozent (`82.92`) -> `/100`, wenn `metric == accuracy` und `target > 1`. MSE/MAE unverändert.
- `tolerance`: feste Regeln wie im Benchmark (CLAUDE.md "Toleranz"): Accuracy ±0.01 (1 Prozentpunkt), MSE/MAE ±0.01 absolut (wie DLinear in `papers.yaml`), sofern das Paper keine Streuung nennt. Vor dem Lauf festlegen und im Zeugnis nennen.
- `text`: aus Methode, Zahl, Task bauen (englisch oder deutsch wie bisher im Zeugnis).
- `source`: `"discovery: <source_url>, Zitat: <quote>"`; das Zitat ist der Beleg.
- `seeds`: Default des Tasks (GCN ~25 s pro Seed; für den Katalog eher 3-5 wegen Kosten, siehe Kosten).
- `fidelity`: leer lassen (Achse 2 bleibt laut CLAUDE.md "nicht geprüft").

### 2. Rolle 2 im Loop nicht doppelt machen

`Orchestrator.resolve_targets` liest den Zielwert per Tavily Extract neu aus dem Paper. Für Discovery-Kandidaten ist der Wert schon belegt (Zitat im Text geprüft). Optionen: Rolle 2 trotzdem laufen lassen (unabhängige Gegenprobe, kostet einen Extract) oder überspringen und das Discovery-Zitat als Beleg übernehmen. Empfehlung: laufen lassen und Abweichung zwischen Discovery-Wert und Rolle-2-Wert im Dossier als Warnung vermerken (die Gegenprobe `agrees=` existiert bereits, `loop.py` ~Zeile 401).

### 3. Status zurückschreiben

Nach dem Loop-Lauf den Kandidaten im Katalog aktualisieren:
- `status`: `geprueft` (Lauf ohne Absturz beendet) oder `lauf_fehlgeschlagen` (Absturz, Budget, Sandbox) -> nicht in den Jury-Katalog.
- `verdict`: Urteil Achse 1 (`reproduziert` / `teilweise` / `nicht reproduziert` / `Agenten-Fehler vermutet` / `nicht reproduzierbar (Setup)`), `measured_mean`, `measured_std`, `seeds`, `run_dir` (Verweis auf `runs/<id>/<ts>/`), `gesamtaussage` (Achse 2 = "nicht geprüft" -> "vorläufig").
- Formulierungen vorsichtig: "mit unserem Setup nicht reproduziert" (CLAUDE.md).

### 4. Skript / Job

`scripts/curate_catalog.py`:
1. Kandidaten aus `logs/discovery_cache.json` (Status `ungeprueft`, gültige `cache_version`) lesen, oder eine Discovery frisch laufen lassen.
2. Filter (Entscheidung E1-E3), dann pro Kandidat Adapter -> `Orchestrator.run`.
3. Status/Urteil in den Katalog schreiben, Kosten pro Kandidat loggen.
4. Harte Kostengrenze pro Skriptlauf (z. B. `--max-usd`), Abbruch bei `BillingBlocked`/`BudgetExceeded`.
5. `--dry-run`: nur Adapter-Ausgabe zeigen, keine Aufrufe.

Nur Token-Factory-Läufe veröffentlichen (CLAUDE.md "Keine Benchmark-Zahlen aus Prototyp").

### 5. UI

- Discovery-Karte: bei `status: geprueft` Urteil + Messwert anzeigen ("CLAIMED 82.92% / MEASURED 81.9%", Delta, Link zur Evidence-Ansicht des Laufs). Label wechselt von "UNVERIFIED CANDIDATE" zu Urteil.
- Katalog-Ansicht (Startseite oder eigene Route): nur geprüfte Papers, Button "Watch replay" (Loop-Events aus `runs/.../events.jsonl`) und "Run quick check" (Schnellprüfung, weniger Seeds, siehe Strategie 3.2).
- `scripts/export_site_data.py` so erweitern, dass Katalog-Läufe mit exportiert werden.

## Entscheidungen (getroffen 2026-09-30, Reviewer im Auftrag des Nutzers)

- **E1 Protokoll unklar -> zulassen, wenn der Task eindeutig ist.** ETTh1 hat genau einen Task, dort laufen `unklar`-Kandidaten mit. Im Dossier und Zeugnis steht "Protokoll im Paper nicht belegt", die Gesamtaussage bleibt höchstens "vorläufig". Cora hat zwei Tasks (Planetoid, Shchur-LCC): ohne `matched_task` kein Lauf, Status `nicht_laufbar` mit Grund. *Warum:* Das Messergebnis ist auch ohne belegtes Protokoll informativ, solange die Einschränkung sichtbar ist; bei Cora würde ein falscher Task das Ergebnis wertlos machen.
- **E2 Spalte ungeprüft -> zulassen mit Warnung.** `column_check: unchecked` läuft, das Zeugnis nennt "Tabellenspalte nicht geprüft, Zahl könnte zu einer Vergleichsmethode gehören" (entspricht der bestehenden Rolle-2-Grenze in CLAUDE.md). `mismatch` wird schon in der Discovery verworfen. *Warum:* Die Spaltenprüfung erkennt nur wenige Kopfzeilenformen; nur `ok` zuzulassen würde fast alle ETTh1-Tabellen ausschließen.
- **E3 Claim -> Hauptergebnis der vorgeschlagenen Methode.** Discovery-Prompt ergänzen: bestes bzw. im Paper hervorgehobenes Ergebnis der Hauptkonfiguration, keine Ablation, keine eingeschränkte Variante (z. B. "frozen", "w/o ..."). Weiterhin ein Claim pro Paper. Kandidaten wie Acacia frozen 58.17 % fallen dadurch nicht weg, bekommen aber beim nächsten Discovery-Lauf ggf. einen anderen Claim. *Warum:* Die Jury versteht "das Paper behauptet X für seine Methode"; Varianten verwirren und sind als Reproduktionsziel schwach.
- **E4 ETTh1 Eingabelänge -> frei (L ≤ 720), nur Horizont 96 zwingend.** Discovery nicht mehr auf "input length 96" einschränken: `DISCOVERY_QUERIES`, `EXTRACT_QUERIES`, `DISCOVERY_PROTOCOL_HINTS` anpassen; `TASK_KEYWORDS` behält "96" (Horizont). Neues Claim-Feld `input_length` (int oder null), wird an den Loop als Setup-Detail übergeben (Quelle: Discovery-Zitat), damit der Agent dieselbe Eingabelänge nutzt. *Warum:* `tasks.py` erlaubt L ≤ 720, DLinear im Benchmark nutzt 336; der Filter auf 96 wirft passende Papers weg.
- **E5 Code -> nicht Pflicht, aber Reihenfolge.** Modus Nachbau braucht kein Repo. Reihenfolge im Kurationslauf: (1) `protocol: passt` + `column_check: ok` + Code, (2) `passt` ohne Code, (3) `unklar`/`unchecked`. Kostengrenze schneidet von hinten ab. *Warum:* Mit Code liefert Rolle 4 belegte Hyperparameter, das erhöht die Chance auf ein faires Urteil.

## Kosten und Zeit (Schätzung, vor dem ersten Lauf prüfen)

- Discovery: ca. 0.02 $ Token Factory + ca. 16 Tavily-Credits pro Lauf (Cora + ETTh1), 60-90 s.
- Loop pro Paper: laut CLAUDE.md ca. 0.09 $ für GCN mit Rollen 2 und 4 (Stand vor Umstellung Rolle 4 auf Super), plus Tavily (Rollen 2, 3, 4). Sandbox (Nebius) kostenlos. Laufzeit hängt an Seeds x Trainingszeit (GCN ~25 s pro Seed, ETTh1-Modelle evtl. länger; 10-Minuten-Grenze pro Claim auf CPU beachten, Nicht-Ziel "kein GPU-Training").
- **Pilot (entschieden):** zuerst **ein** Kandidat, SL-GCN auf Cora-LCC (2511.08094; Protokoll und Spalte belegt, Code vorhanden), mit **10 Seeds** statt 3. *Warum:* Das Paper nennt ±1.39 Streuung; bei 3 Splits liegt der Standardfehler des Mittelwerts bei ca. 0.8 Prozentpunkten, zu nah an der Toleranz von 1 Prozentpunkt für ein belastbares Urteil. GCN-artige Modelle brauchen ca. 25 s pro Seed, also wenige Minuten. Toleranz vorab: ±0.01 (1 Prozentpunkt) auf den Mittelwert. Erst nach dem Pilot weitere Kandidaten.

## Bekannte Grenzen der Discovery (für das README/Video ehrlich benennen)

- Ausbeute schwankt: dieselben Papers ergeben je Lauf 1-4 Kandidaten (LLM-Urteil nicht stabil trotz Temperatur 0; ein zweiter Versuch mit Temperatur 0.3 ist eingebaut).
- Protokoll `passt` ist LLM-Urteil + wörtliches Zitat im Text; der Code prüft das Zitat, nicht die Bedeutung. Schutz: Kennwörter unterstützter Tasks verhindern falsche Rejects.
- Spaltenprüfung erkennt nur die Kopfzeilenformen aus `table_check.py`.
- Repo-Suche verfehlt Repos: Die Community-Suche fand für MoHETS `github.com/evortigosa/mohe_forecast`, die Repo-Suche nicht. Idee: GitHub-Treffer der Community-Suche als Repo-Kandidaten prüfen.
- Tavily-Extract liefert gelegentlich fast leeren Text (16-483 Zeichen); ein Wiederholungsversuch ist eingebaut.

## Arbeitsweise, die sich bewährt hat

- Worker (Sonnet) implementiert mit Tests über Fakes, keine echten Aufrufe; Reviewer prüft Code, Suite und Browser-Screenshots (Playwright) und macht die echten Läufe.
- Jede Behauptung "erledigt" gegen Datei:Zeile und Testausgabe prüfen.
- Echte Läufe mit `LLM_PROVIDER=tokenfactory` nur nach grüner Suite; Kosten aus `logs/costs.jsonl` (brutto = netto x 1.19) berichten.
