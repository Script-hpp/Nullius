# recall cg v1: Vorab-Festlegung (Plan A + Placebos)

Stand 2026-10-09, Branch `Script-hpp/recall-cg-v1-run`. Diese Datei, `scripts/recall_cg_v1.py` (Laufplan, Gate, Zaehlregel, Deckel), `scripts/run_recall_cg_v1.py` (Planer, Worker, Vorbereitung), `scripts/run_recall_cg_v1_mechanics.py`, `scripts/eval_recall_cg_v1.py`, `tests/test_recall_cg_v1.py` und der Placebo-Claim `rlinear_etth1_96_mse_placebo` in `benchmark/papers.yaml` werden **vor dem ersten Modell-Aufruf** committet. Danach aendern sich Einheiten, Konfiguration, Regeln, Schwellen, Deckel und Zaehlung nicht mehr. Erlaubt sind nur dokumentierte Infrastruktur-Korrekturen (Abschnitt „Abbrueche“) und redaktionelle Aenderungen dieser Datei.

Entscheidung der Nutzerin/des Nutzers vom 2026-10-09, 02:15: **Plan A + Placebos**, Kostendeckel **9,00 $ brutto**, harter Stopp auf Aufruf-Ebene bei **8,75 $ brutto**, Messweg Nebius-Sandbox. Grundlage: `docs/recall_analysis_3.md` Abschnitt 5 „Plan A“ und `docs/recall_plan.md` Abschnitt 6 (Stufen-Gate).

## Einordnung

- **Name:** „recall cg v1“. Kein v5 (das v5-Gate ist gescheitert und bleibt so berichtet). Benchmark v4 (Arm B 2 von 8) bleibt der Vergleich und die Hauptzahl, **nie summiert**.
- **Modus „Nachbau aus Paper und Autorencode“** (offen benannt, `code_grounded=True`): der rohe Repo-Ausschnitt (Modelldatei, importierte Schichten, aufgerufene Hilfsfunktionen, Start-Skript, Konfigurationen) steht in Plan-, Code- und Reparatur-Prompt, Zahlenwerte erzwingt der Steckbrief (S2) deterministisch. Das Urteil heisst: „die Methode, wie Paper **und Code** sie festlegen, neu implementiert unter unserem Harness“. Es ist kein unabhaengiger Nachbau „aus Worten“ mehr und kein Nachlauf des Original-Codes (Daten, Metrik, Ein-/Ausgabe sind unsere). **method fidelity not checked** (Achse 2 bleibt „nicht geprueft“).
- **Vorgaenger:** G0 (`recall_cg_gate.md`, lokales Docker) wurde gestoppt und ist **unvollstaendig** (50 von 54 Proben, Ergebnis dort: Regel (a) knapp nicht erfuellt). G1 (`recall_cg_gate_g1.md`, Branch `Script-hpp/gat-analysis`, Vorab-Festlegung `436b4b2`, Ergebnis `64e7d40`, Verweis `34c4777`) ist 18 von 18 fertig und nach seiner Regel **bestanden** (Entwicklung `cg` 10 gegen `heute` 4, getragen allein von Shchur-GAT `cg` 2 von 3; Veličković-GAT 0 von 3 in beiden Armen; RLinear `cg` 2 von 3; 0,6053 $ brutto). G1 war ein schwaches Signal (ein Treffer Vorsprung, In-Sample-Fixes, anderer Messweg als G0) und traegt nur die Entscheidung, Plan A zu bezahlen. Recall cg v1 ist die Messung dieser Entscheidung.
- **Code-Stand:** Basis `436b4b2` plus der Nach-Lauf-Fix F2 `dfde860` (Stack-Kuerzung vom Anfang statt vom Ende, Abdeckungs-Frames weglassen; im Branch als `98ab24f`). G1 wurde mit der fehlerhaften Kuerzung gemessen; recall cg v1 laeuft mit der korrigierten. Jede Ergebniszeile traegt `code_commit`.

## Frage

Hebt das Code-Grounding (Autorencode roh im Prompt, Zahlen erzwungen, Ausgabehelfer, Fixes F1 bis F4 und F8) im **vollen Produkt-Loop** die Uebereinstimmung der Mehrheitsurteile mit den Referenzurteilen gegenueber v4 (Arm B: 2 von 8), und haelt es auf einem Paper, das beim Entwurf der Fixes nicht betrachtet wurde (RLinear)? Und bleibt die Negativkontrolle (Placebos) sauber?

## Konfiguration (fest)

- **Arm B only** (mit Tavily; `tavily_mode="fixture"`, gespeicherte Ergebnisse; **kein Tavily-Aufruf**, die Zugriffsfunktion des Orchestrators wirft im Lauf). Rollen 2 und 4 aus Fixtures, Rolle 5 (Fehlertext-Suche) **nicht live** (`error_search_live=False`, wie v2; v3 und v4 hatten Rolle 5 live im Arm B).
- **Orchestrator:** `graph_repair=True`, `fresh=True`, `llm_cache=False`, `code_grounded=True` (setzt Steckbrief-Modus und `setup_v2`), `io_helper=True`. Voller Produkt-Loop (Planung, Kandidaten, Reparatur, Graph-Ablation, Urteil), **nicht** das Gate-Skript. `MAX_ATTEMPTS_PER_CLAIM=4`, `MAX_PARALLEL_SEEDS=3`.
- **Modelle:** Token Factory, Rollen wie `.env` (Prod): `reasoning` Ultra (Plan), `code` Super, `fast` Lightning. (G0 und G1 liefen mit Super in allen Rollen; hier gilt die Produktkonfiguration.)
- **Messweg:** Nebius-Sandbox, 4 vCPU und 4 GB je Branch, Netzsperre `unshare -n`, Harness frisch kopiert. **Das ist derselbe Weg wie v2 bis v4** (v2 bis v4 liefen auf Nebius, `benchmark_v2.md`), nicht der von G0 (lokales Docker, 12 GB je GAT-Container).
- **Zeitlimit je Seed:** 600 s, ausser Veličković-GAT 660 s und Shchur-GCN 900 s (Regel `timeouts.train_timeout`, aus der Referenzlaufzeit auf Nebius, `recall_cg_gate_g1_timeouts.json`). Trainingsbefehl `timeout -s ABRT` mit `faulthandler` (F2, weicher Timeout 15 s vor dem harten Limit).
- **Kostenkappe je Lauf:** `MAX_COST_PER_RUN_USD=0.75` (v2 bis v4: 0,50). Begruendung: Generation 0 enthaelt zwei Claims in einem Lauf, die cg-Prompts sind laenger (v4 Arm B: Mittel 0,24 $ brutto, Maximum 0,36 $ je Lauf mit Placebo). Die Gesamtkappe unten bleibt die bindende.
- **Fixtures:** `benchmark/fixtures` bleibt unberuehrt. Ein Overlay (`runs/recall_cg_v1_fixtures/`) = Kopie davon plus die committeten Zusaetze in `benchmark/experiments/recall_cg_v1_fixtures/`. Die Zusaetze entstehen einmalig in der Vorbereitung (Schritt 0), danach sehen alle Laeufe dieselben Dateien:
  - **S4-Repo-Auszug je Einheit** (neun Dateien `<Repo>__<Einheit>.json`): Klon in der Nebius-Sandbox, Dateiwahl durch `select_files` (ein kleiner Modell-Aufruf je Einheit), Pflicht-Modelldatei, Importe zwei Ebenen. G0 und G1 nutzten stattdessen die Dateiliste des gespeicherten v4-Laufs mit lokalen Klonen (die Auszuege koennen sich unterscheiden).
  - **RLinear:** Paper-Text aus dem gespeicherten Lauf vom 2026-10-02 (Tavily Extract, kein neuer Aufruf), README **aus dem Repo-Auszug** (`README.md` des Klons) statt aus einem Tavily-Extract, Zielwert-Beleg (Rolle 2) per Modell-Aufruf einmalig.
  - Schritt 0 kostet geschaetzt 0,05 bis 0,15 $ und zaehlt in den Deckel.
- **Seeds:** wie in `benchmark/papers.yaml` (Kipf 10, SGC 10, DGI 10, Veličković-GAT 10, DLinear 5, NLinear 5, RLinear 5, Shchur-GCN und Shchur-GAT je 30 Splits × 1). Toleranz ±0,01 (Genauigkeit relativ 0,01, MSE absolut 0,01), Mittelwert entscheidet.

## Einheiten, Claims, Referenzurteile

Die acht Entwicklungseinheiten von Benchmark v2 (Referenzen und Herkunft wie `benchmark_v2.md`), dazu **RLinear als Held-out** (Li et al. 2023, Referenzlauf `rlinear_reference.py` im Harness, Mittel 0,3704, Zielwert 0,366 aus Tabelle 2; `b1_candidates.md`). Kein Referenzurteil wird nach Sicht der Ergebnisse geaendert.

```yaml
reference_verdicts:
  tolerance: 0.01
  units:
    kipf_gcn_2017:        {claim: gcn_cora_acc,         placebo: gcn_cora_acc_placebo,         target: 0.815, placebo_target: 0.88,  reference_mean: 0.8136, origin: R}
    wu_sgc_2019:          {claim: sgc_cora_acc,         placebo: sgc_cora_acc_placebo,         target: 0.810, placebo_target: 0.88,  reference_mean: 0.8081, origin: R}
    zeng_dlinear_2023:    {claim: dlinear_etth1_96_mse, placebo: dlinear_etth1_96_mse_placebo, target: 0.375, placebo_target: 0.300, reference_mean: 0.3809, origin: R}
    shchur_pitfalls_2018: {claim: shchur_gcn_cora_acc,  placebo: shchur_gcn_cora_acc_placebo,  target: 0.815, placebo_target: 0.88,  reference_mean: 0.8107, origin: "R+H"}
    velickovic_gat_2018:  {claim: gat_cora_acc,         placebo: gat_cora_acc_placebo,         target: 0.830, placebo_target: 0.88,  reference_mean: 0.8280, origin: R}
    velickovic_dgi_2019:  {claim: dgi_cora_acc,         placebo: dgi_cora_acc_placebo,         target: 0.823, placebo_target: 0.88,  reference_mean: 0.8047, origin: R}
    shchur_gat_2018:      {claim: shchur_gat_cora_acc,  placebo: shchur_gat_cora_acc_placebo,  target: 0.818, placebo_target: 0.88,  reference_mean: 0.8140, origin: "R+H"}
    zeng_nlinear_2023:    {claim: nlinear_etth1_96_mse, placebo: nlinear_etth1_96_mse_placebo, target: 0.374, placebo_target: 0.300, reference_mean: 0.3739, origin: R}
    rlinear_2023:         {claim: rlinear_etth1_96_mse, placebo: rlinear_etth1_96_mse_placebo, target: 0.366, placebo_target: 0.300, reference_mean: 0.3704, origin: R}
```

Referenzurteile aus `judge(reference_mean, target, 0.01)`: alle „reproduziert“ ausser DGI „teilweise“. Placebos: Referenzurteil „nicht reproduziert“ (Abstand ≥ 0,052). Placebo-Zielwerte stehen als `origin: placebo` in `papers.yaml`; Rolle 2 laeuft nur fuer die echten Claims.

## Ablauf

**Schritt 0 (Vorbereitung, einmalig):** `scripts/run_recall_cg_v1.py --prep` baut die Overlay-Fixtures (siehe oben) und kopiert die neuen Dateien nach `benchmark/experiments/recall_cg_v1_fixtures/` (werden mit dem Ergebnis committet).

**Stufe 1 (16 Laeufe, gezaehlt):** die acht Entwicklungseinheiten × Generation 0 und 1, echter Claim; **in Generation 0 je Einheit zusaetzlich der Placebo-Claim** im selben Lauf (eigener Plan, Code, Versuche). Reihenfolge: erst alle Generationen 0 (langsame Einheiten zuerst), dann alle Generationen 1; bis zu 5 Laeufe gleichzeitig.

**Stufen-Gate** (`recall_plan.md` Abschnitt 6): F = Entwicklungseinheiten, bei denen **beide** Generationen die Referenzklasse treffen; U = Einheiten, bei denen genau eine sie trifft. **Weiter nur bei F ≥ 2 und F + U ≥ 4.** Das Gate begrenzt den Verlust, es beweist keine Wirkung (bei reinem Zufall geht es in etwa jedem vierten Fall weiter). Fehlt eine Stufe-1-Zelle (Kostenstopp, Abbruch), ist das Gate **nicht entschieden**, Stufe 2 startet nicht, das Ergebnis heisst „unvollstaendig“.

**Stufe 2 (nur wenn das Gate besteht), in dieser Reihenfolge, damit ein Kostenstopp das Unwichtigste zuletzt trifft:**
1. RLinear Generation 0 (mit Placebo) und 1;
2. Entwicklungs-Generation 2 **nur** bei Einheiten, deren Generation 0 und 1 in der Klasse abweichen;
3. RLinear Generation 2, nur wenn Generation 0 und 1 abweichen.

**Faellt das Gate:** Stopp, nichts mehr ausgeben, Bericht „recall cg v1 stoppte am vorab festgelegten Gate nach Stufe 1 (F von 8)“. RLinear wird dann **nicht** gemessen (das Held-out liegt in Stufe 2).

## Zaehlregel und Kennzahlen

- **Klassen** (Achse 1): reproduziert / teilweise / nicht reproduziert; alles andere („nicht reproduzierbar (Setup)“, „Agenten-Fehler vermutet“, „abgebrochen“, „ungueltig“) ist **Ausfall**, getrennt nach Ursache berichtet.
- **Einheitsklasse:** Stimmen Generation 0 und 1 ueberein, **steht diese Klasse** (keine Generation 2). Weichen sie ab, entscheidet Generation 2 (mindestens 2 von 3 gleich); ohne zweite gleiche Klasse „keine Mehrheit“, ohne Generation 2 „unvollstaendig“. Ausfall, „keine Mehrheit“ und „unvollstaendig“ sind **nie** ein Treffer.
- **Treffer** = Einheitsklasse gleich Referenzurteil.
- **Drei getrennte Zahlen, nie summiert:**
  1. **Entwicklung X von 8** („code-grounded rebuild“, In-Sample);
  2. **neues Paper Y von 1** (RLinear);
  3. **Placebos bestaetigt Z von 9** (Placebo-Urteil „reproduziert“; Ziel 0). Placebos werden **nur in Generation 0** gestellt (v2 bis v4: jede Generation, 48 Placebos je Version).
- **Vergleich:** v4 Arm B 2 von 8 (Kipf-GCN, Shchur-GCN). Die Formulierung im Bericht: „matched the reference verdict“, nicht „target reached“; immer „method fidelity not checked“.
- **Lesregel (vorab):** Eine Aussage „Recall verbessert“ ist erlaubt nur, wenn **S1** Entwicklung ≥ 4 von 8 **und S2** RLinear 1 von 1 **und S3** kein Placebo „reproduziert“ und DGI nicht mehrheitlich „reproduziert“ gelten, und selbst dann nur als „in-sample code-grounded rebuild plus ein Held-out“, nicht als Gewinn des unabhaengigen Nachbaus. Ein Ergebnis von 3 von 8 ist **keine** Verbesserung (es tritt in 27 bis 41 % der Welten ohne Effekt auf, `recall_plan.md`). Fehlschlag von S3 wird zuerst und deutlich berichtet; die Aussage „bestaetigte nie einen falschen Claim“ faellt dann aus README, Devpost-Text und Video.
- **Sekundaer (beschreibend, nicht entscheidend):** Einzelurteile je Generation, „nah“ (reproduziert oder teilweise), Median des Abstands zum Zielwert (post hoc), Ausfaelle nach Ursache, Graph-Ablation, Versuche und Reparaturen, Kosten je Einheit und Stufe, `target_matches_yaml`.

## Budget, Deckel, Stopp

- **Deckel 9,00 $ brutto** fuer den ganzen Lauf (Schritt 0 eingeschlossen). **Harter Stopp auf Aufruf-Ebene bei 8,75 $ brutto:** der Planer setzt fuer alle Worker `BUDGET_TOTAL_USD = 8,75 / 0,95` und `BUDGET_SINCE` = Beginn der Zaehlung; der LLM-Client verweigert dann jeden weiteren Aufruf (`BudgetExceeded`, unabhaengig von der Parallelitaet). Gezaehlt wird das gemeinsame Kosten-Log (`logs/costs.jsonl`): **Verbrauch anderer Sitzungen in diesem Zeitraum zaehlt mit** (sichere Richtung).
- **Reservierung:** ein weiterer Lauf startet nur, wenn Verbrauch seit Beginn + Reserve aller laufenden Jobs (0,60 $ mit Placebo, 0,40 $ ohne) + Reserve des neuen ≤ 8,75 $. Abbruch wegen Budget schreibt **keine** Zeile: die betroffene Zelle bleibt „unvollstaendig“ und ist nie ein Treffer.
- **Schaetzung:** Stufe 1 ≈ 8 × 0,45 + 8 × 0,31 = **6,1 $**, Stufe 2 ≈ 0,76 (RLinear) + ≈ 4 × 0,31 (Entwicklungs-Generation 2) + 0,31 ≈ **2,3 $**, zusammen ≈ **8,4 $** (Plan A: ≤ 8 $; die Placebos kosten 1,1 $ mehr als dort). Der Stopp bei 8,75 $ kann Stufe 2 kuerzen; gekuerzte Einheiten sind „unvollstaendig“, nie ein Treffer.
- **Kontostand Token Factory** vor dem Lauf: im Bericht (`scripts/cost_report.py`). Das Log ist eine Schaetzung, massgeblich ist die Konsole (Abgleich offen).

## Abbrueche und Wiederholung

- **Agenten-Fehler bleiben in der Zaehlung** (Ausfall, nie stillschweigend wiederholt): Reparatur ohne Code, Versuchs-Limit, Harness-Check fehlgeschlagen, Kostenkappe je Lauf (Zeile „abgebrochen“).
- **Infrastruktur-Abbrueche** (Sandbox-Setup/-Timeout, Transportfehler, Prozess tot): die Generation wird **vollstaendig neu gestartet** (halbe Laeufe schreiben keine Zeilen), hoechstens 3 Mal, protokolliert in `runs/recall_cg_v1_incidents.jsonl` und im Ergebnisbericht. Eine Code-Aenderung nur fuer Infrastrukturfehler ist erlaubt (Commit und Begruendung im Bericht), ohne Aenderung an Prompts, Harness oder Messung.
- **`BillingBlocked`:** Stopp, Rueckfrage an die Koordination, kein Retry.
- **Mehr als 25 % der Laeufe einer Einheit** aus Infrastrukturgruenden abgebrochen: Einheit stoppen und als Task-/Harness-Problem berichten.
- **Ein gezaehlter Lauf.** Kein Wiederholen und kein zweiter Versuch nach einer Code-Aenderung. Der Planer laesst sich mit der Datei `runs/recall_cg_v1.STOP` anhalten (keine neuen Starts).

## Mechanik-Pruefung vor dem Lauf (ohne Modell-Aufruf)

`scripts/run_recall_cg_v1.py --mechanics`: der echte Orchestrator mit `graph_repair`, `io_helper`, `code_grounded` laeuft auf der Nebius-Sandbox (RLinear, 2 Seeds), das „Modell“ ist ein Skript (Persistenz-Vorhersager ueber `nullius_io`). Beobachtet: Basis-Snapshot, Ausgabehelfer im Branch, Formpruefung, drei Kandidaten, Fehler-Reparatur, **weicher Timeout im Produktpfad (Exit 124, Reparaturtext mit Stack)**, Harness, Urteil; Modellkosten 0 $. Ergebnis im Ergebnisbericht. Die Parallelitaet (5 Laeufe × 3 Seeds = 15 Branches, 16 getestet in G1) wird nicht mit Modell-Aufrufen getestet.

## Grenzen (vorab, werden im Ergebnis wiederholt)

1. **In-Sample.** Die Fixes F1 bis F4 und F8 wurden nach Sicht der G0-Ergebnisse (GAT-Einheiten) entworfen; alle acht Entwicklungseinheiten wurden mehrfach von Hand analysiert. Nur RLinear ist ein Paper, auf das nichts abgestimmt wurde (und selbst RLinear ging in G0 und G1 als Nichtverschlechterungs-Einheit ein).
2. **Ein Held-out, ein Paper.** Y von 1 traegt keine Verallgemeinerung.
3. **Anderer Modus.** Das Ergebnis ist „Nachbau aus Paper und Autorencode“, nicht der unabhaengige Nachbau aus Worten von v2 bis v4; die beiden Zahlen messen verschiedene Dinge.
4. **Zaehlung abweichend von v2 bis v4:** nur Arm B, Placebo nur in Generation 0 (9 statt 48), adaptive Generation 2 (zwei Generationen gelten, wenn sie in der Klasse uebereinstimmen), Stufen-Gate. Kennzahl und Referenzurteile sind dieselben; sie ist **nicht** unter exakt den v2-Regeln gezaehlt.
5. **Konfigurationsunterschiede zu v4 Arm B:** Rolle 5 nicht live, Kappe je Lauf 0,75 $, `code_grounded` und `io_helper`, korrigierte F2-Kuerzung, S4-Auszug neu gewaehlt, RLinear-Fixtures aus gespeicherten Quellen.
6. **G0 unvollstaendig, G1 niedrige Huerde** (siehe Einordnung). Veličković-GAT war in G1 in beiden Armen 0 von 3 (Setup-Ausfaelle durch Zeitlimits); Erwartung fuer diese Einheit daher niedrig.
7. **Zielwerte** stammen fuer RLinear aus dem gespeicherten Paper-Text (Rolle 2 belegt ihn gegen Zitat und Spalte); `papers.yaml` und Beleg werden verglichen (`target_matches_yaml`).
8. **Kein Tavily im Lauf** (Fixtures); der Live-Beleg fuer Tavily stammt aus den Produktlaeufen, nicht aus diesem Experiment.
9. **Log gegen Konsole:** der Deckel rechnet gegen das Log (Schaetzung); die Konsole ist seit 2026-09-29 nicht abgeglichen. Das Log hat in der Vergangenheit stark von der Konsole abgewichen (`cost_report.py`); der Deckel ist daher eine Obergrenze nach Log, nicht nach Rechnung.
10. **Keine Signifikanzaussagen.** Was hier nicht steht, ist post hoc und so gekennzeichnet.
11. **Nichts davon ins README oder Video, bevor die Nutzerin/der Nutzer entschieden hat.** Keine Prototyp-Zahlen; dies sind Prod-Laeufe (Token Factory, Nebius).

## Was nachtraeglich nicht geaendert wird

Einheiten, Claims, Zielwerte, Placebo-Werte, Toleranzen, Seeds, Referenzurteile, Konfiguration, Zaehlregel, Gate, Lesregel, Deckel und Stopp, Reservierung, Abbruch-Regeln. Jede Auswertung, die hier nicht steht, wird als post hoc gekennzeichnet.

## Ablauf (Befehle)

```
uv run python scripts/run_recall_cg_v1.py --dry-run
LLM_PROVIDER=tokenfactory uv run python scripts/run_recall_cg_v1.py --prep
nohup env LLM_PROVIDER=tokenfactory uv run python scripts/run_recall_cg_v1.py --run > runs/recall_cg_v1_planner.log 2>&1 &
uv run python scripts/eval_recall_cg_v1.py > benchmark/experiments/results/recall_cg_v1.md
```
