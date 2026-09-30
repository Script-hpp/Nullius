# Benchmark v2: Vorab-Festlegung (ENTWURF, nicht freigegeben)

Entwurf von worker-2, Stand 2026-10-01. Gilt erst, wenn der Reviewer freigibt und die Datei als „final“ committet ist, **vor dem ersten v2-Lauf**. Nichts wird nach Sicht der Ergebnisse geändert. Prod-Modelle (Token Factory); Ergebnisse dürfen ins README. Vorgänger: `benchmark_protocol.md` (v1, eingefroren, historischer Stand). Plan und Begründungen: `docs/benchmark_plan.md`.

## Was v2 gegenüber v1 ändert (Entscheidungen des Nutzers, 2026-10-01)
1. **Produktkonfiguration:** `graph_repair=True` (Plan-Prüfung, Fehler-Reparatur, Graph-Ablation mit Reparatur nach Seed 0, Gesamtgrenze 3 Runden) in **beiden** Armen. v1 lief ohne; v1 bleibt als historischer Stand im README, alle Papers laufen in v2 neu.
2. **Zählregel:** Zahl-Urteil (Achse 1), **Mehrheit je Paper-Claim über 3 Generierungen**, Rohzählung sekundär. Das ersetzt für v2 ausdrücklich die CLAUDE.md-Regel „Benchmark zählt nur ‚Paper reproduziert‘ und ‚nicht reproduziert‘“ (siehe Zählregeln, Begründung dort).
3. Mehr Claims (DGI, GAT, Shchur-GAT, NLinear), Placebo auch für GAT.
4. Code-Stand enthält den Fix 530f748 (Code-Schreiben: Wiederholung bei abgeschnittener Antwort mit mehr `max_tokens`).

## Hypothesen (vorab)
- **H1 (primär, deskriptiv):** Arm B (Rolle 2 und 4 aus gespeicherten Tavily-Ergebnissen) erzielt bei den echten Claims in mehr Paper-Claim-Einheiten die Mehrheit „gleich Referenzurteil“ als Arm A. Erwartung nach v1 (6 gegen 3 von 12 Urteilen): ja, aber getragen von Rolle 4 bei wenigen Papers; Zielwerte sind in beiden Armen gleich. Keine Signifikanzaussage (8 Einheiten).
- **H2:** Alle Placebo-Claims werden in beiden Armen als „nicht reproduziert“ erkannt (Urteilslogik und Isolation lassen falsche Zielwerte nicht durch; v1: 12 von 12).
- Beide Hypothesen können scheitern; das wird berichtet wie es ist.

## Papers und Claims (Zielwerte aus den Papers, nicht aus dem Gedächtnis)

Claims stehen in `benchmark/papers.yaml` (v1-Einträge unverändert; neue Einträge am Ende, GAT-Placebo im GAT-Eintrag). Toleranz überall ±0,01 (feste Regel Accuracy; MSE: 0,01 absolut), Mittelwert über Seeds entscheidet.

| Eintrag (Task) | Echter Claim: Ziel | Belegstelle (Fixture) | Seeds | Placebo | Referenz (Mittel) | Referenzurteil | Herkunft |
|---|---|---|---|---|---|---|---|
| `kipf_gcn_2017` (Cora) | GCN 0,815 | Tabelle 2, `1609.02907.json` | 10 | 0,88 | 0,8136 (`results/cora_references.txt`) | reproduziert | (R) |
| `wu_sgc_2019` (Cora) | SGC 0,810 | Tabelle 2, `1902.07153.json` | 10 | 0,88 | 0,8081 (`cora_references.txt`) | reproduziert | (R); Paper misst SGC selbst |
| `zeng_dlinear_2023` (ETTh1) | DLinear MSE 0,375 (Horizont 96) | Tabelle 2, `2205.13504.json` | 5 | 0,300 | 0,3809 (`etth1_harness_reference.txt`) | reproduziert | (R) |
| `shchur_pitfalls_2018` (Cora-LCC) | GCN 0,815 | Tabelle 1, `1811.05868.json` | 30 Splits × 1 | 0,88 | 0,8107 (`lcc_harness_reference_30x1.txt`) | reproduziert | (R) und (H): Kritik-Paper misst GCN 81,5 ± 1,3 selbst |
| `velickovic_gat_2018` (Cora) | GAT 0,830 | Paper Tabelle 2, `1710.10903.json`; auch Wu Tabelle 2 | 10 | 0,88 (neu) | 0,8280 (`cora_references_holdout.txt`) | reproduziert | (R) |
| `velickovic_dgi_2019` (Cora) | DGI 0,823 | Tabelle 2, `1809.10341.json` | 10 | 0,88 | 0,8047 (`cora_references_holdout.txt`) | **teilweise** | (R); Abstand 0,0183 |
| `shchur_gat_2018` (Cora-LCC) | GAT 0,818 | Tabelle 1, `1811.05868.json` (Spaltenprüfung ok: Spalte CORA, Position 1 von 4) | 30 Splits × 1 | 0,88 | **offen**, Referenzlauf nötig | aus Referenzlauf | (H) Kritik-Paper misst GAT 81,8 ± 1,3; (R) offen |
| `zeng_nlinear_2023` (ETTh1) | NLinear MSE 0,374 (Horizont 96) | Tabelle 2, `2205.13504.json` (Spaltenprüfung ok: MSE Position 3, MAE Position 4) | 5 | 0,300 | **offen**, Referenzlauf nötig | aus Referenzlauf | (R) offen |

Maschinenlesbar für die Auswertung (`scripts/eval_benchmark_v2.py` liest diesen Block; Referenzurteil = `judge(reference_mean, target, tolerance)` mit Toleranz 0,01; `null` = Referenzlauf offen, die Auswertung bricht dann ab, bis der Wert eingetragen ist):

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
    shchur_gat_2018:      {claim: shchur_gat_cora_acc,  placebo: shchur_gat_cora_acc_placebo,  target: 0.818, placebo_target: 0.88,  reference_mean: null,   origin: "R offen, H"}
    zeng_nlinear_2023:    {claim: nlinear_etth1_96_mse, placebo: nlinear_etth1_96_mse_placebo, target: 0.374, placebo_target: 0.300, reference_mean: null,   origin: "R offen"}
```

**6 Papers** (Kipf, Wu, Zeng, Shchur, Veličković GAT, DGI), **8 Paper-Claim-Einheiten** (Zeng und Shchur je zweimal). Das README nennt beide Zahlen getrennt.

**Referenzherkunft:** (R) = eigene Referenzlösung im eigenen Harness (`tests/reference/`), (H) = unabhängige Neumessung in einem Kritik-Paper (Fixture). Ein Referenzurteil (R) ist **kein** Fremdurteil und nicht unabhängig von unseren Protokollentscheidungen; das README sagt das offen. Die Formulierung „menschliche Referenzimplementierung im selben Harness“ ist wörtlich zu nehmen, nicht „menschliche Reproduktionsstudie“.

**Referenzurteil der beiden offenen Claims:** Es entsteht aus dem Referenzlauf des Reviewers (Befehle unten) nach derselben Regel (≤ Toleranz reproduziert, ≤ 3 × Toleranz teilweise, sonst nicht reproduziert), **vor** dem ersten Agentenlauf dieses Claims. Ein Claim wird nur dann gestrichen, wenn die Referenz keinen gültigen Lauf liefert, **nie** wegen des Referenzurteils. Gestrichene Claims werden im Bericht genannt.

**Placebos:** konstruierte Negativkontrollen (falscher Zielwert, kein Paper-Wert), Referenzurteil „nicht reproduziert“ (Abstand von Referenz und Placebo > 3 × Toleranz: GAT 0,052, DGI 0,075, Shchur-GAT mit Referenz noch zu prüfen, NLinear mit Referenz noch zu prüfen).

## Dacrema: nicht gebaut (Entscheidung des Nutzers)
- **Befund:** Der Text liegt als Fixture vor (`benchmark/fixtures/1907.06902.json`). Tabelle 6 („Experimental results for NCF“): Auf MovieLens-1M wurde NeuMF reproduziert (HR@10 **0,7120**, NDCG@10 0,4369) und liegt klar vor den einfachen Baselines (UserKNN 0,6705, ItemKNN 0,6596, P3α 0,6464, RP3β 0,6715); nur SLIM (0,7161) ist gleichauf. Die einfachen Baselines gleichauf oder besser liegen nur auf Pinterest (z. B. ItemKNN 0,8744 gegen NeuMF 0,8719). MovieLens liefert deshalb **keinen Negativfall**.
- **Aufwand:** 3 bis 5 Personentage (Harness-Vertrag, Metrik, Protokoll, Lizenz, zwei Referenzlösungen; `docs/benchmark_plan.md` Abschnitt 2.1).
- **Folge:** Es gibt keinen echten negativen Fall im Benchmark. Die **Negativkontrolle sind die Placebos** (konstruiert, ersetzen den echten Negativfall nicht). Das README sagt das offen.

## Konfiguration (für alle Läufe gleich)
- **Arme:** A = ohne Tavily (`use_tavily=False`: keine Rolle 2, keine Rolle 4). B = mit Tavily (Rolle 2: Zielwert belegt mit Spaltenprüfung; Rolle 4: Setup-Details aus Paper und, wenn verlinkt, README/Repo). Rollen 1, 3, 5, 6, 7 sind nicht im Loop und nicht Teil des Benchmarks (Stand `loop.py`, `benchmark_plan.md` 4.1).
- **Tavily-Modus:** `fixture` in beiden Armen (`scripts/run_benchmark.py`, Kwarg `tavily_mode="fixture"`), gleicher gespeicherter Text für beide Arme und alle Wiederholungen. Kein Live-Aufruf im Benchmark; der Live-Beleg für Tavily stammt aus den Produktläufen (Katalog, Demo). Vorteil: Vergleichbarkeit, reproduzierbar, keine Credits. Nachteil: der Benchmark belegt keinen Live-Aufruf.
- **Flags:** `graph_repair=True`, `fresh=True` (LLM-Cache aus), `MAX_ATTEMPTS_PER_CLAIM=4`, `MAX_PARALLEL_SEEDS=3`, Sandbox `nebius`, Modelle wie in `.env` (Prod, Token Factory). Gleiche Flags in beiden Armen. Auf ETTh1 gilt die Fehler-Reparatur, der Graph-Teil entfällt (kein Graph).
- **Generierungen:** 3 frische je Paper-Claim-Einheit und Arm; je Generierung ein Lauf mit echtem Claim und Placebo (je eigener Plan, Code, Versuche). 8 Einheiten × 2 Arme × 3 = **48 Läufe**, 48 echte + 48 Placebo-Urteile.
- **Reihenfolge:** je Paper Generierung 0 A, 0 B, 1 A, 1 B, …; Neustart setzt bei der ersten fehlenden Generierung fort; Ergebnisse `runs/benchmark_v2.jsonl` (Feld `config: v2_graph_repair`, `code_commit`).
- **Gleiche Zielwerte in beiden Armen:** Arm A nutzt `papers.yaml`, Arm B den belegten Wert; Abweichung wird gemeldet (`target_matches_yaml`).
- **Modus „Nachbau“, gleicher Harness, gleiche Isolation, Formprüfung, Graph-Ablation** wie v1. Achse 2 (Methodentreue) bleibt „nicht geprüft“ (endgültig, CLAUDE.md).

## Zählregeln
- **Urteilsklassen:** reproduziert / teilweise / nicht reproduziert (Achse 1, dreistufig exakt). Ausfall: „nicht reproduzierbar (Setup)“, „Agenten-Fehler vermutet“, „abgebrochen“, „ungültig“ (alle als eine Klasse „Ausfall“, getrennt nach Ursache berichtet).
- **Primär: Mehrheit je Paper-Claim-Einheit.** Über die 3 Generierungen des echten Claims je Arm: Mehrheit = mindestens 2 gleiche Klassen (Ausfall zählt als eigene Klasse). Die Einheit **stimmt überein**, wenn die Mehrheitsklasse gleich dem Referenzurteil ist. Keine Mehrheit (drei verschiedene Klassen) und Mehrheit „Ausfall“ zählen **nicht** als Übereinstimmung und werden getrennt berichtet. Kennzahl: „Arm B: X von 8 Einheiten stimmen überein, Arm A: Y von 8“.
- **Placebos:** je Einheit Mehrheit über 3 Generierungen; „erkannt“ = Mehrheit „nicht reproduziert“. Kennzahl „P von 8 gegen Q von 8“. Getrennt von den echten Claims.
- **Sekundär:** Rohzählung (24 echte Urteile je Arm: Übereinstimmungen von 24; Ausfälle; Verteilung); „nah“ = reproduziert oder teilweise; Mittelwert der Abweichung vom Zielwert; Graph-Ablation-Verteilung; Versuche und Reparaturen je Claim; Kosten.
- **Begründung der Abweichung von der CLAUDE.md-Zählregel:** CLAUDE.md zählt nur „Paper reproduziert“/„nicht reproduziert“. Diese Gesamtaussagen setzen Achse 2 = ja voraus. Die Eichungen des Methodentreue-Prüfers v1 bis v3 scheiterten; Achse 2 ist deshalb endgültig „nicht geprüft“, jede Gesamtaussage wäre „vorläufig“, und streng genommen zählte **kein** Urteil. Das v1-Protokoll zählte bereits Achse 1; v2 macht das ausdrücklich zur Regel.
- **Konsequenz für die Kennzahl (README, englisch):** Die Kennzahl misst „target value reached according to the harness“ (Zahl), **nicht** „method reproduced“. Jeder Satz im README trägt „method fidelity not checked“. Das Wort „reproduced“ steht nie ohne diesen Zusatz. Für GCN, SGC, Shchur-GCN und Shchur-GAT gilt zusätzlich die Graph-Ablation (9/10 validiert).
- **Keine Signifikanzaussagen.** 8 Einheiten sind Zählwerte mit Vorbehalt. Alles, was nicht in dieser Datei steht, ist **post hoc** und so gekennzeichnet.

## Agenten-Fehler und Abbrüche
- **Agenten-Fehler bleiben in der Zählung** (Ausfall, getrennt berichtet), werden nie stillschweigend wiederholt. Dazu zählen Reparatur ohne Code, `MAX_ATTEMPTS_PER_CLAIM` ausgeschöpft, Harness-Check fehlgeschlagen, Kostenlimit je Lauf (`MAX_COST_PER_RUN_USD` brutto, Zeile „abgebrochen“).
- **Infrastruktur-Abbrüche** (Sandbox-Timeout `ApiTimeoutError`, Prozess tot, Laptop aus) sind kein Ausfall des Agenten: die Generierung wird **vollständig neu gestartet** (halbe Läufe schreiben keine Zeilen), vermerkt als „Infrastruktur-Abbruch, wiederholt“ in `benchmark_v2_incidents.md` (analog `results/benchmark_incidents.md`: Zeitpunkt, Fehlertext, Code-Stand).
- **Abrechnungs-/Auth-Fehler** (`BillingBlocked`): Prozess bricht ab, Reviewer prüft die Konsole, danach Neustart; kein Ausfall des Agenten.
- **Code-Stand:** `src/` bleibt während des Benchmarks unverändert. Eine Änderung nur für Infrastrukturfehler ist erlaubt, wird mit Commit und Begründung in `benchmark_v2_incidents.md` festgehalten und ändert keine Prompts, keinen Harness und keine Messung. Jede Zeile trägt `code_commit` (mit `+dirty`, falls `src` unsauber); der Bericht weist Stände aus.
- **Einzelne Generierungen mit instabilem Code** (Beispiel v1 GCN Arm A Gen. 0) bleiben in den Daten.

## Budget und Abbruchkriterien
- **Schätzung** (`scripts/run_benchmark.py --v2 --dry-run`, Mittel je Lauf aus v1 mal Faktor 1,5 für Reparaturschleifen): 48 Läufe ≈ 2,6 $ netto ≈ **3,1 $ brutto**; Dauer grob 20 h seriell, ≈ 7 h mit 3 parallelen Prozessen (Annahme 25 min je Lauf, **nicht gemessen**).
- **Harte Grenze: 6,00 $ brutto** kumuliert im v2-Store. Das Skript startet keinen neuen Lauf mehr, wenn die Summe (Nettokosten × 1,19, je Lauf einmal gezählt) die Grenze erreicht hat; zusätzlich gelten `MAX_COST_PER_RUN_USD` und das globale Budget (`BUDGET_TOTAL_USD`, Warnung 80 %, Stopp 95 %) im LLM-Client.
- **Stopp und Rückfrage beim Reviewer**, wenn: die Budgetgrenze erreicht ist; `BillingBlocked`; dieselbe Generierung dreimal an Infrastruktur scheitert; eine Referenzlösung keinen gültigen Lauf liefert (Claim streichen, siehe oben); mehr als 25 % der Läufe eines Papers „abgebrochen“/Setup-Ausfall sind (Hinweis auf Fehler in Task oder Harness, nicht auf das Paper).
- **Tavily:** Fixture-Modus, keine Credits im Lauf. Einzige bekannte Fixture-Lücke: README `PetarV-/DGI` (Arm B würde beim ersten Lauf einmal live extrahieren und die Fixture schreiben). Vorher einmalig durch den Reviewer holen (Befehl unten), damit **beide Arme und alle Wiederholungen denselben Text sehen**.

## Vorbereitung vor dem ersten Lauf (Reviewer)
1. Referenzläufe der zwei neuen Claims in der Sandbox, Ergebnis nach `benchmark/experiments/results/` (Befehle unten), Referenzurteile hier eintragen, bevor der erste Agentenlauf startet:
   ```
   uv run python scripts/validate_v2_references.py gat_lcc 30 | tee benchmark/experiments/results/v2_reference_gat_lcc.txt
   uv run python scripts/validate_v2_references.py nlinear 5  | tee benchmark/experiments/results/v2_reference_nlinear.txt
   ```
2. Die übrigen Referenzen liegen vor (GCN, SGC, DLinear, Shchur-GCN, GAT, DGI; siehe Tabelle). Unverändert.
3. DGI-README einmalig (1 Tavily-Extract, wird als Fixture gespeichert):
   ```
   uv run python -c "from repro_agent.config import load_settings; from repro_agent.repo_source import fetch_readme; from repro_agent.tavily_tools import TavilyTools; fetch_readme('https://github.com/PetarV-/DGI', lambda: TavilyTools(load_settings()), mode='live')"
   ```
4. Dry-Run prüfen: `uv run python scripts/run_benchmark.py --v2 --dry-run` (muss „Fixtures vollständig“ für alle Papers zeigen).
5. Auswertungsskript für v2 (Mehrheitsregel, Referenzurteile aus dieser Datei) **vor den Daten committen** (noch zu liefern; `scripts/eval_benchmark.py` kennt nur v1, v1-Referenzen und Zeilen ohne `config`).
6. Start: `LLM_PROVIDER=tokenfactory MAX_PARALLEL_SEEDS=3 uv run python scripts/run_benchmark.py --v2 [<paper_id> ...]` (mehrere Prozesse mit getrennten Paper-Listen möglich).

## Grenzen (vorab)
- 6 Papers, 8 Einheiten; Erwartungen fast alle „reproduziert“ (DGI „teilweise“). **Kein echter Negativfall** (Dacrema nicht gebaut), nur konstruierte Placebos.
- Referenzurteile (R) stammen aus unseren Referenzlösungen im eigenen Harness. Nur Shchur-GCN und Shchur-GAT haben zusätzlich eine unabhängige Neumessung (H).
- **Zielwerte sind in beiden Armen gleich**, Rolle 2 ändert das Urteil nicht. Ein Unterschied zwischen den Armen kommt aus Rolle 4 und der Promptlänge (Störgröße).
- Zielwerte in `papers.yaml` stammen für Shchur, SGC, DGI, Shchur-GAT, NLinear aus Fixture-Belegen (früher mit Tavily geholt), für GCN und DLinear „Gedächtnis, ungeprüft“ (in Arm B belegt ersetzt, gleiche Zahl). Arm A ist damit „ohne Laufzeitquellen“, nicht „ohne Vorwissen“; das Modell kennt die Papers aus dem Training.
- Shchur-GCN und Shchur-GAT teilen das Paper und die Splits (nicht unabhängig); Zeng DLinear und NLinear ebenso.
- **DGI war als Prüfmenge für Tavily v2 reserviert** (`tavily_v2_holdout.md`). Tavily v2 bleibt bis nach der Abgabe pausiert; das README vermerkt, dass DGI im Benchmark benutzt wurde und Tavily v2 deshalb nicht mehr sauber auf DGI gemessen werden kann.
- Achse 2 nicht geprüft; ein „Zielwert erreicht“ heißt nicht „Methode reproduziert“ (Beispiel Ridge-Regression bei DLinear, v1).
- Der Agent sieht den (Placebo-)Zielwert im Prompt; der Harness verhindert Fälschung, nicht Feinabstimmung auf Validierungsdaten.
- Dauer- und Kostenschätzung beruhen auf v1-Mittelwerten und einer Annahme; v2 mit Reparaturen kann teurer und länger sein.

## Was nachträglich nicht geändert wird
Hypothesen, Claims, Zielwerte, Toleranzen, Seeds, Anzahl Generierungen, Arme, Tavily-Modus, Konfiguration, Zählregeln (Mehrheit, Klassen, Ausfall-Definition), Placebo-Werte, Budgetgrenze, Abbruchkriterien, Referenzurteile (nach Eintrag). Erlaubt sind nur dokumentierte Infrastrukturkorrekturen (`benchmark_v2_incidents.md`) und rein redaktionelle Änderungen dieser Datei. Jede Auswertung, die hier nicht steht, wird als post hoc gekennzeichnet.
