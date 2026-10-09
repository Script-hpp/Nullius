# RLinear in recall cg v1: warum kein Treffer? (Analyse, keine Modellaufrufe)

Stand 2026-10-09, Branch `Script-hpp/recall-cg-v1-run`. Reine Aktenanalyse: gelesen wurden Dossiers, Ereignisprotokolle, Plaene und Codes der Laeufe, die G0/G1-Proben und die Fixtures. Es gab keinen LLM-, Tavily- oder Token-Factory-Aufruf, keinen Agenten-Code auf dem Host und keinen Push. Das vorab festgelegte Ergebnis von recall cg v1 (S2 nicht erfuellt, "recall improved" nicht erlaubt) bleibt unveraendert; diese Datei erklaert es nur.

## Kurzbefund

1. **Der Messweg ist es nicht.** Im selben Plan-A-Lauf (gleicher Harness, gleiche Daten, Nebius) erreichte der Placebo-Code der Generation 0 fuer RLinear 0,3725 (Seed-Streuung 4e-5); G1 lief ebenfalls auf Nebius und traf 0,370. Kein Hinweis auf Harness, Task, Seeds oder Timeouts (alle Versuche `abweichung`, keine Zeitueberschreitung, kein `ungueltig`).
2. **Der Hauptunterschied im Code ist der Lernraten-Plan.** Jeder RLinear-Code mit Treffer (alle 3 G0-cg-Proben, alle 3 G1-Proben, der Placebo-Code in Plan A) halbiert die Lernrate je Epoche (`lradj=type1`, wie `exp/exp_main.py` des Autoren-Repos es tut). Beide Plan-A-Codes der echten Claims (Gen 0: 0,487, Gen 1: 0,432) trainieren mit **konstanter** Lernrate 0,005. Das ist mit hoher Wahrscheinlichkeit der Hauptgrund (Beleg unten), aber kein Beweis; ein Ablationslauf fehlt (Vorschlag unten).
3. **Warum der Plan den Schedule wegliess, ist offen.** Es ist eine Mischung aus Generationsvarianz und Eingabe-Unterschieden, die sich mit n = 5 Plaenen nicht trennen laesst. Belegt ist: In Plan A sagen beide Plaene der echten Claims, `adjust_learning_rate` sei "defined but not called" bzw. "none used"; der Plan des Placebos im selben Lauf (gleiche Eingaben bis auf den Claim-Text) wendet `lradj='type1'` an.
4. **Drei echte Eingabefehler** liegen unabhaengig davon vor und gehoeren in den Befund (Abschnitt 4): Batch-Groesse 5 aus einer Fussnoten-Markierung des Paper-Texts (richtig: 128), ein auf 4500 Zeichen abgeschnittener `exp_main.py`-Ausschnitt genau vor dem `adjust_learning_rate`-Aufruf, und ein `script.md`-Eintrag (Batch 128, lr 0,005, `--rev`), den der Filter "andere Modellwahl (Linear)" verwirft.
5. **Der Repo-Abgleich (A2) kann den Fehler nicht sehen**, weil seine feste Aspektliste keinen Lernraten-Plan, keinen Optimierer und keine Batch-Groesse enthaelt (`src/repro_agent/repo_diff.py:17`).

## 1. Die Laeufe im Vergleich

Alle Pfade relativ zum Worktree `tavily-ui`, ausser G0/G1 (siehe Kopf der Tabelle).

| | G0 (lokal, Gate-Skript) | G1 (Nebius, Gate-Skript) | Plan A Gen 0 (`runs/rlinear_2023/20261009-060608`) | Plan A Gen 1 (`.../20261009-060609`) | Plan A Placebo Gen 0 (gleicher Lauf wie Gen 0) |
|---|---|---|---|---|---|
| Ergebnis (5 Seeds) | 0,3698 / 0,3698 / 0,3717 (3 Treffer) | 0,3707 / 0,3782 / 0,3696 (2 Treffer, 1 teilweise) | **0,4871** +/- 0,0093 | **0,4321** +/- 0,0182 | 0,3725 +/- 4e-5 (Ziel 0,300, "nicht reproduziert") |
| lr | 0,001 (argparse) | 0,001 (argparse) | 0,005 (Paper) | 0,005 (Paper) | 0,005 |
| Batch | 16 | 16 | 5 | 16 | 5 |
| Lernraten-Plan | halbieren je Epoche | halbieren je Epoche | **keiner** | **keiner** | halbieren (`type1`) |
| RevIN | an (`final_code_cg_0.2.py:167`, `_0.8.py:171`; `facts.revin` im JSON sagt `false`, nicht nachgeprueft) | an (laut Plan) | an (affine) | **aus** | aus |
| Epochen / Geduld | 15 / 10 | 15 / 10 | 20 / 10 | 15 / 10 | 20 / 10 |
| Trainingsdaten | Train (+Val fuer Val-Fenster) | wie G0 | nur Train | nur Train | Train **und** Val (`full_train`) |
| Plaene | 1 Plan, 3 Temperaturen | 1 Plan, 3 Temperaturen | 1 Plan, 3 Kandidaten | 1 Plan, 3 Kandidaten | eigener Plan, 3 Kandidaten |

Belege:

- G0-Proben: `<worktree cto>/benchmark/experiments/results/recall_cg_gate_samples/rlinear_2023/`: `final_cg_{0.2,0.5,0.8}.json` (Mittel, `facts.halve_each_epoch: true` in allen drei), `final_code_cg_0.2.py:38,171,244` (`adjust_lr`, `LEARNING_RATE = 0.001`, Aufruf je Epoche), `final_code_cg_0.8.py:103,173,177` (`adjust_learning_rate`, `lradj = 'type1'`), `plan_cg.md` (lr 0,001, Batch 16, "type1 (halve LR each epoch)").
- G1-Proben: `<worktree gat-analysis>/benchmark/experiments/results/recall_cg_gate_g1_samples/rlinear_2023/`: `final_cg_{0.2,0.5,0.8}.json`, `final_code_cg_0.2.py:37,179,190` (`CONFIG` lr 0,001, `adjust_lr(epoch)` vor jeder Epoche), `plan_cg.md` ("Learning-rate schedule: type 1 ... lr x 0.5^(epoch-1)").
- Plan A Gen 0: `rlinear_etth1_96_mse_ad41edf03b8c.py:17,23,25,245` (lr 0,005, Batch 5, `rev: True`, `Adam(... lr=CONFIG["lr"])` ohne Schedule); Plan `rlinear_etth1_96_mse_plan.md:20` ("defined but not called").
- Plan A Gen 1: `rlinear_etth1_96_mse_f271b6fa5d32.py:15,21,202` (`batch_size 16`, `revin False`, `Adam` ohne Schedule); Plan `rlinear_etth1_96_mse_plan.md:25` ("none used"). `events.jsonl:29` (`spec_override`: Batch 5 durch Steckbrief auf 16 gesetzt).
- Plan A Placebo: `rlinear_etth1_96_mse_placebo_10940e45b9de.py:29,30,153-156,185,250` (kein RevIN, Batch 5, `adjust_learning_rate` mit `type1`, Aufruf je Epoche, Training auf Train+Val); Plan `rlinear_etth1_96_mse_placebo_plan.md:7` ("use the adjust_learning_rate function with lradj='type1'").

**Wichtige Einschraenkung zu G0/G1:** Das Gate-Skript schreibt **einen** Plan je Einheit und zieht darauf drei Code-Proben (Temperaturen 0,2/0,5/0,8). "3/3" (G0) und "2/3" (G1) sind also zwei unabhaengige Plaene, nicht fuenf. Plan A zog zwei Plaene fuer den echten Claim (beide ohne Schedule) und einen fuer den Placebo (mit Schedule). Die Plan-Stichprobe ueber alle Laeufe ist 3 von 5 mit Schedule, zusaetzlich der volle Produktlauf vom 2026-10-02 (`<main checkout>/runs/rlinear_2023/20261002-105416`, "teilweise" 0,378 +/- 0,006), dessen Setup-Details "adjust_learning_rate called each epoch" enthielten.

## 2. Wie stark ist der Schedule-Befund?

Fuer "Schedule ist der Hauptgrund" sprechen drei Dinge, die sich nicht gegenseitig erklaeren:

1. **Alle Treffer haben ihn, kein Fehlschlag hat ihn** (ausser dem Placebo-Code, der trifft). Das gilt ueber unterschiedliche lr (0,001/0,005), Batch (5/16) und RevIN (an/aus) hinweg; diese anderen Unterschiede sortieren die Ergebnisse nicht.
2. **Seed-Streuung.** Mit Schedule 4e-5 bis 4e-3 (G1 Probe 0.5: 3,6e-3), ohne Schedule 0,009 bis 0,018 (Gen 0: 0,4777 / 0,4941 / 0,4861 / 0,4989 / 0,4789; Gen 1: 0,4396 / 0,4321 / 0,4243 / 0,4077 / 0,4568). Das ist das erwartete Bild: Adam mit konstant 0,005 auf einem linearen Modell rauscht am Ende, halbierte Lernrate beruhigt es.
3. **Kandidaten brachten keine Vielfalt.** Alle drei Kandidaten je Plan folgen dem Plan: Gen 0 Seed 0 `0,5239 / 0,4777 / 0,5093`, Gen 1 `0,4234 / 0,4396 / 0,5264`, Placebo `0,3698 / 0,3725 / 0,3698`. Die Auswahl nach Validierungs-MSE (117 Val-Fenster) ordnet grob richtig; sie traf in Gen 0 den Test-besten, in Gen 1 (0,4396 statt 0,4235) und im Placebo (0,3725 statt 0,3698) knapp daneben. Die Abstaende (0,016 und 0,003) sind klein gegen die Luecke zum Ziel (0,07 bis 0,12). Sie kann nur zwischen Kandidaten mit demselben Plan waehlen. Die Kandidatenauswahl ist kein Grund.

**Was das nicht beweist:** Es gibt keinen Ablationslauf. Der Placebo-Code hat zusaetzlich kein RevIN, trainiert auf Train+Val (`:250`, anders als das Repo) und haelt `best_state = model.state_dict()` ohne Kopie (`:200`, also faktisch das Gewicht der letzten Epoche); sein 0,3725 ist deshalb **kein sauberer Treffer** und zaehlt nicht (anderer Zielwert, eine Generation, nachtraegliche Sicht). Er zeigt nur, dass derselbe Lauf mit demselben Messweg 0,37 erreichen kann. Ob RevIN affine (Gen 0) gegen Batch 5/16 gegen Schedule die 0,43 gegen 0,49 trennt, ist offen: Gen 1 (kein RevIN, Batch 16, 0,432) ist besser als Gen 0 (RevIN, Batch 5, 0,487), beide ohne Schedule.

Ein Referenzwert zur Orientierung: die Referenzloesung (`tests/reference/rlinear_reference.py:4-5,19,87`) nutzt Batch 128, lr 0,005, RevIN, **Halbierung je Epoche** und kommt auf 0,3704.

## 3. Agent, Eingabe oder Messweg?

**Messweg: nein** (Punkt 1 im Kurzbefund). Ausserdem lief G1 auf demselben Sandbox-Typ (Nebius) und derselben Harness-Kopie wie Plan A.

**Eingabe: ja, teilweise belegt.** Die Eingaben unterscheiden sich real:

| Eingabe | Gate (G0/G1) | Plan A |
|---|---|---|
| Paper-Angaben im Prompt | keine (Setup-Details stammen aus dem Lauf vom 2026-10-02, der nur Repo-Quellen auswertete; `format_code_grounded` in `src/repro_agent/spec.py:645` nimmt nur Details mit `source == "paper"`) | lr 0,005 ("initial learning rate"), Batch **5**, Epochen 20 (nur Gen 0) aus `dossier.json` `setup_details` |
| Steckbrief (erzwungen) | lr 0,001, Epochen 15, Batch 16, Dropout 0,1, Geduld 10 (alle argparse-Defaults, G1: `spec.json`) | Gen 0: lr 0,005, Epochen 20, Dropout, Geduld, seq_len 336; Gen 1: lr 0,005, Epochen 15, Batch 16, Dropout, Geduld |
| Dateiliste des Klons | 13 Dateien, **ohne** `script.md`, README leer (`<main checkout>/benchmark/fixtures/repos/plumprc_RTSF.json`, `..._readme.json` mit `"text": ""`) | 14 Dateien mit `script.md`, README 5179 Zeichen (`runs/recall_cg_v1_fixtures/repos/plumprc_RTSF__rlinear_2023.json`) |
| Repo-Ausschnitt | 15 533 Zeichen | 13 936 Zeichen; gleiche Kernstuecke (RLinear.py, exp_main.py, run.py, tools.py-Rumpfe, Invertible.py), andere Rumpf-Auswahl |
| Paper-Text und Zielwert | identisch (`sha256 b4ce09d1...`, Zielwert 0,366), Desktop- und Overlay-Fixture stimmen ueberein | |

Die Desktop-Fixtures (`benchmark/fixtures/2305.10721*.json`, `repos/plumprc_RTSF*.json`, alle vom 2026-10-02) sind damit der Gate-Eingabe zuzuordnen, nicht der von Plan A. Der Paper-Text ist byte-identisch; verschieden sind Repo-Dateiliste (kein `script.md`), README und die daraus gezogenen Setup-Details.

Ob diese Unterschiede den Schedule-Entscheid verursachten, ist **nicht entscheidbar**: Der Placebo-Plan in Plan A sah dieselben Eingaben wie die Plaene der echten Claims (nur der Claim-Text und der Zielwert unterscheiden sich) und entschied anders. Das spricht fuer Generationsvarianz im Plan; eine Stichprobe von 5 Plaenen traegt aber keinen Test.

**Agent (Generationsvarianz): wahrscheinlich ein grosser Teil.** Beleg: gleicher Prompt-Aufbau, gleicher Ausschnitt, drei Plaene, zwei verschiedene Schedule-Entscheidungen. Der Ausschnitt legt die Fehlinterpretation allerdings nahe (Abschnitt 4, Punkt b), sodass "Varianz" und "Eingabefehler" hier nicht sauber getrennt werden koennen.

## 4. Konkrete Eingabefehler (unabhaengig vom Ausgang)

a) **Batch-Groesse 5 aus einer Fussnote.** Der Paper-Text hat an der Stelle "...we set the initial learning rate to 0.005 and the batch size to" einen Fussnoten-Block (`> 5 https://www.ncei.noaa.gov/...`, `> 6 https://archive.ics.uci.edu/...`) und erst danach "128" (`runs/recall_cg_v1_fixtures/2305.10721.json`, Text um "batch size to"). Rolle 4 extrahierte "batch size = 5" mit dem Zitat `"...the batch size to\n> 5"` (`dossier.json` `setup_details`, beide Laeufe); das Zitat steht woertlich im Text, der Zitat-Check greift also nicht. Alle drei Plaene schreiben "Batch size = 5 (paper)". In Gen 1 setzte `spec_override` (`events.jsonl:29`) Batch 5 auf den argparse-Default 16. Richtig ist 128 (Paper und `script.md`).

b) **`exp_main.py` abgeschnitten vor dem Aufruf.** Der Ausschnitt schneidet `exp/exp_main.py` bei etwa 4500 Zeichen ab (`SCRIPT_CHARS`, `src/repro_agent/spec.py:62`), der Aufruf `adjust_learning_rate(model_optim, epoch + 1, self.args)` steht bei Zeichen 4913 (Datei 8608 Zeichen). Der Ausschnitt enthaelt den Import, den `def`-Rumpf und `--lradj` aus `run.py`, aber nicht den Aufruf. Rekonstruiert mit `focused_excerpt` ohne LLM aus den Overlay-Fixtures; die Abschnittsgrenzen stimmen mit dem gespeicherten G0-Ausschnitt ueberein (`exp_main.py` jeweils 4539 Zeichen). Der Plan-A-Plan "defined but not called" ist diese Fehlinterpretation. Die Dossier-Details enthalten den Aufruf ("adjust_learning_rate called each epoch", Gen 1) bzw. eine Beschreibung (Gen 0), aber `format_code_grounded` gibt dem Prompt nur Paper-Details, nicht Repo-Details.

c) **`script.md` verworfen.** Das echte Startskript fuer ETTh1 (`--batch_size 128 --learning_rate 0.005 --model Linear --rev --seq_len 336`) wurde in beiden Laeufen vom Filter "andere Modellwahl (Linear)" entfernt (`dossier.json` `spec.dropped`; `src/repro_agent/spec.py:428`). Das Skript nennt das Modell "Linear" (Altname im Repo), die Modelldatei heisst `RLinear`. Gen 0 verlor lr und Batch aus dem Skript, Gen 1 den `seq_len`. Das Skript ist zudem kein `.sh`, wird daher von `drop_run_script_defaults` ignoriert und steht nicht im Ausschnitt.

d) **Repo-Abgleich blind fuer das Thema.** `ASPECTS` in `src/repro_agent/repo_diff.py:17` nennt Dropout, Merkmals-/Adjazenz-Normalisierung, Initialisierung, Regularisierung und Abbruchkriterium. Lernraten-Plan, Optimierer, Batch-Groesse stehen nicht drin (die Liste stammt aus der GNN-Arbeit). Gen 1 meldete `no_deviation` (`events.jsonl:44`), Gen 0 nur zwei kosmetische Abbruch-Abweichungen (`events.jsonl:30`).

## 5. Vorschlag (nichts gebaut)

Reihenfolge nach Nutzen pro Aufwand; alles ohne Auswirkung auf das vorab festgelegte Ergebnis von recall cg v1.

1. **Ablation ohne LLM zuerst.** Die menschliche Referenz `rlinear_reference.py` auf der Nebius-Sandbox mit vier Varianten, je 5 Seeds: (A) wie Referenz, (B) ohne Schedule (konstant 0,005, Batch 128), (C) ohne Schedule, Batch 16, (D) ohne Schedule, Batch 5, kein RevIN. Kosten 0 $ (nur Sandbox). Das trennt "Schedule" von "Batch/RevIN" und ist die einzige Messung, die Punkt 2 des Kurzbefunds belegen oder widerlegen kann. Ich habe sie hier nicht ausgefuehrt: Auf dem Host fehlt `torch`, und ein Lauf auf der Sandbox faellt unter "Modell-/Sandbox-Aufrufe", die dieser Auftrag nicht vorsieht.
2. **Wenn recall ueber v1 hinaus weiterverfolgt wird** (jeweils eigene Vorab-Festlegung, kein Nachbessern von v1): (i) Trainings-Schleife des Autoren-Repos so in den Ausschnitt nehmen, dass der Schedule-Aufruf nie abgeschnitten wird (Rumpf der aufgerufenen Funktion plus die Aufrufzeile; kein Abschneiden mitten in `train()`), (ii) Fussnoten-Zahlen im Paper-Text vor der Extraktion entfernen oder Werte mit `\n> <Zahl>` als Zitat ablehnen, (iii) Modell-Alias ("Linear" in `script.md` gehoert zu `RLinear`) im Filter "andere Modellwahl" zulassen, wenn der Dateiname des Modells das Wort enthaelt, (iv) `ASPECTS` im Repo-Abgleich um "learning-rate schedule", "optimiser" und "batch size" erweitern, (v) Schedule-Entscheid im Plan nicht dem Zufall ueberlassen: Wenn Repo-Code einen Schedule aufruft und das Paper "initial learning rate" sagt, steht beides als Pflichtangabe im Steckbrief.
3. **Nicht tun:** weitere Gate-artige Proben mit einem Plan und drei Temperaturen als Beleg fuer Treffer zaehlen (siehe Einschraenkung in Abschnitt 1), und den Placebo-Code als Treffer verbuchen.

## 6. Was offen bleibt

- Ob Schedule allein oder zusammen mit Batch/RevIN die 0,43 bis 0,49 erklaert (Ablation fehlt).
- Wie gross die Generationsvarianz im Plan wirklich ist (n = 5 Plaene, 2 ohne Schedule).
- Ob ein Plan mit Schedule im vollen Produkt-Loop (Super fuer `reasoning`, Kandidatenwahl, Reparatur, Repo-Abgleich) trifft. Belegt ist nur: der Produktlauf vom 2026-10-02 mit Schedule im Prompt kam auf 0,378 ("teilweise"), und der Placebo-Code in Plan A auf 0,3725 bei abweichendem Protokoll.
- Dieser Bericht prueft keine Wirkung der Ultra-Entfernung auf den Plan (Plan A lief mit Super als `reasoning`, G0/G1 ebenfalls mit Super in allen Rollen; hier also kein Unterschied).
