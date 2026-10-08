# Plan: Steckbrief erzwingen statt Fakten nur liefern (Recall)

Status: **Entwurf**, 2026-10-08. Nichts gebaut, kein LLM-Aufruf, `src/` unverändert. Grundlage: `docs/recall_root_cause.md`, `docs/recall_plan.md` (Rev. 2) und das Adversarial Review dazu (Zählung der 19 Fehlschläge in Arm B nach R, R*, E, D, O). Zahlen ohne Quelle sind Schätzungen und als solche markiert.

## 1. Problem in einem Satz

Der Agent hat die entscheidenden Angaben oft schon im Prompt und **befolgt sie nicht** (DLinear: `--learning_rate 0.005`, `--seq_len 336` und die ganze `adjust_learning_rate` wörtlich im Prompt, der Code nahm andere Werte; NLinear g0: Plan sagt lr 0,005, Code schreibt 5e-4). Mehr Text im Prompt (F2) behebt das nicht. Das zeigt A2: dasselbe Modell, dieselben Repo-Dateien, und die fehlende Feature-Normalisierung wurde in 4 von 6 Fällen übersehen.

**Idee:** Wir *hoffen* nicht mehr, dass das Modell die Angaben nutzt. Wir machen daraus einen kurzen, festen **Steckbrief**, **erzwingen** die Zahlen technisch und **prüfen** den Rest einzeln.

## 2. Die vier Bausteine

### S1: Steckbrief (Spec) statt Zitatberg

- **Eingabe:** die schon verifizierten Setup-Details (Rolle 4, `setup_details.py`, v2-Pfad mit `source_rank`) und ein **kleiner** Repo-Ausschnitt (siehe S4: Modelldatei des Claims, importierte Layer, Rümpfe der aufgerufenen Vorverarbeitungs- und Schedule-Funktionen per `function_body`, Startskript für den Datensatz; nur Quelltext, höchstens 12 000 bis 15 000 Zeichen).
- **Ein** LLM-Aufruf (Code-Rolle) füllt ein festes JSON-Schema. Jeder Eintrag hat `value`, `quote` und `source`.
- **Feste Schlüssel je Familie**, nicht je Paper (gegen Überanpassung an die Dev-Einheiten):
  - gemeinsam: `optimizer`, `lr`, `weight_decay`, `epochs`, `batch_size`, `lr_schedule` (Formel), `early_stopping` (`monitor`, `patience`), `dropout`, `hidden`, `layers`, `activation`, `init`, `model_core` (1 bis 3 Sätze zur Kernrechnung, z. B. „letzten Wert des Fensters abziehen, am Ende wieder addieren“)
  - Graph: `feature_norm`, `adj_norm`, `self_loops`, `heads`
  - Zeitreihe: `seq_len`, `revin_or_norm`, `channel_mode`
- **Deterministische Prüfung ohne LLM:** Zitat steht im Dokument (`quote_in_text`), Zahlen des Werts stehen im Zitat (`numbers_in_quote`). Fällt die Prüfung durch, wird der Eintrag verworfen und nicht geraten.
- **Widersprüche löst eine Regel, bevor ein Modell sie sieht:** Startskript > Paper-Text > README > argparse-Default (Reihenfolge wie `source_rank`). Ein argparse-Default fliegt aus dem Steckbrief, sobald ein Startskript dieselbe Flag setzt (D-Klasse). **Modellname-Regel:** Der Modellname aus dem Claim schlägt jede Modellwahl-Flag im Skript (`model_name=DLinear` im NLinear-Lauf).
- Ausgabe: höchstens etwa 25 Zeilen. Der Steckbrief ersetzt im Plan- und Code-Prompt die heutige Liste aus bis zu 116 Zitaten. Die volle Liste bleibt im Dossier.

### S2: Zahlen erzwingen (deterministisch)

- Pflicht für `train.py`: ein literaler Block `CONFIG = {...}` ganz oben mit allen numerischen Schlüsseln des Steckbriefs.
- Prüfer `spec_check.config_check(code, spec)` ohne LLM, per `ast`:
  1. `CONFIG` existiert und ist ein Literal-Dict.
  2. Jeder Wert stimmt mit dem Steckbrief überein. Bei Abweichung wird der Literalwert **automatisch überschrieben** (AST-Patch) und das als Ereignis `spec_override` protokolliert.
  3. Jeder Schlüssel wird im Code auch **gelesen** (`CONFIG["lr"]` kommt außerhalb des Blocks vor). Ein ungenutzter Schlüssel löst eine gezielte Reparatur aus (S3-Nachricht nur für diesen Schlüssel).
  4. Keine konkurrierenden Literale für dieselbe Größe, z. B. `lr=5e-4` im Optimizer-Aufruf statt `CONFIG["lr"]`.
- Damit sind lr, Epochen, Fenster, Hidden, Dropout, Weight Decay und Patience **per Konstruktion** richtig. Das deckt die D-Fälle DLinear g0/g1 und NLinear g0 ab.

### S3: Nicht-Zahlen einzeln prüfen

- Für jeden nicht-numerischen Eintrag (`feature_norm`, `layers`/Architektur, `activation`, `model_core`, `lr_schedule`, `self_loops` …) eine **kleine** Frage an die Code-Rolle: *„Setzt dieser Code X um? Antworte ja/nein und zitiere die Zeilen.“*
- Das Zitat muss wörtlich im Code stehen (deterministisch geprüft). Fehlt es, gilt die Antwort als „nein“.
- Bei „nein“: **eine** gezielte Reparatur nur für diesen Punkt (Steckbrief-Eintrag mit Quelle plus Zeilen), danach S2 und S3 erneut. Höchstens 2 Runden je Kandidat.
- Warum kleine Fragen: A2 prüft fünf Aspekte in einem großen Prompt und hat dabei 4 von 6 Normalisierungsfehlern verpasst. Eine Frage je Punkt ist leichter zu beantworten und leichter zu prüfen.
- **Auswahlregel:** Unter lauffähigen Kandidaten gewinnt der steckbriefkonforme, auch wenn ein nicht konformer auf Validierung besser ist (GAT g1: der Code mit Leck auf die Validierungslabels war auf Validierung besser). Erst unter konformen Kandidaten entscheidet die Validierung (A1).

### S4: Datei-Klempnerei (F1 aus `recall_plan.md`, ergänzt)

In `repo_source.py`:

- Fixture-Schlüssel je Einheit (Paper + Claim) statt je Repo (`_repo_key`). Im Live-Modus nie eine Fixture lesen, die für einen anderen Claim angelegt wurde (heute `read_repo`, Prüfung bei :153-155).
- Relative Imports auflösen (`from . import gat`), zwei Ebenen tief, Groß-/Kleinschreibung egal, Zeichenbudget statt fester Dateizahl.
- Filter für Nicht-Quelltext: binäre oder nicht UTF-8-lesbare Dateien und Datenordner weglassen (SGC: sechs Pickles, rund 153 000 Zeichen).
- Pflichtdatei: Die Datei mit der Klasse des Claim-Modells muss gelesen werden. Fehlt sie, wird das als Ereignis `model_file_missing` gemeldet und nicht still ignoriert.

## 3. Was das auf die 19 Fehlschläge abbildet

| Klasse (Review) | Läufe | Baustein | Erwartung |
|---|---|---|---|
| D: geliefert, nicht befolgt | DLinear g0, g1, NLinear g0, SGC g2 | S1-Regeln + S2 (+ S3 für SGC-Normalisierung) | hoch, weil deterministisch |
| E: im Repo, nicht herausgezogen | DGI ×3, SGC g0, g1, GAT g2 | S4-Ausschnitt → S1 → S3 | mittel; hängt daran, dass S1 die Angabe herauszieht |
| R: Datei nie gelesen | Shchur-GAT g0, g1 | S4 → S1 → S3 | mittel; TF1-Code muss übersetzt werden |
| R*: falsche Datei übernommen | NLinear g1, g2 | S4-Schlüssel + Modellname-Regel | mittel bis hoch |
| O: Bugs, Protokoll, Absturz | 5 | B1/B2 schon gebaut; Rest nicht abgedeckt | gering |

**Schätzung, nicht gemessen:** heute 2 von 8 Dev-Einheiten. Mit S1 bis S4: 4 bis 6 von 8. Die Obergrenze durch Referenzrauschen liegt bei etwa 7,7 von 8 (`ceiling.py`). Was fehlt, sind Architektur-Fehler außerhalb des Steckbriefs, Bugs und Abstürze.

## 4. Was sich nicht ändert

- **Modus bleibt Nachbau.** Der Steckbrief enthält Werte, Formeln und kurze Beschreibungen mit Beleg, keine ganzen Autorendateien. Das passt zu CLAUDE.md („übernimmt nur belegte Angaben aus Paper und Repo“). Der Repo-Ausschnitt geht nur in den S1-Aufruf, nicht in den Code-Prompt. Die Frage, ob der Code-Prompt ganze Dateien bekommen soll, stellt sich damit nicht.
- **Arm A bleibt unverändert** (kein Paper, kein Repo, also kein Steckbrief). Der A-gegen-B-Vergleich misst danach „Steckbrief vs. Gedächtnis“. Das steht offen im Bericht.
- Toleranzen, Referenzurteile, Seeds und Zählregel bleiben gleich. Keine Testwerte in Prompts.

## 5. Kostenloser Vortest vor jedem bezahlten Lauf

**V0, Liefer-Audit (offline, ohne LLM).** `scripts/review/fact_audit.py` erweitern: Für jede Einheit und jeden vorab gelisteten entscheidenden Fakt prüfen, ob er im neuen S4-Ausschnitt steht. Erwartung: alle R- und E-Fakten enthalten. Sonst zuerst S4 korrigieren.

**V1, Umsetzungstest (OpenRouter `nvidia/nemotron-3-super-120b-a12b:free`, zählt nicht für den Benchmark).**

- Einheiten: DLinear, NLinear, SGC, GAT, DGI, Shchur-GAT und die neuen Papers SparseTSF und RLinear.
- Drei Arme aus gespeicherten Dossiers, ohne neue Extraktion außer dem S1-Aufruf:
  - **heute:** `format_for_prompt_v2` + Plan + Code
  - **Steckbrief:** S1 im Prompt, ohne S2/S3
  - **Steckbrief + Prüfung:** S1 + S2 + S3
- Je Arm und Einheit 3 Code-Proben (Temperatur 0,2 / 0,5 / 0,8).
- Bewertung mit deterministischen Fakt-Checks, die **vor** V1 committet werden (AST/Regex), z. B.: NLinear zieht `x[:, -1:, :]` ab und addiert es wieder; DLinear/NLinear haben `seq_len` 336, lr 0,005 und lr × 0,5 je Epoche; SGC teilt die Features durch die Zeilensumme (L1) und standardisiert keine Spalten; GAT/DGI normalisieren die Features zeilenweise; DGI hat eine Graph-Faltung, PReLU, 512; Shchur-GAT hat additive Attention, Self-Loops und beide Kantenrichtungen. Was kein Check entscheiden kann, bewertet eine Person, die den Arm nicht kennt.
- Optional: Sandbox-Smoke (`SMOKE=1`) zählt Abstürze je Arm.

**Entscheidungsregel (vor V1 committen):** Weiter zum bezahlten Lauf nur, wenn

1. im Arm „Steckbrief + Prüfung“ mindestens **0,7** der R-, E- und D-Fakten im Code stehen **und** das mindestens **0,25** über dem Arm „heute“ liegt,
2. auf den neuen Papers der Anteil nicht unter „heute“ fällt,
3. höchstens 2 von 24 Code-Aufrufen im Arm „Steckbrief + Prüfung“ leer oder abgeschnitten enden,
4. `spec_override` den Code in keiner Probe zum Absturz bringt.

Sonst: Stopp und Bericht, an welchem Baustein es hängt (S1 hat den Fakt nicht herausgezogen / S2 oder S3 haben nicht gegriffen).

V1 sagt nichts über die Accuracy. Es zeigt nur, ob die Fakten im Code ankommen.

## 6. Betroffene Dateien

| Datei | Änderung |
|---|---|
| `src/repro_agent/spec.py` (neu) | Schema je Familie, `build_spec(llm, details, excerpt)`, `verify_spec` (Zitat/Zahl), `resolve(spec)` (Rangregel, Default-Wegfall, Modellname-Regel), `format_spec` |
| `src/repro_agent/spec_check.py` (neu) | `config_check` (AST, Override, Nutzungsprüfung), `item_questions`, `verify_item_answer` (Zitat im Code), `repair_message_for(item)` |
| `src/repro_agent/repo_source.py` | S4: Schlüssel je Einheit, relative Imports zwei Ebenen, Quelltext-Filter, Pflicht-Modelldatei, Zeichenbudget |
| `src/repro_agent/setup_details.py` | `focused_excerpt(docs, claim)`, nutzt `function_body` und `call_bodies` |
| `src/repro_agent/loop.py` | Flag `spec_mode`: `plan` / `write_code` / `repair_code` (:345-424) nehmen `format_spec` statt `_details_prompt`; CONFIG-Pflicht im Code-Prompt; S2/S3 nach jedem Kandidaten in `_run_candidates` (:976); Auswahlregel „konform vor Validierung“; explizites `max_tokens` beim ersten Code-Aufruf (:384-388) |
| `tests/test_spec.py`, `tests/test_spec_check.py` (neu) | Unit-Tests ohne LLM: Rangregel, Default-Wegfall, Modellname-Regel, AST-Override, ungenutzte Schlüssel, Zitatprüfung |
| `scripts/review/fact_audit.py`, `scripts/review/uptake_probe.py` (neu) | V0 und V1 |

Hinter dem Flag `spec_mode` (Default aus), damit v4 reproduzierbar bleibt.

## 7. Reihenfolge und Aufwand (geschätzt)

1. S4 + Unit-Tests: ½ Tag
2. S1 (`spec.py`) + Tests: 1 Tag
3. S2 (`config_check`) + Tests: ½ Tag
4. V0 laufen lassen, S4/S1 nachbessern: ½ Tag
5. Fakt-Checks und Entscheidungsregel für V1 committen, dann V1 laufen lassen (mehrere Stunden Rechenzeit, 0 $)
6. S3 + Einbau in `loop.py`: 1 Tag (kann parallel zu 5 laufen; V1-Arm 3 braucht S3)
7. Nur wenn V1 besteht: vorab festgelegtes v6 mit eigener Datei (Gate wie `recall_plan.md`), Budget neu rechnen

**Kosten im bezahlten Lauf (Schätzung):** S1 ist 1 Aufruf je Claim mit etwa 15 k Tokens rein. S3 sind etwa 5 bis 8 kleine Aufrufe je Kandidat. Dafür wird der Prompt kürzer als heute (Steckbrief statt bis zu 116 Zitate). Größenordnung +0,02 bis +0,05 $ je Claim. Vor v6 aus V1-Token-Zahlen nachrechnen.

## 8. Risiken

- **S1 zieht falsch heraus:** Wird ein Fakt falsch extrahiert, erzwingt S2 einen falschen Wert, und das schadet mehr als heute. Gegenmittel: Zitatpflicht, Zahl-im-Zitat-Prüfung, V0. Jeder Override steht im Dossier und im Zeugnis.
- **Überanpassung:** Die Schlüssel sind je Familie generisch. Die Wirkung belegen nur SparseTSF und RLinear, nicht die Dev-Einheiten. Im Bericht so sagen.
- **Architektur außerhalb des Schemas** (z. B. fünf verschiedene GAT-Fehler) fängt nur `model_core` + S3 ab, und nur teilweise.
- **Reparaturen brechen funktionierenden Code:** höchstens 2 Runden, danach greift die Auswahl über konforme und lauffähige Kandidaten. Kipf (3/3) und Shchur-GCN (2/3) müssen in v6 halten; das wird als eigener Punkt im Gate geprüft.

## 9. Offene Entscheidungen für den Nutzer

1. Ansatz freigeben (S1 bis S4 hinter `spec_mode`)?
2. Auswahlregel „steckbriefkonform vor Validierung“: einverstanden? Sie ändert A1 im Produktpfad.
3. Schwellen der V1-Regel (0,7 / +0,25) so festlegen?
4. Soll `spec_mode` später auch im Produkt (API) laufen oder zuerst nur im Benchmark?
