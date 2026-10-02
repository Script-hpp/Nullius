# Nachlauf M1: Notizen zum Bau (Auftrag 35, Branch `nachlauf-m1`)

Regeln P1 bis P4 aus `benchmark/experiments/nachlauf_v1.md` unverändert umgesetzt. Kein LLM, kein Tavily, kein Sandbox-Aufruf; nichts ausgeführt außer Tests (Fake-Sandbox).

## Gebaut

| Datei | Inhalt |
|---|---|
| `src/repro_agent/nachlauf.py` | `UnitSpec`, `run_unit(sandbox, spec, seeds, offsets=…, out_dir=…)`, `check_command` (P1), `forbidden_flags`, `self_check` (P3), `env_diff`, `make_seed_copy`, Snapshot-Befehle, Dossier und Ereignisse |
| `src/repro_agent/harness/independence.py` | P2: `random_offsets`, `make`, `compare`, `summarize`, `parse_predictions` (Orchestrator-Seite, numpy) |
| `src/repro_agent/dossier.py` | nur neues Feld `nachlauf: dict = {}` (Default leer, `load` nutzt `data.get("nachlauf", {})`); `mode` gab es schon |
| `scripts/run_nachlauf_manual.py` | Hand-Specs `rlinear` und `dlinear`, `--unit --seeds --offsets --dry-run` |
| `scripts/nachlauf_manual_adapters/{rtsf,ltsf}_predict.py` | Hand-Adapter nach dem Vertrag (nur Modellklasse, `NACHLAUF_CKPT`, `test_inputs[:, -336:, :]`, kein Datensatzcode des Repos) |
| `tests/test_nachlauf.py` | 26 Tests, Fake-Sandbox |

Ablauf je Seed (`run_unit`): Trainings-Branch (Seed-Kopie prüfen, Start unter `unshare -n` 600 s, Eigenausgabe lesen, `git status`/`git diff --stat` leer, genau ein Checkpoint per `get_file`) → Vorhersage-Branch für `SMOKE=1` → Vorhersage-Branch für den vollen Lauf (`unshare -n`), danach Harness frisch nach `/opt/harness-<zufall>` und `etth1_eval.py` unverändert → P3 → P2 (je Versatz ein frischer Branch, Probe ersetzt `/data/etth1_public.npz`, Prüfung, dass nur die Probe im Branch liegt). Einmal je Einheit: statische Adapterprüfung (`code_check`, vor jedem Snapshot), beide Snapshots, `find / -name 'ETTh1*'` im Vorhersage-Snapshot (Ausgabe im Log, nicht leer = Abbruch), P1 und `env_diff`.
Dossier: `mode="nachlauf"`, `nachlauf[claim_id]` mit allen verlangten Feldern (zusätzlich `predict_snapshot_raw_files`), `isolation` = `NACHLAUF_ISOLATION` (fünf Schichten), Warnungen für Startbefehl-Abweichung, Umgebung, „Zuordnung nicht geprüft", „eine Messung, keine Streuung". Ereignisse: `nachlauf_train`, `nachlauf_predict`, `nachlauf_independence`, `nachlauf_self_check`.

## Anders gelöst als im Auftrag beschrieben

1. **Urteil:** `run_unit` ruft `Orchestrator._verdict(None, …)` unverändert auf (Methode nutzt `self` nicht), statt eine zweite Regel zu schreiben. `baselines=None`: der Test „Agenten-Fehler vermutet" (schlechter als triviale Baseline) soll im Nachlauf nicht greifen, weil dort der Code der Autoren gemessen wird, nicht der des Agenten. Ungültige Seeds (P1/P2/P3, Repo geändert) zählen nicht in das Mittel; ohne gültigen Seed lautet das Urteil „nicht reproduzierbar (Setup)" mit den Gründen in der Erklärung.
2. **Bestehende Testdatei geändert:** `tests/test_droplast.py::test_no_prompt_path_reads_droplast_fields` listet die Module, die `metrics_droplast` kennen dürfen; `nachlauf.py` ist dort ergänzt (eine Zeile). Ein neuer Test in `test_nachlauf.py` stellt sicher, dass `nachlauf.py` keinen LLM-/Prompt-Pfad hat. Sonst keine Änderung an bestehenden Dateien außer `dossier.py`.
3. **Volle öffentliche Datei für P2** erzeugt der Orchestrator selbst (`_load_public_local`: `etth1_data.download` plus `prepare_public`, dieselbe Funktion wie im Snapshot) statt 50 MB per `get_file` zu holen. Das lädt die CSV auf dem Rechner des Reviewers (Netz nötig, kein Sandbox-Aufruf). Die Gleichheit mit der Datei im Snapshot wird nicht eigens geprüft (gleicher Code, gleiche Quelle).
4. **P2 läuft für jeden Seed** (eigener Checkpoint). Das sind 8 kleine Branches je Seed, bei 5 Seeds 40 Branches je Einheit. `--offsets N` senkt das für Entwicklungsläufe.
5. **Seed-Weg „flag"** hängt `<flag> <seed>` an den Startbefehl an. Der Befehl der Seed-Läufe weicht damit vom dokumentierten ab; P1 prüft den Basisbefehl (Repo-Seed). In M1 gebaut, aber nur getestet über `kind="none"` und `kind="copy"`.
6. **`--root_path`, `--data_path`** sind in den erlaubten Abweichungen ohne Meldung (wie verlangt); `--features`, `--data` usw. sind es nicht.
7. **Hand-Adapter** liegen unter `scripts/nachlauf_manual_adapters/` (nicht in `src/`). `spec.adapter_source` ist ihr Text; die statische Prüfung läuft auf genau diesem Text.
8. **Snapshot-Befehle** in `nachlauf.py` doppeln `sandbox_run.py` aus dem Vorversuch (Repo jetzt unter `/work/repo`, ein Repo je Einheit statt zwei). Der Vorhersage-Snapshot nutzt `etth1_task().setup_cmds` (lädt die Rohdatei in ein Temp-Verzeichnis, erzeugt die npz, löscht die Rohdatei); `numpy<2` wird davor installiert, damit der spätere `pip install numpy` es nicht hebt.

## Offene Fragen und nicht in der echten Sandbox geprüft

- **Shell-Details nur im Fake getestet:** Zeitmarker (`date +%s%N`), `ls -1 …/checkpoints/*/…`, `grep -c` für den Seed-Diff, `find / -xdev -name 'ETTh1*'`, die Probe-Prüfung (`np.load` plus `find … | wc -l`). Ein Fehler dort zeigt sich im ersten echten Lauf; Rohausgaben stehen im Log.
- **`find`-Muster `ETTh1*`** trifft Dateien in Python-Paketen nur, wenn sie so heißen; falls ein Repo selbst eine `ETTh1.csv` mitbringt, bricht der Vorhersage-Snapshot-Check ab (gewollt).
- **Eigenausgabe LTSF:** Test-Loader hat dort `drop_last=False`, P3 vergleicht daher mit allen Fenstern; RTSF mit `metrics_droplast["128"]`. Batchgrößen außerhalb von `droplast.DROPLAST_BATCHES` (32, 64, 128, 256) ergeben „nicht geprüft" mit Grund.
- **P1 „wörtlich":** Gleichheit der Schalter (Name und Wert) nach Auflösen von `$var`, Reihenfolge egal; die dokumentierte Zeile mit der kleinsten Abweichung zählt. Positionsargumente und Pipes/`&&` in der Dokumentzeile sind nicht modelliert.
- **Verbotsliste** (`FORBIDDEN_FLAG_NAMES`, `FORBIDDEN_HELP_PATTERNS`) ist nicht vollständig, im Code so kommentiert. LTSF-`--train_only` trifft über Name und Hilfetext.
- **M2 (Negativkontrollen)** sind hier als Unit-Tests abgedeckt (Cheater-Adapter fällt durch P2, `--train_only` durch P1); die Variante „vertauschte Fensterreihenfolge" fehlt noch als Fixture.
- **Zeit:** `run_unit` hat kein Gesamtlimit; ein Seed braucht ca. 25 s Training, Vorhersage und Harness je ca. 10 s, dazu P2.

## Startbefehl für den Reviewer

```
cd ../research-prestudy && uv run python scripts/run_nachlauf_manual.py --unit rlinear --dry-run   # Befehle ansehen
cd ../research-prestudy && uv run python scripts/run_nachlauf_manual.py --unit rlinear --seeds repo --offsets 2   # erster echter Lauf, klein
cd ../research-prestudy && uv run python scripts/run_nachlauf_manual.py --unit rlinear                           # fünf Seeds, 8 Versätze
```
Danach `--unit dlinear`. Erwartung aus dem Vorversuch (nur zum Vergleich, nicht als Kriterium): RLinear Harness alle Fenster ca. 0,3698 (Repo-Seed), Eigenausgabe 0,3660 gegen `drop_last` 128; DLinear 0,3841 gegen 0,3841.
Ausgabe: `runs/nachlauf_manual/<unit>/<zeitstempel>/{dossier.json,events.jsonl}`.

## M2: Negativkontrollen (Auftrag 36, Branch `nachlauf-m2` von `nachlauf-m1` `ead80a3`)

Nichts in einer echten Sandbox gelaufen. Fixtures liegen unter `scripts/nachlauf_manual_adapters/negative/` (jeweils der Hand-Adapter RTSF plus ein markierter Block `# --- negative control X: begin/end`; der Test prüft, dass sie sich nur dort vom ehrlichen Adapter unterscheiden und die statische Prüfung bestehen, damit P2/P3 getestet werden und nicht `code_check`).

| Kontrolle | Einheit | Fixture / Spec | erwartet | Ergebnis Fake-Test |
|---|---|---|---|---|
| (a) Abschreiber | RLinear | `a_copy_neighbors_rtsf.py`: `pred[:n-96] = x[96:, -96:, :]` (Fenster i+96 enthält das Ziel von i), sonst ehrliches Modell; in der Probe fehlt der Nachbar, Rückfall aufs Modell | P2 | P2 **und** P3 lehnen ab; Grund „prediction depends on other windows" |
| (b) Befehl | DLinear | Spec mit `--train_only True` | P1 | P1 lehnt vor jedem Training ab, kein Befehl mit `unshare`, der Trainings-Snapshot-Branch liest nur Dateien |
| (c) vertauscht | RLinear | `c_shifted_rtsf.py` (`np.roll(pred, 1, axis=0)`), `c_permuted_rtsf.py` (feste Permutation) | P3 oder P2 | Verschiebung: P2 immer; P3 nur mit Eigenausgabe und MSE-Änderung > 0,001. Permutation: P3 und P2 |

Skript: `scripts/run_nachlauf_manual.py --negative a|b|c [--variant shift|permute] [--dry-run]` (Repo-Seed, 8 Versätze). Ausgabe eine Zeile `negative X (…): abgelehnt: ja/nein | Pruefung: … (erwartet: …) | Grund: …`. Exit 0 nur, wenn abgelehnt **und** durch eine erwartete Prüfung (a: P2, b: P1, c: P3 oder P2).

**Änderung an M1 (`nachlauf.py`):** P2 läuft jetzt auch, wenn P3 schon ungültig meldet; alle ablehnenden Prüfungen stehen im Grund und in `nachlauf[claim]["rejected_by"][seed]` (auch `["P1"]`, `["static"]`). Vorher hätte (a) wegen MSE ≈ 0 schon an P3 geendet und P2 wäre nie gelaufen; die Vorab-Festlegung verlangt aber, dass der Abschreiber an P2 scheitert.

**Befund zu (c), welche Prüfung greift.**
- **P2 greift strukturell** bei jeder Reihenfolgeänderung: P2 vergleicht je Position der Probe mit dem vollen Lauf; die Vorhersage unter der ID von Fenster i stammt nach der Verschiebung von Fenster i-1, und in der Probe liegt dort ein anderes Fenster oder Rauschen.
- **P3 greift hier ebenfalls**, aber nur wenn der Autorencode eine eigene Metrik ausgibt. Echte Zahlen, lokal aus den Vorhersagen von Schritt 1 gerechnet (RLinear, Seeds 1024, 0, 1, 2, 3; kein neuer Lauf, kein Sandbox-Aufruf; Rechenweg: `parse_predictions`, `np.roll`, `etth1_data.targets`): Verschiebung um 1 Fenster hebt den MSE von 0,370 auf 0,428 (alle Fenster) bzw. von 0,3660 auf 0,4238 (`drop_last` 128, Seed 1024); Abstand zur Eigenausgabe 0,055 bis 0,058 gegen die Grenze 0,001. Zufällige Permutation: 1,21, Abstand 0,85. Der ehrliche Wert stimmt auf 0,0000.
- Also **kein Befund gegen die Vorab-Festlegung**: Verschiebung um 1 wird von P2 immer und von P3 (bei RLinear) mit großem Abstand abgelehnt. Grenze: Bei einem Repo ohne Eigenausgabe („nicht geprüft") oder einem Modell mit sehr glatten Vorhersagen (Verschiebung ändert den MSE um < 0,001) bleibt P2 allein. Im Fake-Test mit synthetischem Signal ist das als Fall „ohne Eigenausgabe: nur P2" abgedeckt.
- Dieselbe Rechnung für (a) mit dem Idealfall „Vorhersage = Ziel für Fenster 0 bis 2688": MSE 0,0164 (alle Fenster), 0,0000 (`drop_last` 128), Abstand zur Eigenausgabe 0,366; P3 würde ebenfalls ablehnen.

**Weitere Änderungen:**
- `FORBIDDEN_HELP_PATTERNS` um `(on|with) (the )?full (input )?data` ergänzt: der echte LTSF-Hilfetext von `--train_only` („perform training on full input dataset without validation and testing") passte auf kein altes Muster (nur der Name fing ihn ab). Test mit anderem Flag-Namen und dem echten Hilfetext.
- Beide Skripte (`run_nachlauf_manual.py`, `prestudy_nachlauf/sandbox_run.py`) haben `default_factory()` mit der Importzeile der echten Fabrik; `tests/test_nachlauf_m2.py::test_scripts_import_the_real_sandbox_factory_path` führt sie ohne Client aus. Gegenprobe: mit der alten Zeile `from repro_agent.sandbox import get_sandbox` ist der Test für `run_nachlauf_manual.py` rot (`ImportError`).
- Zu (b): Das Trainings-Snapshot wird vor P1 noch gebaut und ein Branch liest Repo-Dateien (Hilfetexte für P1). „Kein Trainings-Branch" heißt: kein Branch führt den Startbefehl aus.

**Startbefehle für den Reviewer** (je eine Zeile Ausgabe, Exit 0 nur wenn abgelehnt):
```
cd ../research-m2 && uv run python scripts/run_nachlauf_manual.py --negative a --dry-run
cd ../research-m2 && uv run python scripts/run_nachlauf_manual.py --negative a
cd ../research-m2 && uv run python scripts/run_nachlauf_manual.py --negative b
cd ../research-m2 && uv run python scripts/run_nachlauf_manual.py --negative c
cd ../research-m2 && uv run python scripts/run_nachlauf_manual.py --negative c --variant permute
```
(b) bricht vor dem Training ab und braucht nur die beiden Snapshots und einen Lese-Branch; (a) und (c) laufen einmal Training (ca. 25 s) plus Vorhersage, Harness und 8 P2-Versätze.

## M3: Agenten-Teil (Auftrag 38, Branch `nachlauf-m3` von `main` `9b52d5e`, Worktree `../research-m3`)

Entwickelt nur mit Fake-LLM und Fake-Sandbox an einem **erfundenen Repo** („gizmo"); kein echter LLM-, Tavily- oder Sandbox-Aufruf; die vier neuen Einheiten wurden nicht angefasst (kein Klon, keine Fixture, kein Name in Tests).

### Neu

| Datei | Inhalt |
|---|---|
| `src/repro_agent/nachlauf_agent.py` | Prompts (Konstanten), `NachlaufAgent.run`, `read_repo_at_commit`, `verify_answer` (Zitatprüfung), `CostGuard`/`BudgetStop`, `Asker` (alle LLM-Aufrufe), Erkundung von Training und Adapter |
| `scripts/run_nachlauf.py` | Einstieg `--paper --repo --commit [--claim --seeds --offsets --store --budget-stop --dry-run]`, Store-Zeile je Lauf |
| `src/repro_agent/nachlauf.py` (Ergänzungen) | P2 parallel (Thread-Pool, `MAX_PARALLEL_SEEDS`, Standard 4), Versätze je Seed neu gezogen, öffentliche Datei einmal geladen, `new_dossier`, `run_unit(..., snapshots=, event_log=, parallel=)`, `UnitSpec.claim_source` |
| `tests/test_nachlauf_agent.py` | 24 Tests (Fake-LLM, Fake-Sandbox) |

`loop.py`, `tasks.py` und der Nachbau-Pfad sind unverändert (`git diff main -- src/repro_agent/loop.py` leer).

### Ablauf und Prompts

`NachlaufAgent.run`: (1) Lese-Snapshot: Klon auf dem festen Commit, `git ls-files`, das Modell wählt bis zu 14 Dateien (`FILE_SELECT_*`, Rolle `extract` = Modell von `code`), dazu Abhängigkeitsdateien und lokal importierte Module, je Datei höchstens 16 000, insgesamt 120 000 Zeichen im Prompt. (2) `start_command` (Rolle `code`, JSON, Konstanten `COMMAND_SYSTEM`, `COMMAND_USER`, `COMMAND_FORMAT`): Befehl mit Quelle, `seed`, `checkpoint_glob`, `test_loader.drop_last` und `.batch_size` mit Datei und Zitat, `input_len`, `output_len`, `needs_time_features` mit Zitat, `model_file`, `training_file`. Jedes Zitat wird gegen die genannte Datei geprüft (`setup_details.quote_in_text`, normalisiert); unbelegte Angaben werden **einmal** mit `COMMAND_REASK` nachgefragt, danach verworfen (`agent.discarded` im Dossier). (3) Erkundung am Repo-Seed: Training mit Reparatur (P1 vor jedem Lauf), dann Adapter schreiben und prüfen (statisch, `SMOKE=1`, voller Lauf, P3, P2, je mit Reparaturrunde). (4) Der feste Stand läuft über alle Seeds mit `nachlauf.run_unit` (8 Versätze je Seed) und liefert das Dossier.

Feste Rückmeldungen (Wortlaut in `P3_SENTENCE`, `P2_SENTENCE`, `P1_SENTENCE`, `COMMAND_SHAPE_SENTENCE`, `PACKAGE_SENTENCE`): P3 und P2 wie im Auftrag, **ohne jede Zahl**. Reparatur nach Trainingsfehler: Ende von stderr (1500 Zeichen), nie stdout, nie eine Zusammenfassung. Reparatur nach Formfehler: Meldung der Formprüfung (Form, keine Testwerte) oder Ende von stderr. Prompts enthalten keinen Repo-, Modell- oder Dateinamen und keinen Schalter der Entwicklungs-Repos; Beispiele sind als erfunden gekennzeichnet. Wortlisten-Test über alle Prompt-Konstanten und alle Prompts eines vollen Ablaufs mit erfundenem Repo.

Ereignisse: `nachlauf_command` (Befehl, Zitat-Ergebnisse, Seed-Art, Test-Loader), `nachlauf_adapter` (Hash, Stufe, Probleme), `nachlauf_repair` (Art, Runde, Grund), dazu die M1-Ereignisse. Adapter-Quelltexte als `adapter_<hash>.py` und die Befehlsantwort mit Prüfung als `command_answer.json` im Laufordner.

### Abweichungen vom Auftrag (mit Grund)

1. **JSON-Felder `input_len` und `output_len` statt `seq_len` und `pred_len`:** beides sind Schalternamen der Entwicklungs-Repos und dürfen in keinem Prompt stehen. Zusätzlich `model_file` und `training_file` (Eingabe des Adapter-Aufrufs).
2. **Zwei Läufe am Repo-Seed:** Die Erkundung (mit 2 P2-Versätzen) trainiert und prüft am Repo-Seed, der Endlauf rechnet dann alle Seeds inklusive Repo-Seed noch einmal mit 8 Versätzen. Das kostet etwa 25 s Training und die Prüfungen doppelt, hält aber `run_unit` (M1, abgenommen) unverändert die einzige Stelle für Messung und Urteil. Fehlgeschlagene Erkundungsversuche stehen vor den Seeds im Dossier.
3. **Versuchsgrenze:** `MAX_ATTEMPTS_PER_CLAIM` (4) gilt getrennt für Trainingsversuche (Befehle, einschließlich abgelehnter) und Adapter-Versionen. „Gesamtgrenze wie oben" habe ich so gelesen; ein gemeinsamer Zähler wäre eine Zeile.
4. **Standardpakete:** Beide Snapshots bekommen `pandas`, `scikit-learn`, `matplotlib` (`BASE_PIP`). Ein Repo ohne Abhängigkeitsdatei (RTSF hat keine) könnte sonst nicht laufen, und die Reparatur darf nur Pakete aus Abhängigkeitsdateien nachinstallieren. Das ist eine Umgebungsentscheidung für diese Task-Familie; bitte bestätigen.
5. **`read_repo` nicht wiederverwendet:** Es klont flach den Hauptzweig und cacht eine Fixture je Repo. Für den Nachlauf zählt der feste Commit, daher `read_repo_at_commit` (nutzt `repo_source._last_string_list` und `imported_files`).
6. **P2 läuft auch nach P3-Fehler** (Änderung aus M2, hier unverändert); bei Reparatur gilt „eine Runde je Prüfung": Wird dieselbe Prüfung nach der Runde erneut ungültig, endet die Adaptersuche.
7. **Seed `flag`:** Der Repo-Seed ist der `default=` des Schalters, falls er in der Argumentdefinition steht, sonst `None` (der Lauf ohne Schalter). Nur mit `copy` und `none` im Fake getestet (`flag` über die Prüfung der Argumentdefinition).
8. **Befehle mit Shell-Operatoren** (`;`, `|`, `&&`, Umleitung ausser der Log-Umleitung, Variablenpräfix) werden mit festem Satz zurückgewiesen und zählen als Versuch.

### Kosten, Parallelität

- **LLM-Aufrufe je Lauf:** Normalfall 3 (Dateiauswahl, Startbefehl, Adapter); mit nachgefragten Zitaten 4; höchstens etwa 11 (1 + 2 + 3 Befehls-Reparaturen + 1 + 3 Adapter-Reparaturen, dazu Wiederholungen bei abgeschnittenen Antworten). Prompts bis rund 30 000 Token Eingabe; bei den Preisen von Super (0,30 / 0,90 $ je Million Token) grob 0,01 bis 0,03 $ je Aufruf, **Schätzung, nicht gemessen**. `MAX_COST_PER_RUN_USD` (0,50 brutto) gilt je Lauf zusätzlich.
- **Budgetstopp** (`CostGuard.check`, vor **jedem** LLM-Aufruf): Kosten dieses Prozesses (soweit nicht schon im Store) plus Store (Zeilen mit `cost_usd_run` netto, brutto × 1,19) ≥ `--budget-stop` (Standard 3 $) → `BudgetStop`, Store-Zeile mit Urteil „abgebrochen". Kosten eines fertigen Laufs gehen erst nach dem Schreiben der Zeile in den Store-Anteil (`note_stored`), kein Doppelzählen. Tests: Prozesskosten, Store, Doppelzählung. LLM-Cache ist im Skript aus.
- **Parallele Sandbox-Sitzungen:** Der Loop fährt Seeds schon parallel (`ThreadPoolExecutor`, `MAX_PARALLEL_SEEDS`, Standard 4) auf einer gemeinsamen `NebiusSandbox`; jeder Branch gehört einem Thread, `_sessions[branch]` wird nur von diesem Thread geschrieben (verschiedene Schlüssel, CPython-Wörterbuch). P2 nutzt dasselbe Muster. Proben werden vorher nacheinander gebaut (gemeinsamer Zufallsgenerator), die Ergebnisse stehen in Versatz-Reihenfolge. Test: Ergebnis mit 1 und 4 Threads gleich.

### Offene Fragen und Risiken (nicht in der echten Sandbox geprüft)

- **P3 ohne belegten Test-Loader** vergleicht mit allen Fenstern. Hat das Repo `drop_last=True`, liegt die Eigenausgabe um den `drop_last`-Effekt daneben (Vorversuch RLinear: 0,0038 > 0,001): P3 wäre dann ungültig, obwohl nichts falsch ist, und die feste P3-Rückmeldung hilft dem Modell nicht. Das folgt der Vorab-Festlegung; ich habe es nicht verändert. Ein belegter Loader vermeidet es.
- **stderr in der Reparatur:** Druckt ein Repo Testmetriken nach stderr (z. B. über `logging`) und scheitert danach, stehen sie im Prompt. Der Auftrag verlangt das Ende von stderr unverändert; ein Filter wäre eine Ergänzung.
- **Zitatprüfung beweist Vorkommen, nicht Bedeutung:** `drop_last = True` kommt in Ladern oft mehrfach vor (Test, Vorhersage, Training); das Zitat belegt die Zeile, nicht den Zweig. P3 fängt Fehlzuordnungen.
- **Wortlisten-Wache** ist nur so gut wie die Liste (`NAMES` im Test); neue Repos brauchen neue Einträge, bevor sie in einen Test kommen.
- **Ladezeit der öffentlichen Datei:** `_load_public_local` lädt die CSV auf dem Rechner des Reviewers (einmal je Lauf).
- Fake-Tests ersetzen nicht den ersten echten Lauf (Fehlerquelle M1: Importzeile der Fabrik; `default_factory` in `run_nachlauf.py` ist über den Import-Test abgedeckt).

### Startbefehle für den Reviewer

```
# erst Plan ansehen (kein LLM, keine Sandbox)
cd ../research-m3 && uv run python scripts/run_nachlauf.py --paper rlinear_2023 --repo https://github.com/plumprc/RTSF --commit 0fec00104f754f4fbf795b9b4da5fa2459b32e76 --store runs/nachlauf_dev.jsonl --dry-run
cd ../research-m3 && uv run python scripts/run_nachlauf.py --paper zeng_dlinear_2023 --repo https://github.com/cure-lab/LTSF-Linear --commit 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6 --store runs/nachlauf_dev.jsonl --dry-run
# echter Lauf (Token Factory, Sandbox); klein beginnen
cd ../research-m3 && LLM_PROVIDER=tokenfactory uv run python scripts/run_nachlauf.py --paper rlinear_2023 --repo https://github.com/plumprc/RTSF --commit 0fec00104f754f4fbf795b9b4da5fa2459b32e76 --store runs/nachlauf_dev.jsonl --seeds repo --offsets 2 --budget-stop 1.0
cd ../research-m3 && LLM_PROVIDER=tokenfactory uv run python scripts/run_nachlauf.py --paper zeng_dlinear_2023 --repo https://github.com/cure-lab/LTSF-Linear --commit 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6 --store runs/nachlauf_dev.jsonl --seeds repo --offsets 2 --budget-stop 1.0
```
(Entwicklungsläufe in `runs/nachlauf_dev.jsonl`, nicht im Store der Prüfung `runs/nachlauf_v1.jsonl`.)

### M3b (Auftrag 38b): robuste Modell-Antworten

Anlass: Im ersten echten Lauf lieferte das Modell in der Nachfrage-Antwort `seed` als String; `verify_answer` stürzte mit `AttributeError` ab.

- **Normalisierung** (`verify_answer`, `_obj`, `_text`, `_intval`, `_quote_ok`): Jede Angabe mit falschem Typ oder falscher Form (String, Liste, Zahl, null, `{}`, fehlender Schlüssel, Zahl als Liste, Datei als Liste) gilt als **nicht belegt** und geht in denselben Rückfall wie ein nicht wörtliches Zitat (einmal nachfragen, dann verwerfen, Warnungen wie bisher). Gilt für `start_command`, die Nachfrage (Merge ohne Typannahmen), `repair_command` (Befehl, `command_source`, `install`, `checkpoint_glob`) und die Dateiauswahl (unbrauchbare oder leere Auswahl → deterministischer Rückfall nach Namen, `fallback_files`).
- **Gerettet** (sinnvolle Fälle): `seed: "none"` als Wort = keine Seed-Methode (belegt); `seed: "copy"` oder `"flag"` ohne Objekt/Zeile = nicht belegt, also kein Seed-Weg (Warnung „eine Messung, keine Streuung"); `needs_time_features` als nacktes `false`/`null` = nein; `true` ohne belegtes Zitat zählt nie als „ja"; `batch_size.value` als Ziffernfolge (`"128"`) wird als Zahl gelesen, das Zitat entscheidet. Ein Zitat direkt als String statt `{file, quote}` ist nicht belegt.
- **Rohantworten:** `Asker._dump` schreibt jede Antwort sofort nach dem Aufruf, vor dem Parsen, als `llm_<schritt>_<n>.txt` in den Laufordner (nie in einen Prompt; Test).
- **Unerwartete Fehler:** `NachlaufAgent.run` fängt alles ausser `BudgetStop`, `BillingBlocked`, `BudgetExceeded` ab (dazu zählt auch `AgentAnswerError`, wenn die Antwort nach drei Versuchen unbrauchbar bleibt). Ergebnis: Dossier mit Urteil „nicht reproduzierbar (Setup)", Grund in der Erklärung, `nachlauf[claim]["agent"]["error"]` (Typ, Meldung, Traceback-Ende), `error.txt` im Laufordner, Ereignis `nachlauf_repair` mit `kind: crash`. `run_nachlauf.py` schreibt dazu die Store-Zeile mit den Kosten bis zum Absturz (Test).
- **Tests** (`tests/test_nachlauf_agent.py`, jetzt 120): parametrisiert über falsche Formen je Feld (`command_source`, `seed`, `test_loader.drop_last`, `.batch_size`, `needs_time_features`), der gemeldete Fall (Nachfrage mit `seed` als String, `command_source` als String, `batch_size` als String), Nachfrage ohne Objekt, `repair_command` mit falschen Formen, unbrauchbare Dateiauswahl (5 Formen), Rohantworten, unbrauchbare Antwort → Urteil statt Traceback, unerwarteter Fehler, Budget-/Abrechnungsfehler laufen durch, Store-Zeile nach Absturz.
- Offen: Eine schmale, aber gültige Dateiauswahl (z. B. nur README) führt zu wenigen Zitaten; der Lauf endet dann mit weniger belegten Angaben (Warnungen), nicht mit einem Fehler. Ein zweiter Auswahlversuch ist nicht gebaut.
