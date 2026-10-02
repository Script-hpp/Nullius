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
