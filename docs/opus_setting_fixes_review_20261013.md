# Review: Setting-Fixes fuer FITS (Quelle -> Spec -> CONFIG)

Stand **2026-10-13, 17:42 UTC**, `ceo` `f9ad1cb` plus uncommitteter Diff `5159a37c…` (`experiment_settings.py`,
`spec.runtime_items`, `repo_source.forecast_selection`, `loop`-Zulassung, Tests/Fixture). Scratch-Kopie, `ceo`
unveraendert, keine Calls, keine Downloads. Probe: `docs/audits/opus_settings_probe_20261013.{py,json}`.

## Ergebnis

**Zwei Blocker vor Push.** Beide werden durch die neue Regel verschaerft, dass unter `independent_selection` nur
Kandidaten mit voller Spec-Erfuellung zugelassen werden (`_admit_final_code`, Kandidatenfilter): Ein falsch oder
unnoetig erzwungener Wert fuehrt jetzt zu `MethodImplementationBlocked` statt nur zu einem schlechteren Score.

Bestaetigt: FITS-Kette liefert seq 360, batch 128, lr 0.0005, epochs 50, patience 20, cut 106 aus dem
Original-CLI; Header-/Kommentar-Belege werden verworfen (`\%` bleibt); Discovery-Zitat wird als `paper` gefuehrt,
wenn es im aktiven Papertext steht; Graph-Familie erhaelt keine Laufzeit-Settings (0 Eintraege); Autorencode wird
nicht ausgefuehrt (nur `ast.parse`, `shlex`, `literal_eval` auf CLI-Token, begrenzte Arithmetik ohne Potenz,
Zahlen < 1e9). Seit 17:33 behoben: `set_defaults` (Literal), andere Namespace-Namen (`opt`), bedingte
Shell-Zuweisungen (`if`/`case`).

## Blocker

**B1 Nicht modellierte Mutationen erzwingen falsche Werte.** `python_settings` invalidiert nur `args.X`-Stores in
Top-Level-`If/For/While/Try/With`. Diese Faelle liefern weiter den Default/CLI-Wert statt UNKNOWN (CLI
`--batch_size 64`, wirklicher Wert 128 bzw. aus Konfiguration):

| Muster | Ergebnis |
| --- | --- |
| `def main(): args = parser.parse_args(); args.batch_size *= 2` | 64 |
| `for k, v in cfg.items(): setattr(args, k, v)` (YAML) | 64, lr 0.001 |
| `setattr(args, 'batch_size', 128)` auf Modulebene | 64 |
| `args.__dict__.update(...)`, `vars(args).update(...)` | 64 |
| `args, _ = parser.parse_known_args(); args = load_config(args.config, args)` | 64 |

Minimaler Fix: Attribut-Stores auf Dest-Namen **ueberall** im Modul (auch in Funktionen/Klassen/Lambdas)
invalidieren; bei `setattr(`, `vars(`, `__dict__`, Neuzuweisung des Namespace durch einen anderen Aufruf als
`parse_args` oder Konfigurationsladern alle Werte dieser Datei UNKNOWN setzen. Dazu Tests fuer jede Zeile oben.

**B2 Explizite, nirgends gelesene CLI-Flags werden Pflichtschluessel.** Gesamtsuite auf der Kopie:
1 failed, 1536 passed — `test_code_grounded.py::test_build_spec_numeric_only_prompt_has_no_text_keys_and_keeps_raw_excerpt`.
`--e 1 --f 2` werden als erzwungene Keys `e`, `f` aufgenommen (`name in explicit`). In geteilten TSLib-Skripten sind
das z. B. `label_len`, `factor`, `moving_avg`, `d_ff`, `enc_in`. Ein ungenutzter Pflichtschluessel ergibt
`unused`, Reparaturdruck „use CONFIG[...] where applied“ und jetzt den harten Ausschluss. Fix: explizite Nicht-
Standardflags nur, wenn `args.<name>` ausserhalb von `add_argument` im Startskript, seinen Importen oder im
Modellkern gelesen wird (FITS: `train_mode`, `cut_freq` bleiben; `e`, `f` fallen weg). Test dann bewusst anpassen.

## Weitere Punkte (keine Blocker)

- **Widerspruechliche Startskripte:** Zwei passende Skripte mit seq 360/720 -> `seq_len` wird nicht erzwungen,
  aber mit Wert 360 (erstes Skript) im Steckbrief gefuehrt. Besser als Konflikt ohne Wert ausweisen.
- **Unaufgeloeste Variablen passen:** `call_matches` akzeptiert `--data $data`, `--model $model_name`,
  `--pred_len $pred_len`. Richtig fuer feste Werte ueber Schleifen; Werte, die je Datensatz in anderen Dateien
  gesetzt werden, bleiben unsichtbar. Bekannt aus E09/E10, hier nur noch wirksamer.
- **`forecast_selection` stoppt bei mehr als vier Datensatzskripten** ohne README-Kennzeichnung (`SourceContextBlocked`).
  Fuer andere Katalog-Repos ohne Downloads nicht pruefbar; sicherer waere: Laufzeit-Settings auslassen und warnen
  statt den ganzen Lauf zu stoppen.
- **Graph und andere Paper:** Laufzeit-Settings nur fuer `forecast`; `active_paper_text` wirkt auf alle LaTeX-Papers
  (Kommentare sind kein Beleg mehr), gewollt.

## Nachtrag: Schlussreview der B1/B2-Nacharbeit

Stand **2026-10-13, 17:51 UTC**, `ceo` `f9ad1cb` plus Diff `8a4ea8f1…` (Quellen danach unveraendert, nur Tests).
Scratch-Kopie, keine Calls/Downloads. 178 fokussierte Spec-/Repo-/Setting-Tests passed (inkl. `test_code_grounded`).
Proben: `docs/audits/opus_settings_probe2_20261013.{py,json}`, `opus_real_script_probe_20261013.{py,json}`.

**B2 behoben.** `--e 1 --label_len 48` ohne Lesestelle werden nicht mehr Pflicht; FITS behaelt `train_mode`/`cut_freq`.

**B1 im Kern behoben:** Mutation in `def main()`, `setattr`-Schleife, `__dict__`/`vars`-Update,
`parse_known_args` + Neuzuweisung -> UNKNOWN; `set_defaults` und `opt`-Namespace korrekt.

### Neuer Blocker A: Horizont- und Datensatzschleifen stoppen jeden Lauf

Die neue Identitaetspruefung stoppt mit `SourceContextBlocked`, wenn ein passender Aufruf `--pred_len $pred_len`
oder `--data $data` aus einer Schleife traegt; `call_matches` laesst solche Aufrufe aber zu. Die **echten**
offiziellen Startskripte aus frueheren lokalen Laeufen haben genau diese Form:

| Skript (aus `assistant/runs/…/dossier.json`) | Ergebnis |
| --- | --- |
| `scripts/SparseTSF/linear/etth1.sh` (`seq_len=720`, `for pred_len in 96 192 336 720`) | BLOCKED: prediction horizon unresolved |
| `scripts/SegRNN/etth1.sh` (gleiches Muster) | BLOCKED: prediction horizon unresolved |

Damit wuerden SparseTSF und SegRNN (frueher reproduziert bzw. gemessen) ohne Experiment enden. Fix: Schleifen
ueber Identitaetsvariablen (Horizont, Datensatz, Modell) aufloesen; enthaelt die Liste den Zielwert, gilt fuer diese
Instanz genau dieser Wert, sonst passt der Aufruf nicht. Andere Schleifen (z. B. `seq_len`-Gitter) bleiben unbekannt.
Regressionstest mit beiden echten Skripten.

Folgerisiko fuer SegRNN: Das Repo hat zusaetzlich `scripts/SegRNN/Lookback_96/etth1.sh` (seq 96). Nach dem
Aufloesen trifft der Discovery-Wert 720 auf einen Laufzeitwert 96 und loest den Konfliktstopp aus. Wenn ein
Discovery-Wert fuer `seq_len` belegt ist, sollte er Aufrufe auswaehlen (wie der Horizont), nicht mit allen
anderen Skripten kollidieren; Stopp nur, wenn kein Aufruf passt.

### Blocker B (kleiner, gleiche Klasse wie B1): drei Mutationspfade erzwingen weiter den CLI-Wert

| Muster | Ergebnis (CLI 64, wirklich 256) |
| --- | --- |
| `self.args.batch_size = 256` in einer Klasse derselben Datei | 64 |
| `hp = args; hp.batch_size = 256` | 64 |
| `self.args.batch_size *= 4` im importierten `exp/exp_main.py` | 64 |

Fix: Stores auf `*.args.<dest>` und auf Aliase des Namespace invalidieren und gelesene Repo-Dateien (Importe des
Startskripts) nach solchen Stores durchsuchen. Unter der neuen Pflicht zur vollen Spec-Erfuellung wird ein solcher
falscher Wert dem Agenten aufgezwungen.

Fail-closed bei mehr als vier Datensatzskripten und Gleichrang-Konflikten ohne Paperurteil: als bewusste
Entscheidung akzeptiert; Blocker A zeigt aber, dass „unbekannt -> Stopp“ nur fuer wirklich unbestimmte Werte gelten darf.

## Nachtrag 2: Schlusspruefung nach A/B-Nacharbeit

Stand **2026-10-13, 17:58 UTC**, `ceo` `410fd9f` plus Diff `4621a43c…`. Scratch-Kopie, keine Calls/Downloads.
**Gesamtsuite: 1551 passed, 20 skipped, 5 xfailed.** **Kein Blocker mehr.**

| Pruefung | Ergebnis |
| --- | --- |
| Echte Skripte `SparseTSF/linear/etth1.sh`, `SegRNN/etth1.sh` (Horizontschleife) | kein Stopp; SparseTSF seq 720, batch 256, lr 0.02, epochs 30, patience 5, period 24; SegRNN seq 720, batch 64, lr 0.0003, seg 24, d_model 512 |
| Zweites Skript nur mit Horizonten 336/720 neben konkretem 96-Aufruf | kein Stopp, Werte des 96-Aufrufs |
| Mutationen: `def main`, `setattr`, `vars`/`__dict__`, `parse_known_args` + Neuzuweisung, `self.args` in Klasse | UNKNOWN |
| Alias `hp = args; hp.batch_size = 256` (Modulebene) | 256 (richtig ausgewertet) |
| `self.args`-Mutation im importierten `exp/exp_main.py` | Stopp „unresolved model parameters“ (fail-closed, akzeptiert) |

Cross-Paper-Gate mit echten gecachten Autoren-Repos (`docs/audits/opus_cross_paper_probe_20261013.{py,json}`,
deterministischer Teil von `build_spec`, Modellantwort leer):

| Fall | committed `410fd9f` | neuer Stand |
| --- | --- | --- |
| FITS (`tests/fixtures/fits_runtime_settings.json`) | lr, patience | seq 360, batch 128, lr 0.0005, epochs 50, patience 20, cut 106, train_mode 1, enc_in 7, pred_len 96 |
| DLinear (`cure-lab_LTSF-Linear`) | lr 0.005, batch 32, seq 336 | dazu epochs 10, patience 3 (argparse-Defaults von `run_longExp.py`, im Skript nicht ueberschrieben), pred_len 96; `enc_in` nicht erzwungen (7 vs 1, siehe unten) |
| NLinear (gleiches Repo, Skript startet DLinear) | lr/batch/seq aus DLinear | unveraendert (kein Laufzeitwert) |
| MoHETS (nur Weather-Skripte) | nichts | nichts |
| GCN Cora (Graph) | nichts | nichts |

Restpunkte (keine Blocker):
1. **Feature-Modus nicht gebunden.** `Linear/univariate/etth1.sh` (`--feature S --enc_in 1`) zaehlt als passender
   DLinear/ETTh1/96-Aufruf. Hier kostet das nur `enc_in` (Gleichstand, nicht erzwungen); bei abweichender lr waere
   auch lr nicht erzwungen. `features`/`feature` = M als Identitaet binden (argparse kuerzt `--feature` zu `--features`;
   `python_settings` fuehrt es als eigenen Namen).
2. **NLinear-Altlast (wie committed):** Die alten Rohregeln nehmen weiter den Aufruf mit `--model $model_name`, obwohl die
   neue Auswertung ihn als DLinear aufloest. Dieselbe Aufloesung sollte den Rohpfad fuer andere Modelle sperren.
3. Gleichrang-Konflikte bleiben nicht erzwungen; ein eigener Stopp dafuer ist im Code nicht zu sehen.
