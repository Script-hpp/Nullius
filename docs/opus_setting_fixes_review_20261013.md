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
