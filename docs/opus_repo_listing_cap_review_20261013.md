# Review: abgeschnittene Repo-Dateiliste im FITS-Live-Lauf 35758c32

Stand **2026-10-13**, Basis `ceo` `be1a643` (Scratch-Export, `ceo` unveraendert). Keine Provider-, Tavily-, Nebius- oder
Download-Aufrufe. Lokaler FITS-Klon `/tmp/nullius-fits-independent-20261013` (HEAD `d040bb015b6299da26d879b90dd19c80fb72c160`,
sauber) statt Netz. Probe `docs/audits/opus_repo_listing_cap_probe_20261013.py`, Ausgabe `…out.txt`.

## 1. Ursache: bestaetigt

- `git ls-files` des gepinnten Repos: **1410 Dateien, 83540 Bytes**. Byte-Offsets: `AD/FITS.py` 11,
  `models/FITS.py` 82903, `run_longExp_F.py` 83078, `scripts/FITS/etth1_best.sh` 83168. 1340 der 1410 Eintraege liegen unter `logs/`.
- `NebiusSandbox.run` setzt `truncate_output_at` nur, wenn der Aufrufer ihn uebergibt. `read_repo` tut das nicht. Damit gilt
  `contree_sdk.config.default_truncate_output_at = 65535`.
- Kopf bis 65535 Bytes = 1089 ganze Zeilen plus **Teilzeile `lo`**. Diese Teilzeile steht als falscher Pfad in der Liste.
- Emulation von `read_repo` mit derselben Kappung liefert die Live-Dateiwahl: `AD/FITS.py`, `AD/main.py`, `README.md`,
  `data_provider/*`, `exp/*`, `layers/utils.py`. Ursache: `model_file_in_listing` findet nur `AD/FITS.py`, also gilt der
  Filter `other_task` nicht. `forecast_selection` findet kein `etth1`-Startskript.
- Voller Listing-Inhalt, sonst identisch: `models/FITS.py`, `scripts/FITS/etth1_best.sh`, `run_longExp_F.py` zuerst;
  `AD/` faellt weg. Die fruehere Probe (`opus_cross_paper_probe_*`) zeigt fuer diese Dateien die volle FITS-Kette
  360/128/0.0005/50/20/106/train_mode1/enc_in 7/pred 96.
- Kopf statt Ende bleibt erhalten: Live wurde `AD/FITS.py` (Offset 11) gelesen, `models/FITS.py` (Offset 82903) fehlte.
  Bei einer Kappung am Ende waere es umgekehrt. Das Serververhalten selbst ist nicht direkt gemessen.

## 2. Budget

| | gekappt (live) | voll |
| --- | --- | --- |
| gelesene Dateien | 9 | 15 |
| Zeichen | 86810 | 111091 (< `IMPORT_CHAR_BUDGET` 120000, keine Kuerzung) |
| `target_docs` fuer Rolle 4 | `AD/main.py`, `exp/*`, `data_provider/*`, `layers/utils.py` | `etth1_best.sh`, `run_longExp_F.py`, `exp/*`, `data_provider/*` |
| Repo-Abschnitte fuer die Extraktion (5000 Zeichen) | 18 | 18 |

- Modellkosten bleiben gleich: `MAX_TARGET_DOCS = 8` begrenzt die Extraktion, der Steckbrief-Ausschnitt bleibt bei `EXCERPT_CHARS = 15000`.
- Transport auf Nebius: ein Metadaten-Run und ein Chunk mit etwa 83 KB, kein Problem.
- Nebenbefund, schon vorher so: `repo-readme` faellt in beiden Faellen aus `target_docs`, weil `picked[:8]` schneidet.

## 3. Design-Pruefung des geplanten Fixes (Datei + Bytezahl/SHA + `get_file`)

Der Ansatz ist richtig. `get_file` auf Nebius prueft SHA und Groesse bereits selbst; eine adapterunabhaengige Pruefung zusaetzlich schadet nicht.

Hinweise:
1. **`git ls-files -z`** benutzen und an NUL trennen. Ohne `-z` setzt Git Pfade mit Nicht-ASCII-Zeichen oder Sonderzeichen
   in Anfuehrungszeichen und Escapes (`core.quotePath`). Diese Pfade loesen sich dann nie auf. FITS hat keine, andere Repos koennen sie haben.
2. Die Anzahl der Eintraege zusaetzlich zu Bytes und SHA pruefen. Bei Abweichung **stoppen**, nicht still weitermachen.
3. **Unterordner-Pfad gleich behandeln:** `git ls-tree -r -l HEAD -- <subdir>` laeuft weiter ueber stdout. Zeilen mit Groesse
   und SHA sind laenger, die Kappung greift also frueher. Zusaetzlich wird die Groessenpruefung `MAX_SUBDIR_TOTAL_BYTES` nur ueber
   die abgeschnittenen Zeilen summiert. Der 200-MB-Schutz ist damit **fail-open**. Fix: auch `-z` ueber eine Datei, oder fail-closed bei Kappung.
4. **Gleiche Fehlerklasse ausserhalb des Listings:** `NebiusSandbox.run` liest `result.truncated` (SDK vorhanden) nie aus.
   - **Training** (`_run_and_eval`, kein `run_agent` auf Nebius): Bei stderr ueber 65535 Bytes (tqdm, Warnungen, Epochen-Logs)
     bleibt nur der Kopf. Dann fehlen die eigentliche Ausnahme, die `WorkerFailure`-Zusammenfassung des Brokers (steht am Ende)
     und der faulthandler-Stack beim Soft-Timeout. `triage` liest dann die letzten 1500 Zeichen aus der Mitte des Logs.
     Der Broker-Fix 33f41fc wirkt auf Nebius also nur unterhalb der Grenze. Lokal und im Docker-Test gibt es diese Grenze nicht.
   - `stdout_tail` stammt ebenfalls aus dem gekappten Kopf.
   - Smoke-/Eval-JSON sind klein und zeigen eine Kappung sichtbar als „unlesbar/ungueltig“ an, nur mit falscher Einordnung.
   - Vorschlag: `RunResult.truncated` aus `result.truncated` setzen. Laeufe, deren Ausgabe als Daten geparst wird, stoppen bei
     Kappung (fail-closed). Fuer das Training den Schwanz erhalten: stderr/stdout im Branch in Dateien umleiten und danach
     `tail -c` ausgeben, oder `truncate_output_at=_MAX_OUTPUT` setzen und die Kappung in `detail` nennen.
     Das ist nicht noetig fuer den Listing-Fix, betrifft aber dieselbe Transportgrenze.
5. **Zweite, unabhaengige Grenze:** Das Modell in `select_files` sieht nur `listing[:MAX_LISTING]` = 600 Zeilen. Bei FITS stehen
   `models/FITS.py` (Index 1381), `run_longExp_F.py` (1391) und `etth1_best.sh` (1395) **auch nach dem Fix** nie im Prompt (Probe:
   `llm_prompt_has_models_FITS: false`). Fuer Forecast rettet `forecast_selection` deterministisch. Andere Familien haben nur
   `model_file_in_listing`; Startskripte und Configs nach Zeile 600 sieht dort niemand. Vorschlag: vor der Kappung Quelltext-/Script-/Config-Pfade
   nach vorne sortieren und `logs/`, Daten und Binaerdateien ausschliessen. Kein Blocker fuer FITS.
6. **Regressionstest:** Kappung **generisch im FakeSandbox** fuer jeden `run` emulieren (Kopf 65535 plus `truncated`-Flag), nicht nur
   fuer den Listing-Befehl. Dann scheitern neue stdout-Parser auch im Test. Echte FITS-Liste mit 1410 Eintraegen und Original-Offsets
   als Fixture statt Toy-Liste. Pruefen: `models/FITS.py`, `etth1_best.sh`, `run_longExp_F.py` gewaehlt, `AD/` ausgeschlossen,
   keine Teilzeile in `listing`.

## Ergebnis

Ursache unabhaengig bestaetigt. Mit vollem Listing ist die Dateiwahl korrekt; Zeichenbudget und Modellkosten bleiben im Rahmen.
Fuer den Fix-Diff pruefe ich: `-z`, fail-closed bei Kappung oder Abweichung, Unterordner-Pfad und die Regression mit echter Liste.
Nicht Teil dieses Fixes, aber derselbe Fehler: Bei Training-stderr ueber 64 KiB geht auf Nebius die primaere Ausnahme verloren.

## Nachtrag 1: Delta-Pruefung `verified_listing` und Helper-Vertrag

Basis: `ceo` `75852b8` plus Arbeitsbaum (8 Dateien, +77/-15, `src` gegen Scratch byte-gleich). Keine Calls, keine Downloads, `ceo` unveraendert.
Probe `docs/audits/opus_repo_listing_delta_probe_20261013.py`, Ausgabe `…out.json`. Der Fake fuehrt die **echten Shell-Befehle**
gegen einen lokalen Git-Baum aus und kappt **jedes** `run`-stdout bei 65535 Bytes.

**Kein Blocker.**

| Fall | Ergebnis |
| --- | --- |
| FITS-Wurzel, echter Klon `d040bb0` | 1410 Eintraege, 83540 Bytes, keine Teilzeile `lo`; zuerst `models/FITS.py`, `scripts/FITS/etth1_best.sh`, `run_longExp_F.py`; kein `AD/` gelesen |
| Unterordner `logs` (163833 Bytes per `ls-tree -l -z`) | 1340 von 1340, also auch die Groessensumme vollstaendig (vorher fail-open) |
| Namen mit Leerzeichen, Umlaut, `"`, Zeilenumbruch, Tab | alle exakt; das alte `git ls-files` lieferte vier davon in Anfuehrungszeichen und mit Escapes |
| Liste ueber 1 MB (5200 tiefe Pfade) | Stopp vor dem Transfer, keine Modellauswahl |
| Fokussierte Tests (Transport, Helper, Repo, Setting-Handoff, Spec-Guards, `io_helper`, Auswahlvertrag) | 164 passed, 3 skipped (Docker) |

Weitere Pruefungen:
- `SourceContextBlocked` aus `read_repo` wird nicht als `SandboxSetupError` gefangen. Es gibt also keinen stillen Rueckfall
  auf „Nachbau aus Paper“.
- Der Helper-Vertrag passt zur echten API `forecast_data(length: int, stride: int = 1) -> dict`, Alias `nio` steht im Vertrag.
  Das Beispiel laeuft im Test gegen den echten Helper, mit und ohne Kalender.

Nicht blockierend:
1. Die Meldung fuer eine Liste ueber 1 MB lautet gleich wie bei Beschaedigung („could not be verified“). Besser die Groesse
   nennen, damit ein grosses Repo nicht wie ein Transportfehler aussieht.
2. Nicht-UTF-8-Pfadbytes fuehren zu `UnicodeDecodeError`, also zum Stopp. Das ist fail-closed und selten.
3. `forecast_data` lehnt `np.int64(360)` ab (`isinstance(..., int)` ist `False`). Die Meldung ist klar, kostet aber einen
   Reparaturversuch, wenn die Laenge aus numpy kommt. Vorschlag: `numbers.Integral`, aber nicht `bool`.
4. Weiter offen aus Abschnitt 3, nicht Teil dieses Deltas: Training-stderr ueber 64 KiB auf Nebius (Ausnahme und
   `WorkerFailure` gehen verloren); das Modell sieht in `select_files` nur `listing[:600]`.
