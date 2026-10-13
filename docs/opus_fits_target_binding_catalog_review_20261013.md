# Review: FITS-Zielbindung und Katalog-Gegenpruefung

Stand **2026-10-13, 15:38 UTC**. Gegenstand: uncommittete Aenderungen in `ceo` auf `e305b48`
(`claims_source.verify_selected_target`, `table_check._check_latex_datasets_horizons`, `latex_tabulars`,
`catalog._input_length_quote`, Verdrahtung in `loop.resolve_targets`, Tests `test_selected_target_table.py`).
Diff-Hash des geprueften Stands `5a7ef902…`. Geprueft auf Scratch-Kopien; `ceo` wurde nicht veraendert.
Keine Modell-, Tavily- oder Nebius-Aufrufe. Gelesen wurden nur freie arXiv-Quellen (E-Print, API) und vorhandene
lokale Quellen (`/tmp/nullius-catalog-audit/`, Laufordner in anderen Worktrees).

## Kurzurteil

**Kein Blocker fuer die lokale Integration.** Der Fix behebt die FITS-Ursache: Der Katalogwert 0.372 wird an genau
`tab:FITSett` gebunden; der Anhangwert 0.368 (`tab:app_ett`) ist ein eigener, getrennt pruefbarer Claim und ersetzt
den Katalogwert nicht mehr. Der Pfad bleibt fail-closed: Mehrdeutige Zitate und Abweichungen stoppen vor dem Experiment.

Vor weiteren Produktionslaeufen auf dem Katalog sollten zwei Katalogeintraege korrigiert oder gesperrt werden
(xCPD, Acacia, siehe Tabelle). Die Befunde B1 bis B6 sind Verbesserungen, keine Integrationsblocker.

## Ausgefuehrt

| Pruefung | Ergebnis |
| --- | --- |
| Ziel-, Tabellen-, Discovery- und Katalogtests auf Kopie des neuesten Stands (9 Dateien) | 232 passed |
| FITS gegen echte v3-Quelle (SHA-256 `9022b12b628f…`, identisch mit Produktion) | 0.372 -> `tab:FITSett`, 0.368 -> `tab:app_ett`, beide `column_check=ok` |
| Alle 17 Katalogeintraege durch `build_paper` und den neuen Binder | siehe Tabelle; `docs/audits/opus_catalog_binding_probe_20261013.{py,json}` |
| Randfaelle des Binders auf den echten FITS-Tabellen | `docs/audits/opus_binder_edge_probe_20261013.{py,json}` |

Hinweis zu den Tests: `tests/test_discovery_target_handoff.py` scheitert, wenn in der Shell `TAVILY_MODE=fixture`
gesetzt ist („Discovery braucht live“). Das ist kein Fehler des Diffs, aber die Tests sollten die Variable selbst setzen.

## Was die FITS-Quelle zeigt

- `tab:FITSett` (Haupttabelle): FITS ETTh1/96 MSE **0.372**. Die ETTh1-Ablation `tab:abl_h1` enthaelt genau
  **0.372101 bei Look-back 360**, COF 6, nur Forecast-Supervision. Der beste Wert bei Look-back 720 ist 0.379216.
  Der Anhangtext sagt: „ETTh1 … does not benefit from the longer look-back window, i.e. 720. Instead, it achieves the
  sota performance at look-back window of 360.“ Das Autorenskript `etth1_best.sh` nutzt ebenfalls 360
  (siehe `ceo/docs/fits_independent_verification_20261013.md`).
- **Der Katalogwert `input_length: 720` fuer FITS ist falsch.** Codex' Aenderung erzwingt ihn nicht mehr; das ist richtig.
  Der Wert bleibt aber in `catalog.yaml` und im Site-Export (`site_export.clean_claim`) stehen. Empfehlung: korrigieren
  oder als ungepruefter Hinweis kennzeichnen.
- `tab:app_ett` (Anhang, 0.368): Gleiches Layout, eigene Baselines (Autoformer, Informer, FiLM …). Das Setting
  dieser Tabelle ist im Paper nicht benannt. Der getrennte Claim ist zulaessig, sein Protokoll bleibt aber ungeklaert.

## Befunde zum Binder (nach Schwere)

**B1 Quellen-Drift faellt still auf den alten Fehlerpfad zurueck.** Der Katalog speichert keinen Hash, keine
Version und kein Tabellenlabel der Discovery-Quelle. `fetch_paper_text` laedt die E-Print-Datei ohne Version.
`latest_arxiv_version` lieferte bei meinen Abrufen fuer 15 von 16 Papers `None` (API-Zeitueberschreitung).
Verschwindet das Discovery-Zitat in einer neuen Fassung, gibt `verify_selected_target` `None` zurueck und
`resolve_targets` faellt auf die generische LLM-Extraktion zurueck. Das ist genau der Pfad, der bei FITS den
Anhang waehlte (Probe `drift_main_row_changed`: Rueckfall statt Stopp). Das Ergebnis ist nur fail-closed, solange
der LLM-Wert vom Katalogwert abweicht.
Empfehlung: Fuer `require_verified_target` mit LaTeX-Zitat stoppen, wenn das Zitat fehlt („Quelle seit Discovery
geaendert“). Discovery-Hash, Version und Tabellenlabel im Katalog speichern. Die E-Print-URL mit Version abrufen.

**B2 Quellenart haengt vom Netz ab.** `_fetch_arxiv_source` nutzt ein Zeitlimit von 10 s und verschluckt jeden
Fehler. Heute lieferten GLinear (3.1 MB), RLinear (3.9 MB) und GNN-Diff (2.5 MB) leeren Text. Die E-Prints existieren
(HTTP 200, gzip), ein Download brach mit einem HTTP/2-Fehler ab. GLinear wurde am 2026-10-10 noch als LaTeX gelesen.
Folge: Derselbe Claim laeuft je nach Netz ueber LaTeX und Binder oder ueber Tavily (Credits) und LLM.
Empfehlung: Retry mit laengerem Zeitlimit. Fuer strikte Katalogclaims mit LaTeX-Zitat keinen stillen Wechsel der
Quellenart; dann stoppen und den Grund nennen.

**B3 Eindeutigkeit zaehlt nur `table`/`table*`.** Kopien des Zitats in `wraptable`, `sidewaystable`, `figure`,
`longtable` oder `tabular*`/`tabularx` werden nicht gezaehlt. Probe `copy_in_wraptable_is_ignored`: Eine zweite
FITS-Zeile mit Caption „Look-back 720“ in einer `wraptable` wird ignoriert, der Binder meldet `tab:FITSett`
ohne Hinweis. Verschachtelte Kopfzellen zaehlt der neueste Stand korrekt (`latex_tabulars`).
Empfehlung: Vorkommen des normalisierten Zitats im ganzen Quelltext zaehlen. Mehr als ein Vorkommen bedeutet ein
laengeres Zitat oder einen Stopp.

**B4 Die Zahl ist gebunden, das Setting nicht.** Der Binder belegt Methode, Datensatz, Horizont, Metrik und Zelle.
Look-back und Hyperparameter der Tabelle belegt er nicht. FITS zeigt den Fall: Beide Tabellen sind echte
FITS-ETTh1/96-Werte unter verschiedenen Settings.
Empfehlung: Tabellenlabel in die Evidenz uebernehmen (schon vorhanden) und die Setup-Suche fuer Eingabelaenge
an Caption und Abschnitt dieser Tabelle koppeln. Ohne Bindung bleibt die Eingabelaenge frei.

**B5 Fail-closed-Stopps bei echten Tabellen.** Diese Faelle stoppen statt zurueckzufallen: `$\pm$` in der Ergebniszelle
(„keine eindeutige Zahl“), Caption mit „MSE and MAE“, Caption ohne Metrikwort. Das ist sicher, aber jede dieser
Formen kostet einen Lauf. Empfehlung: Streuungsanteil (`\pm …`) vor der Zahlensuche entfernen; Metrik aus einer
Metrikzeile akzeptieren, wenn die Caption beide nennt.

**B6 Eingabelaengen-Pruefung zu eng (Regression).** `_input_length_quote` erkennt nur „input length / look-back
[of|is|=|:] N“ direkt hintereinander und prueft nur `input_length_quote` oder `quote`, nicht `protocol_quote`.
Damit verlieren vier Eintraege ihre Eingabelaenge, obwohl das Paper sie woertlich nennt:

| Eintrag | Paper (frische Quelle) | Grund fuer Verlust |
| --- | --- | --- |
| GLinear 336 | „Input Sequence Length for all experiments is 336“ | Woerter zwischen „Length“ und „is“ |
| TSMixer 512 | „input length $L = 512$“ | nur in `protocol_quote`; `L =` und `$` |
| SparseTSF 720 | „input length of 720“ | nicht im Katalogzitat |
| SegRNN 720 | „look-back of 720“ | nicht im Katalogzitat |

Das ist sicher (Warnung, der Agent waehlt frei bis 720), aendert aber die Laufbedingung der bereits reproduzierten
Eintraege TSMixer, SparseTSF und GLinear. Empfehlung: `protocol_quote` mitpruefen, Woerter innerhalb eines Satzes
erlauben, `L = N` akzeptieren. Den Vergleich mit der frischen Quelle auf LaTeX-bereinigtem Text fuehren, sonst passt
ein HTML/PDF-Zitat aus der Discovery nie zur LaTeX-Quelle.

## Katalog: 17 Eintraege

Quelle heute: frisch gelesene LaTeX-Fassung, soweit verfuegbar. „Rueckfall“ heisst: Der Binder greift nicht,
die generische Extraktion (LLM) entscheidet, danach greift die bisherige Abweichungspruefung.

| Eintrag | Quelle heute | Binder | Protokoll-/Eingabelaengenbefund | Einstufung |
| --- | --- | --- | --- | --- |
| Kipf GCN Cora 0.815 | LaTeX | Rueckfall (Zitat aus PDF-Text) | Planetoid, passt | ok (Referenz) |
| Kipf GCN CiteSeer 0.703 | LaTeX | Rueckfall | `matched_task` leer, Task CiteSeer | ok (Referenz) |
| Shchur GCN 0.815 | nur PDF | Rueckfall | LCC, 20/30/Rest, passt | ok (Referenz) |
| HaKAN 0.369 | LaTeX | Rueckfall | Paper: „best-performing look-back reported“ aus {96…720}, Anhang L=336; kein Repo | Setup-Grenze, bleibt „nicht reproduzierbar (Setup)“ |
| MoHETS 0.350 | LaTeX | Rueckfall | 0.350 fuer ETTh1/96 in zwei Tabellen, beide gleich; Look-back nicht belegt | ok, Eingabelaenge frei |
| **Acacia 58.17** | LaTeX | Rueckfall | „graph foundation model, trained on the web graph“; Zeile „Acacia, frozen“ = vortrainiertes, eingefrorenes Modell. `requires_external_weights` greift nicht („does not rely on pretrained LLMs“) | **Konflikt: nicht auf CPU aus Cora trainierbar; sperren** |
| SL-GCN 82.92 | LaTeX | Rueckfall | Shchur-Splits, 100 Splits x Init, LCC | ok |
| SimMLP | - | - | NotRunnable (Task offen) | gesperrt |
| **xCPD 0.357** | LaTeX | Rueckfall | 0.357 steht in der Spalte **„TSMixer + xCPD“**; Caption: „lookback … 96 (512 for TSMixer)“. Katalog: Methode „xCPD“ ohne Backbone, `input_length: 96` | **Konflikt: Backbone fehlt, Eingabelaenge falsch (512); korrigieren oder sperren** |
| TSMixer 0.361 | LaTeX | Rueckfall (Zitat ist Kopfzeile) | L=512 belegt | ok; Eingabelaenge geht durch B6 verloren |
| ARIEL | - | - | NotRunnable (Task offen) | gesperrt |
| RLinear 0.366 | heute unlesbar (B2) | - | L=336, Wert 0.366 in RLinear-Spalte (PDF-Cache v2) | ok, Quelle instabil |
| GNN-Diff 72.37 | heute unlesbar (B2) | - | Methode erzeugt Parameter aus einer Sammlung trainierter Checkpoints (Vergleich Grid/Random/Coarse/C2F); Split nicht belegt | Protokoll unklar, nicht verifiziert |
| SparseTSF 0.359 | LaTeX | **verified** `main_reslut` | „input length of 720“ | ok; Eingabelaenge geht durch B6 verloren |
| **FITS 0.372** | LaTeX | **verified** `tab:FITSett` | Katalog L=720, Paper/Autoren **L=360** | Ziel ok; Katalogwert L korrigieren |
| SegRNN 0.341 | LaTeX | **verified** `main_result` | „look-back of 720“ | ok; Eingabelaenge geht durch B6 verloren |
| GLinear 0.382 | heute unlesbar (B2), am 10.10. LaTeX | - | L=336 im Katalogzitat | ok; Eingabelaenge geht durch B6 verloren |

Die fuenf oeffentlichen Live-Kandidaten aus Codex' Audit habe ich nicht unabhaengig nachgerechnet.
Codex' Ergebnis „SARAF 0.366 blockiert, 0.371 verifiziert“ passt zum bekannten RAFT-Spaltenfehler.

## Minimale Nacharbeit (Vorschlag, keine Fixes durch mich)

1. Katalog: xCPD und Acacia sperren oder korrigieren; FITS `input_length` auf 360 setzen oder als Hinweis kennzeichnen.
2. B1: Fehlendes LaTeX-Zitat bei strikten Katalogclaims stoppt statt LLM-Rueckfall; Discovery-Hash/Version/Label speichern.
3. B6: `protocol_quote` und Satzluecken in `_input_length_quote`; Vergleich auf bereinigtem LaTeX.
4. B2/B3/B5 nach Bedarf.
