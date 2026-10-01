# B1: Kandidaten für mehr Benchmark-Einheiten (Recherche, nur Doku)

Stand 2026-10-02, Branch `b-research`. Keine echten API-Aufrufe (kein Tavily, keine Token Factory, keine Sandbox). Quellen: arXiv-HTML per `curl` geholt und zu Text gewandelt, Zitate sind Auszüge aus diesem Text (Tabellenzellen stehen dort zeilenweise, hier mit `|` verbunden). **Vor Aufnahme in `benchmark/papers.yaml` muss Rolle 2 den Wert wie üblich aus dem Paper-Text belegen** (Fixture anlegen, Spaltenprüfung); diese Datei ersetzt das nicht. Zielwerte stammen nur aus den zitierten Stellen, nicht aus dem Gedächtnis.

Unsere Tasks (Stand `tasks.py`): **Cora** fester Planetoid-Split (140/500/1000), **Cora-LCC** zufällige Splits 20/30/Rest (30 Splits × 1 statt 100 × 20), **ETTh1** Horizont 96, Split 12/4/4 Monate (8640/2880/2880 Zeilen), z-score mit Train-Statistik, MSE über alle 7 Kanäle und **alle 2785 Testfenster**, Eingabelänge frei bis 720. Toleranz wie bisher ±0,01.

## Entscheidende Erkenntnis vorab: der `drop_last`-Fehler bei ETTh1

Mehrere ETTh1-Zahlen aus der Literatur sind mit einem Fehler im Test-Dataloader entstanden (`drop_last=True`, der letzte unvollständige Batch fällt weg). Unser Harness wertet alle 2785 Fenster aus. Das gehört zu jeder ETTh1-Einheit (Details und Belege in `b2_negative_case.md`). Ein Zielwert aus **vor** der Korrektur veröffentlichten Tabellen ist für uns nicht fair vergleichbar:

- FITS-README, Abschnitt „Result Update" (nach Korrektur neu gerechnet), ETTh1-96, MSE: `PatchTST | 0.385`, `Dlinear | 0.384`, `FedFormer | *0.375*`, `TimesNet | 0.384`, `FITS | **0.372**`.
- Dieselben Modelle in den Originalpapers: PatchTST/64 **0,370**, PatchTST/42 und DLinear **0,375** (PatchTST-Paper Tabelle 3, v2).
- Unsere eigene DLinear-Referenz im Harness: 0,3809 (Benchmark v2), passt zu den korrigierten 0,384, nicht zu den veröffentlichten 0,375.

Folge für B1: ETTh1-Kandidaten bevorzugen, deren **Zielwert nach der Korrektur** gerechnet wurde (FITS ICLR-Endfassung, SparseTSF Tabelle 11, TFB), oder Modelle, die nicht betroffen sind (TimesNet, iTransformer testen laut FITS-README mit `batch_size=1`).

## Kandidatentabelle

Zahlen = Zielwert im Paper, MSE für ETTh1, Accuracy für Cora (in %, wir rechnen 0,xxx). „Aufwand R" = Referenzlösung in `tests/reference/` (Personenstunden, Schätzung). „Erwartung" = unbelegte Vermutung, **das Referenzurteil entsteht erst aus dem Referenzlauf** (Regel v2).

| # | Einheit (Task) | Zielwert + Zitat | Protokoll passt? | Repo (offiziell) | CPU < 10 min? | Aufwand R | Erwartung | Empfehlung |
|---|---|---|---|---|---|---|---|---|
| 1 | **FITS** (ETTh1 96) | MSE **0,372**. FITS-Paper v3 Tabelle 1 „Long-term forecasting results on ETT dataset in MSE": Zeile `FITS 0.372 0.404 0.427 0.424 0.271 ...` (Spalte ETTh1-96); gleiche Zahl in der README nach der Korrektur. | Ja: Split 12/4/4 (SparseTSF: „6:2:2 ... following FITS and Autoformer"), z-score. Look-back per Grid {90,180,360,720} gewählt auf Validierung („We conduct grid search on the look-back window of 90, 180, 360, 720 and cutoff frequency"). Unser Task erlaubt L ≤ 720. | github.com/VEWOXIC/FITS | Ja, ca. 10 000 Parameter, komplexes lineares Modell im Frequenzraum | 3 bis 4 h (RevIN, Tiefpass, komplexe Interpolation) | reproduziert oder teilweise (SparseTSF rechnet FITS nach der Korrektur mit L=720, COF=5 neu: 0,382) | **Ja** |
| 2 | **SparseTSF** (ETTh1 96) | MSE **0,362** nach Korrektur. SparseTSF v2 Tabelle 11 „... after fixing code bug": `SparseTSF (ours) 0.362 0.403 0.434 0.426 0.294 ...`. Tabelle 2 (vor Korrektur) nennt 0,359. | Ja: „The dataset splitting follows the procedures of FITS and Autoformer ... 6:2:2", L=720, Adam, 30 Epochen, Patience 5. | github.com/lss-1138/SparseTSF | Ja, rund 1 000 Parameter | 2 bis 3 h (Periodenzerlegung w=24, ein Linear, RevIN) | reproduziert | **Ja** (kleinster Aufwand, korrigierter Wert) |
| 3 | **RLinear** (Li et al., ETTh1 96) | MSE **0,366**. Tabelle 2 „Time series forecasting results. The length of the historical horizon is 336 ...": `ETTh1 | 96 | 0.366 | 0.391 | 0.390 | 0.410 | 0.381 | 0.405 | 0.398 | 0.418 | 0.375 | 0.399` (RLinear MSE, MAE, RMLP, PatchTST, TimesNet, DLinear). | Ja (L=336). **Unklar**, ob vor oder nach der `drop_last`-Korrektur gerechnet (v2 des Papers, Mai 2023, vor dem Fund Dez. 2023): Verdacht „vor" → hoher Wert zu optimistisch. | github.com/plumprc/RTSF | Ja (ein Linear + RevIN) | 1 h (wir haben `nlinear_reference.py`) | teilweise oder nicht reproduziert (wenn betroffen, Abstand bis ca. 0,01 bis 0,02) | **Ja, als „Bug-Fall"** (siehe B2) |
| 4 | **PatchTST** (ETTh1 96) | MSE **0,370** (PatchTST/64, L=512) bzw. **0,375** (PatchTST/42, L=336). Tabelle 3: `ETTh1 | 96 | 0.370 | 0.400 | 0.375 | 0.399 | 0.375 | 0.399 | 0.376 | 0.415 | 0.435 | 0.446 ...` (PatchTST/64, PatchTST/42, DLinear, FEDformer, Autoformer ...). | Ja bis auf `drop_last`: Patchlänge 16, Stride 8, RevIN; „PatchTST/64 ... look-back window L=512". Zielwert aus Zeit vor Korrektur. | github.com/yuqinie98/PatchTST | Ja (3 Schichten, d_model klein, kanalunabhängig; ca. 2 bis 5 min/Lauf geschätzt, nicht gemessen) | 5 bis 6 h (Patching, Transformer-Encoder, RevIN) | teilweise (korrigierter Wert 0,385, siehe B2) | **Ja, als B2-Kandidat**; als normale Einheit nur, wenn B2 verworfen wird |
| 5 | **iTransformer** (ETTh1 96) | MSE **0,386**. Tabelle 1 (Detail): `ETTh1 | 96 | 0.386 | 0.405 | 0.386 | 0.395 | 0.414 | 0.419 | ...` (iTransformer, RLinear, PatchTST ...). Look-back fest 96. | Ja: „we fix the length of the lookback series as 96 in ETT ..." chronologischer Split. FITS-README: testet mit `batch_size=1`, daher vom Fehler nicht betroffen. | github.com/thuml/iTransformer | Ja (2 Schichten, Aufmerksamkeit über 7 Variablen, kleine Matrizen) | 4 bis 5 h | reproduziert | Ja (optional) |
| 6 | **TimesNet** (ETTh1 96) | MSE **0,384**. Tabelle 2 Detail: `ETTh1 | 96 | 0.384 | 0.402 | 0.494 | 0.479 | ...`. Eingabelänge 96. | Ja, nicht vom Fehler betroffen (`batch_size=1`). | github.com/thuml/TimesNet | **Unsicher**: 2D-Faltungen über Perioden, mehrere Schichten, auf CPU eher 10 bis 20 min | 8 h+ (FFT-Perioden, Inception-Block) | offen | Nein (Aufwand, Laufzeit) |
| 7 | **FEDformer** (ETTh1 96) | MSE 0,376 (FEDformer-f). Tabelle 2: `ETTh1 | 96 | 0.376 | 0.419 | 0.395 | 0.424 | 0.449 | 0.459 ...`; Eingabe 96. | Ja, aber vom `drop_last`-Fehler betroffen (FITS-README: gelistet); Re-Run: 0,375. | github.com/MAZiqing/FEDformer | Nein/unsicher (Fourier- und Wavelet-Blöcke, Zerlegung) | 12 h+ | offen | Nein |
| 8 | **TiDE** (ETTh1 96) | MSE 0,375. Tabelle 2: `ETTh1 | 96 | 0.375 | 0.398 | 0.379 | 0.401 ...`. | **Nein**: nutzt Zeitkovariaten („time-covariates"), unsere Daten haben nur die 7 Werte, kein Datum. | (GluonTS-Verweis, kein eigenes Repo im Text) | ja | 4 h | – | **Verwerfen** (Protokoll) |
| 9 | **GCNII** (Cora) | Accuracy **85,5 ± 0,5** (64 Schichten). Tabelle 2 „Summary of classification accuracy (%) results on Cora, Citeseer, and Pubmed": `GCNII | 85.5 ± 0.5 (64) | 73.4 ± 0.6 (32) | 80.2 ± 0.4 (16)`. | Ja: „standard fixed training/validation/testing split (Yang et al., 2016) ... with 20 nodes per class for training, 500 nodes for validation and 1,000 nodes for testing"; lr 0,01, Patience 100, 100 Läufe. arXiv v1. | github.com/chennnM/GCNII | Wahrscheinlich ja (64 dünne Schichten, 2708 Knoten, dünn besetzte Matrix), nicht gemessen | 4 h (anfängliche Residuen + Identity-Mapping, 64 Schichten; Initialisierung und Gewichtszerfall je Schicht gruppiert) | offen, **hoher Zielwert 85,5**; GCN-Familien landen oft bei 83 bis 84 | **Ja** (spannend, feine Details, Repo vorhanden → A2 greift) |
| 10 | **ARMA** (Cora) | Accuracy **83,4 ± 0,6**. Tabelle 1 „Node classification accuracy": `ARMA | 83.4 ± 0.6 | 72.5 ± 0.4 | 78.9 ± 0.3 | 90.5 ± 0.3` (Cora, Citeseer, Pubmed, PPI). v7. | Teilweise belegt: „The models are trained using 20 labels per document class"; Validierungs-/Testgröße im Text nicht gefunden (Planetoid vermutet). Hyperparameter laut Anhang: K=2, T=1, Dropout 0,75, L2 5e-4, lr 0,01, 2000 Epochen, Early Stopping 50. | Im Paper-Text **kein** Repo-Link zum Modell gefunden (Spektral-Bibliothek, nicht geprüft) | Ja (2 Schichten) | 3 h | offen | Ja, nach Prüfung des Splits |
| 11 | **MixHop** (Cora) | Accuracy **81,9 ± 0,40**. Tabelle 1: `MixHop: learned architecture (ours) | 81.9 ± 0.40 | ...`; Tabelle 1 nennt auch `Vanilla GCN (Kipf & Welling, 2017) | 81.5`. v3. | **Nein im Kern**: „we run all models 100 times, sort by the validation accuracy, and finally report the test accuracy for the top 50 runs". Das ist ein anderes Auswahlprotokoll als unser Mittelwert über Seeds; der Wert ist systematisch optimistisch. | github.com/samihaija/mixhop | Ja | 3 h | teilweise (Vermutung) | Verwerfen als normale Einheit; als Protokoll-Beispiel in B2 erwähnt |
| 12 | **APPNP** (Cora) | **Verwerfen.** Eigene Tabelle 2 nutzt Cora-ML, nicht Planetoid: `APPNP | Citeseer: 75.73 ... | Cora-ML: 85.09 ...`, eigener Split („1500 nodes ... visible set", Early-Stopping-Set 500 Knoten). Der Wert 83,3 auf Planetoid steht nur in GCNII Tabelle 2 (`APPNP | 83.3 | 71.8 | 80.1`), dort Fremdmessung. | Nein | github.com/klicpera/ppnp | – | – | – | Nein |
| 13 | **JKNet** (Cora) | Nicht geprüft: arXiv-HTML lieferte nur eine Stub-Seite (1,7 KB). In GCNII Tabelle 2 als Fremdmessung `JKNet | 81.1 (4)`. | unbekannt | – | – | – | – | Nein (nicht belegbar) |
| 14 | **GraphSAGE** (Cora, Original) | **Verwerfen.** Das Originalpaper evaluiert nicht auf Planetoid-Cora. | – | – | – | – | – | Nein |
| 15 | **ChebNet** (Cora, Original) | **Verwerfen.** Original nutzt MNIST und 20NEWS; ein Cora-Wert (79,5 ± 1,2) steht nur in der ARMA-Tabelle 1 als Fremdmessung. | – | – | – | – | – | Nein |
| 16 | **MoNet** (Cora-LCC) | **81,3 ± 1,3**, Fremdmessung im Kritik-Paper: Shchur v2 Tabelle 1: `MoNet 81 .3 ± 1.3 71 .2 ± 2.0 78 .6 ± 2.3 59 .8 ± 0.8` (CORA, CiteSeer, PubMed, CORA Full), Fixture `1811.05868.json` liegt vor. | Ja (gleiches Protokoll wie unsere Shchur-Einheiten, 100 Splits × 20 Inits, wir 30 × 1). | Original: Monti et al., Code im Kritik-Paper-Repo | Ja | 5 h (Gauss-Kerne, Pseudokoordinaten) | offen | Nein (Aufwand, Modell eher unbekannt) |
| 17 | **GraphSAGE-mean** (Cora-LCC) | **79,2 ± 7,7** (Shchur Tabelle 1 `GS-mean 79 .2 ± 7.7 71 .6 ± 1.9 ...`). Hohe Streuung. | Ja | – | Ja | 3 h (Mittelwert-Aggregator, Vollbatch) | wegen std 7,7 und 30 Splits: Mittelwert kann stark schwanken, Urteil instabil | Nein/Risiko (std 7,7 erzeugt „keine Mehrheit") |
| 18 | **MLP** (Cora-LCC) | **58,2 ± 2,1** (Shchur Tabelle 1 `MLP 58 .2 ± 2.1 59 .1 ± 2.3 70 .0 ± 2.1 36 .8 ± 1.0`). | Ja | – | Ja | 1 h | reproduziert | Nein als Paper-Einheit (Baseline, kein Paper-Claim); als Kalibrier-Einheit denkbar |

## Empfehlung (Stand der Recherche, vor den Referenzläufen; Entscheidungen nach den Läufen: Abschnitt „Referenzläufe“ unten)

Reihenfolge nach Nutzen minus Aufwand. Alle brauchen vorab Rolle-2-Fixture und Referenzlauf.

1. **SparseTSF ETTh1-96, 0,362** (korrigierter Wert, billigste Referenz, hohe Chance „reproduziert").
2. **FITS ETTh1-96, 0,372** (korrigierter Wert; Repo vorhanden; nahe an der Toleranzgrenze, weil SparseTSF mit L=720 und COF=5 0,382 misst: Referenzlauf entscheidet).
3. **RLinear ETTh1-96, 0,366** (Verdacht „vor Korrektur", deshalb informativ; sehr geringer Aufwand durch `nlinear_reference.py`).
4. **PatchTST ETTh1-96** nur falls B2 den Fall nicht übernimmt (siehe dort). Sonst dort verwenden.
5. **GCNII Cora, 85,5** (Planetoid-Split belegt, Repo vorhanden, 64 Schichten: dichtes Material für A2 und Plan-Prüfung).
6. **ARMA Cora, 83,4** (Split-Aussage noch zu schließen: Validierungs-/Testgröße im Paper-Text prüfen, sonst verwerfen).

Zusatz (optional, falls 7. und 8. nötig): iTransformer ETTh1-96 (0,386, vom Fehler unbetroffen).

**Nicht empfohlen:** FEDformer, TimesNet (Aufwand/Laufzeit), TiDE (Kovariaten), MixHop (Top-50-Protokoll), APPNP, GraphSAGE-Original, ChebNet (kein Cora-Wert im Original), JKNet (nicht belegbar), MLP/GS-mean (Baseline bzw. zu hohe Streuung).

## Offene Punkte für den Reviewer

- Alle Zielwerte sind mit Zitat belegt, aber **nicht durch Rolle 2** (kein Tavily). Beim Anlegen der Einheit läuft die übliche Spaltenprüfung; das SparseTSF-Layout (Tabellen 2 und 11 nebeneinander) braucht besondere Vorsicht (Wert „vor" und „nach" der Korrektur).
- Laufzeiten sind Schätzungen, nicht gemessen.
- ARMA-Split und GCNII-Laufzeit prüfen, bevor die Einheit in die Vorab-Festlegung kommt.
- Ob RLinear und andere frühe Zahlen betroffen sind, steht nicht im Paper; nur der Referenzlauf klärt das.

## Referenzläufe (2026-10-02, lokal auf der CPU, Branch `b-references`)

Eigene Referenzlösungen im Harness-Vertrag (`tests/reference/`), gerechnet mit `scripts/run_reference_local.py` (lokal, 16 Threads, kein Sandbox-Aufruf, kein LLM; Daten und Metrik aus `src/repro_agent/harness/`). Hyperparameter nur aus dem Paper bzw. dem offiziellen Repo (Quelle im Kopf jeder Datei), **nicht auf den Zielwert getunt**; ein Lauf pro Seed, Seeds 0 bis 4. Toleranz ±0,01 wie bei DLinear.

| Einheit | Datei | Seeds | Mittel ± Std | Ziel | Abstand | Laufzeit je Seed | Referenzurteil | Ergebnisdatei |
|---|---|---|---|---|---|---|---|---|
| SparseTSF ETTh1-96 (MSE) | `sparsetsf_reference.py` | 5 | **0,3607 ± 0,0011** | 0,362 (Tabelle 11, nach Korrektur) | 0,0013 | 6 s (max 7 s) | **reproduziert** | `results/b1_reference_sparsetsf.txt` |
| RLinear ETTh1-96 (MSE) | `rlinear_reference.py` | 5 | **0,3704 ± 0,0008** | 0,366 (Tabelle 2) | 0,0044 | 7 s (max 7 s) | **reproduziert** | `results/b1_reference_rlinear.txt` |
| GCNII Cora (Accuracy) | `gcnii_cora_reference.py` | 1 (Seed 0, Lauf abgebrochen) | 0,8480 | 0,855 (Tabelle 2) | 0,0070 | **463 s** (869 Epochen, 16 Threads) | (nur Seed 0: reproduziert) | – |

Einzelwerte SparseTSF: 0,3590, 0,3616, 0,3605, 0,3606, 0,3618. RLinear: 0,3699, 0,3697, 0,3713, 0,3712, 0,3699. GCNII Seed 0: bester Validierungsloss 0,8131, Validierungsgenauigkeit 0,8340.

### Entscheidungen des Reviewers

- **SparseTSF und RLinear sind im Benchmark** (Ziel wie oben, Referenzurteil wie gemessen, Toleranz wie DLinear). Eintrag für die Vorab-Festlegung: `sparsetsf_2024` Ziel 0,362, Referenzmittel 0,3607, Seeds 5, Herkunft (R); `rlinear_2023` Ziel 0,366, Referenzmittel 0,3704, Seeds 5, Herkunft (R).
- **GCNII verworfen für den Benchmark:** 463 s je Seed bei 16 Threads (869 Epochen, 64 Schichten). Die Sandbox hat weniger Kerne, der Agenten-Timeout liegt bei 600 s (`TRAIN_TIMEOUT_S`), die Aufgabe nennt „< 5 minutes". Die 10-Seeds-Läufe wurden abgebrochen; es gibt nur den Seed-0-Wert und die Laufzeit. Der Seed-0-Wert (0,848 gegen 0,855) zeigt, dass die Referenz das Paper trifft, aber ein Agent würde fast sicher in den Timeout laufen.
- **ARMA verworfen:** Der Paper-Text (v7) nennt für die Zitationsnetze nur „The models are trained using 20 labels per document class"; Größe von Validierungs- und Testset und der Name des Splits stehen nicht im Text. Die Daten stammen aus den Planetoid-Dateien von Kipf (`github.com/tkipf/gcn`). Das Spektral-Beispiel (`examples/node_prediction/citation_arma.py`) nutzt die Standardmasken, aber andere Hyperparameter als Tabelle 6 und Abschnitt 7.1 des Papers (Beispiel: L2 5e-5, 20 000 Epochen, Geduld 100; Paper: L2 5e-4, 2000 Epochen, Early Stopping nach 50 Epochen). Der Zielwert 83,4 ist so nicht eindeutig unserem Task zuzuordnen.

### Beleg für die Ursache der Abweichung bei RLinear (und SparseTSF): `drop_last` im Test-Loader

Der Fehler ist im Code der Autoren sichtbar. Die Referenzen wurden zusätzlich so ausgewertet, als würde der Test-Loader den letzten unvollständigen Batch verwerfen (`scripts/run_reference_local.py`, Funktion `run_one`: erste `(2785 // 128) * 128 = 2688` Fenster, also 97 von 2785 Fenstern weniger; Batchgröße 128 wie in den Skripten der Autoren). Das ist eine **Nachbildung**, kein Lauf des Autorencodes.

| | Code der Autoren | Zeile | Paperwert | Unsere Referenz (alle 2785 Fenster) | Nachbildung `drop_last`, Batch 128 |
|---|---|---|---|---|---|
| RLinear | `github.com/plumprc/RTSF`, Stand `0fec00104f` (2023-09-13), `data_provider/data_factory.py:16-18`: `if flag == 'test':` ... `drop_last = True`; Skript `script.md:3` (ETTh1: `--batch_size 128 ... --seq_len 336 --pred_len 96`, ohne `--drop`, also Dropout 0,1) | `data_factory.py:18` | **0,366** | 0,3704 | **0,3665** (Urteil: reproduziert; Abstand 0,0005 zum Paperwert) |
| SparseTSF | `github.com/lss-1138/SparseTSF`, Stand `b8c2740eec` (2026-03-04), `data_provider/data_factory.py:18-20`: `if flag == 'test':` ... `drop_last = False  # fix bug` (Fehler im Repo korrigiert); Skript `scripts/SparseTSF/linear/etth1.sh` (Batch 256, lr 0,02, L=720) | `data_factory.py:20` | 0,359 (Tabelle 2, vor Korrektur) bzw. 0,362 (Tabelle 11, nach Korrektur) | 0,3607 | 0,3587 (Nachbildung bei Batch 128; das Repo nutzt Batch 256, also fiele mehr weg: 2785 mod 256 = 225 Fenster) |

Lesart: Unsere ehrliche Auswertung liegt bei RLinear 0,0044 über dem Paperwert, die Nachbildung des Fehlers trifft ihn auf 0,0005. Das stützt die Aussage, dass der Paperwert unter dem `drop_last`-Fehler entstand. Es **beweist** sie nicht (Batchgröße und Hyperparameter der Autoren sind nicht bitgleich nachgebildet, ein Lauf des Autorencodes fand nicht statt). Beide Abstände liegen innerhalb der Toleranz 0,01, das Urteil bleibt in beiden Fällen „reproduziert". Bei SparseTSF sind beide Paperwerte (vor/nach Korrektur) für uns „reproduziert".

### Offen

- GCNII bleibt als Referenzlösung im Repo (`tests/reference/gcnii_cora_reference.py`), aber nicht im Benchmark; ein Lauf mit weniger Schichten oder Epochen wäre eine andere Einheit (nicht gewollt).
- Die Referenzläufe laufen lokal, nicht in der Nebius-Sandbox; Laufzeiten dort können abweichen (RLinear und SparseTSF mit 6 bis 7 s weit unter dem Timeout).
