# Ursachenanalyse Benchmark v4 (Auftrag 24)

Stand 2026-10-02. Grundlage: `runs/benchmark_v4.jsonl` (96 Urteile, 48 Läufe), je Lauf `dossier.json`, `events.jsonl` und der gewählte Agenten-Code; `results/benchmark_v4.md`, `results/failure_log_v4.md`. Reine Auswertung: kein LLM-, Tavily- oder Sandbox-Aufruf, keine Änderung an `src/`, `scripts/`, `web/`, `benchmark/papers.yaml`. Alles hier ist **post hoc**; Regeln, Toleranzen und Referenzurteile von v4 bleiben unverändert.

**Kurzfassung.** Die Abweichungen haben keine gemeinsame Ursache, sondern sechs wiederkehrende, die sich je Einheit unterscheiden. H1 in der gestellten Form ("Setup-Details kommen nicht im Code an") ist nicht bestätigt: Von 106 Angaben, die im Dossier stehen, übernimmt der Code 89 (84 %). Die Lücke liegt davor: Die entscheidende Angabe fehlt im Dossier (40 von 146 anwendbaren Zellen, vor allem Feature-Normalisierung und Lernraten-Plan) oder sie steht dort im falschen Zusammenhang (README-Ratschlag, `argparse`-Default, induktive statt transduktive Einstellung) und wird dann treu übernommen. In Arm A ersetzt das Modell fehlende Angaben durch Standardwerte eines anderen Papers (Kipf-Werte für Shchur, Veličković-Werte für Shchur-GAT) oder baut die falsche Methode (NLinear, DGI). 24 lokale Gegenproben mit den Referenzlösungen sind fertig, 2 offen (Abschnitt 3); 14 davon sind als "belegt" gewertet (Richtung und Größe passen zum Agentenwert des zugehörigen Laufs, Einzelwerte in der Tabelle), eine widerlegt eine Vermutung (L2-Form). H4 (Auswahl über Validierung schadet systematisch) ist nicht bestätigt, hat aber eine Nebenwirkung: Die Auswahl bevorzugt Code, der auf den Validierungslabels trainiert.

## 1. Je Einheit und Arm: Abstand, Hauptursache, Beleg

Median-Abstand = Median der drei Generierungen (Mittel über Seeds minus Zielwert; MSE: plus = schlechter). "Dossier [n]" = Eintrag n in `setup_details` des genannten Laufs. Pfade relativ zu `runs/`.

| Einheit | Arm | Mittel Gen 0/1/2 | Median-Abstand | Hauptursache | Beleg |
|---|---|---|---|---|---|
| Kipf GCN | A | 0,7365 / 0,7476 / 0,7987 | −0,067 | Gen 0, 1: Early Stopping mit Geduld 10 auf Validierungs-**Accuracy** bricht in der flachen Anfangsphase ab (Seed-Std 0,093 und 0,057, einzelne Seeds 0,56 bis 0,62). Gen 2: Merkmale nicht zeilennormalisiert. Hyperparameter stimmen in allen drei (aus dem Gedächtnis). | `kipf_gcn_2017/20261001-233305/gcn_cora_acc_3e7f8616831d.py:97-103,169`; `.../20261001-235153/gcn_cora_acc_df194304877f.py:111,139-145`; `.../20261002-001630/gcn_cora_acc_cfdda2760338.py:127-136` (keine Normalisierung der Merkmale, nur `normalize_adj`). Gegenproben G1, G2. |
| Kipf GCN | B | 0,8194 / 0,8170 / 0,8178 | +0,003 | keine (reproduziert). Angaben übernommen; Gen 1 nutzt 400 Epochen ohne Early Stopping (Dossier [23], [24]). | `.../20261001-235758/gcn_cora_acc_b22d60c17bab.py:61,177-182` |
| Wu SGC | A | 0,8215 / 0,8530 / 0,7851 | +0,012 | Drei verschiedene Protokolle. Gen 0: keine Merkmalsnormalisierung, LBFGS, λ-Gitter (knapp zu hoch). Gen 1: nach der λ-Wahl **Neutraining auf Train + Validierung** (640 statt 140 Labels, Std 0,0). Gen 2: Adam lr 0,01, kein Weight Decay, 2000 Epochen. | `wu_sgc_2019/20261002-003527/sgc_cora_acc_6fe7c5d170d4.py:78,127-128,162`; `.../20261002-010019/sgc_cora_acc_9b135be6dff8.py:97-98`; `.../20261002-011230/sgc_cora_acc_3d0d74cead37.py:117,123-124`. Gegenproben S5, S6. |
| Wu SGC | B | 0,7489 / 0,7970 / 0,7336 | −0,061 | **Falsche Merkmalsnormalisierung aus dem README.** Gen 0 und 2 standardisieren spaltenweise (Mittel 0, Std 1), Gen 1 normalisiert zeilenweise mit L2 statt L1. Quelle: README-Satz "normalize features to zero mean and unit standard deviation (optional)". `row_normalize` aus `utils.py` steht in Gen 0 nicht im Dossier, in Gen 2 steht beides, der Code folgt dem README. lr 0,2, 100 Epochen, Weight Decay 5e-6 sind übernommen. A2-Abgleich in allen drei Läufen `error`. | `.../20261002-004306/sgc_cora_acc_43ebdb09da08.py:53-57,140`, Dossier [36]; `.../20261002-010524/sgc_cora_acc_b29fac5317d3.py:31-32,134`, Dossier [26], [55]; `.../20261002-012056/sgc_cora_acc_c83a4890dcdf.py:57`, Dossier [28], [33]. Gegenproben S2, S3, S4. |
| Zeng DLinear | A | 0,3863 / 0,3834 / 0,3840 | +0,009 | Konstante Lernrate 1e-3, 20 Epochen statt 0,005 mit Halbierung je Epoche. Liegt an der Toleranzgrenze (zwei von drei reproduziert). | `zeng_dlinear_2023/20261001-203448/dlinear_etth1_96_mse_670df12d8799.py:22-25`. Gegenprobe E2. |
| Zeng DLinear | B | 0,3866 / 0,3930 / Ausfall | +0,015 | **`argparse`-Defaults statt Werte aus `etth1.sh`:** Eingabelänge 96 statt 336, lr 1e-4 statt 0,005. Das Dossier enthält beide (Paper "96" für die Transformer-Baselines, `run_longExp.py` Default 0,0001; `etth1.sh` 336 und 0,005). Gen 2: kein lauffähiger Code (Form von `val_predictions.csv`, `ValueError` nach 4 Versuchen). | `.../20261001-204505/dlinear_etth1_96_mse_9fcc3646f34a.py:135,143-144`, Dossier [0], [19], [111]; `.../20261001-215952/dlinear_etth1_96_mse_7f5d4656db55.py:31,35-37`, Dossier [4], [12], [15], [98]; `.../20261001-223820/zeugnis.md:6`. Gegenprobe E1. |
| Shchur GCN | A | 0,7957 / 0,7942 / 0,7849 | −0,021 | Kipf-Werte statt Shchur-Werte: Hidden 16, Dropout 0,5, L2 5e-4 statt 64 / 0,8 / 1e-3; dazu 200 Epochen (Gen 1, 2) und Geduld 100 bzw. 20 statt 50 auf Gesamt-Validierungsloss. Ohne Tavily nicht wissbar. | `shchur_pitfalls_2018/20261001-203448/shchur_gcn_cora_acc_8780b2eaa8b2.py:149-153`; `.../20261001-220311/shchur_gcn_cora_acc_896340d1295c.py:154-161`; `.../20261001-224641/shchur_gcn_cora_acc_c70d7fae553d.py:155-158,179-180`. Gegenprobe L1. |
| Shchur GCN | B | 0,7984 / 0,8086 / 0,8092 | −0,006 | Gen 1, 2 reproduziert (alle Angaben übernommen). Gen 0: Gewichte mit `torch.randn` statt Glorot initialisiert. | `.../20261001-205908/shchur_gcn_cora_acc_c52f9c3a3fb5.py:63-65`. Gegenprobe L2. |
| Veličković GAT | A | 0,8122 / 0,8022 / 0,7801 | −0,028 | Gen 1: keine Merkmalsnormalisierung. Gen 2: 4 Köpfe × 4 statt 8 × 8, keine Selbstschleifen, keine Normalisierung. Gen 0: alles nach Paper, trotzdem 0,812, Ursache offen. | `velickovic_gat_2018/20261002-035810/gat_cora_acc_2578199bd06a.py:150-152,177`; `.../20261002-054502/gat_cora_acc_5067507acdbd.py:144`; `.../20261002-015850/gat_cora_acc_256646d1f222.py:37,73,179`. Gegenproben T1, T2. |
| Veličković GAT | B | 0,8148 / 0,8440 / 0,8014 | −0,015 | Gen 1: **Training auf Validierungslabels** (Loss über alle Labels ≥ 0, also Train + Val; Validierungs-Accuracy 0,964 bis 1,0), deshalb "zu hoch". Der A2-Abgleich fand die fehlende Normalisierung, der korrigierte Code wurde verworfen, weil seine Validierung schlechter war als die des Leak-Codes. Gen 2: keine Merkmalsnormalisierung (A2 meldet sie nicht). Gen 0: nach A2 alles nach Repo, 0,8148, Ursache offen. | `.../20261002-043629/gat_cora_acc_4ae6e57de951.py:195-201,229-233`, `dossier.json` `repo_diff.gat_cora_acc` (`kept`, `val_old` 1,0, `val_new` 0,964); `.../20261002-063447/gat_cora_acc_6c79c698d128.py:189,203`; `.../20261002-032153/gat_cora_acc_c7d81943622b.py:149,161-169`. Gegenprobe T1. |
| Veličković DGI | A | 0,6722 / 0,7506 / 0,7811 | −0,072 | Falscher Encoder in allen drei: zwei GCN-Schichten mit ReLU und Dropout 0,5 statt einer Schicht mit PReLU; keine Merkmalsnormalisierung; Gen 1 Hidden 64; Gen 2 zeilennormalisierte Adjazenz; Gen 0 Klassifikator mit SGD und Weight Decay 1,0. | `velickovic_dgi_2019/20261002-032153/dgi_cora_acc_c5375802f803.py:86,122,146,166`; `.../20261002-035319/dgi_cora_acc_39661be49d94.py:80,87,100`; `.../20261002-044340/dgi_cora_acc_e2e106e02e9a.py:29-33,64,108`. Gegenproben D2 bis D4. |
| Veličković DGI | B | 0,6897 / 0,7754 / 0,6898 | −0,133 | **Induktive Angaben für den transduktiven Claim.** Das Dossier hat nur Paper-Angaben (5 bis 20 Einträge, keine Repo-Datei), darunter "512 features in each MP layer", "minibatch 256", "sample 10, 10, 25 neighbors", "two-layer DGI encoder", "dropout applied to input features of sampled graph" (Reddit/PPI). Gen 0 baut MLP + 3 Message-Passing-Schichten mit Minibatches über die 140 Trainingsknoten, Gen 2 zwei Schichten mit Dropout, Gen 1 Hidden 256 mit ReLU. lr, Encoder (eine GCN-Schicht, PReLU) und Merkmalsnormalisierung stehen nicht im Dossier. | `.../20261002-033734/dgi_cora_acc_5ce7e2aa4b68.py:98-101,186-192`, Dossier [3], [4], [5]; `.../20261002-050651/dgi_cora_acc_c989090b099a.py:64-78,115`, Dossier [4], [10]; `.../20261002-040607/dgi_cora_acc_832263aab412.py:33,50,116`. |
| Shchur GAT | A | 0,7917 / 0,7778 / 0,7913 | −0,027 | Veličković-Werte statt Shchur-Tabelle 4: lr 0,005, L2 5e-4, Attention-Dropout 0,6, Geduld 100 (im Code als "uncertain" markiert); Gen 2 zusätzlich 2 Köpfe, keine Selbstschleifen. | `shchur_gat_2018/20261001-203448/shchur_gat_cora_acc_fc551ed6e783.py:223-230`; `.../20261001-222635/shchur_gat_cora_acc_71de5f367595.py:197-207`; `.../20261002-001352/shchur_gat_cora_acc_7e88a5f7e150.py:160-168`. Gegenprobe L3. |
| Shchur GAT | B | 0,8043 / 0,7938 / 0,7062 | −0,024 | Gen 2: **Kanten nur in einer Richtung** (`u < v`, eindeutig, dann nur diese Richtung aggregiert): halber Graph. Gen 0 und 2: "Hidden size for GAT 64" (Tabellenzeile) als 64 **je Kopf** gelesen (512 statt 8 × 8). Gen 0 Geduld 100 statt 50. Gen 1: alle Werte richtig, 0,794, Ursache offen (A2-Abgleich übernommen: binäre Merkmale, Selbstschleifen, Validierungsloss). Die Repo-Auswahl enthält `gcn.conf.yaml`, nicht `gat.conf.yaml`. | `.../20261002-012908/shchur_gat_cora_acc_391c18e49811.py:47-57,230-231`; `.../20261001-210628/shchur_gat_cora_acc_c8c6fe381021.py:178-186,199`, Dossier [11]; `.../20261001-231903/shchur_gat_cora_acc_c06a2f52f591.py:42-58`. Gegenproben L4 bis L6. |
| Zeng NLinear | A | 0,8423 / 0,7689 / 0,8385 | +0,465 | **Falsche Methode:** eine lineare Schicht über alle Kanäle (`Linear(L·C, H·C)`, 1,6 Mio. Gewichte) ohne Abzug des letzten Werts (Gen 0, 2) bzw. mit Abzug des Fenstermittels (Gen 1). Das ist kein NLinear. Wie Muster M2 aus v3 (DLinear ohne Zerlegung). | `zeng_nlinear_2023/20261001-225043/nlinear_etth1_96_mse_f0fbf5c371fa.py:47-53,147`; `.../20261001-231908/nlinear_etth1_96_mse_7be29e4e9674.py:100-105`; `.../20261002-003205/nlinear_etth1_96_mse_a8b6254191ce.py:32-38`. Gegenprobe N1. |
| Zeng NLinear | B | 0,3854 / 0,4304 / 0,4595 | +0,056 | Gen 1, 2: Code baut **DLinear** (Trend/Saison-Zerlegung) für den NLinear-Claim und trainiert mit konstanter lr 0,005 ohne die Halbierung je Epoche (`lradj type1`); Gen 2 ohne Early Stopping. Gen 0: richtiges NLinear, aber lr 5e-4 konstant ("reduced learning rate to avoid divergence"). Die Halbierung steht im Dossier nur als Funktionsname (`adjust_learning_rate`), nicht als Regel. | `.../20261001-232931/nlinear_etth1_96_mse_8db259403c9e.py:63-70,83-85,97`; `.../20261002-010849/nlinear_etth1_96_mse_8ebdecbf1740.py:37-44,67,82-85`; `.../20261001-230318/nlinear_etth1_96_mse_ab98b875dc43.py:152-154`, Dossier [12], [34]. Gegenproben N2, N3, E3. |

## 2. H1: Kommen die Setup-Angaben im Code an?

Von Hand aus den Codezeilen und `setup_details` zugeordnet (Fehler von ein, zwei Zellen möglich). Arm B: 23 Läufe mit Code (DLinear Gen 2 ohne Code) × 7 Angaben = 161 Zellen.

- **übernommen**: Angabe steht im Dossier, Code nutzt sie.
- **abweichend**: Angabe steht im Dossier, Code nutzt einen anderen Wert.
- **fehlt im Code**: Angabe steht im Dossier, Code setzt sie nicht um.
- **nicht im Dossier**: Rolle 4 hat die Angabe nicht geliefert (Code nimmt einen Wert aus dem Gedächtnis oder lässt sie weg).

| Angabe (Arm B) | übernommen | abweichend | fehlt im Code | nicht im Dossier | nicht anwendbar |
|---|---|---|---|---|---|
| Lernrate | 13 | 3 | 0 | 7 | 0 |
| Weight Decay / L2 | 14 | 1 | 0 | 3 | 5 |
| Epochen (Maximum) | 13 | 1 | 0 | 9 | 0 |
| Dropout | 16 | 0 | 0 | 2 | 5 |
| Hidden / Köpfe (ETTh1: Eingabelänge) | 15 | 4 | 0 | 4 | 0 |
| Early Stopping / Geduld | 13 | 5 | 1 | 1 | 3 |
| Normalisierung (Merkmale; NLinear: letzter Wert) | 5 | 1 | 1 | 14 | 2 |
| **Summe** | **89** | **15** | **2** | **40** | **15** |

Lesart: 89 von 106 vorhandenen Angaben (84 %) werden übernommen. Die 15 Abweichungen: DLinear lr und Eingabelänge (je 2, `argparse`-Default statt Skriptwert), Geduld 7 statt 3 (DLinear ×2, aus der Klasse `EarlyStopping` in `tools.py`), Shchur-GAT Hidden 64 je Kopf (×2), Geduld 100 statt 50 (×1), L2 doppelt und auf allen Parametern (×1), NLinear lr 5e-4 (×1) und Geduld 5 (×1), Kipf Gen 2 mit 200 Epochen und Geduld 10 gegen die Dossier-Angabe "400 Epochen ohne Early Stopping" (×2; hier ist der Code näher am Hauptexperiment als das Dossier), SGC L2- statt L1-Zeilennorm (×1). "Fehlt im Code": NLinear Gen 1 ohne Abzug des letzten Werts, NLinear Gen 2 ohne Early Stopping.

**Mindestens 9 der 89 übernommenen Angaben sind irreführend** (richtig zitiert, falscher Zusammenhang) und senken das Ergebnis: SGC-Standardisierung (2), DGI-Dropout und zwei Schichten bzw. 3 MP-Schichten und Minibatch (3), DLinear Geduld 7 (bereits unter abweichend gezählt), Kipf "400 Epochen ohne Early Stopping" (2, ohne Schaden), Shchur-GAT Geduld 100 (GAT-Original statt einheitlicher Regel).

Größte Lücke je Angabe: **Normalisierung steht in 14 von 21 anwendbaren Läufen nicht im Dossier.** Der Task-Text nennt die Merkmale ausdrücklich "binary, not normalised" (`src/repro_agent/tasks.py:59,196`), der Agent muss die Normalisierung also selbst kennen oder aus dem Repo erfahren.

**Arm A** hat keine Setup-Angaben. Vergleich mit den Werten der Referenzlösung (24 Läufe):

| Angabe (Arm A) | wie Referenz | abweichend | fehlt | nicht anwendbar |
|---|---|---|---|---|
| Lernrate | 12 | 12 | 0 | 0 |
| Weight Decay / L2 | 8 | 6 | 1 | 9 |
| Epochen | 9 | 15 | 0 | 0 |
| Dropout | 6 | 9 | 0 | 9 |
| Hidden / Köpfe / Eingabelänge | 18 | 6 | 0 | 0 |
| Early Stopping / Geduld | 6 | 15 | 0 | 3 |
| Normalisierung | 4 | 1 | 10 | 9 |

In Arm A stimmen die Werte dort, wo das Paper "kanonisch" ist (Kipf, Veličković GAT), und fehlen dort, wo das Paper eigene Werte setzt (Shchur) oder die Methode wenig bekannt ist (NLinear, SGC-Protokoll, DGI).

## 3. Gegenproben

Referenzlösung `tests/reference/<name>_reference.py`, lokal auf der CPU, mit **einer** gezielten Änderung (Textersetzung im Referenzcode). Daten und Metrik wie der Harness (`cora_data`, `cora_lcc_data`, `etth1_data`, `metrics`). Seeds 0 bis n−1; Cora-LCC: Split k mit Seed k. Skript `results/failure_analysis_v4_gp.py` (Kopie von `gp.py` aus dem Scratchpad; Pfade `ROOT` und `CACHE` oben im Skript anpassen, `CACHE` braucht `raw/ind.cora.*` und `ett/ETTh1.csv`), Rohausgabe aller fertigen Läufe mit Einzelwerten: `results/failure_analysis_v4_gegenproben.txt`. Eigene Umgebung mit torch 2.14.1 CPU, numpy 2.5.2, scipy (`uv venv gpvenv --python 3.12; uv pip install --python gpvenv/bin/python torch numpy scipy --index-url https://download.pytorch.org/whl/cpu`), weil in `.venv` kein torch ist. Scratchpad und Umgebung lagen unter `/tmp` und sind nach dem Neustart weg. Die unveränderten Referenzen treffen die gespeicherten Werte (`gcn_ref` 0,8136; `sgc_ref` 0,8081; `nlinear_ref` 0,3739; `dlinear_ref` 0,3809 mit `LR_HALVE=1`, 0,4278 ohne; `dgi_ref` 0,8053 statt 0,8047; `gcnlcc_ref` 0,8117 statt 0,8107), die lokale Umgebung ist also vergleichbar.

Befehl je Zeile: `gpvenv/bin/python gp.py <Variante> [n]`.

| # | Variante | Änderung an der Referenz | n | Mittel ± Std (min bis max) | Referenz | Agentenwert dazu | Befund |
|---|---|---|---|---|---|---|---|
| G1 | `gcn_es_valacc10` | Early Stopping: Geduld 10 auf Validierungs-Accuracy, bestes Modell zurück | 10 | 0,6999 ± 0,1833 (0,270 bis 0,821) | 0,8136 ± 0,0051 | Kipf A Gen 0: 0,7365 ± 0,093; Gen 1: 0,7476 ± 0,057 | **belegt**: gleiche Größenordnung und gleiche Seed-Ausreißer (Stopp nach 13 Epochen) |
| G2 | `gcn_nonorm` | keine Zeilennormalisierung der Merkmale | 10 | 0,7920 ± 0,0136 | 0,8136 | Kipf A Gen 2: 0,7987 | **belegt** |
| S1 | `sgc_wd5e-6` | Weight Decay fest 5e-6 statt Gitter | 10 | 0,8054 ± 0,0021 | 0,8081 ± 0,0009 | – | Repo-Default allein schadet nicht |
| S2 | `sgc_std_wd5e-6` | spaltenweise Standardisierung, wd 5e-6 | 10 | 0,7300 ± 0,0035 | 0,8081 | SGC B Gen 0: 0,7489; Gen 2: 0,7336 | **belegt** |
| S3 | `sgc_l2norm_wd5e-6` | L2- statt L1-Zeilennorm, wd 5e-6 | 10 | 0,7932 ± 0,0011 | 0,8081 | SGC B Gen 1: 0,7970 | **belegt** |
| S4 | `sgc_nonorm_wd5e-6` | keine Normalisierung, wd 5e-6 | 10 | 0,7850 ± 0,0075 | 0,8081 | – | Normalisierung fehlt: −2,3 Pp |
| S5 | `sgc_nonorm_grid` | keine Normalisierung, wd aus Gitter (wählt 5e-3) | 10 | 0,8164 ± 0,0016 | 0,8081 | SGC A Gen 0: 0,8215 | **belegt**: ohne Normalisierung und mit starker Regularisierung liegt SGC über dem Paperwert |
| S6 | `sgc_nonorm_trainval` | wie S5, Training auf Train + Validierung | 10 | 0,8384 ± 0,0010 | 0,8081 | SGC A Gen 1: 0,8530 ± 0,0 | **belegt** (Richtung und Größe; Agent nutzt LBFGS) |
| T1 | `gat_nonorm` | keine Zeilennormalisierung | 10 | 0,8069 ± 0,0082 | 0,828 (gespeichert) | GAT A Gen 1: 0,8022; B Gen 2: 0,8014 | **belegt** |
| T2 | `gat_4x4_noloops_nonorm` | 4 Köpfe × 4, keine Selbstschleifen, keine Normalisierung | 10 | 0,7963 ± 0,0086 | 0,828 | GAT A Gen 2: 0,7801 | wahrscheinlich (erklärt 3,2 von 4,8 Pp) |
| D1 | `dgi_200ep` | höchstens 200 Epochen | 10 | 0,8025 ± 0,0080 | 0,8053 ± 0,0073 | – | Epochenzahl ist **nicht** die Ursache |
| D2 | `dgi_nonorm` | keine Zeilennormalisierung | 10 | 0,7784 ± 0,0123 | 0,8053 | DGI A Gen 2: 0,7811; B Gen 1: 0,7754 | **belegt** (−2,7 Pp; trifft alle 6 DGI-Läufe) |
| D3 | `dgi_nonorm_200ep` | D1 + D2 | 10 | 0,7800 ± 0,0142 | 0,8053 | wie D2 | wie D2 |
| D4 | `dgi_h64_200ep` | Hidden 64, 200 Epochen (mit Normalisierung) | 10 | 0,7179 ± 0,0150 | 0,8053 | DGI A Gen 1: 0,7506 (Hidden 64, zwei Schichten) | Einbettungsgröße wirkt stark; Zuordnung zum Lauf nur wahrscheinlich |
| L1 | `gcnlcc_kipf_hp` | Hidden 16, Dropout 0,5, L2 5e-4 | 30 | offen | 0,8117 ± 0,0133 | Shchur GCN A: 0,7957 / 0,7942 / 0,7849 | **offen**: Lauf nicht fertig (Rechner-Neustart). Befehl: `gpvenv/bin/python gp.py gcnlcc_kipf_hp 30` |
| L2 | `gcnlcc_randn_init` | Gewichte `torch.randn` statt Glorot | 30 | offen | 0,8117 | Shchur GCN B Gen 0: 0,7984 | **offen**: Lauf nicht fertig (Rechner-Neustart). Befehl: `gpvenv/bin/python gp.py gcnlcc_randn_init 30` |
| L3 | `gatlcc_velickovic_hp` | lr 0,005, L2 5e-4, Attention-Dropout 0,6, Geduld 100 | 30 | 0,8000 ± 0,0170 | 0,8140 ± 0,0115 (gespeichert) | Shchur GAT A: 0,7917 / 0,7778 / 0,7913 | wahrscheinlich (erklärt 1,4 von 2,6 Pp) |
| L4 | `gatlcc_onedir` | nur Kanten mit `src < dst` plus Selbstschleifen | 10 | 0,7257 ± 0,0189 | 0,8127 ± 0,0084 (gleiche 10 Splits) | Shchur GAT B Gen 2: 0,7062 | **belegt** |
| L5 | `gatlcc_hid64perhead` | 64 Einheiten je Kopf (512) | 10 | 0,8052 ± 0,0148 | 0,8127 (gleiche 10 Splits) | Shchur GAT B Gen 0: 0,8043 | wahrscheinlich (−0,8 Pp, innerhalb der Split-Streuung) |
| L6 | `gatlcc_l2x2_allparams` | L2 doppelt so stark, auch auf Bias | 30 | 0,8128 ± 0,0106 | 0,8140 | – | **widerlegt**: die L2-Form erklärt Gen 2 nicht |
| E1 | `dlinear_L96_lr1e-4` | Eingabelänge 96, lr 1e-4 konstant, Geduld 7 | 5 | 0,3944 ± 0,0020 | 0,3809 ± 0,0139 | DLinear B: 0,3866 / 0,3930 | **belegt** |
| E2 | `dlinear_lr1e-3_20ep` | lr 1e-3 konstant, 20 Epochen, Geduld 5 | 5 | 0,3864 ± 0,0069 | 0,3809 | DLinear A: 0,3863 / 0,3834 / 0,3840 | **belegt** |
| E3 | `dlinear_lr5e-3_nohalve` | lr 0,005 konstant (keine Halbierung), Geduld 5 | 5 | 0,4266 ± 0,0089 | 0,3809 | NLinear B Gen 1 (DLinear-Code): 0,4304; Gen 2: 0,4595 | **belegt** (deckt sich mit `results/etth1_harness_reference.txt`: konstante Lernrate 0,4278) |
| N1 | `nlinear_flat_nolast_lr1e-3` | `Linear(L·7, H·7)` ohne Abzug des letzten Werts, lr 1e-3, 50 Epochen | 5 | 0,8678 ± 0,0574 | 0,3739 ± 0,0013 | NLinear A: 0,8423 / 0,7689 / 0,8385 | **belegt** |
| N2 | `nlinear_lr5e-4` | lr 5e-4 konstant, 20 Epochen, Geduld 7 | 5 | 0,3859 ± 0,0030 | 0,3739 | NLinear B Gen 0: 0,3854 ± 0,004 | **belegt** (auf 0,0005 genau) |
| N3 | `nlinear_lr5e-3_nohalve` | lr 0,005 konstant | 5 | 0,4136 ± 0,0122 | 0,3739 | – | Halbierung der Lernrate ist für beide Linear-Modelle entscheidend (+0,04 MSE) |

## 3a. H3 (Toleranz gegen Streuung) und H4 (Auswahl über Validierung), post hoc

**H3.** Die Regel wird nicht geändert; das hier ist nur eine Einordnung.
- Seed-Std der Referenzlösungen (lokal): GCN 0,0051; SGC 0,0009; DGI 0,0073; GAT-LCC 0,0115 (gespeichert); GCN-LCC 0,0133; NLinear 0,0013; **DLinear 0,0139** (5 Seeds: 0,371 bis 0,405; ein Seed allein verschiebt das Mittel um 0,006). Bei DLinear ist die Seed-Streuung der Referenz größer als die Toleranz 0,01; das Referenzurteil "reproduziert" (Abstand 0,0059) hängt an 5 Seeds.
- Seed-Std der Agenten-Läufe: Cora 0,003 bis 0,010 (ohne die Abbruch-Läufe), Cora-LCC 0,014 bis 0,021, ETTh1 0,001 bis 0,006.
- Std der Mittelwerte über die drei Generierungen: 0,001 (Kipf B, DLinear A) bis 0,046 (DGI A), Median 0,016. **Die Streuung zwischen Generierungen ist größer als die Toleranz und meist größer als die Seed-Streuung**: das Urteil hängt stärker davon ab, welchen Code das Modell schreibt, als vom Seed.
- Toleranz = 2 × Std des Papers (Shchur GCN/GAT 1,3 Pp, GAT 0,7 Pp, DGI 0,6 Pp laut `papers.yaml`; sonst keine Std im Paper, dort 0,01): "reproduziert" in A 4 statt 2, in B 9 statt 5 von 24 Einzelurteilen; Mehrheit je Einheit A 2 von 8 (Shchur GCN, DLinear), B 3 von 8 (Kipf, Shchur GCN, Shchur GAT). Mit Blick auf das Referenzurteil (DGI: teilweise) bleibt es bei höchstens 3 von 8. **Die Toleranz ist nicht die Hauptursache**; nur die Shchur-Einheiten liegen im Bereich der Paper-Streuung.

**H4.** Seed 0, je Kandidat Validierungs- und Testwert aus dem Dossier (`attempts[].val`, `attempts[].measured`); Testwerte nur hier.
- 44 Läufe mit mindestens zwei lauffähigen Kandidaten: Der gewählte Kandidat ist in 29 der test-beste, in 7 der test-schlechteste. Verlust gegenüber dem test-besten: Median 0,000, Mittel 0,0027, Maximum 0,019. Über 0,01: GAT A Gen 0 (0,011) und Gen 2 (0,016), SGC A Gen 1 (0,019), DLinear A Gen 0 (0,012).
- Kandidatenpaare innerhalb eines Laufs: Validierung und Test ordnen in 73 Paaren gleich, in 21 entgegengesetzt.
- **Nicht bestätigt**: Die Auswahl schadet nicht systematisch. Sie hilft aber auch selten, weil die drei Kandidaten meist denselben Fehler teilen (gleicher Plan, gleiche fehlende Angabe): Die Spanne der Testwerte zwischen Kandidaten ist klein gegen den Abstand zum Ziel.
- **Nebenwirkung (belegt, 2 Läufe):** Ein Kandidat, der auf den Validierungslabels trainiert, hat die beste Validierungs-Metrik und gewinnt die Auswahl (SGC A Gen 1: Val 0,998; GAT B Gen 1: Val 0,964 bis 1,0). Im GAT-Lauf verhindert derselbe Vergleich die A2-Korrektur. Das Ergebnis ist dann "zu hoch" und zählt als Nicht-Übereinstimmung. Die Validierungslabels liegen im Agenten-Datensatz (Standard-Split), der Code-Check prüft das nicht.
- Im Durchschnitt hat nur etwa die Hälfte der Kandidaten überhaupt einen lauffähigen Seed-0-Lauf (in 4 Läufen genau einer, in DLinear B Gen 2 keiner): Die Auswahl entscheidet oft zwischen zwei Kandidaten.

## 3b. H5: Einzelfälle

- **DGI (−7 bis −13 Pp):** kein einzelner Fehler. Belegt: fehlende Merkmalsnormalisierung (−2,7 Pp, alle 6 Läufe). Dazu je Lauf ein anderer Encoder als im Paper (zwei Schichten mit Dropout, MLP + 3 MP-Schichten mit Minibatch, Hidden 64 oder 256, ReLU statt PReLU, zeilennormalisierte Adjazenz). Pretraining-Dauer (200 Epochen) ist nicht die Ursache (D1). Readout und Diskriminator sind in allen gelesenen Codes richtig (Mittel + Sigmoid, bilinear). Die logistische Regression weicht ab (SGD mit Weight Decay 1,0 in A Gen 0; Auswahl des Encoders nach Validierungs-Accuracy in A Gen 0 und B Gen 0), Anteil nicht getrennt gemessen. Die Referenz selbst erreicht nur 0,805 (Ziel 0,823, Referenzurteil "teilweise").
- **NLinear (MSE 0,8 in Arm A):** kein Normalisierungsfehler in den Daten, sondern falsches Modell (N1). In Arm B zwei von drei Läufen DLinear statt NLinear plus konstante Lernrate (E3, N3).
- **Shchur-GAT 0,706 (B Gen 2):** Kanten nur in einer Richtung (L4). Der Graph-Check sieht das nicht (Ablation: Graph wird genutzt). Der einzige lauffähige Kandidat des Laufs; der A2-Abgleich (Aspekt "to_undirected", "eliminate_self_loops") hat den Code ersetzt, die Validierung blieb gleich (0,743).

## 4. Rangliste der Ursachen

Gezählt werden echte Claims, 48 Läufe, 41 davon nicht "reproduziert" (inkl. 1 Ausfall). Ein Lauf kann zwei Ursachen haben (Summe > 41). "Belegt" = Codezeile und Gegenprobe; "wahrscheinlich" = Codezeile, Gegenprobe erklärt einen Teil oder fehlt; "offen" = keine Erklärung.

| Rang | Ursache | Läufe | Betroffen | Status |
|---|---|---|---|---|
| 1 | **Falsche Methode oder Architektur** (kein NLinear; DLinear statt NLinear; DGI-Encoder; GAT 4 × 4) | 12 | NLinear A ×3, NLinear B Gen 1, 2; DGI A ×3, B ×3; GAT A Gen 2 | NLinear belegt (N1, E3); DGI und GAT wahrscheinlich (D4, T2) |
| 2 | **Merkmalsnormalisierung fehlt oder falsch** (Hauptursache in 7, Mitursache in 7) | 14 | Kipf A Gen 2; SGC A Gen 0, B ×3; GAT A Gen 1, 2, B Gen 2; DGI ×6 | belegt (G2, S2 bis S5, T1, D2) |
| 3 | **Hyperparameter eines anderen Papers** (Arm A, ohne Setup-Angaben) | 8 | Shchur GCN A ×3; Shchur GAT A ×3; SGC A Gen 2; DLinear A Gen 0 | Codezeilen belegt; Gegenprobe DLinear belegt (E2), Shchur GAT wahrscheinlich (L3 erklärt 1,4 von 2,6 Pp), Shchur GCN **offen** (L1 nicht fertig) |
| 4 | **Irreführende Tavily-Angabe treu übernommen** (Arm B; überschneidet sich mit 1 und 2) | 7 | SGC B Gen 0, 2; DLinear B Gen 0, 1; DGI B Gen 0, 2; Shchur GAT B Gen 0 | belegt für SGC und DLinear (S2, E1); DGI und Shchur GAT wahrscheinlich (L5) |
| 5 | **Lernraten-Plan fehlt** (konstante Lernrate statt Halbierung je Epoche) | 4 | NLinear B ×3; DLinear A Gen 0 | belegt (E2, E3, N2, N3) |
| 6 | **Early Stopping zu früh** (Geduld 10 auf Validierungs-Accuracy) | 2 | Kipf A Gen 0, 1 | belegt (G1) |
| 7 | **Training auf Validierungslabels** (Ergebnis zu hoch), von der A1-Auswahl begünstigt | 2 | SGC A Gen 1; GAT B Gen 1 | belegt (Code, S6) |
| 8 | **Graph-Implementierungsfehler** (Kanten nur eine Richtung) | 1 | Shchur GAT B Gen 2 | belegt (L4) |
| 9 | Initialisierung `randn` statt Glorot | 1 | Shchur GCN B Gen 0 | wahrscheinlich (Codezeile); Gegenprobe **offen** (L2 nicht fertig) |
| 10 | Kein lauffähiger Code (Form `val_predictions.csv`) | 1 | DLinear B Gen 2 | belegt (`zeugnis.md`) |
| 11 | **Knapp daneben, Ursache nicht gefunden** | 3 | GAT A Gen 0 (0,812), GAT B Gen 0 (0,815), Shchur GAT B Gen 1 (0,794) | offen |

Zur Einordnung: Die GAT-Referenz liegt mit 0,828 nur 0,2 Pp unter dem Ziel 0,83, die DLinear-Referenz 0,6 Pp, die Shchur-GCN-Referenz 0,4 Pp, die Shchur-GAT-Referenz 0,4 Pp. Schon ein kleiner Unterschied im Detail reicht dort für "teilweise".

## 5. Vorschläge für ein vorab festlegbares v5

Kein Vorschlag bringt einen Test- oder Validierungswert in einen Prompt, keiner ändert Toleranz, Zielwerte oder Referenzurteile. Reihenfolge nach erwartetem Effekt.

| # | Was sich ändert | Betroffene Einheiten | Erwarteter Effekt | Leak-Risiko |
|---|---|---|---|---|
| V1 | **Rolle 4 bekommt eine feste Pflichtliste je Task-Familie** (Graph: Merkmalsnormalisierung, Adjazenz-Normalisierung, Initialisierung, Lernraten-Plan, Early-Stopping-Regel, Epochen; Zeitreihe: Eingabelänge, Lernraten-Plan, Epochen, Geduld). Fehlt ein Punkt nach der ersten Extraktion, liest Rolle 4 gezielt die Datei, die die Daten lädt (`utils.py`, `process.py`, `data_loader.py`) und das Startskript. Fehlende Punkte stehen im Plan-Prompt ausdrücklich als "not found in paper or repository". | alle in Arm B, vor allem SGC, GAT, DGI, NLinear | schließt die 14 fehlenden Normalisierungs-Angaben und die fehlende Halbierung der Lernrate; größter Einzelhebel in Arm B (bis zu 9 Läufe) | keines (nur Paper und Repo; Liste ist allgemein, nicht je Paper) |
| V2 | **Rangfolge der Quellen im Plan-Prompt:** Startskript des Experiments (`scripts/.../etth1.sh`, `execute_cora.py`) vor Konfigurationsdatei vor `argparse`-Default vor README-Fließtext. README-Sätze mit "optional", "for example", "can" werden als "optional advice" markiert. Erweitert A3-M7. | DLinear B, SGC B, NLinear B | DLinear B: 0,394 auf etwa 0,381 (E1 gegen Referenz); SGC B: 0,73 auf 0,80 (S2 gegen S1) | keines |
| V3 | **Zuordnung der Angabe zum Claim prüfen:** Jede Setup-Angabe bekommt beim Extrahieren das Feld "gilt für" (Modell, Datensatz, Einstellung transduktiv/induktiv), mit Zitat. Angaben für ein anderes Modell oder einen anderen Datensatz gehen nicht in den Plan. Repo-Dateiauswahl nennt den Modellnamen des Claims (`gat.conf.yaml` statt `gcn.conf.yaml`). | DGI B, Shchur GAT B, DLinear B (Eingabelänge 96 der Baselines) | DGI B: Encoder nach Abschnitt "transductive"; Shchur GAT: 8 × 8 statt 64 je Kopf | keines; Risiko einer falschen Filterung (Angabe verworfen, die doch gilt), deshalb verworfene Angaben im Dossier ausweisen |
| V4 | **Statische Prüfung "Validierungslabels im Training"** in `harness/code_check.py` oder als Lauf-Prüfung: Validierungs-Metrik des Harness ≥ 0,95 bei Trainings-Metrik-ähnlichem Wert, oder zweiter Lauf mit maskierten Validierungslabels (wie die Graph-Ablation: fällt die Testgenauigkeit, wurden sie genutzt). Treffer = Kandidat scheidet aus der A1-Auswahl aus, Reparaturhinweis ohne Zahlen. | SGC, GAT (2 Läufe in v4), jede Einheit mit A1 | entfernt die "zu hoch"-Läufe und stellt die A2-Korrektur wieder scharf | keines (Validierungslabels lädt weiter nur der Harness; die Schwelle ist eine feste Regel, kein Wert im Prompt) |
| V5 | **Deterministische Graph-Prüfung "Kanten symmetrisch"** im Smoke-Lauf: Der Harness gibt einen kleinen gerichteten Testgraphen vor und prüft, ob eine Änderung an Knoten j die Ausgabe von Nachbar i in beiden Richtungen ändert. | alle Graph-Einheiten | fängt den Fall Shchur GAT B Gen 2 (1 Lauf in v4); billig | keines |
| V6 | **Kandidaten aus verschiedenen Plänen statt aus einem** (A1 heute: ein Plan, drei Temperaturen). Drei Pläne mit je einem Code-Kandidaten, Auswahl wie bisher über die Validierung. | alle | die Kandidaten teilen heute denselben Fehler; verschiedene Pläne erhöhen die Chance, dass einer die Normalisierung oder den Lernraten-Plan enthält. Effekt unsicher. | keines; Kosten etwa +2 Plan-Aufrufe je Claim |
| V7 | **Reparaturhinweis bei instabilem Training:** Seed-Std über einer festen Schwelle (z. B. 5 × Toleranz) oder einzelne Seeds nahe der Mehrheits-Baseline lösen eine Reparaturrunde mit festem Text aus ("early stopping may trigger too early; use validation loss and restore the best weights"), ohne Zahlen. | Kipf A (2 Läufe), allgemein | fängt G1; kleiner Effekt | gering: Die Schwelle nutzt Testwerte nur als Ja/Nein-Signal für "instabil". Sauberer: Schwelle auf der Streuung der **Validierungs**-Metrik über Seeds. |
| V8 | **Berichten statt beheben:** Arm A ohne Setup-Angaben kann Shchur-Werte nicht kennen (Rang 3). Im README als erwartete Grenze nennen ("ohne Recherche übernimmt der Agent Standardwerte des Ursprungspapers"); das ist der eigentliche Beleg für den Nutzen von Tavily (Shchur GCN: A 0 von 3, B 2 von 3). | Shchur GCN, Shchur GAT | kein Effekt auf die Zahl, aber auf die Aussage | keines |

Nicht empfohlen: Methodenwissen von uns in den Prompt schreiben ("NLinear subtrahiert den letzten Wert", "DGI nutzt eine GCN-Schicht"). Das wäre Wissen aus der Referenzlösung und kein Ergebnis des Agenten. Der richtige Ort dafür ist Rolle 4 (V1, V3) und die Methodentreue-Achse.

Offene Punkte dieser Analyse: GAT (Standard-Split) Gen 0 in beiden Armen und Shchur GAT B Gen 1 liegen 1,5 bis 2,4 Pp unter dem Ziel, obwohl alle geprüften Angaben stimmen; dafür wäre ein zeilenweiser Vergleich mit der Referenz nötig (Attention-Richtung, Dropout-Stellen, Bias, Initialisierungs-Gain). Die H1-Zählung ist Handarbeit an 47 Codes und 23 Dossiers ohne Zweitprüfung.
