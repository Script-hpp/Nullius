# B2: Suche nach einem echten Negativfall (Recherche, nur Doku)

Stand 2026-10-02, Branch `b-research`. Keine echten API-Aufrufe; Websuche und `curl` auf arXiv-HTML, GitHub-README und GitHub-Issue-API. Zitate sind Auszüge aus dem geholten Text.

## Ergebnis in einem Satz

**Teilfund, kein starker Negativfall.** Es gibt einen gut dokumentierten Messfehler (`drop_last=True` im Test-Dataloader), der veröffentlichte ETTh1-Zahlen verbessert hat. Für **PatchTST ETTh1 Horizont 96** ist der veröffentlichte Wert 0,370 und der nach der Korrektur neu gemessene 0,385: Abstand 0,015 = 1,5 × unsere Toleranz, also im Urteil **„teilweise"**, nicht „nicht reproduziert". Ein Fall mit Abstand > 3 × Toleranz (0,03) wurde nicht gefunden.

## Der Fall: PatchTST, ETTh1, Horizont 96, MSE

| | Wert | Beleg |
|---|---|---|
| **Behauptet** | MSE **0,370** (PatchTST/64, Look-back 512); PatchTST/42 (Look-back 336): **0,375** | PatchTST-Paper (arXiv 2211.14730 v2), Tabelle 3 „Multivariate long-term forecasting results with supervised PatchTST", Zeile `ETTh1  \|  96 \|  0.370 \|  0.400 \|  0.375 \|  0.399 \|  0.375 \|  0.399 \|  0.376 \|  0.415 \|  0.435 \|  0.446 ...` (PatchTST/64, PatchTST/42, DLinear, FEDformer, Autoformer ...). Text: „PatchTST/64 implies the number of input patches is 64, which uses the look-back window L=512. PatchTST/42 means the number of input patches is 42, which has the default look-back window L=336." |
| **Neu gemessen** | MSE **0,385** | FITS-Repository, README, Abschnitt „Result Update" (Tabelle nach Korrektur neu gerechnet): `\| PatchTST  \| 0.385 \| *0.413* \| *0.44* \| *0.456* \| ...` (ETTh1-96, -192, -336, -720). Gleiche Zahl in FITS-Paper v3 Tabelle 1: `PatchTST 0.385 0.413 0.440 0.456 0.274 0.338 ...`; SparseTSF v2 Tabelle 11 („after fixing code bug") führt PatchTST ebenfalls mit `0.385`. |
| **Protokoll der Neumessung** | ETTh1, Split 6:2:2 (12/4/4 Monate), Horizont 96, **ohne** `drop_last`. Look-back und Hyperparameter der PatchTST-Neuläufe stehen im README-Auszug **nicht**. | siehe Grenzen |
| **Ursache** | Der Test-Dataloader verwirft den letzten unvollständigen Batch. | FITS-README: „The bug stems from an incorrect implementation in the data loader. Specifically, the test dataloader uses `drop_last=True`, which may exclude a significant portion of test data, particularly with large batch sizes, leading to unfair model comparisons." Als betroffen gelistet: „PatchTST (ICLR 2023)", „DLinear (AAAI 2022 reported version)", „Informer", „Autoformer (NIPS 2021 reported version)", „Fedformer (ICML 2022)", „FiLM". Ausdrücklich **nicht** betroffen (spätere Korrektur): TimesNet und iTransformer („uses batch_size=1 during testing"). |
| **Unabhängige Bestätigung des Fehlers** | TFB (Qiu et al. 2024, arXiv 2403.20150 v4) | „implementations of existing methods often employ a “Drop Last” trick in the testing phase (Zhou et al., 2022b; Nie et al., 2022; Wang et al., 2022) ... if we discard the last incomplete batch with fewer instances than the batch size, this causes unfair comparisons." Beispiel dort ETTh2: Batchgrößen 32, 64, 128 verwerfen 17, 49, 113 Testbeispiele; „Table 4 shows the testing results of PatchTST, DLinear and FEDformer on the ETTh2 with different batch sizes and the “Drop Last” trick turned on. We observe that the performance of the methods changes when varying the batch size." |
| **Wirkung laut SparseTSF** | „This bug significantly improved the performance on ETTh1 and ETTh2 datasets when the batch size was large, while the impact on other datasets was relatively minor." | SparseTSF v2, Anhang D.1 „Comparison Results after Fixing the Code Bug" |

Eigene Rechnung (nicht aus den Quellen): Bei unserem ETTh1-Test gibt es 2880 − 96 + 1 = **2785** Fenster. Mit Batchgröße 128 fielen 2785 mod 128 = **97** Fenster (3,5 %) weg, mit 64 33 Fenster, mit 32 ein Fenster. Die Batchgröße der PatchTST-Läufe ist hier nicht belegt.

## Bezug zu unserem Benchmark

- Unser Harness wertet **alle** 2785 Fenster aus. Er verhält sich wie die korrigierte Pipeline. Ein Agent, der den Zielwert 0,370 reproduzieren will, rechnet ohne den Fehler und landet erwartungsgemäß näher an 0,385.
- Gleiches Muster steckt schon in v2/v3: DLinear-Zielwert 0,375 (Tabelle 2 des DLinear-Papers), unsere Referenzlösung im Harness misst **0,3809**, passt zu den korrigierten **0,384** der FITS-Tabelle. Abstand Paper zu Referenz 0,006: innerhalb der Toleranz, deshalb „reproduziert". Die Referenz stützt also die Fehlerthese, ist aber keine unabhängige Quelle (eigene Lösung).
- Erwartetes Referenzurteil für PatchTST/64 (unbelegt, erst Referenzlauf): **teilweise** (Abstand ≈ 0,015 bei korrigiertem Wert 0,385). Für PatchTST/42 (Ziel 0,375): Abstand 0,010, an der Grenze reproduziert/teilweise.

## Grenzen (ehrlich)

1. **Abstand zu klein für einen harten Negativfall.** 0,015 liegt zwischen 1 × und 3 × Toleranz. Selbst mit korrigierter Messung wäre das Urteil „teilweise". Wer „nicht reproduziert" braucht, muss die Toleranz absenken, das widerspräche der Vorab-Festlegung.
2. **Neumessung nicht ganz unabhängig.** Die 0,385 stammen aus dem Repository der FITS-Autoren, die ihr eigenes Modell gegen PatchTST vergleichen (Interessenkonflikt möglich). Die Fehlerursache selbst bestätigt unabhängig TFB, aber TFB belegt im geprüften Text nur den ETTh2-Batchgrößenvergleich; eine PatchTST-Zahl für ETTh1-96 wurde dort nicht gesucht (nicht geprüft).
3. **Look-back der Neuläufe unbekannt.** Es ist offen, ob 0,385 zu L=336 oder L=512 gehört; das Original nennt zwei Werte (0,370/0,375).
4. **Kein Urteil der PatchTST-Autoren gefunden.** Ob das PatchTST-Repository den Fehler behoben oder die Tabelle korrigiert hat, wurde nicht geprüft (nur der Link auf die betroffene Datei steht in der FITS-README).

## Verworfene Funde (Suchweg)

| Fund | Warum kein Negativfall |
|---|---|
| **GAT-Issue** PetarV-/GAT #19 „About the accuracy on the Cora dataset" (geschlossen, 2019-01-21): README nennt „84.4% accuracy", Paper 83,0 ± 0,7 | Antwort des Autors: „The pre-trained model corresponds to a single training run of GAT ... the value reported in the paper is averaged after training 100 times (with different random seeds). So you can think of it as: training and evaluating GAT on Cora (with the given train/test split) will give you ~83% accuracy on average." Zahl hält, Unterschied erklärt (Einzellauf gegen Mittel). |
| **Shchur et al. 2018** (arXiv 1811.05868) | Zeigt, dass die Rangfolge vom Split abhängt, nicht, dass eine Zahl nicht hält. GCN 81,5 ± 1,3 und GAT 81,8 ± 1,3 stehen schon als Einheiten im Benchmark (reproduziert). |
| **MixHop** (arXiv 1905.00067 v3) | Protokoll: „we run all models 100 times, sort by the validation accuracy, and finally report the test accuracy for the top 50 runs". Systematisch optimistischer als ein Mittel, aber keine Neumessung gefunden. |
| **APPNP** (arXiv 1810.05997 v6) | Eigener Split (Cora-ML), kein Planetoid; Zahl 85,09 hängt am eigenen Protokoll. Kein Widerspruch dokumentiert. |
| **ML Reproducibility Challenge / ReScience** | Websuche liefert Reproduktionsberichte zu Erklär- und Fairness-Methoden, nichts zu Cora-Genauigkeit eines Klassifikators mit belegter Abweichung. Nichts Belastbares. |
| **DGI** in unserem Benchmark (Referenz 0,8047 gegen 0,823: teilweise) | Eigene Referenz, keine fremde Quelle; kein Beleg. Könnte nur als „intern beobachtet" erwähnt werden. |
| **SparseTSF/FITS-Nachrechnung** (SparseTSF Tabelle 11: FITS 0,382 gegen 0,372 im FITS-Paper) | Abstand 0,010, andere Konfiguration (L=720, COF=5); kein klarer Widerspruch. |

Suchanfragen (WebSearch): (1) „cannot reproduce Cora accuracy GitHub issue paper reported 85.5 OR 83.0 ...", (2) „reproducibility study graph neural network Cora node classification could not reproduce ... ML Reproducibility Challenge", (3) „ETTh1 reported MSE inflated drop_last bug re-evaluation PatchTST DLinear ...". Danach gezielt: FITS-README (raw.githubusercontent.com), SparseTSF v2 Anhang D.1, TFB v4, PatchTST-Paper, GAT-Issue #19 über die GitHub-API. Nicht gesucht: OpenReview-Kommentare, Papers-with-Code-Diskussionen, weitere Repos (PatchTST-Issues).

## Empfehlung

- **Kein harter Negativfall vorhanden.** Im README ehrlich: „no real negative case; negative control = placebos".
- **PatchTST ETTh1-96 (Ziel 0,370)** als „Messfehler-Fall" aufnehmen, wenn der Reviewer einen Fall mit belegtem Anlass will. Erwartetes Urteil teilweise; Nutzen: zeigt, dass Nullius eine durch einen dokumentierten Pipeline-Fehler überhöhte Zahl nicht bestätigt. Vor Aufnahme klären: (a) Look-back-Konfiguration der Neuläufe (FITS-Skripte `scripts/` lesen), (b) PatchTST-Repository auf Korrektur prüfen, (c) Referenzlauf.
- **Nächste Suche, falls Zeit:** OpenReview-Reviews/Kommentare zu PatchTST, TFB-Ergebnistabellen (Github `decisionintelligence/TFB`) für ETTh1-96 mit gleichen Look-backs, Issues im LTSF-Linear-Repository zu `drop_last`, CORA-bezogene Reproduktionen in „Reproducibility and Geometric Intrinsic Dimensionality" (arXiv 2403.08438, nur als Fundstelle der Websuche, nicht gelesen).
