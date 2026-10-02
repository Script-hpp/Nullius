# Vorversuch Nachlauf, Teil „Schritt 1, lokal“ (Auftrag 33, 2026-10-02)

Handarbeit, lokal auf CPU (16 Kerne), kein Agent, kein LLM, kein Tavily, keine Sandbox. Vorab-Festlegung: `docs/prestudy_nachlauf.md` (final). Skripte und genaue Befehle: `scripts/prestudy_nachlauf/` (README). Rohausgaben: `nachlauf_prestudy_rlinear.txt`, `nachlauf_prestudy_dlinear.txt`, `nachlauf_prestudy_rlinear_literal.txt`. Diese Zahlen sind kein Agenten-Ergebnis und zählen in keinem Benchmark. Ich fälle keine Entscheidung „bauen oder nicht“ (das prüft der Reviewer in Schritt 2).

## Kurzfassung

| | RLinear (RTSF) | DLinear (LTSF-Linear) |
|---|---|---|
| Commit | `0fec00104f75…` | `0c113668a3b8…` |
| Start wie dokumentiert lauffähig? | **Nein** (`script.md:3`: `--model Linear` → `KeyError: 'Linear'`); mit `--model RLinear` ja | Ja, `etth1.sh` Zeile 96 |
| Abhängigkeit, die nicht zur Repo-Angabe passt | numpy 2 scheitert (`np.Inf`), nötig: `numpy<2` | keine Änderung nötig (`requirements.txt` nennt `torch==1.9.0`, lief mit torch 2.14.1+cpu, numpy 1.26.4) |
| Zuordnungsprüfung (2785 × 96 × 7) | bestanden, max. Abweichung 1,5e-14 | bestanden, max. Abweichung 1,5e-14 (nach Adapter-Fix, siehe unten) |
| Repo-Datei geändert | nein | nein |
| Eigenausgabe Repo-Seed (Repo-Test-MSE) | **0,3660** (Seed 1024) | **0,3841** (Seed 2021) |
| Harness, alle Fenster, Repo-Seed | 0,3698 | 0,3841 |
| Harness `drop_last`-Wert, Repo-Seed | 0,3660 (Batch 128) | 0,3841 (Batch 32) |
| Abstand Harness-`drop_last` zu Repo-Ausgabe (alle 5 Seeds) | ≤ 0,00004 | ≤ 0,00007 |
| Harness alle Fenster, 5 Seeds: Mittel ± Std (ddof=1) | **0,3708 ± 0,0013** | **0,3781 ± 0,0082** |
| Zielwert (Tabelle 2) / Abstand Mittel | 0,366 / 0,0048 | 0,375 / 0,0031 |
| Urteil nach bestehender Regel (Toleranz 0,01) | reproduziert | reproduziert |
| Laufzeit je Seed lokal (Training + Test) | 23 bis 44 s | 21 bis 33 s |

## Gemeinsame Angaben

- Daten: `ETTh1.csv` von der Quelle in `harness/etth1_data.py` (`raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv`), 17420 Datenzeilen, SHA-256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; identische Datei in beiden Klonen (`dataset/`).
- Umgebung: Python 3.11.16 (uv venv), torch 2.14.1+cpu, numpy 1.26.4, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2. Volle Liste: `prestudy_work/pkgs_freeze.txt` (Kopfzeilen in den Rohdateien). **Versionsänderung, nicht still:** numpy wurde von 2.4.6 auf 1.26.4 gesenkt, weil `utils/tools.py:38` in RTSF `np.Inf` nutzt (in numpy 2.0 entfernt). DLinear lief nur mit 1.26.4 (mit numpy 2 nicht getestet).
- CPU: beide Repos wählen `args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False`. Ohne CUDA fällt der Code selbst auf CPU (`Use CPU` in der Ausgabe). `--use_gpu` hat zwar `type=bool`, wird aber nicht gesetzt (Default True), `--gpu 3` ist ohne GPU wirkungslos. Kein Eingriff nötig.
- Klone unverändert: `git status --short` und `git diff --stat` in beiden Klonen leer (Kopfzeilen der Rohdateien). Die Läufe legen nur ignorierte Pfade an (`dataset/`, `checkpoints/`, `test_results/`, `results/`, `__pycache__/`, `result.txt`, `logs/`), alle in den `.gitignore` der Repos; `ETTh1.csv` liegt in `dataset/`.
- Seed-Kopien: `scripts/prestudy_nachlauf/seed_copies/{rtsf_run,ltsf_run}_seed{0..3}.py`, Diff gegen das Original je Kopie genau eine Zeile:
  - RTSF: `8c8  < fix_seed = 1024  ---  > fix_seed = N`
  - LTSF: `8c8  < fix_seed = 2021  ---  > fix_seed = N`
  Der Repo-Seed-Lauf nutzte das unveränderte Startskript.
- Std ist die Stichproben-Std (ddof=1) über 5 Seeds. Das Urteil beruht auf dem Mittel.

## RLinear (RTSF, `plumprc/RTSF`, `0fec00104f754f4fbf795b9b4da5fa2459b32e76`)

Startbefehl (cwd Klon): `python -u run.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --channel 7 --itr 1 --batch_size 128 --learning_rate 0.005 --model RLinear --rev --seq_len 336 --pred_len 96 --gpu 3`, einzige Abweichung von `script.md:3`: `RLinear` statt `Linear` (Modellname; kein Hyperparameter). Repo-Seed `fix_seed = 1024`.

Ausgabe des Autorencodes: `mse:0.3660, mae:0.3909` (Repo-Seed). Test-Loader des Repos: `drop_last = True`, Batch 128 (`data_factory.py:16-18`): die Ausgabe beruht auf 2688 von 2785 Fenstern.

| Seed | Repo-Ausgabe | Harness alle 2785 | Harness `drop_last` 128 (2688) | \|drop_last − Repo\| |
|---|---|---|---|---|
| 1024 (Repo) | 0,3660 | 0,3698 | 0,3660 | 0,00004 |
| 0 | 0,3691 | 0,3729 | 0,3691 | 0,00004 |
| 1 | 0,3665 | 0,3704 | 0,3665 | 0,00003 |
| 2 | 0,3659 | 0,3698 | 0,3659 | 0,00004 |
| 3 | 0,3671 | 0,3710 | 0,3671 | 0,00003 |
| **Mittel ± Std** | 0,3669 | **0,3708 ± 0,0013** | 0,3669 | |

- Zuordnungsprüfung: Zielwerte des Repo-Testsatzes (`Dataset_ETT_hour`, `flag='test'`, ohne `drop_last`, ohne Mischen) gegen `etth1_data.targets`: Form (2785, 96, 7) beide, größte absolute Abweichung **1,5e-14** (Grenze 1e-4). Bestanden, bei jedem der fünf Seeds gleich.
- Selbstprüfung des Adapters (Punkt 4 der Entscheidungsregel, lokal): Der `drop_last`-128-Wert des Harness trifft die Eigenausgabe des Repos auf ≤ 0,00004 (Ziel ≤ 0,001) für alle fünf Seeds, Repo-Seed: 0,36596 gegen 0,3660.
- Befund zu `b1_candidates.md`: Die dort **nachgebildete** Erwartung (alle Fenster ca. 0,370, mit `drop_last` ca. 0,3665) ist hier mit dem Lauf des Autorencodes bestätigt: alle Fenster 0,3698 (Seed 0 bis 4 Mittel 0,3708), `drop_last` im Repo-Seed 0,3660 und damit **identisch zum Paperwert 0,366**; Mittel der `drop_last`-Werte über die fünf Seeds 0,3669. Die Referenzlösung (`b1_reference_rlinear.txt`, 0,3704 ± 0,0008) liegt 0,0004 unter unserem Nachlauf-Mittel. Das stützt die Aussage, dass der Paperwert unter dem `drop_last`-Test-Loader entstand. Ich habe `b1_candidates.md` nicht verändert (Auftrag: nur diese Ergebnisdatei); eine datierte Ergänzung dort ist Sache des Reviewers.
- Urteil nach bestehender Regel: |0,3708 − 0,366| = 0,0048 ≤ 0,01 → reproduziert (auf allen Fenstern). Einzelne Seeds: Abstand 0,0038 bis 0,0069.
- Laufzeit je Seed (wall, Training + Test, 16 Threads, Repo druckt 15 Epochen, kein Early Stopping): 43,7 s (Seed 1024, erster Lauf, kalter Start), 22,7 / 23,8 / 35,8 / 24,3 s (Seeds 0 bis 3). Adapter plus Harness je Seed zusätzlich ca. 8 s. Mit weniger Kernen in der Sandbox ist die Laufzeit unbekannt (Schritt 2).

## DLinear (LTSF-Linear, `cure-lab/LTSF-Linear`, `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`)

Startbefehl: Zeile für Horizont 96 aus `scripts/EXP-LongForecasting/Linear/etth1.sh` (`seq_len=336`, `model_name=DLinear`): `python -u run_longExp.py --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id ETTh1_336_96 --model DLinear --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --des 'Exp' --itr 1 --batch_size 32 --learning_rate 0.005` (nur diese Zeile, nicht das ganze Skript; die Log-Umleitung des Skripts entfällt). Repo-Seed `fix_seed = 2021`.

Ausgabe des Autorencodes: `mse:0.3841443955898285, mae:0.40471312403678894`. Test-Loader des Repos: `drop_last = False` (`data_factory.py:20` in diesem Commit), also alle 2785 Fenster; die Repo-Ausgabe ist damit direkt mit dem Harness-Wert „alle Fenster“ vergleichbar.

| Seed | Repo-Ausgabe | Harness alle 2785 | Harness `drop_last` 32 (2784) | \|alle − Repo\| |
|---|---|---|---|---|
| 2021 (Repo) | 0,3841 | 0,3841 | 0,3841 | < 1e-7 |
| 0 | 0,3895 | 0,3895 | 0,3895 | < 1e-7 |
| 1 | 0,3739 | 0,3739 | 0,3739 | < 1e-7 |
| 2 | 0,3717 | 0,3717 | 0,3717 | < 1e-7 |
| 3 | 0,3713 | 0,3713 | 0,3713 | < 1e-7 |
| **Mittel ± Std** | 0,3781 | **0,3781 ± 0,0082** | 0,3781 | |

- Zuordnungsprüfung: Form (2785, 96, 7), größte absolute Abweichung **1,5e-14**. Bestanden. **Haken:** Der Loader dieses Repos liefert `batch_y` mit `label_len + pred_len = 144` Schritten (`label_len` Default 48); der erste Adapterlauf schlug die Prüfung deshalb an (`repo_shape [2785, 144, 7]`, kein Schreiben von Vorhersagen, Ausgabe im Lauf-Log). Das Repo schneidet selbst `[-pred_len:]` (`exp_main.py:260-261`); der Adapter tut dasselbe (eine Zeile). Das war ein Adapterfehler, keine abweichende Datenaufbereitung: nach dem Schnitt stimmen die Labels bis 1,5e-14.
- Selbstprüfung des Adapters: Harness „alle Fenster“ trifft die Eigenausgabe des Repos auf < 1e-7 (gleicher Seed), der `drop_last`-32-Wert (nur 1 Fenster weniger) auf ≤ 0,00007. Ziel ≤ 0,001 erreicht.
- Abstand zum Ziel: Repo-Seed 0,3841, **0,0091 über dem Paperwert 0,375**, knapp innerhalb der Toleranz 0,01. Die Seeds streuen stark (0,3713 bis 0,3895); Mittel 0,3781, Abstand 0,0031 → reproduziert nach bestehender Regel. Der einzelne Repo-Seed wäre allein ebenfalls „reproduziert“ (≤ 0,01), aber nur knapp. Referenzlösung (`benchmark_v2.md`): 0,3809; unser Nachlauf-Mittel liegt 0,0028 darunter.
- Hinweis zur Einordnung: LTSF-Linear trainiert bis Early Stopping (Patience 3); Seed 2021 stoppte nach 7 Epochen. Das ist Verhalten des Repos, nichts wurde getunt.
- Laufzeit je Seed (wall, 16 Threads): 32,8 s (Seed 2021), 21,4 / 26,4 / 31,1 / 27,8 s (Seeds 0 bis 3); Adapter plus Harness je Seed ca. 8 s.

## Adapter (Frage 3)

| Datei | Zeilen | Inhalt |
|---|---|---|
| `adapter_rtsf.py` | 60 | Checkpoint laden, Loader, Zuordnungsprüfung, Schreiben |
| `adapter_ltsf.py` | 60 | dito |
| `common.py` | 71 | Zuordnungsprüfung, `predictions.csv`-Schreiber, Harness-Aufruf (für beide gleich) |
| `evaluate.py` | 9 | Harness-Aufruf |

- Zwischen den beiden Adaptern sind **49 von 60 Zeilen identisch** (`diff` zeigt 11 geänderte Zeilen: 2 Docstring-/Aufrufzeilen, Standard-Batchgröße, Modell-Import, Namespace, Modellaufruf, Dataset-Aufruf und die `[-pred_len:]`-Zeile; ca. 6 davon sind Code). `common.py` und `evaluate.py` sind komplett gleich (80 weitere Zeilen).
- Was man über das Repo wissen musste: (1) Modellklasse und ihr Importpfad (`models.RLinear.Model` bzw. `models.DLinear.Model`), (2) welche Konfigurationsfelder die Modellklasse liest (RTSF: `seq_len, pred_len, channel, individual, drop, rev`; LTSF: `seq_len, pred_len, enc_in, individual`), (3) Signatur der Datenklasse `Dataset_ETT_hour` (RTSF `size=[seq, pred]`, LTSF `size=[seq, label, pred]` plus `train_only`), (4) Aufruf des Modells (`model(x, y)` gibt `(pred, loss)` zurück, bzw. `model(x)`), (5) Form von `batch_y` (LTSF: `label_len` vorne abschneiden), (6) Pfad des Checkpoints (`checkpoints/<setting>/checkpoint.pth`), (7) dass `drop_last` im Test-Loader eines Repos schon im Repo steht (RTSF `True`, LTSF `False`). Punkte 1 bis 5 sind genau die Stellen, an denen ein LLM-Adapter bei einem fremden Repo scheitern könnte.
- Beide Repos stammen aus derselben Autoformer-Familie; das erklärt den hohen Anteil gleicher Zeilen. Zu Graph-Repos sagt das nichts.
- LTSF schreibt selbst `results/<setting>/pred.npy` (`exp_main.py:300`, alle 2785 Fenster, Form 2785×96×7 in standardisierten Einheiten); das wäre eine adapterlose Quelle für DLinear, RTSF schreibt keine Vorhersagen. Nicht benutzt, nur als Beobachtung.

## Eingriff (Frage 4) und Hakeln (Arbeitszeit)

Keine Dateiänderung in den Klonen nötig, nur Kommandozeile, Seed-Kopien außerhalb des Klons und die neuen Adapterdateien. Aber: die Startbefehle wurden **nicht** in jedem Fall unverändert übernommen (RTSF `--model RLinear`, numpy-Version).

Arbeitszeit (Schätzung, nicht mit Stoppuhr gemessen; Wanduhr für Läufe siehe oben):
- RTSF: ca. 25 min Handarbeit. Hakeln: (1) `script.md` nennt `--model Linear`, der Code kennt nur `RLinear`, `RMLP`, `STD` (`exp_main.py:21-25`): `KeyError`, durch Lesen des Codes gelöst. (2) `np.Inf` unter numpy 2 (`AttributeError`): numpy auf 1.26.4 gesenkt. (3) Zuordnungsprüfung und Harness liefen auf Anhieb. (4) Eigenes Hakeln: der Adapter chdirt ins Repo, relative Pfade brachen einmal.
- DLinear: ca. 15 min. Hakeln: (1) `batch_y` mit 144 statt 96 Schritten (Zuordnungsprüfung schlug an, Adapter-Zeile ergänzt, Checkpoints neu ausgewertet, nicht neu trainiert; der Runner `run_ltsf_seed.sh` enthält die Korrektur). (2) sonst nichts; `requirements.txt` mit `torch==1.9.0` musste nicht erfüllt werden.
- Unabhängig vom Autorencode nur eine Hürde im Harness-Aufruf: `etth1_eval.py` hat den Pfad `/output/predictions.csv` fest eingebaut und lädt ETTh1 selbst; lokal wird nur `open()` für diesen einen Pfad umgebogen (`common.run_harness`), die Datei selbst läuft unverändert per `runpy`. In der Sandbox gibt es `/output/`, dort entfällt das.

## Eigene Einschätzung zum Leak-Schutz

Welche Dateien liegen im Nachlauf-Lauf vor, die Testwerte enthalten:
1. `dataset/ETTh1.csv` komplett, also auch die Testzeilen 11520 bis 14400. Der Autorencode liest sie über seinen Test-Loader (`flag='test'`); das ist beim Original-Code nicht vermeidbar (Schicht 3 „Testlabels nie im Agenten-Datensatz“ entfällt, wie die Vorab-Festlegung sagt).
2. Der Autorencode berechnet in beiden Repos den Test-Verlust in **jeder** Epoche (RTSF `exp_main.py:122`, LTSF `exp_main.py:191`: `test_loss = self.vali(..., test_loader, ...)`, nur zum Ausdrucken), Early Stopping und Checkpoint-Wahl nutzen aber nur den Validierungsverlust. Das Standardisieren nutzt in beiden nur Trainingszeilen. Der Autorencode wählt also nicht auf dem Test aus.
3. LTSF schreibt `pred.npy` (Vorhersagen), nicht die Labels (`true.npy` ist auskommentiert); die PDFs in `test_results/` enthalten einzelne Testfenster.

Reicht „fester Commit, unveränderte Repo-Dateien, geprüfter Adapter“? Aus dem Vorversuch: **für den Autorencode ja, für den Adapter nicht ohne weitere Maßnahmen.**
- Der Autorencode ist durch Commit-Hash und `git status` gedeckt, und die Diff-Prüfung ist trivial (beide Klone leer).
- Der Adapter ist die einzige Stelle, an der Code läuft, den kein Mensch geschrieben hat (im Modus wäre er LLM-generiert), und er **muss** Zugriff auf die Testdaten haben (er ruft den Test-Loader, gerade das lieferte hier die Zuordnungsprüfung). Ein Adapter könnte statt `model(x)` die Labels als Vorhersage schreiben. Der Harness würde das nicht ablehnen: die Plausibilitätsprüfungen (vollständig, eindeutig, nicht konstant) bestehen, und ein MSE von 0 sieht der Harness nicht als „ungültig“. Die Zuordnungsprüfung vergleicht nur Labels des Repo-Loaders mit Harness-Labels, nicht die Vorhersagen.
- Die Selbstprüfung (Punkt 4, `drop_last`-Wert gegen die Eigenausgabe des Repos) hilft hier tatsächlich: sie vergleicht die Harness-Auswertung der Adapter-Vorhersagen mit der MSE, die der Autorencode selbst ausgibt (4e-5 Abstand bei RLinear). Ein Label-kopierender Adapter fiele dort sofort auf (MSE ≈ 0 gegen 0,366). Sie braucht dafür den Eigenwert aus der Ausgabe des Repos (Parsen der stdout; das Format der Repos ist verschieden: `mse:0.3660, …`), und sie greift nicht, wenn ein Repo keine eigene Test-MSE ausgibt.
- Weitere naheliegende Maßnahmen (nur Vorschlag, nicht gebaut oder geprüft): Adapter ohne Netz und ohne Schreibzugriff außer `/output`; Abgleich der Vorhersagen mit einer zweiten Rechnung des Modells durch den Harness (z. B. Eingaben leicht verändern, Vorhersage muss sich ändern); statische Prüfung, dass der Adapter die Labelspalte des Loaders (`batch_y`) nicht in die Ausgabe leitet (ist bei meinem Adapter nur für die Zuordnungsprüfung nötig und könnte vom Harness getrennt werden, indem der Adapter nur Eingaben bekommt).
- Offen für Schritt 2: Lässt sich in der Sandbox das Training vom Adapter trennen (Training unter `unshare -n`, Adapter in einem zweiten Prozess ohne Zugriff auf Testlabels, nur auf Eingabefenster)? Das wäre die stärkste Maßnahme, ich habe sie nicht gebaut.

## Offene Punkte / Grenzen

- Nur lokal, nur 16 Kerne und 16 Threads; keine Aussage zu Sandbox-Laufzeit oder Snapshot-Größe (Schritt 2). Größe der Python-Umgebung lokal: 1,2 GB (venv mit torch CPU und Zubehör), Checkpoints und Daten klein (Checkpoints 1,9 MB gesamt, Daten 2,5 MB).
- Sandbox-Image `python:3.11-slim` hat numpy und torch vorinstalliert (neueste Versionen, numpy 2 wahrscheinlich): RTSF braucht dort `numpy<2` im Snapshot, das ist eine Änderung an der Umgebung, nicht an Repo-Dateien.
- Die Zahlen gelten für diese beiden Repos auf diesen Commits mit torch 2.14.1 statt `torch==1.9.0`.
