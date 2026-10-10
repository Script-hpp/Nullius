# Reproduktions-Zeugnis: disc_2405_00946

## Hinweise zur Belegbarkeit
- Tabellenspalte nicht geprueft, Zahl koennte zu einer Vergleichsmethode gehoeren.
- Rolle 2 konnte den Zielwert im Paper nicht gegenpruefen (Modellantwort enthaelt kein JSON); es gilt der Wert der Discovery 0.359.

## Claim disc_2405_00946_mse: reproduziert
Gemessen: 0.3653 (std 0.0034)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
The paper's number matches our measurement almost exactly if the last 225 of 2785 test windows are skipped, as a data loader with drop_last=True does at batch size 256 (the batch size of the original SparseTSF ETTh1 script). With all 2785 windows we measure 0.3653, which is still within tolerance of the claimed 0.3590. With the last windows skipped we measure 0.3573. This is an observation, not a criticism.
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0034. Mit unserem Setup gemessen (3 von 3 Seeds, Zielwert 0.359 +/- 0.01, discovery: https://arxiv.org/abs/2405.00946, Zitat: \multirow{2}{*}{SparseTSF (ours)} & \textbf{0.359} & \textbf{0.397} & \textbf{0.404} & \textbf{0.417}): Mittel 0.3653. Code-Kandidaten: 3, gewaehlt Kandidat 0 (Code ba8d6b237495) nach Validierungs-Metrik (0: mse 0.7120, 1: mse 0.7292, 2: mse 2.0162); Rueckfall ohne Val-Wert; Repo-Abgleich: keine geprueften Abweichungen
Zielwert 0.359: discovery: https://arxiv.org/abs/2405.00946, Zitat: \multirow{2}{*}{SparseTSF (ours)} & \textbf{0.359} & \textbf{0.397} & \textbf{0.404} & \textbf{0.417}
Quellen:
- https://arxiv.org/abs/2405.00946

## What others found (Tavily Rolle 3, nur Bericht, nicht Teil des Urteils)
- [protocol note] Dropping the last incomplete batch during testing leads to unfair comparisons unless all methods use identical batch sizes. — "Unless all methods use the same batch size, discarding the last batch of test samples is unfair, as the actual usage length of the test set is inconsistent." (https://github.com/decisionintelligence/TFB)
- [reproduction report] The authors acknowledge a long‑standing bug in the code framework that affected results, implying prior reported numbers may not be reproducible. — "🚩 News (2024.07): We have now fixed a long-standing bug (see description in FITS and TFB ) in the code framework and supplemented the full results (including MSE and MAE) of SparseTSF after fixing the bug in this table." (https://github.com/lss-1138/SparseTSF)
- [protocol note] The paper’s evaluation may be flawed because a bug caused the last batch of test data to be dropped, potentially affecting reported performance. — "We recently fixed a longstanding bug in this code framework (see description in TFB), which caused the last batch of data to be dropped during the testing phase." (https://github.com/lss-1138/SparseTSF/issues/10)
- [reproduction report] Reproduction attempts show significant discrepancies for sequence lengths 336 and 720, questioning the reproducibility of the reported results. — "when I used the command sh scripts/SparseTSF/etth1.sh to reproduce the results from the paper, I found that the predictions with sequence lengths of 336 and 720 differ significantly from the results reported in the paper." (https://github.com/lss-1138/SparseTSF/issues/10)
- [other] The evaluation bug affected the test results on ETTh1 and ETTh2, raising doubts about the reported numbers. — "After the paper was accepted, we noticed the evaluation bug you mentioned, which primarily affected the test results on the ETTh1 and ETTh2 datasets." (https://github.com/lss-1138/SparseTSF/issues/9)
- [reproduction report] Showing buggy results instead of the actual metrics is disingenuous and harms reproducibility. — "I find this borderline disingenuous, you should have presented the actual metrics instead of the buggy one, regardless of your messaging." (https://github.com/lss-1138/SparseTSF/issues/9)
Abfrage: 2026-10-09T20:17:36+00:00 (live), 12 Seiten gelesen, 2 Aussagen verworfen (Zitat nicht im Text).

Modus: nachbau_autorencode. Rebuild from paper and authors' code: the agent reads the authors' code as text and writes its own code. The authors' code is never executed; the number is measured by our harness.
Repo gelesen: https://github.com/lss-1138/SparseTSF @ b8c2740eec

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] input length: 720 — "seq_len=720"
- [repo:README.md@b8c2740] input length: Hard to model long periods without extending the input length (solved by CycleNet) — "Hard to model long periods without extending the input length **(solved by CycleNet)**"
- [repo:models/Transformer.py@b8c2740] input length: seq_len — "self.seq_len = configs.seq_len" (not in the plan: other model (file name Transformer.py))
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] learning_rate: 0.02 — "--learning_rate 0.02"
- [repo:exp/exp_main.py@b8c2740] learning rate: lr=self.args.learning_rate — "model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)"
- [paper] learning-rate schedule: initial learning rate 0.02 — "Additionally, the learning rate needs to be set relatively large (i.e., 0.02) due to the very small number of learnable parameters in SparseTSF."
- [paper] batch size: 256 — "For datasets with fewer than 100 channels (such as ETTh1), the batch size is set to 256"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] batch size: 256 — "--batch_size 256"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] epochs: 30 — "--train_epochs 30 \"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] patience: 5 — "--patience 5 \"
- [repo:utils/tools.py@b8c2740] patience: 7 — "def __init__(self, patience=7, verbose=False, delta=0):"
- [repo-readme] other settings: period_len can be set to [2-6] when the period length is excessively large — "Alternatively, you can set it to [2-6] when the period length is excessively large, as mentioned in the paper."
- [repo-readme] other settings: run the script sh scripts/SparseTSF/linear/etth1.sh to obtain results on ETTh1 — "sh scripts/SparseTSF/linear/etth1.sh;"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] other settings: model_name=SparseTSF — "model_name=SparseTSF"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] other settings: data_name=ETTh1 — "data_name=ETTh1"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] other settings: root_path_name=./dataset/ — "root_path_name=./dataset/"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] other settings: data_path_name=ETTh1.csv — "data_path_name=ETTh1.csv"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] other settings: model_id_name=ETTh1 — "model_id_name=ETTh1"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] other settings: period_len=24 — "--period_len 24"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] other settings: enc_in=7 — "--enc_in 7"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] other settings: features=M — "--features M"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] other settings: train_epochs=30 — "--train_epochs 30"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] other settings: patience=5 — "--patience 5"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] other settings: itr=1 — "--itr 1"
- [repo:scripts/SparseTSF/linear/etth1.sh@b8c2740] other settings: is_training=1 — "--is_training 1"
- [repo:data_provider/data_factory.py@b8c2740] other: timeenc set based on embed argument (0 unless timeF) — "timeenc = 0 if args.embed != 'timeF' else 1"
- [repo:data_provider/data_factory.py@b8c2740] other: test phase: no shuffle, no drop_last, batch_size = args.batch_size — "if flag == 'test':
        shuffle_flag = False
        drop_last = False  # fix bug
        batch_size = args.batch_size
        freq = args.freq"
- [repo:data_provider/data_factory.py@b8c2740] other: prediction phase: batch_size fixed to 1, uses Dataset_Pred — "elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred"
- [repo:data_provider/data_factory.py@b8c2740] other: training/validation: shuffle enabled, drop_last enabled, batch_size = args.batch_size — "else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq"
- [repo:data_provider/data_factory.py@b8c2740] other: dataset instantiated with seq_len, label_len, pred_len, features, target, timeenc, freq — "data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )"
- [repo:data_provider/data_factory.py@b8c2740] other: DataLoader uses batch_size, shuffle flag, num_workers from args, and drop_last — "data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)"
- [repo:README.md@b8c2740] other settings: period_len = [2-6] — "Alternatively, you can set it to [2-6] when the period length is excessively large, as mentioned in the paper."
- [discovery] input length: 720 — "\multirow{2}{*}{SparseTSF (ours)} & \textbf{0.359} & \textbf{0.397} & \textbf{0.404} & \textbf{0.417}"

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch