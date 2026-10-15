# Reproduktions-Zeugnis: disc_2307_03756

Code-Auswahl und Reparaturen: Validierung und strukturelle Pruefungen. Testmessungen steuerten keine Code-Aenderung.

Selection protocol withheld_validation_v1: 21 of 117 validation windows/nodes were withheld from fitting and used for candidate selection. Checkpoints used the remaining visible validation data. Original training/test splits unchanged. This selection protocol differs from older runs.

Forecast input contract causal_stream_v2: histories exposed chronologically after fitting, 8 repeated-history spot checks. Method fidelity remains unchecked.

Past calendar features: year, month, day, weekday, hour. No future measured values exposed.

## Hinweise zur Belegbarkeit
- Tabellenspalte nicht geprueft, Zahl koennte zu einer Vergleichsmethode gehoeren.
- Independent candidate selection uses 21 of 117 validation windows/nodes with hidden answers. Checkpoints use the remaining visible validation data. Training/test splits are unchanged; this selection protocol differs from older runs.

## Claim disc_2307_03756_mse: reproduziert
Gemessen: 0.3720 (std 0.0001)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0001. Mit unserem Setup gemessen (3 von 3 Seeds, Zielwert 0.372 +/- 0.01, tab:FITSett, arXiv v3 (latex)): Mittel 0.3720. Code-Kandidaten: 3, gewaehlt Kandidat 0 (Code de18d490e610) nach Validierungs-Metrik (0: mse 0.4697, 1: mse 0.4697, 2: mse 0.4703); Repo comparison incomplete: files were truncated or findings could not be verified. Method fidelity remains unchecked.; Independent selection used the final 20% of validation observations, withheld from fitting; checkpoint selection used the remaining visible validation data. Original train/test splits unchanged.; Forecast histories were streamed chronologically to an unprivileged predictor after fitting; each forecast was recorded before the next history was exposed.
Zielwert 0.372 belegt: "FITS 0.372 0.404 0.427 0.424 0.271 0.331 0.354 0.377 0.303 0.337 0.366 0.415 0.162 0.216 0.268 0.348" (tab:FITSett, https://export.arxiv.org/src/2307.03756v3, Version v3, abgerufen 2026-10-14T08:05:42+00:00, SHA-256 9022b12b628f)
Spaltenzuordnung: an Kopfzeile geprüft (Position von Methode und Metrik stimmt).
Quellen:
- https://export.arxiv.org/src/2307.03756v3

## What others found (Tavily Rolle 3, nur Bericht, nicht Teil des Urteils)
- [protocol note] The test dataloader uses drop_last=True, which may exclude a significant portion of test data and lead to unfair model comparisons. — "Specifically, the test dataloader uses drop_last=True, which may exclude a significant portion of test data, particularly with large batch sizes, leading to unfair model comparisons." (https://github.com/VEWOXIC/FITS) [Repo-Abgleich: widerlegt]
- [other] The reported results are still preliminary and the authors are rerunning parameter search, ablation studies and multi-runs for final results. — "This is still preliminary results for FITS. We are rerunning the parameter search, ablation study and multi-runs for the final results." (https://github.com/VEWOXIC/FITS)
- [protocol note] Comparing FITS to other works may be compromised because those works potentially leak information from the test set when selecting thresholds. — "we still compare FITS with the results reported in their original paper which may have potential information leakage." (https://github.com/VEWOXIC/FITS)
Abfrage: 2026-10-13T17:08:14+00:00 (cache), 4 Seiten gelesen, 3 Aussagen verworfen (Zitat nicht im Text).

## Objections checked (Tavily role 3 -> own experiments, nur Bericht, nicht Teil des Urteils)
- "Specifically, the test dataloader uses drop_last=True, which may exclude a significant portion of test data, particularly with large batch sizes, leading to unfair model comparisons." (https://github.com/VEWOXIC/FITS) -> same predictions, test windows as a drop_last=True loader keeps them: largest change 0.0072 (batch size 32: 0.3720 on 2784 of 2785 windows vs 0.3720 on all, 3 seeds) -> bestaetigt
- "This is still preliminary results for FITS. We are rerunning the parameter search, ablation study and multi-runs for the final results." (https://github.com/VEWOXIC/FITS) -> not measurable: Quote notes results are preliminary and will be updated after further experiments, without mentioning any specific varying factor or providing a measured number. -> nicht pruefbar
- "we still compare FITS with the results reported in their original paper which may have potential information leakage." (https://github.com/VEWOXIC/FITS) -> not measurable: The quote raises a concern about potential information leakage affecting fair comparison, which does not correspond to any of the measurable objection types. -> nicht pruefbar
bestaetigt/widerlegt gilt nach der vorab festgelegten Regel auf diesem Task; die Autorin des Einwands kann ein anderes Setup benutzt haben.

Modus: nachbau_autorencode. Rebuild from paper and authors' code: the agent reads the authors' code as text and writes its own code. The authors' code is never executed; the number is measured by our harness.
Repo gelesen: https://github.com/VEWOXIC/FITS @ d040bb015b

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [repo:scripts/FITS/etth1_best.sh@d040bb0] input length: 360 — "for seq_len in 360"
- [repo:scripts/FITS/etth1_best.sh@d040bb0] learning rate: 0.0005 — "--learning_rate 0.0005"
- [repo:run_longExp_F.py@d040bb0] learning rate: 0.0001 — "parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')"
- [repo:scripts/FITS/etth1_best.sh@d040bb0] learning-rate schedule: 0.0005 — "python -u run_longExp_F.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --train_mode $m \
  --H_order $H_order \
  --gpu 0 \
  --seed $seed \
  --patience 20 \
  --itr 1 --batch_size $bs --learning_rate 0.0005 | tee logs/FITS_fix/etth1_abl/$m'_'$model_name'_'Etth1_$seq_len'_'96'_H'$H_order'_bs'$bs'_s'$seed.log"
- [repo:run_longExp_F.py@d040bb0] learning-rate schedule: type3 — "parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')"
- [repo:exp/exp_main_F.py@d040bb0] learning-rate schedule: adjust_learning_rate(model_optim, epoch + 1, self.args) — "adjust_learning_rate(model_optim, epoch + 1, self.args)"
- [repo:README.md@d040bb0] batch size: 128 — "- ~~**FITS benefits from large batch size. Our latest version uses batch size of 128.** Some results are not updated due to limited time.~~"
- [repo:run_longExp_F.py@d040bb0] epochs: 50 — "parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')"
- [repo:exp/exp_main.py@d040bb0] epochs: self.args.train_epochs — "for epoch in range(self.args.train_epochs):"
- [repo:scripts/FITS/etth1_best.sh@d040bb0] patience: 20 — "--patience 20 \"
- [repo:run_longExp_F.py@d040bb0] patience: 3 — "parser.add_argument('--patience', type=int, default=3, help='early stopping patience')"
- [repo:exp/exp_main_F.py@d040bb0] patience: self.args.patience — "train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()"
- [repo:utils/tools.py@d040bb0] patience: 7 — "def __init__(self, patience=7, verbose=False, delta=0):"
- [repo:models/FITS.py@d040bb0] other settings: seq_len — "self.seq_len = configs.seq_len"
- [repo:models/FITS.py@d040bb0] other settings: pred_len — "self.pred_len = configs.pred_len"
- [repo:models/FITS.py@d040bb0] other settings: individual — "self.individual = configs.individual"
- [repo:models/FITS.py@d040bb0] other settings: enc_in (channels) — "self.channels = configs.enc_in"
- [repo:models/FITS.py@d040bb0] other settings: cut_freq (dominance_freq) — "self.dominance_freq=configs.cut_freq # 720/24"
- [repo:models/FITS.py@d040bb0] other settings: length_ratio — "self.length_ratio = (self.seq_len + self.pred_len)/self.seq_len"
- [repo:scripts/FITS/etth1_best.sh@d040bb0] other_settings: patience=20 — "--patience 20"
- [repo:scripts/FITS/etth1_best.sh@d040bb0] other_settings: itr=1 — "--itr 1"
- [repo:scripts/FITS/etth1_best.sh@d040bb0] other_settings: features=M — "--features M"
- [repo:scripts/FITS/etth1_best.sh@d040bb0] other_settings: enc_in=7 — "--enc_in 7"
- [repo:scripts/FITS/etth1_best.sh@d040bb0] other_settings: des='Exp' — "--des 'Exp'"
- [repo:scripts/FITS/etth1_best.sh@d040bb0] other_settings: gpu=0 — "--gpu 0"
- [repo:scripts/FITS/etth1_best.sh@d040bb0] other_settings: root_path=./dataset/ — "--root_path ./dataset/"
- [repo:scripts/FITS/etth1_best.sh@d040bb0] other_settings: data_path=ETTh1.csv — "--data_path ETTh1.csv"
- [repo:scripts/FITS/etth1_best.sh@d040bb0] other_settings: is_training=1 — "--is_training 1"
- [repo:scripts/FITS/etth1_best.sh@d040bb0] other_settings: model_name=FITS — "model_name=FITS"
- [repo:scripts/FITS/etth1_best.sh@d040bb0] other_settings: CUDA_VISIBLE_DEVICES=4 — "export CUDA_VISIBLE_DEVICES=4"
- [repo:README.md@d040bb0] other settings: batch size of 128 — "- ~~**FITS benefits from large batch size. Our latest version uses batch size of 128.** Some results are not updated due to limited time."
- [discovery] input length: 360 — "look-back window of 360"

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch