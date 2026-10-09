# Reproduktions-Zeugnis: disc_2303_06053

## Hinweise zur Belegbarkeit
- Protokoll im Paper nicht belegt: Der Task wurde angenommen, nicht aus dem Paper belegt. Die Gesamtaussage bleibt hoechstens vorlaeufig.
- Tabellenspalte nicht geprueft, Zahl koennte zu einer Vergleichsmethode gehoeren.

## Claim disc_2303_06053_mse: reproduziert
Gemessen: 0.3624 (std 0.0003)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0003. Mit unserem Setup gemessen (3 von 3 Seeds, Zielwert 0.361 +/- 0.01, Table 3, arXiv v5 (pdf)): Mittel 0.3624. Code-Kandidaten: 3, gewaehlt Kandidat 1 (Code d2c318c6b71f) nach Validierungs-Metrik (0: no val value, 1: mse 0.6545, 2: no val value); Rueckfall ohne Val-Wert; Repo-Abgleich: keine geprueften Abweichungen
Zielwert 0.361 belegt: "ETTh1 96 0.361 0.392 0.674 0.634 0.376 0.415 0.435 0.446 0.941 0.769 0.359 0.391 0.368 0.392 0.370 0.400" (Table 3, https://arxiv.org/pdf/2303.06053v5, Version v5, abgerufen 2026-10-09T18:12:31+00:00, SHA-256 f12af1d77b25)
ACHTUNG Zielwert: Methodenname 'TSMixer' steht nicht im Zitat: Der Check belegt nur, dass die Zahl im Text steht, nicht dass sie zur richtigen Spalte/Zeile gehoert (Zuordnung nicht automatisch geprueft, Kopfzeile ggf. von Hand gegenlesen).
Quellen:
- https://arxiv.org/pdf/2303.06053v5

## What others found (Tavily Rolle 3, nur Bericht, nicht Teil des Urteils)
- [reproduction report] The paper relies on a single random seed, which may not capture variability and hinders reproducibility. — "All results are reported from a single random seed (42). Multi-seed evaluation with statistical significance testing would provide stronger evidence for the observed differences, particularly on ETTh1 where the improvements are small." (https://arxiv.org/html/2605.02689v2)
- [protocol note] The paper points out that literature baselines may have used different hyperparameters, training seeds, and data preprocessing details, limiting strict head-to-head comparison. — "We report literature baselines from their original publications, which may use different hyperparameters, training seeds, and data preprocessing details. The five Transformer baseline results are included for contextualisation rather than strict head-to-head comparison." (https://arxiv.org/html/2605.02689v1)
- [reproduction report] The paper only reports results from a single random seed, leaving uncertainty about the stability and reproducibility of the reported performance across different initializations. — "Single seeds: We release one checkpoint per configuration; seed-to-seed variance is not reported." (https://arxiv.org/html/2609.31669v1)
- [reproduction report] User reports problems running the project, indicating possible reproducibility issues. — "I'm having some problems running this project." (https://github.com/google-research/google-research/issues/1034)
- [reproduction report] User expresses disbelief at many reported problems, hinting at concerns about reliability/reproducibility. — "I can't imagine that this project from Google has reported 496 problems so far" (https://github.com/google-research/google-research/issues/1034)
Abfrage: 2026-10-09T18:57:19+00:00 (live), 13 Seiten gelesen, 3 Aussagen verworfen (Zitat nicht im Text).

Modus: nachbau_autorencode. Rebuild from paper and authors' code: the agent reads the authors' code as text and writes its own code. The authors' code is never executed; the number is measured by our harness.
Repo gelesen: https://github.com/google-research/google-research/tree/HEAD/tsmixer @ 6e6b1ff747

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [repo:tsmixer/tsmixer_basic/run.py@6e6b1ff] input length: 336 — "parser.add_argument(
      '--seq_len', type=int, default=336, help='input sequence length'
  )"
- [paper] learning_rate: 0.0001 — "ETTh1 Search space Learing rate Blocks Dropout Hidden size Heads Model T 0.001, 0.0001 1, 2, 4, 6, 8 0.1, 0.3, 0.5, 0.7, 0.9 8, 16, 32, 64 4, 8 TSMixer 96 0.0001 6 0.9 512 relu"
- [repo:tsmixer/tsmixer_basic/run.py@6e6b1ff] learning rate: 0.0001 — "'--learning_rate',
      type=float,
      default=0.0001,
      help='optimizer learning rate',"
- [repo:tsmixer/tsmixer_basic/run.py@6e6b1ff] learning-rate schedule: 0.0001 — "parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.0001,
      help='optimizer learning rate',
  )"
- [paper] epochs: 100 (maximum epochs, with early stopping after 5 epochs without improvement) — "We train each model with a maximum 100 epochs and do early stopping if the validation loss is not improved after 5 epochs."
- [repo:tsmixer/tsmixer_basic/run.py@6e6b1ff] epochs: 100 — "parser.add_argument(
      '--train_epochs', type=int, default=100, help='train epochs'
  )"
- [repo:tsmixer/tsmixer_basic/run.py@6e6b1ff] patience: 5 — "parser.add_argument(
      '--patience', type=int, default=5, help='number of epochs to early stop'
)"
- [paper] input length: L = 512 — "We set the input length L = 512 as suggested in Nie et al. (2023) and evaluate the results for prediction lengths of T = {96, 192, 336, 720}."
- [paper] normalization: reversible instance normalization — "We apply reversible instance normalization (Kim et al., 2022) to ensure a fair comparison with the state-of-the-art PatchTST (Nie et al., 2023)."
- [repo:tsmixer/tsmixer_basic/run.py@6e6b1ff] other settings: --seed=0 (default) — "parser.add_argument('--seed', type=int, default=0, help='random seed')"
- [repo:tsmixer/tsmixer_basic/run.py@6e6b1ff] other settings: --target=OT (default) — "parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')"
- [repo:tsmixer/tsmixer_basic/run.py@6e6b1ff] other settings: --pred_len=96 (default) — "parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')"
- [repo:tsmixer/tsmixer_basic/run.py@6e6b1ff] other settings: --dropout=0.05 (default) — "parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')"
- [repo:tsmixer/tsmixer_basic/run.py@6e6b1ff] other settings: --batch_size=32 (default) — "parser.add_argument('--batch_size', type=int, default=32, help='batch size of input data')"
- [repo:tsmixer/tsmixer_basic/requirements.txt@6e6b1ff] other settings: numpy — "numpy"
- [repo:tsmixer/tsmixer_basic/requirements.txt@6e6b1ff] other settings: matplotlib — "matplotlib"
- [repo:tsmixer/tsmixer_basic/requirements.txt@6e6b1ff] other settings: pandas — "pandas"
- [repo:tsmixer/tsmixer_basic/requirements.txt@6e6b1ff] other settings: scikit-learn — "scikit-learn"
- [repo:tsmixer/tsmixer_basic/requirements.txt@6e6b1ff] other settings: tensorflow — "tensorflow"
- [paper] learning rate: 0.0001 — "Learing rate Blocks Dropout Hidden size Heads ... TSMixer 96 0.0001"
- [paper] number of blocks / layers: 6 — "Learing rate Blocks Dropout Hidden size Heads ... TSMixer 96 0.0001 6"
- [paper] dropout: 0.9 — "Learing rate Blocks Dropout Hidden size Heads ... TSMixer 96 0.0001 6 0.9"
- [paper] hidden size: 512 — "Learing rate Blocks Dropout Hidden size Heads ... TSMixer 96 0.0001 6 0.9 512"
- [repo:tsmixer/tsmixer_basic/run_tuned_hparam.sh@6e6b1ff] dataset: $1 — "data=$1"
- [repo:tsmixer/tsmixer_basic/run_tuned_hparam.sh@6e6b1ff] input sequence length: 512 — "seq_len=512"
- [repo:tsmixer/tsmixer_basic/run_tuned_hparam.sh@6e6b1ff] forecast horizon: 96 — "pred_len=96"
- [discovery] input length: 512 — "| Models | TSMixer | TFT | FEDformer | Autoformer | Informer | TMix-Only | Linear | PatchTST | | Metric | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | | ETTh1 | 96 | 0.361 | 0.392 | 0.674 | 0.634 | 0.376 | 0.415 | 0.435 | 0.446 | 0.941 | 0.769 | 0.359 | 0.391 | 0.368 | 0.392 | 0.370 | 0.400"
- Not found in the sources: batch size

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch