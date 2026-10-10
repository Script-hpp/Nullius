# Reproduktions-Zeugnis: disc_2501_01087

## Hinweise zur Belegbarkeit
- Tabellenspalte nicht geprueft, Zahl koennte zu einer Vergleichsmethode gehoeren.
- Rolle 2 konnte den Zielwert im Paper nicht gegenpruefen (Modellantwort enthaelt kein JSON); es gilt der Wert der Discovery 0.382.

## Claim disc_2501_01087_mse: reproduziert
Gemessen: 0.3796 (std 0.0003)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0003. Mit unserem Setup gemessen (3 von 3 Seeds, Zielwert 0.382 +/- 0.01, discovery: https://arxiv.org/html/2501.01087v1, Zitat: TABLE II : Multivariate Forecasting performance comparison across datasets and models (lower MSE/MAE is better). Input Sequence Length for all experiments is 336 and a range of Prediction Length {12, 24, 48, 96, 192, 336, 720} are used. The top-performing results are marked in bold . The second-best results are underlined . Lookup Window (Input Sequence Length) = 336 Learning Rate 0.001 Methods | Autoformer | NLinear | DLinear | RLinear | GLinear Dataset / Output Horizon (Prediction Length) | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE ETTh1 | 12 | 0.3991 | 0.4422 | 0.3069 | 0.3564 | 0.2976 | 0.3494 | 0.2862 | 0.3412 | 0.2848 | 0.3448 24 | 0.4759 | 0.4733 | 0.3474 | 0.3842 | 0.3194 | 0.3627 | 0.3090 | 0.3559 | 0.3142 | 0.3654 48 | 0.5046 | 0.4831 | 0.3553 | 0.3845 | 0.3477 | 0.3803 | 0.3454 | 0.3766 | 0.3537 | 0.3869 96 | 0.5392 | 0.4979 | 0.3731 | 0.3941 | 0.3705 | 0.3919 | 0.3901 | 0.4054 | 0.3820 | 0.4025): Mittel 0.3796. Code-Kandidaten: 3, gewaehlt Kandidat 1 (Code 3b8cabaf40ba) nach Validierungs-Metrik (0: mse 0.7206, 1: mse 0.6859, 2: mse 0.7080); Rueckfall ohne Val-Wert; Repo-Abgleich: 1 Abweichungen, Code 3b8cabaf40ba durch 67dae24762de ersetzt
Zielwert 0.382: discovery: https://arxiv.org/html/2501.01087v1, Zitat: TABLE II : Multivariate Forecasting performance comparison across datasets and models (lower MSE/MAE is better). Input Sequence Length for all experiments is 336 and a range of Prediction Length {12, 24, 48, 96, 192, 336, 720} are used. The top-performing results are marked in bold . The second-best results are underlined . Lookup Window (Input Sequence Length) = 336 Learning Rate 0.001 Methods | Autoformer | NLinear | DLinear | RLinear | GLinear Dataset / Output Horizon (Prediction Length) | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE ETTh1 | 12 | 0.3991 | 0.4422 | 0.3069 | 0.3564 | 0.2976 | 0.3494 | 0.2862 | 0.3412 | 0.2848 | 0.3448 24 | 0.4759 | 0.4733 | 0.3474 | 0.3842 | 0.3194 | 0.3627 | 0.3090 | 0.3559 | 0.3142 | 0.3654 48 | 0.5046 | 0.4831 | 0.3553 | 0.3845 | 0.3477 | 0.3803 | 0.3454 | 0.3766 | 0.3537 | 0.3869 96 | 0.5392 | 0.4979 | 0.3731 | 0.3941 | 0.3705 | 0.3919 | 0.3901 | 0.4054 | 0.3820 | 0.4025
Quellen:
- https://arxiv.org/html/2501.01087v1

## What others found (Tavily Rolle 3, nur Bericht, nicht Teil des Urteils)
No public discussion found.
Abfrage: 2026-10-10T14:32:29+00:00 (live), 2 Seiten gelesen, 0 Aussagen verworfen (Zitat nicht im Text).

Modus: nachbau_autorencode. Rebuild from paper and authors' code: the agent reads the authors' code as text and writes its own code. The authors' code is never executed; the number is measured by our harness.
Repo gelesen: https://github.com/t-rizvi/GLinear @ f54bc961ea

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [repo:models/GLinear.py@f54bc96] input length: configs.seq_len — "self.seq_len = configs.seq_len"
- [repo:models/DNGLinear.py@f54bc96] input length: configs.seq_len — "self.seq_len = configs.seq_len"
- [repo:exp/exp_main.py@f54bc96] learning-rate schedule: Adam optimizer with learning rate = args.learning_rate — "model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)"
- [repo:exp/exp_main.py@f54bc96] learning-rate schedule: Learning rate adjusted each epoch via adjust_learning_rate — "adjust_learning_rate(model_optim, epoch + 1, self.args)"
- [repo:utils/tools.py@f54bc96] learning-rate schedule: type1: lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1)) — "lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}"
- [repo:utils/tools.py@f54bc96] learning-rate schedule: type2: lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8} — "lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }"
- [repo:exp/exp_main.py@f54bc96] patience: self.args.patience — "early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)"
- [repo:utils/tools.py@f54bc96] patience: 7 — "class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience"
- [repo:models/DNGLinear.py@f54bc96] other settings: kernel_size = 25 — "kernel_size = 25"
- [repo:models/RLinear.py@f54bc96] other settings: rev=True — "configs.rev  = True" (not in the plan: other model (file name RLinear.py))
- [repo:models/RLinear.py@f54bc96] other settings: individual=True — "configs.individual  = True" (not in the plan: other model (file name RLinear.py))
- [repo:models/RLinear.py@f54bc96] other settings: dropout=0.1 — "self.dropout = nn.Dropout(0.1)" (not in the plan: other model (file name RLinear.py))
- [paper] learning_rate: 0.001 — "extbf{Lookup Window (Input Sequence Length) = 336 Learning Rate 0.001}"
- [discovery] input length: 336 — "TABLE II : Multivariate Forecasting performance comparison across datasets and models (lower MSE/MAE is better). Input Sequence Length for all experiments is 336 and a range of Prediction Length {12, 24, 48, 96, 192, 336, 720} are used. The top-performing results are marked in bold . The second-best results are underlined . Lookup Window (Input Sequence Length) = 336 Learning Rate 0.001 Methods | Autoformer | NLinear | DLinear | RLinear | GLinear Dataset / Output Horizon (Prediction Length) | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE | MSE | MAE ETTh1 | 12 | 0.3991 | 0.4422 | 0.3069 | 0.3564 | 0.2976 | 0.3494 | 0.2862 | 0.3412 | 0.2848 | 0.3448 24 | 0.4759 | 0.4733 | 0.3474 | 0.3842 | 0.3194 | 0.3627 | 0.3090 | 0.3559 | 0.3142 | 0.3654 48 | 0.5046 | 0.4831 | 0.3553 | 0.3845 | 0.3477 | 0.3803 | 0.3454 | 0.3766 | 0.3537 | 0.3869 96 | 0.5392 | 0.4979 | 0.3731 | 0.3941 | 0.3705 | 0.3919 | 0.3901 | 0.4054 | 0.3820 | 0.4025"
- Not found in the sources: batch size; epochs

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch