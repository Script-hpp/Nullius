# Reproduktions-Zeugnis: disc_2410_09348_d8991277dd33d43f

Code-Auswahl und Reparaturen: Validierung und strukturelle Pruefungen. Testmessungen steuerten keine Code-Aenderung.

Selection protocol withheld_validation_v1: 100 of 500 validation windows/nodes were withheld from fitting and used for candidate selection. Checkpoints used the remaining visible validation data. Original training/test splits unchanged. This selection protocol differs from older runs.

## Hinweise zur Belegbarkeit
- Tabellenspalte nicht geprueft, Zahl koennte zu einer Vergleichsmethode gehoeren.
- Info: Das Paper nennt Streuung +-0.62 (ueber Splits/Laeufe, nicht Toleranz des Mittelwerts); die Toleranz +-0.01 bleibt fest.
- Independent candidate selection uses 100 of 500 validation windows/nodes with hidden answers. Checkpoints use the remaining visible validation data. Training/test splits are unchanged; this selection protocol differs from older runs.
- Steckbrief: Datei des Modells BANGS nicht im Repo-Ausschnitt

## Claim disc_2410_09348_d8991277dd33d43f_accuracy: teilweise
Gemessen: 0.8193 (std 0.0065)
Methodentreue: nicht geprueft (Methodentreue-Pruefer nicht geeicht, Eichung gescheitert). Gesamtaussage: vorlaeufig
Graph-Ablation (Daten ohne Kanten, Seed 0): Graph genutzt (Genauigkeit 0.8190 mit Graph, 0.6350 ohne Kanten)
Urteil beruht auf dem Mittelwert ueber die Seeds, Streuung (std) 0.0065. Mit unserem Setup gemessen (3 von 3 Seeds, Zielwert 0.8423 +/- 0.01, tab:GCN, arXiv v1 (latex)): Mittel 0.8193. Code-Kandidaten: 3, gewaehlt Kandidat 2 (Code 02e0ab745121) nach Validierungs-Metrik (0: accuracy 0.5400, 1: no val value, 2: accuracy 0.7700); Repo comparison incomplete: files were truncated or findings could not be verified. Method fidelity remains unchecked.; Independent selection used the final 20% of validation observations, withheld from fitting; checkpoint selection used the remaining visible validation data. Original train/test splits unchanged.
Zielwert 0.8423 belegt: "Cora 80.82±0.14 81.40±0.29 83.06±0.11 83.04±0.38 83.16±0.10 83.72±0.52 84.23±0.62*" (tab:GCN, https://export.arxiv.org/src/2410.09348v1, Version v1, abgerufen 2026-10-15T13:30:33+00:00, SHA-256 73fbc25999bd)
Spaltenzuordnung: an Kopfzeile geprüft (Position von Methode und Metrik stimmt).
Quellen:
- https://export.arxiv.org/src/2410.09348v1

## What others found (Tavily Rolle 3, nur Bericht, nicht Teil des Urteils)
No public discussion found.
Abfrage: 2026-10-15T14:48:20+00:00 (live), 1 Seiten gelesen, 0 Aussagen verworfen (Zitat nicht im Text).

## Possible explanations für disc_2410_09348_d8991277dd33d43f_accuracy (Tavily Rolle 6, ungeprüfte Hypothesen, nicht Teil des Urteils)
None found.

Modus: nachbau_autorencode. Rebuild from paper and authors' code: the agent reads the authors' code as text and writes its own code. The authors' code is never executed; the number is measured by our harness.
Repo gelesen: https://github.com/fangxin-wang/BANGS @ 5ca22d2c9f

## Setup-Details aus dem Paper (Tavily-Rolle 4, wörtlich belegt)
- [repo:calib_src/data/data_utils.py@5ca22d2] feature_normalisation: NormalizeFeatures transform applied to dataset — "transform = NormalizeFeatures()
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='./data/', name=name, transform=transform)"
- [paper] adjacency_normalisation: normalized adjacency matrix with added self-loop — "With ${\sV^p_r}'$ selected, the student model $f_{r}$ logit estimation $\hat{\mH}({\sV^p_r}') = \alpha (\mI - (1-\alpha) \Tilde{\mA} )^{-1} \hat{\mX}$, where $\alpha \in (0,1]$ is a teleportation probability, and $\Tilde{\mA} \in \mathbb{R}^{N \times N}$ is the normalized adjacency matrix with added self-loop."
- [repo:ST_src/data.py@5ca22d2] adjacency_normalisation: add self-loop to adjacency matrix — "g = dgl.add_self_loop(g)"
- [repo:ST_src/banzhaf.py@5ca22d2] adjacency_norm: influence_matrix = torch.transpose(influence_matrix, 0, 1)  # Let rows of influence_matrix sum up to 1 — "# Let rows of influence_matrix sum up to 1
# print( 'row',torch.sum(influence_matrix,dim=0), ' col',torch.sum(influence_matrix,dim=1))
influence_matrix = torch.transpose(influence_matrix, 0, 1)" (not in the plan: other model (file name banzhaf.py))
- [repo:ST_src/train_test.py@5ca22d2] learning rate: args.lr — "optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)"
- [repo:main_node.py@5ca22d2] learning rate: 0.001 — "parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')" (not in the plan: other model (file name main_node.py))
- [repo:main_node.py@5ca22d2] weight decay: 5e-4 — "parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')" (not in the plan: other model (file name main_node.py))
- [repo:main_node.py@5ca22d2] hidden size: 128 — "parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')" (not in the plan: other model (file name main_node.py))
- [repo:main_node.py@5ca22d2] hidden size: 32 — "parser.add_argument("--hid_dim_1", type=int, default=32, help="Hidden layer dimension")" (not in the plan: other model (file name main_node.py))
- [repo:main_node.py@5ca22d2] heads: 8 — "parser.add_argument('--nb_heads', type=int, default=8)" (not in the plan: other model (file name main_node.py))
- [repo:main_node.py@5ca22d2] heads: 8 — "parser.add_argument('--nb_out_heads', type=int, default=8)" (not in the plan: other model (file name main_node.py))
- [repo:ST_src/train_test.py@5ca22d2] epochs: args.epochs — "if not FT:
        epochs = args.epochs"
- [repo:ST_src/train_test.py@5ca22d2] epochs: args.epochs_ft — "else:
        state_dict = torch.load('./save_model/tmp.pth')
        model.load_state_dict(state_dict)
        epochs = args.epochs_ft"
- [paper] early_stopping_rule: max iteration number = 40, early stopping using validation data — "We set the max iteration number as 40, and use validation data to early stop."
- [paper] early_stopping_rule: test accuracy obtained by early stopping using validation data — "For our method, we present the test accuracy obtained by early stopping using validation data."
- [paper] early_stopping_rule: our method reports test accuracy by early stopping (while baselines report final test accuracy) — "baseline methods report final test accuracy, our method reports test accuracy by early stopping."
- [repo:ST_src/train_test.py@5ca22d2] early stopping rule: Training stops if validation accuracy does not improve for args.patience consecutive epochs. — "if bad_counter == args.patience:
            # print('early stop')
            break"
- [repo:main_node.py@5ca22d2] early stopping rule: --patience 500 (default) — "parser.add_argument('--patience', type=int, default=500)" (not in the plan: other model (file name main_node.py))
- [repo:main_node.py@5ca22d2] early stopping rule: Early stopping based on validation accuracy (select best test accuracy where validation accuracy is max) — "logger.info('Best Acc Early Stopped by Valid Acc: {:.5f}'.format(test_acc_l[np.argmax(val_acc_l)]))" (not in the plan: other model (file name main_node.py))
- [repo:ST_src/banzhaf.py@5ca22d2] sampling_probability: 0.5 — "torch.rand(len(indices)) < 0.5" (not in the plan: other model (file name banzhaf.py))
- [repo:ST_src/banzhaf.py@5ca22d2] APPNP_alpha: 0.15 — "alpha=0.15" (not in the plan: other model (file name banzhaf.py))
- [repo:ST_src/banzhaf.py@5ca22d2] entropy_epsilon: 1e-9 — "torch.log(class_prob + 1e-9)" (not in the plan: other model (file name banzhaf.py))
- [repo:ST_src/banzhaf.py@5ca22d2] loss_reduction: mean — "reduction='mean'" (not in the plan: other model (file name banzhaf.py))
- [repo:ST_src/banzhaf.py@5ca22d2] loss_reduction: sum — "reduction='sum'" (not in the plan: other model (file name banzhaf.py))
- [repo:calib_src/data/data_utils.py@5ca22d2] other: samples_in_one_fold=5, k_fold=3, test_samples_per_class=85 — "# Run at console -> python -c 'from calib_src.data.data_utils import *; split_data("Cora", 5, 3, 85)'"
- [repo:calib_src/data/data_utils.py@5ca22d2] other: split_type=5_3f_85 — "# Run at console -> python -c 'from calib_src.data.data_utils import *; generate_node_to_nearest_training("Cora", "5_3f_85")'"
- [repo:calib_src/data/data_utils.py@5ca22d2] other: bfs_depth=10 — "def generate_node_to_nearest_training(name: str, split_type: str, bfs_depth=10):"
- [repo:main_node.py@5ca22d2] dropout: 0.3 — "parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate in training")" (not in the plan: other model (file name main_node.py))
- [repo:main_node.py@5ca22d2] input dropout: 0.5 — "parser.add_argument('--input_droprate', type=float, default=0.5, help='dropout rate of input layer')" (not in the plan: other model (file name main_node.py))
- [repo:main_node.py@5ca22d2] hidden dropout: 0.5 — "parser.add_argument('--hidden_droprate', type=float, default=0.5, help='dropout rate of hidden layer')" (not in the plan: other model (file name main_node.py))
- [repo:main_node.py@5ca22d2] weight_decay: 5e-3 — "cal_wdecay = 5e-3" (not in the plan: other model (file name main_node.py))
- [repo:main_node.py@5ca22d2] heads: 2 — "heads, bias = 2, 1" (not in the plan: other model (file name main_node.py))
- Not found in the sources: weight decay; dropout; hidden size / heads

## Isolation (aktive Schutzschichten)
- Trainingslauf ohne Netz (unshare -n); Setup vorher im Basis-Snapshot
- Statische Codepruefung auf Netz/Download/Prozess (harness/code_check.py)
- Testlabels dem Agenten vorenthalten, Harness laedt sie frisch