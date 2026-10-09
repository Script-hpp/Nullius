

**Reproduction plan (≤15 lines)**  

- **Model**: `RLinear` – a single linear mapping (or one per channel if `individual=True`) applied after optional RevIN normalization and dropout.  
- **Input / output lengths**: `seq_len = 336` (last L hours of each test window), `pred_len = 96`.  
- **Number of channels**: `C = 7` (ETTh1 variables).  
- **Individual linear layers**: **(uncertain)** – set `individual = False` (shared weight) unless ablation shows per‑channel works better.  
- **RevIN**: **(uncertain)** – enable (`rev = True`) as the paper reports using reversible instance normalization.  
- **Dropout**: `drop = 0.1`.  
- **Optimizer**: Adam with learning rate `lr = 0.001`.  
- **Batch size**: `B = 16` (as in the default run script).  
- **Epochs**: `train_epochs = 15` (early‑stopping may stop earlier).  
- **Early stopping**: patience = 10 epochs on validation MSE.  
- **Learning‑rate schedule**: **(uncertain)** – default `lradj = 'type1'` (halve LR each epoch) as defined in `utils.tools.adjust_learning_rate`.  
- **Weight initialization**: PyTorch default for `nn.Linear`.  
- **Loss**: MSE (used both for training and validation).  
- **AMP**: disabled (CPU‑only run).  
- **Random seed**: read from environment variable `SEED` and set `numpy`, `torch`, and `random` seeds accordingly.  
- **Inference**: for each test window `i`, take the last `seq_len` rows of `test_inputs[i]` (shape `[336,7]`), feed through the trained model, and write the 96×7 predictions in the required `id,prediction` format.  

All other settings (e.g., number of data loader workers) follow the defaults in the provided run script but are not critical for reproducing the reported MSE.