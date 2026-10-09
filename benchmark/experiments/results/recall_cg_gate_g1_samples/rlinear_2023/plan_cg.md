

- Model: RLinear – a single `nn.Linear(seq_len, pred_len)` (shared across the 7 channels) preceded by RevIN normalization and followed by dropout 0.1; per‑channel linear only if `individual=True` (we keep shared).  
- seq_len = 336 (input window) **(uncertain)**.  
- pred_len = 96 (forecast horizon).  
- channel = 7 (ETTh1 variables).  
- individual = False (shared linear) **(uncertain)**.  
- RevIN = True (apply reversible instance normalization) **(uncertain)**.  
- dropout = 0.1.  
- learning_rate = 0.001 (Adam optimizer; weight_decay = 0) **(uncertain)**.  
- batch_size = 16.  
- max epochs = 15, early‑stopping patience = 10 (monitor validation MSE).  
- Learning‑rate schedule: type 1 from `utils.tools` (lr ← lr × 0.5^{epoch‑1}) **(uncertain)**.  
- Loss: MSE (`nn.MSELoss`).  
- Training: shuffle training batches, gradient step per batch, validate each epoch, stop early.  
- Inference: for each test window *i* use the last L = seq_len = 336 steps of `test_inputs[i]` (no cross‑window leakage) to predict the next 96 steps for all 7 channels.  
- Output: write predictions in standardized units to `/output/predictions.csv` with id format "`{window}_{step}_{channel}`".