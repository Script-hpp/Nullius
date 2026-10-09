

**Reproduction plan for NLinear (ETTh1, horizon 96)**  

- **Model**: `NLinear` (Normalization‑Linear) – subtract last timestep, apply a single linear layer (`nn.Linear(seq_len, pred_len)`) shared across channels (`individual=False` *(uncertain)*).  
- **Input / output lengths**: `seq_len = 336` (given), `pred_len = 96` (task).  
- **Channels**: `enc_in = 7` (multivariate).  
- **Optimizer**: Adam *(uncertain)*.  
- **Learning rate**: `lr = 0.005` (given).  
- **Learning‑rate schedule**: `adjust_learning_rate` from `utils/tools.py`; specific `lradj` type not stated *(uncertain)*.  
- **Batch size**: `32` (given).  
- **Weight decay**: `0.0` *(uncertain, typical default)*.  
- **Dropout**: none in the model → `0.0` *(uncertain)*.  
- **Epochs**: train until early stopping; maximum epochs not reported *(uncertain)*; we will cap at `20` *(uncertain)*.  
- **Early stopping patience**: not specified *(uncertain)*; we will use `5` *(uncertain)*.  
- **Loss**: Mean Squared Error (MSE).  
- **Pre‑processing inside model**: `x = x - x[:, -1:, :]` (detached) before the linear layer, then add back the subtracted value after the linear projection (as in the provided `forward`).  
- **Validation**: after each epoch compute MSE on the validation split; stop if no improvement for `patience` epochs, restore best weights.  
- **Testing**: for each test window use the last `L = seq_len = 336` steps of `test_inputs[i]` (as allowed) to produce the 96‑step forecast; repeat for all windows and channels, write predictions to `/output/predictions.csv`.  

All values marked *(uncertain)* are chosen because they are not explicitly defined in the supplied sources; standard defaults are noted where needed.