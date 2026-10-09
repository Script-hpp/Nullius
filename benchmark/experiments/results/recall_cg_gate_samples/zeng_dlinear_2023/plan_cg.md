

- Model: DLinear (seq_len=336, pred_len=96, enc_in=7, individual=False, moving‑avg kernel = 25)【certain】  
- Loss: MSE【certain】  
- Optimizer: Adam (β₁=0.9, β₂=0.999)【uncertain】  
- Learning rate: 0.005【certain】  
- Weight decay: 0【uncertain】  
- Dropout: 0.05 (defined but not used in DLinear)【certain】  
- Batch size: 32【certain】  
- Epochs: max 10【certain】  
- LR schedule: adjust_learning_rate with type 1 → lr × 0.5^{epoch‑1}【uncertain】  
- Early stopping: patience = 7 on validation MSE, restore best weights【certain】  
- Training: shuffle training batches, validate each epoch, stop early if no improvement.  
- Inference: for each test window i, use the last seq_len = 336 steps of test_inputs[i], run model in eval mode, predict 96 steps for all 7 channels.  
- Output: write `/output/predictions.csv` with rows `id="{window}_{step}_{channel}", prediction`.