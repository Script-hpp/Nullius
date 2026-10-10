

- Model: GLinear (RevIN → permute(0,2,1) → Linear(seq_len,seq_len) → GeLU → Linear(seq_len,pred_len) → permute → RevIN denorm).  
- seq_len = 336, pred_len = 96, enc_in = 7 (channels).  
- Optimizer: Adam(lr=0.001, weight_decay=(uncertain)).  
- Loss: MSELoss.  
- Learning‑rate schedule: adjust_learning_rate with lradj=(uncertain) (e.g., ‘type1’).  
- Max epochs = (uncertain) (training stops early via patience).  
- Batch size = (uncertain).  
- Early stopping: patience = 7, monitor validation MSE.  
- Validation each epoch; stop after 7 epochs without improvement.  
- Gradient clipping: (uncertain) – none specified.  
- AMP usage: (uncertain) – only if args.use_amp true.  
- Seed: read SEED env var, set numpy.random.seed and torch.manual_seed.  
- Test input: use last L = seq_len = 336 steps of each test_inputs[i] (L ≤ 720, choose 336).  
- Forecast horizon 96 steps, 7 channels → output rows id = “{window}_{step}_{channel}”.  
- Total predictions required: 2785 × 96 × 7 = 1,871,520 rows.