

1. Read `SEED` from `os.environ`, set `numpy.random.seed(SEED)` and `torch.manual_seed(SEED)`.  
2. Load `/data/etth1_public.npz`; obtain `train` ([8640,7]), `val` ([2880,7]), `test_inputs` ([2785,720,7]).  
3. **Model (DLinear)**: decompose each channel with moving‑average of kernel `K` (uncertain, e.g., 25); feed trend and seasonal parts to two separate `nn.Linear(L, H)` layers; output = trend̂ + seasonal̂.  
4. Set input length `L = 336` (as claimed) and horizon `H = 96`.  
5. Optimiser: `Adam(lr=0.001)` (uncertain); loss = MSE.  
6. Training batch size = 32 (uncertain); max epochs = 10 (uncertain); early‑stop on val loss with patience = 3 (uncertain).  
7. For each batch, take the last `L` steps of the multichannel series, predict the next `H` steps for all 7 channels.  
8. Validate similarly each epoch; keep best model weights.  
9. Inference: for each test window `i`, use the last `L` steps of `test_inputs[i]` (if `L<720`; otherwise use full 720) to forecast the next `H` steps.  
10. Collect predictions → shape `(2785,96,7)`.  
11. Flatten to rows with `id = f"{window}_{step}_{channel}"` and write to `/output/predictions.csv` (header `id,prediction`).  
12. Ensure CPU‑only execution (`torch.set_num_threads(1)`).  
13. Expected runtime < 5 minutes on the given environment.  
14. All uncertain hyper‑parameters are marked above; adjust if needed to match the reported MSE ≈ 0.375.