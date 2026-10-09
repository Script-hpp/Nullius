

1. Read SEED from environment, seed numpy and torch.  
2. Load `/data/etth1_public.npz`; extract `train`, `val`, `test_inputs`.  
3. Set input length **L=336**, horizon **H=96**, channels **C=7** (as reported).  
4. Form training/validation samples by sliding windows: `x ∈ ℝ^{L·C}`, `y ∈ ℝ^{H·C}`.  
5. Model = `nn.Linear(L·C, H·C, bias=True)`  # RLinear (linear mapping).  
6. Loss = `nn.MSELoss()`.  
7. Optimizer = `torch.optim.Adam(model.parameters(), lr=1e-3(uncertain), weight_decay=0(uncertain))`.  
8. Batch size = **256(uncertain)**; shuffle each epoch.  
9. Max epochs = **30(uncertain)**; early‑stop if validation loss does not improve for **5(uncertain)** epochs.  
10. Train, validate each epoch, retain the best model (lowest val MSE).  
11. Test: for each window *i*, take the last **L** steps of `test_inputs[i]` (or the full 720 if L<720), compute `ŷ = model(x).reshape(H, C)`.  
12. Write `/output/predictions.csv` with header `id,prediction`; each row id=`"{window}_{step}_{channel}"`, prediction as float32.  
13. Total rows = 2785 × 96 × 7 = 1,871,520.  
14. Expected runtime < 5 minutes on CPU.