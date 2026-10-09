

**Reproduction plan for SGC on Cora (Planetoid split)**  

- **Model**: `SGC` – a single linear layer `W ∈ ℝ^{1433×7}` (no hidden units, dropout = 0).  
- **Pre‑processing**:  
  1. Load adjacency `A` and feature matrix `X` (binary bag‑of‑words, shape `[2708,1433]`).  
  2. Apply **AugNormAdj** normalization → `\hat{A}=D^{-1/2}(A+I)D^{-1/2}` (symmetric normalized with self‑loops).  
  3. Compute propagated features `\tilde{X}= \hat{A}^{k} X` with `k = degree` (default = 2) via repeated `torch.spmm`.  
  4. Row‑normalize `\tilde{X}` (each row sums to 1).  
- **Hyper‑parameters** (as specified):  
  - Learning rate `lr = 0.2`  
  - Weight decay `wd = 5e‑6`  
  - Epochs `= 100`  
  - Dropout `= 0`  
  - Hidden units `= 0` (SGC has none)  
  - Batch size: **full‑batch** (uncertain)  
  - Optimizer: Adam (`torch.optim.Adam`) with the above `lr` and `wd`.  
  - Loss: cross‑entropy.  
- **Training protocol**:  
  - Set random seeds from environment variable `SEED` for NumPy and PyTorch.  
  - For each epoch: forward pass on pre‑computed `\tilde{X}`, compute loss, back‑prop, optimizer step.  
  - After training, evaluate on validation set (`val_idx`) to select the model with highest validation accuracy (no early stopping; patience not used – uncertain).  
  - Final predictions: run the selected model on `\tilde{X}` for nodes in `test_idx`, output class IDs.  
- **Output**: CSV `/output/predictions.csv` with header `id,prediction` (one row per test node).  

*Uncertain items marked:* batch size, patience/early‑stopping, exact form of AugNormAdj, degree value (though default = 2). All other values follow the paper/code exactly.