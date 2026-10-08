1. Read SEED from `os.environ`, set `np.random.seed(SEED)` and `torch.manual_seed(SEED)`.  
2. Load `/data/etth1_public.npz`; obtain `train`, `val`, `test_inputs` (all float32, already z‑scored).  
3. Fix hyper‑parameters from the paper: input length **L = 336**, horizon **H = 96**, channels **C = 7** (certain).  
4. For each channel *c*: create sliding‑window datasets  
   - `X_train` (N_train, L) and `y_train` (N_train, H) from `train[:,c]` (step = 1)  
   - same for validation (`X_val`, `y_val`).  
5. Initialise a per‑channel model `nn.Linear(L, H, bias=True)`, loss `nn.MSELoss()`, optimizer `torch.optim.Adam(model.parameters(), lr=0.001)` **(uncertain)**.  
6. Train up to **max_epochs = 20** **(uncertain)**, batch size **256** **(uncertain)**, shuffling; stop early if validation loss does not improve for **patience = 5** epochs **(uncertain)**.  
7. Keep the best model (lowest val loss) for each channel.  
8. Inference: for each test window *i* (0…2784) and channel *c*:  
   - take the last **L** steps `seq = test_inputs[i, -L:, c]`  
   - compute `pred = model(torch.from_numpy(seq).unsqueeze(0)).squeeze(0).numpy()` → shape (H,)  
   - write rows with `id = f"{i}_{step}_{c}"` and `prediction = pred[step]` for step = 0…95.  
9. Concatenate all rows and save to `/output/predictions.csv` with header `id,prediction`.  

All uncertain values are marked; the rest follow the paper’s description.