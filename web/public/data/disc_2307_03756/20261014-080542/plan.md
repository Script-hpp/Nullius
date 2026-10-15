

- Model: FITS (non‑individual) with `seq_len=360`, `pred_len=96`, `enc_in=7`; `cut_freq = int(seq_len//base_T+1)*H_order+10 = 106` (`base_T=24`, `H_order=6`); `length_ratio=(seq_len+pred_len)/seq_len`.  
- Hyper‑parameters: `lr=0.0005`, `batch_size=128`, `epochs=50`, `patience=20`, optimizer = Adam, `weight_decay=(uncertain)`.  
- LR schedule: `type3` → `lr * 0.95^{(epoch‑1)}` (utils/tools).  
- Loss: MSE on the predicted horizon only (`train_mode=1` in exp/main_F).  
- Normalization: RevIN inside `forward` (subtract mean, divide √(var+1e‑5)); after irFFT multiply by `length_ratio·√var` and add mean.  
- Training: fit on `data["train"]`; early‑stopping on the visible 80 % of `data["val"]`; final 20 % of val withheld for unbiased selection (not used for fitting).  
- Validation for checkpoint: compute MSE on `nio.forecast_data(length=360)` (`train_inputs/targets` and `val_inputs/targets`).  
- Input processing: crop history to last `seq_len` steps; if `calendar` supplied, crop similarly; convert to torch tensor, add batch dimension.  
- Inference: `model.eval()`; forward; if output length = `seq_len+pred_len` keep last `pred_len` steps; return `np.ndarray[96,7]`.  
- Seeding: read `SEED` env var, set `numpy.random.seed`, `torch.manual_seed`.  
- No file writing; temporary checkpoints stored under `/tmp/forecast-agent`.  
- Predictor signature: `predictor(history: np.ndarray[, calendar: np.ndarray]) → np.ndarray[96,7]`.  
- (uncertain) hidden size / number of layers – FITS does not use them; assume default values.  
- Reported parameter count ≈ 10 k.  
- Expected validation MSE on ETTh1 horizon 96 ≈ 0.372.