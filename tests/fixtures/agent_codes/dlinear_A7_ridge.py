import os
import numpy as np
import torch

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
seed = int(os.environ.get('SEED', 0))
np.random.seed(seed)
torch.manual_seed(seed)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
data = np.load('/data/etth1_public.npz')
train = data['train'].astype(np.float32)   # (8640, 7)
val   = data['val'].astype(np.float32)     # (2880, 7)
test_inputs = data['test_inputs'].astype(np.float32)  # (2785, 720, 7)

num_channels = train.shape[1]
lookback = 336          # L
horizon = 96            # H
lam = 1e-4              # ridge lambda

# ------------------------------------------------------------
# Helper: ridge regression for one channel
# ------------------------------------------------------------
def fit_channel(series):
    """Return beta (lookback, horizon) solving Y = X @ beta with ridge."""
    n = len(series)
    if n < lookback + horizon:
        raise ValueError("Series too short for given lookback/horizon")
    # Build X and Y using sliding windows (no overlap? we use stride 1)
    X_list = []
    Y_list = []
    for i in range(n - lookback - horizon + 1):
        X_list.append(series[i:i+lookback])
        Y_list.append(series[i+lookback:i+lookback+horizon])
    X = np.stack(X_list)  # (n_samples, lookback)
    Y = np.stack(Y_list)  # (n_samples, horizon)
    A = X.T @ X + lam * np.eye(lookback, dtype=X.dtype)
    B = X.T @ Y
    beta = np.linalg.solve(A, B)  # (lookback, horizon)
    return beta

# ------------------------------------------------------------
# Fit models per channel on training data
# --------------------------------------------------------->
betas = []  # list length num_channels, each (lookback, horizon)
for c in range(num_channels):
    beta_c = fit_channel(train[:, c])
    betas.append(beta_c)

# ------------------------------------------------------------
# Validation MSE
# ------------------------------------------------------------
def compute_mse(series, beta):
    n = len(series)
    if n < lookback + horizon:
        return 0.0
    X_list = []
    Y_list = []
    for i in range(n - lookback - horizon + 1):
        X_list.append(series[i:i+lookback])
        Y_list.append(series[i+lookback:i+lookback+horizon])
    X = np.stack(X_list)
    Y = np.stack(Y_list)
    pred = X @ beta
    mse = np.mean((pred - Y) ** 2)
    return mse

val_mses = []
for c in range(num_channels):
    mse_c = compute_mse(val[:, c], betas[c])
    val_mses.append(mse_c)
overall_val_mse = np.mean(val_mses)
print(f'Validation MSE (average over channels): {overall_val_mse:.6f}')

# ------------------------------------------------------------
# Test inference
# --------------------------------------------------------->
num_windows = test_inputs.shape[0]   # 2785
# Take last `lookback` steps from each window's 720-length history
X_test = test_inputs[:, -lookback:, :]   # (num_windows, lookback, 7)
preds = np.empty((num_windows, horizon, num_channels), dtype=np.float32)
for c in range(num_channels):
    Xc = X_test[:, :, c]                 # (num_windows, lookback)
    preds[:, :, c] = Xc @ betas[c]       # (num_windows, horizon)

# ------------------------------------------------------------
# Write predictions.csv
# --------------------------------------------------------->
out_path = '/output/predictions.csv'
chunk_size = 10000
with open(out_path, 'w') as f:
    f.write('id,prediction\n')
    chunk = []
    for w in range(num_windows):
        for t in range(horizon):
            for c in range(num_channels):
                pid = f'{w}_{t}_{c}'
                val = preds[w, t, c]
                chunk.append(f'{pid},{val:.6f}')
                if len(chunk) >= chunk_size:
                    f.write('\n'.join(chunk) + '\n')
                    chunk.clear()
    if chunk:
        f.write('\n'.join(chunk) + '\n')

print('Finished writing predictions to', out_path)
