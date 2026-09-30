```python
import numpy as np
import os
import sys

# Seed everything
seed = int(os.environ.get('SEED', 42))
np.random.seed(seed)

# Load data
data = np.load('/data/etth1_public.npz')
train = data['train'].astype(np.float32)      # [8640, 7]
test_inputs = data['test_inputs'].astype(np.float32)  # [2785, 720, 7]

# Fit VAR(p) on training data
p = 24  # 1 day of hourly lags
T, C = train.shape
# Build design matrix X [T-p, p*C] and target Y [T-p, C]
X = np.lib.stride_tricks.sliding_window_view(train, (p, C))[:-1].reshape(T-p, p*C)
Y = train[p:]

# Ridge regression: B = (X^T X + λI)^{-1} X^T Y
lam = 1e-4
XTX = X.T @ X
XTY = X.T @ Y
B = np.linalg.solve(XTX + lam * np.eye(p*C, dtype=np.float32), XTY)  # [p*C, C]

# Batched recursive forecasting for all test windows
W = test_inputs.shape[0]  # 2785
H = 96
# Initialize history buffer: last p steps for each window [W, p, C]
hist = test_inputs[:, -p:, :].copy()
preds = np.empty((W, H, C), dtype=np.float32)

for h in range(H):
    # Flatten history: [W, p*C]
    x = hist.reshape(W, p*C)
    # Predict next step: [W, C]
    y_hat = x @ B
    preds[:, h, :] = y_hat
    # Roll history and append prediction
    hist = np.roll(hist, -1, axis=1)
    hist[:, -1, :] = y_hat

# Write predictions.csv
out_path = '/output/predictions.csv'
with open(out_path, 'w') as f:
    f.write('id,prediction\n')
    # Pre-format all lines for faster writing
    lines = []
    for w in range(W):
        for h in range(H):
            base = f'{w}_{h}_'
            row = preds[w, h]
            for c in range(C):
                lines.append(f'{base}{c},{row[c]:.6f}')
    f.write('\n'.join(lines) + '\n')

print(f'Written {W*H*C} predictions to {out_path}', file=sys.stderr)
```