# Plan step -> implemented in: class SparseTSF
# Plan step -> implemented in: __init__ of SparseTSF
# Plan step -> implemented in: CONFIG dict and usage
# Plan step -> implemented in: train loop function
# Plan step -> implemented in: inference section

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import nullius_io as nio

# Hyperparameter dictionary as required
CONFIG = {
    "lr": 0.02,
    "epochs": 30,
    "batch_size": 256,
    "patience": 5,
    "seq_len": 720
}
# Additional hyperparameters
CONFIG.update({
    "pred_len": 96,
    "enc_in": 7,
    "period_len": 24,
    "d_model": 64,
    "model_type": "linear"
})

# Seed handling
seed = int(os.getenv("SEED", "0"))
np.random.seed(seed)
torch.manual_seed(seed)

# Smoke mode detection
smoke = os.getenv("SMOKE") == "1"

# Load data
data = nio.load()
train_raw = data["train"].astype(np.float32)          # [8640, 7]
val_raw = data["val"].astype(np.float32)              # [2880, 7]
test_inputs = data["test_inputs"].astype(np.float32)  # [2785, 720, 7]

# Helper to create sliding windows
def create_windows(arr, seq_len, pred_len, stride=1):
    xs = []
    ys = []
    for i in range(0, len(arr) - seq_len - pred_len + 1, stride):
        xs.append(arr[i:i + seq_len])
        ys.append(arr[i + seq_len:i + seq_len + pred_len])
    return np.stack(xs), np.stack(ys)

# Training windows (stride 1)
x_train, y_train = create_windows(train_raw, CONFIG["seq_len"], CONFIG["pred_len"], stride=1)

# Validation windows: use combined train+val with stride 24, then take first 117
combined = np.concatenate([train_raw, val_raw], axis=0)
x_val_full, y_val_full = create_windows(combined, CONFIG["seq_len"], CONFIG["pred_len"], stride=24)
x_val = x_val_full[:117]
y_val = y_val_full[:117]

# Convert to torch tensors
x_train_t = torch.from_numpy(x_train)
y_train_t = torch.from_numpy(y_train)
x_val_t = torch.from_numpy(x_val)
y_val_t = torch.from_numpy(y_val)

# Smoke mode: reduce dataset sizes
if smoke:
    x_train_t = x_train_t[:32]
    y_train_t = y_train_t[:32]
    x_val_t = x_val_t[:8]
    y_val_t = y_val_t[:8]

# Datasets and loaders
train_ds = TensorDataset(x_train_t, y_train_t)
val_ds = TensorDataset(x_val_t, y_val_t)
batch_size = CONFIG["batch_size"]
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# Model definition (SparseTSF linear variant)
class SparseTSF(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        assert self.model_type in ['linear', 'mlp']
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2,
            padding_mode="zeros",
            bias=False
        )
        if self.model_type == 'linear':
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_y)
            )
    def forward(self, x):
        batch_size = x.shape[0]
        seq_mean = torch.mean(x, dim=1, keepdim=True)          # (B,1)
        x_norm = (x - seq_mean).permute(0, 2, 1)               # (B,C,S)
        # Conv1d per channel
        x_conv = self.conv1d(x_norm.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len)
        x = x_conv + x_norm                                    # (B,C,S)
        # Downsample: B,C,S -> BC,seg_num_x,period_len -> BC,period_len,seg_num_x
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        # Sparse forecasting
        if self.model_type == 'linear':
            y = self.linear(x)                                 # (BC,period_len,seg_num_y)
        else:
            y = self.mlp(x)
        # Upsample: BC,period_len,seg_num_y -> BC,seg_num_y,period_len -> B,C,PRED_LEN
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) + seq_mean                      # (B,PRED_LEN,C)
        return y

# Configuration object for model
class ModelConfig:
    pass
model_cfg = ModelConfig()
model_cfg.seq_len = CONFIG["seq_len"]
model_cfg.pred_len = CONFIG["pred_len"]
model_cfg.enc_in = CONFIG["enc_in"]
model_cfg.period_len = CONFIG["period_len"]
model_cfg.d_model = CONFIG["d_model"]
model_cfg.model_type = CONFIG["model_type"]

model = SparseTSF(model_cfg).float()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
criterion = nn.MSELoss()

# Training loop with early stopping
best_val = float('inf')
patience_wait = 0
best_state = None

for epoch in range(CONFIG["epochs"]):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb = xb.float()
        yb = yb.float()
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    avg_train = np.mean(train_losses)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.float()
            yb = yb.float()
            preds = model(xb)
            loss = criterion(preds, yb)
            val_losses.append(loss.item())
    avg_val = np.mean(val_losses) if val_losses else 0.0

    print(f'Epoch {epoch+1}/{CONFIG["epochs"]} - train loss: {avg_train:.6f}, val loss: {avg_val:.6f}')

    if avg_val < best_val:
        best_val = avg_val
        best_state = model.state_dict()
        patience_wait = 0
    else:
        patience_wait += 1
        if patience_wait >= CONFIG["patience"]:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break

    if smoke and epoch == 0:
        # After one epoch in smoke mode, stop
        break

# Load best model
if best_state is not None:
    model.load_state_dict(best_state)

print(f'Best validation MSE: {best_val:.6f}')

# Inference
model.eval()
with torch.no_grad():
    # Test predictions (all windows)
    test_tensor = torch.from_numpy(test_inputs).float()
    test_pred = model(test_tensor).cpu().numpy()  # [2785,96,7]

    # Validation predictions (117 windows)
    val_pred = model(x_val_t.float()).cpu().numpy()  # [117,96,7]

# Write predictions via helper
nio.write_predictions(test_pred, val_pred)
