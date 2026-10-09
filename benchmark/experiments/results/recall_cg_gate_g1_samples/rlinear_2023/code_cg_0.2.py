# Plan step -> implemented in: function or class name
# Import libraries and set seeds -> lines 13-20
# Load data via nullius_io -> lines 23-26
# Define hyperparameters in CONFIG -> lines 29-38
# Define RevIN class -> lines 41-71
# Define RLinear Model class -> lines 74-106
# Create sliding window dataset helper -> lines 109-124
# Prepare training, validation, test data -> lines 127-152
# Training loop with early stopping and LR schedule -> lines 155-210
# Inference for validation and test -> lines 213-230
# Print validation MSE and write predictions -> lines 233-238

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Seed handling
seed_str = os.environ.get('SEED')
if seed_str is not None:
    seed = int(seed_str)
    np.random.seed(seed)
    torch.manual_seed(seed)

import nullius_io as nio

# Load data
data = nio.load()
train_raw = data['train'].astype(np.float32)      # [8640, 7]
val_raw   = data['val'].astype(np.float32)       # [2880, 7]
test_inputs_raw = data['test_inputs'].astype(np.float32)  # [2785, 720, 7]

# Hyperparameters
CONFIG = {
    "lr": 0.001,
    "epochs": 15,
    "batch_size": 16,
    "dropout": 0.1,
    "patience": 10,
    # Additional fixed hyperparameters for RLinear
    "seq_len": 336,
    "pred_len": 96,
    "channel": 7,
    "individual": False,
    "rev": True,
}

# RevIN implementation (verbatim from source)
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

# RLinear model (verbatim from source)
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.channel = configs['channel']
        self.individual = configs['individual']
        self.drop = configs['dropout']
        self.rev_flag = configs['rev']

        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channel)
            ])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        self.dropout = nn.Dropout(self.drop)
        self.rev = RevIN(self.channel) if self.rev_flag else None

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y):
        # x: [B, L, D]; y: [B, pred_len, D]
        if self.rev:
            x = self.rev(x, 'norm')
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros_like(y)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        if self.rev:
            pred = self.rev(pred, 'denorm')
        return pred, self.forward_loss(pred, y)

# Helper to create sliding window dataset
def create_sliding_windows(data, seq_len, pred_len):
    T = data.shape[0]
    X = []
    Y = []
    for i in range(T - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len:i+seq_len+pred_len])
    X = np.stack(X).astype(np.float32)
    Y = np.stack(Y).astype(np.float32)
    return X, Y

# Prepare training data from train_raw
train_X, train_Y = create_sliding_windows(train_raw, CONFIG["seq_len"], CONFIG["pred_len"])
train_dataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_Y))
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)

# Prepare validation windows (117 windows) using combined train+val
combined = np.concatenate([train_raw, val_raw], axis=0)  # [11520, 7]
val_start_base = 8640  # first validation row in combined
val_inputs = []
val_targets = []
for w in range(117):
    start = val_start_base + 24 * w
    inp = combined[start - CONFIG["seq_len"]: start]
    tar = combined[start: start + CONFIG["pred_len"]]
    val_inputs.append(inp)
    val_targets.append(tar)
val_inputs = np.stack(val_inputs).astype(np.float32)   # [117, seq_len, 7]
val_targets = np.stack(val_targets).astype(np.float32) # [117, pred_len, 7]
val_dataset = TensorDataset(torch.from_numpy(val_inputs), torch.from_numpy(val_targets))
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

# Prepare test inputs: use last seq_len of each test_inputs[i]
L = min(CONFIG["seq_len"], test_inputs_raw.shape[1])  # ensure <=720
test_seq = test_inputs_raw[:, -L:, :]                  # [2785, L, 7]
# If L < CONFIG["seq_len"], we need to pad or adjust; but we set seq_len=336 <=720 so L==seq_len
assert L == CONFIG["seq_len"], "seq_len must be <=720"
test_tensor = torch.from_numpy(test_seq)               # [2785, seq_len, 7]

# Device
device = torch.device('cpu')

# Model, optimizer
model = Model(CONFIG).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=0.0)

# Learning rate schedule (type1): lr *= 0.5^{epoch}
def adjust_lr(epoch):
    lr = CONFIG["lr"] * (0.5 ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training with early stopping
best_val_loss = float('inf')
best_state = None
epochs_no_improve = 0

for epoch in range(CONFIG["epochs"]):
    adjust_lr(epoch)
    model.train()
    epoch_losses = []
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        _, loss = model(batch_x, batch_y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    avg_train_loss = np.mean(epoch_losses)

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device)
            vy = vy.to(device)
            _, loss = model(vx, vy)
            val_losses.append(loss.item())
    avg_val_loss = np.mean(val_losses)

    print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= CONFIG["patience"]:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

# Load best model
if best_state is not None:
    model.load_state_dict(best_state)

# Final validation MSE (using the same validation set)
model.eval()
with torch.no_grad():
    all_val_preds = []
    all_val_targets = []
    for vx, vy in val_loader:
        vx = vx.to(device)
        vy = vy.to(device)
        pred, _ = model(vx, vy)
        all_val_preds.append(pred.cpu().numpy())
        all_val_targets.append(vy.cpu().numpy())
    val_pred = np.concatenate(all_val_preds, axis=0)   # [117, 96, 7]
    val_true = np.concatenate(all_val_targets, axis=0)
    final_val_mse = np.mean((val_pred - val_true) ** 2)
    print(f"Final Validation MSE: {final_val_mse:.6f}")

# Test predictions
model.eval()
test_batch_size = CONFIG["batch_size"]
test_preds = []
with torch.no_grad():
    for i in range(0, test_tensor.shape[0], test_batch_size):
        batch = test_tensor[i:i+test_batch_size].to(device)
        pred, _ = model(batch, torch.zeros_like(batch[:, :CONFIG["pred_len"], :]))  # dummy y
        test_preds.append(pred.cpu().numpy())
test_pred = np.concatenate(test_preds, axis=0)  # [2785, 96, 7]

# Ensure float32
test_pred = test_pred.astype(np.float32)
val_pred = val_pred.astype(np.float32)

# Write predictions via helper
nio.write_predictions(test_pred, val_pred)
