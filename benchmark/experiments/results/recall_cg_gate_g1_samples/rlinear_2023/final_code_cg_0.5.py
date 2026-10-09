# Plan step -> imported libraries and set seeds
# Plan step -> load data via nullius_io
# Plan step -> define hyperparameters in CONFIG
# Plan step -> define RevIN class
# Plan step -> define RLinear model class
# Plan step -> define training dataset and data loader
# Plan step -> define validation loss function
# Plan step -> training loop with early stopping and LR schedule
# Plan step -> load best model and generate validation predictions
# Plan step -> generate test predictions
# Plan step -> write predictions via nullius_io
# Plan step -> print validation MSE

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# deterministic seeds
seed = int(os.environ.get("SEED", "0"))
np.random.seed(seed)
torch.manual_seed(seed)

import nullius_io as nio

# plan step -> define hyperparameters in CONFIG
CONFIG = {"lr": 0.001, "epochs": 15, "batch_size": 16, "dropout": 0.1, "patience": 10}
# additional hyperparameters needed for the model/task
CONFIG["seq_len"] = 336
CONFIG["pred_len"] = 96
CONFIG["channel"] = 7
CONFIG["individual"] = False
CONFIG["rev"] = True

# plan step -> load data via nullius_io
data = nio.load()
train = data["train"]          # [8640,7] float32
val = data["val"]              # [2880,7] float32
test_inputs = data["test_inputs"]  # [2785,720,7] float32

# combine train+val for validation window construction
combined = np.concatenate([train, val], axis=0)  # [11520,7]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# plan step -> define RevIN class
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

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

# plan step -> define RLinear model class
class RLinear(nn.Module):
    def __init__(self, configs):
        super(RLinear, self).__init__()
        self.seq_len = configs["seq_len"]
        self.pred_len = configs["pred_len"]
        self.channel = configs["channel"]
        self.individual = configs["individual"]
        self.drop = nn.Dropout(configs["dropout"])
        self.rev = RevIN(self.channel) if configs["rev"] else None
        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channel)
            ])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y):
        # x: [B, L, D]; y: [B, pred_len, D]
        if self.rev is not None:
            x = self.rev(x, 'norm')
        x = self.drop(x)
        if self.individual:
            pred = torch.zeros_like(y)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        if self.rev is not None:
            pred = self.rev(pred, 'denorm')
        return pred, self.forward_loss(pred, y)

# plan step -> define training dataset and data loader
class TrainDataset(Dataset):
    def __init__(self, data_np, seq_len, pred_len):
        self.data = torch.from_numpy(data_np).float()
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return self.data.shape[0] - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return x, y

seq_len = CONFIG["seq_len"]
pred_len = CONFIG["pred_len"]
batch_size = CONFIG["batch_size"]

train_dataset = TrainDataset(train, seq_len, pred_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# plan step -> define validation loss function
def compute_val_loss(model, combined_np, seq_len, pred_len):
    model.eval()
    total_loss = 0.0
    count = 0
    # validation windows: start at 8640 + 24*w, w=0..116
    for w in range(117):
        start = 8640 + 24 * w
        start_input = start - seq_len
        x_np = combined_np[start_input:start]          # [seq_len,7]
        y_np = combined_np[start:start+pred_len]       # [pred_len,7]
        x = torch.from_numpy(x_np).unsqueeze(0).to(device)  # [1,seq_len,7]
        y = torch.from_numpy(y_np).unsqueeze(0).to(device)  # [1,pred_len,7]
        with torch.no_grad():
            _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count,1)

# plan step -> training loop with early stopping and LR schedule
model = RLinear(CONFIG).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=0.0)

# LR schedule helper (type1)
def adjust_lr(optimizer, epoch, base_lr):
    # epoch is 1-indexed as in original code
    lr = base_lr * (0.5 ** ((epoch - 1) // 1))
    for pg in optimizer.param_groups:
        pg["lr"] = lr

best_val = float('inf')
patience_counter = 0
best_state = None

smoke = os.environ.get("SMOKE") == "1"
if smoke:
    # reduce epochs to 1 and limit training data for quick check
    CONFIG["epochs"] = 1
    limited_loader = list(train_loader)[:2]  # at most 2 batches
    use_loader = limited_loader
else:
    use_loader = train_loader

for epoch in range(CONFIG["epochs"]):
    model.train()
    epoch_loss = 0.0
    nbatches = 0
    for xb, yb in use_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        nbatches += 1
    # adjust learning rate after each epoch (epoch index 0 -> treat as epoch+1)
    adjust_lr(optimizer, epoch+1, CONFIG["lr"])
    avg_train_loss = epoch_loss / max(nbatches,1)
    # validation loss
    val_loss = compute_val_loss(model, combined, seq_len, pred_len)
    print(f"Epoch {epoch+1}: train loss {avg_train_loss:.6f}, val loss {val_loss:.6f}")
    # early stopping
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

if best_state is not None:
    model.load_state_dict(best_state)

print(f"Best validation MSE: {best_val:.6f}")

# plan step -> load best model and generate validation predictions
def predict_windows(input_array, seq_len):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(seq_len, input_array.shape[0] - pred_len + 1):
            x = input_array[start-seq_len:start]  # [seq_len,7]
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # [1,seq_len,7]
            dummy_y = torch.zeros((1, pred_len, CONFIG["channel"]), device=device)
            pred, _ = model(x_t, dummy_y)
            pred_np = pred.squeeze(0).cpu().numpy()  # [pred_len,7]
            preds.append(pred_np)
    return np.stack(preds, axis=0)  # [num_windows, pred_len,7]

# validation predictions: using combined array, windows start at 8640+24*w
val_pred_list = []
for w in range(117):
    start = 8640 + 24 * w
    x_hist = combined[start-seq_len:start]  # [seq_len,7]
    x_t = torch.from_numpy(x_hist).unsqueeze(0).to(device)
    dummy_y = torch.zeros((1, pred_len, CONFIG["channel"]), device=device)
    with torch.no_grad():
        pred, _ = model(x_t, dummy_y)
    val_pred_list.append(pred.squeeze(0).cpu().numpy())
val_pred = np.stack(val_pred_list, axis=0)  # [117,96,7]

# plan step -> generate test predictions
test_pred_list = []
for i in range(test_inputs.shape[0]):
    x_hist = test_inputs[i, -seq_len:, :]  # [seq_len,7]
    x_t = torch.from_numpy(x_hist).unsqueeze(0).to(device)
    dummy_y = torch.zeros((1, pred_len, CONFIG["channel"]), device=device)
    with torch.no_grad():
        pred, _ = model(x_t, dummy_y)
    test_pred_list.append(pred.squeeze(0).cpu().numpy())
test_pred = np.stack(test_pred_list, axis=0)  # [2785,96,7]

# plan step -> write predictions via nullius_io
nio.write_predictions(test_pred, val_pred)
