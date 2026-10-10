

# Plan step -> implemented in: class GLinear
# Plan step -> implemented in: hyperparameters (CONFIG, pred_len, enc_in)
# Plan step -> implemented in: optimizer creation
# Plan step -> implemented in: criterion = nn.MSELoss()
# Plan step -> implemented in: adjust_learning_rate function call
# Plan step -> implemented in: training loop with max_epochs and early stopping
# Plan step -> implemented in: DataLoader batch_size
# Plan step -> implemented in: manual early stopping with patience=7
# Plan step -> implemented in: (no gradient clipping)
# Plan step -> implemented in: (AMP not used)
# Plan step -> implemented in: seeding from SEED env var
# Plan step -> implemented in: using last seq_len steps of test_inputs[i]
# Plan step -> implemented in: helper nio.write_predictions handles ID formatting

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import nullius_io as nio

# ----------------------
# Seeding
# ----------------------
seed_str = os.environ.get("SEED", "0")
try:
    seed = int(seed_str)
except ValueError:
    seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# ----------------------
# Hyperparameters (read from CONFIG as required)
# ----------------------
CONFIG = {"lr": 0.001, "patience": 7, "seq_len": 336}
SEQ_LEN = CONFIG["seq_len"]
PRED_LEN = 96
ENC_IN = 7
LR = CONFIG["lr"]
WEIGHT_DECAY = 0.0  # uncertain, set to 0
BATCH_SIZE = 256    # uncertain, chosen reasonably
MAX_EPOCHS = 100    # uncertain, early stopping will halt sooner
PATIENCE = CONFIG["patience"]
LRADJ = "type1"     # uncertain, pick one from the schedule options

# ----------------------
# Learning rate schedule (as in utils/tools.py)
# ----------------------
def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))  # original comment
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')

# ----------------------
# RevIN implementation (as provided)
# ----------------------
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        # initialize RevIN params: (C,)
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

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

# ----------------------
# GLinear model (as provided)
# ----------------------
class GLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(GLinear, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_num_features = configs.enc_in

        self.Linear = nn.Linear(self.seq_len, self.seq_len)
        self.GeLU = nn.GELU()
        self.Hidden1 = nn.Linear(self.seq_len, self.pred_len)

        self.revin_layer = RevIN(self.in_num_features)

    def forward(self, x):
        # x: [Batch, seq_len, Channel]
        x = self.revin_layer(x, 'norm')
        x3 = x.permute(0,2,1)

        x3 = self.Linear(x3)
        x3 = self.GeLU(x3)
        x3 = self.Hidden1(x3)

        x3 = x3.permute(0,2,1)
        x3 = self.revin_layer(x3, 'denorm')
        return x3 # [Batch, Output length, Channel]

# ----------------------
# Simple args holder for LR scheduler
# ----------------------
class Args:
    def __init__(self):
        self.learning_rate = LR
        self.lradj = LRADJ

args = Args()

# ----------------------
# Load data via nullius_io
# ----------------------
data = nio.load()
train_np = data['train'].astype(np.float32)   # [8640, 7]
val_np = data['val'].astype(np.float32)       # [2880, 7]
test_inputs_np = data['test_inputs'].astype(np.float32)  # [2785, 720, 7]

# ----------------------
# Prepare training and validation datasets (sliding window)
# ----------------------
def create_dataset(data_arr, seq_len, pred_len):
    n = data_arr.shape[0]
    indices = np.arange(0, n - seq_len - pred_len + 1)
    X = np.stack([data_arr[i:i+seq_len] for i in indices], axis=0)   # [N, seq_len, C]
    Y = np.stack([data_arr[i+seq_len:i+seq_len+pred_len] for i in indices], axis=0)  # [N, pred_len, C]
    return torch.from_numpy(X), torch.from_numpy(Y)

train_X, train_Y = create_dataset(train_np, SEQ_LEN, PRED_LEN)
val_X, val_Y = create_dataset(val_np, SEQ_LEN, PRED_LEN)

train_dataset = TensorDataset(train_X, train_Y)
val_dataset = TensorDataset(val_X, val_Y)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ----------------------
# Model, optimizer, loss
# ----------------------
device = torch.device('cpu')
model = GLinear(type('Configs', (), {'seq_len': SEQ_LEN, 'pred_len': PRED_LEN, 'enc_in': ENC_IN})())
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()

# ----------------------
# Training loop with early stopping (matching EarlyStopping logic from utils/tools.py)
# ----------------------
best_val_loss = float('inf')
epochs_no_improve = 0
best_state = None

for epoch in range(MAX_EPOCHS):
    model.train()
    train_losses = []
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    avg_train_loss = np.mean(train_losses)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_losses.append(loss.item())
    avg_val_loss = np.mean(val_losses)

    # LR schedule step
    adjust_learning_rate(optimizer, epoch, args)

    print(f'Epoch {epoch+1}: Train Loss {avg_train_loss:.6f}, Val Loss {avg_val_loss:.6f}')

    # Early stopping check: improvement if current val loss < best val loss (delta=0)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break

# Load best model
if best_state is not None:
    model.load_state_dict(best_state)

print(f'Best validation MSE: {best_val_loss:.6f}')

# ----------------------
# Prediction function for a single sequence
# ----------------------
def predict_from_sequence(seq_np):
    """seq_np shape [seq_len, 7] -> returns [pred_len, 7] np.ndarray"""
    seq_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(device)  # [1, seq_len, 7]
    model.eval()
    with torch.no_grad():
        out = model(seq_tensor)  # [1, pred_len, 7]
    return out.squeeze(0).cpu().numpy()

# ----------------------
# Test predictions
# ----------------------
n_test = test_inputs_np.shape[0]
test_pred = np.zeros((n_test, PRED_LEN, ENC_IN), dtype=np.float32)
for i in range(n_test):
    # Use last SEQ_LEN steps of the provided 720-step history
    seq = test_inputs_np[i, -SEQ_LEN:, :]  # [SEQ_LEN, 7]
    test_pred[i] = predict_from_sequence(seq)

# ----------------------
# Validation predictions (117 windows as described)
# ----------------------
combined = np.concatenate([train_np, val_np], axis=0)  # [11520, 7]
n_val_win = 117
val_pred = np.zeros((n_val_win, PRED_LEN, ENC_IN), dtype=np.float32)
base_val_start = train_np.shape[0]  # 8640
for w in range(n_val_win):
    start_idx = base_val_start + 24 * w          # index in combined where forecast starts
    seq = combined[start_idx - SEQ_LEN: start_idx, :]  # [SEQ_LEN, 7]
    val_pred[w] = predict_from_sequence(seq)

# ----------------------
# Write predictions via helper
# ----------------------
nio.write_predictions(test_pred, val_pred)