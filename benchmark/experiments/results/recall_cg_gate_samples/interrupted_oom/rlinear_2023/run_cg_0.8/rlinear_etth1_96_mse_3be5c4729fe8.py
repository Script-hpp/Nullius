# Plan step -> implemented in: class RLinear
# Plan step -> implemented in: variables seq_len, pred_len
# Plan step -> implemented in: variable channel
# Plan step -> implemented in: configs.individual
# Plan step -> implemented in: configs.rev
# Plan step -> implemented in: configs.drop
# Plan step -> implemented in: optimizer creation (Adam lr=0.001)
# Plan step -> implemented in: variable batch_size
# Plan step -> implemented in: variable train_epochs
# Plan step -> implemented in: early stopping logic (patience=10)
# Plan step -> implemented in: args.lradj and adjust_learning_rate call
# Plan step -> implemented in: PyTorch default weight initialization
# Plan step -> implemented in: forward_loss and criterion (MSE)
# Plan step -> implemented in: AMP disabled (not used)
# Plan step -> implemented in: seed reading from env var SEED
# Plan step -> implemented in: inference functions create_test_input, create_val_input and prediction loops

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ---------- Helper modules ----------
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


class RLinear(nn.Module):
    def __init__(self, configs):
        super(RLinear, self).__init__()
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList([nn.Linear(configs.seq_len, configs.pred_len) for _ in range(configs.channel)])
        else:
            self.Linear = nn.Linear(configs.seq_len, configs.pred_len)
        self.dropout = nn.Dropout(configs.drop)
        self.rev = RevIN(configs.channel) if configs.rev else None

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y):
        # x: [B, L, D], y: [B, pred_len, D]
        if self.rev:
            x = self.rev(x, 'norm')
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros_like(y)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)  # [B, D, pred_len] -> [B, pred_len, D]
        if self.rev:
            pred = self.rev(pred, 'denorm')
        return pred, self.forward_loss(pred, y)


def adjust_learning_rate(optimizer, epoch, args):
    # epoch: 1-indexed as in original code
    lr_adjust = {}
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
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')


class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf  # Fixed: np.Inf -> np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        # we keep the best model state in memory
        self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}


# ---------- Configuration ----------
seed = int(os.getenv('SEED', '0'))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

seq_len = 336          # L
pred_len = 96
channel = 7
individual = False
rev = True
dropout = 0.1
learning_rate = 0.001
batch_size = 16
train_epochs = 15
patience = 10
lradj = 'type1'

# Smoke mode adjustments
if os.getenv('SMOKE') == '1':
    train_epochs = 1
    batch_size = 32
    smoke_limit = 1000  # limit number of training samples

# ---------- Load data ----------
import nullius_io as nio
data = nio.load()
train_np = data['train'].astype(np.float32)      # [8640,7]
val_np = data['val'].astype(np.float32)          # [2880,7]
test_inputs_np = data['test_inputs'].astype(np.float32)  # [2785,720,7]

# ---------- Create sliding window samples ----------
def create_samples(data, seq_len, pred_len):
    n = data.shape[0]
    X = []
    Y = []
    for i in range(0, n - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len:i+seq_len+pred_len])
    X = np.stack(X)
    Y = np.stack(Y)
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()

train_X, train_Y = create_samples(train_np, seq_len, pred_len)
val_X, val_Y = create_samples(val_np, seq_len, pred_len)

if os.getenv('SMOKE') == '1':
    # limit to first smoke_limit samples for faster training
    if train_X.shape[0] > smoke_limit:
        train_X = train_X[:smoke_limit]
        train_Y = train_Y[:smoke_limit]

num_train = train_X.shape[0]
num_val = val_X.shape[0]

# ---------- Model, optimizer, early stopping ----------
configs = type('', (), {})()
configs.seq_len = seq_len
configs.pred_len = pred_len
configs.channel = channel
configs.individual = individual
configs.drop = dropout
configs.rev = rev

device = torch.device('cpu')
model = RLinear(configs).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
early_stopping = EarlyStopping(patience=patience, verbose=False)

# Arguments for LR schedule
args = type('', (), {})()
args.learning_rate = learning_rate
args.lradj = lradj

# ---------- Training loop ----------
for epoch in range(train_epochs):
    epoch_idx = epoch + 1  # 1-indexed for LR adjust
    model.train()
    adjust_learning_rate(optimizer, epoch_idx, args)
    indices = np.arange(num_train)
    np.random.shuffle(indices)
    epoch_loss = 0.0
    for start in range(0, num_train, batch_size):
        end = min(start + batch_size, num_train)
        batch_idx = indices[start:end]
        xb = train_X[batch_idx].to(device)
        yb = train_Y[batch_idx].to(device)
        optimizer.zero_grad()
        pred, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= num_train

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for start in range(0, num_val, batch_size):
            xb = val_X[start:start+batch_size].to(device)
            yb = val_Y[start:start+batch_size].to(device)
            _, loss = model(xb, yb)
            val_losses.append(loss.item() * xb.size(0))
    val_loss = sum(val_losses) / num_val

    print(f'Epoch {epoch_idx}: Train Loss {epoch_loss:.6f}, Val Loss {val_loss:.6f}')
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# Restore best model
model.load_state_dict(early_stopping.best_state)

# ---------- Prediction functions ----------
def predict_from_history(history):
    """history: np.ndarray [seq_len, channel]"""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(history).unsqueeze(0).to(device)   # [1, seq_len, C]
        dummy_y = torch.zeros((1, pred_len, channel)).to(device)
        pred, _ = model(x, dummy_y)                         # [1, pred_len, C]
        pred = pred.squeeze(0).cpu().numpy()                # [pred_len, C]
    return pred

# Validation predictions (117 windows)
combined = np.concatenate([train_np, val_np], axis=0)   # [11520,7]
val_preds = []
for w in range(117):
    start = 8640 + 24 * w
    hist = combined[start - seq_len:start, :]           # [seq_len,7]
    pred = predict_from_history(hist)
    val_preds.append(pred)
val_pred = np.stack(val_preds, axis=0)                  # [117,96,7]

# Test predictions (2785 windows)
test_preds = []
for i in range(test_inputs_np.shape[0]):
    hist = test_inputs_np[i, -seq_len:, :]              # last seq_len steps
    pred = predict_from_history(hist)
    test_preds.append(pred)
test_pred = np.stack(test_preds, axis=0)                # [2785,96,7]

# ---------- Output ----------
print(f'Final validation MSE: {val_loss:.6f}')
nio.write_predictions(test_pred, val_pred)
