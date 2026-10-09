

# Plan step -> implemented in: class RevIN
# Plan step -> implemented in: function adjust_learning_rate
# Plan step -> implemented in: class RLinear
# Plan step -> implemented in: function create_datasets
# Plan step -> implemented in: class Trainer
# Plan step -> implemented in: main()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ------------------ RevIN (from layers/Invertible.py) ------------------
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

# ------------------ Learning rate scheduler (type1) ------------------
def adjust_learning_rate(optimizer, epoch, args):
    # type1: lr = lr * (0.5 ** ((epoch-1)//1))
    lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')

# ------------------ RLinear model (from models/RLinear.py) ------------------
class RLinear(nn.Module):
    def __init__(self, configs):
        super(RLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel = configs.channel
        self.individual = configs.individual
        self.drop = configs.drop
        self.rev = configs.rev

        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channel)
            ])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        self.dropout = nn.Dropout(self.drop)
        self.revin = RevIN(self.channel) if self.rev else None

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y):
        # x: [B, L, D], y: [B, P, D]
        if self.revin:
            x = self.revin(x, 'norm')
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros_like(y)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        if self.revin:
            pred = self.revin(pred, 'denorm')
        return pred, self.forward_loss(pred, y)

# ------------------ Configuration holder ------------------
class Configs:
    def __init__(self, seq_len=336, pred_len=96, channel=7,
                 individual=False, rev=True, drop=0.1,
                 learning_rate=0.001):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channel = channel
        self.individual = individual
        self.rev = rev
        self.drop = drop
        self.learning_rate = learning_rate

# ------------------ Dataset creation ------------------
def create_datasets(seq_len, pred_len):
    import nullius_io as nio
    d = nio.load()
    train = d['train'].astype(np.float32)      # [8640,7]
    val = d['val'].astype(np.float32)          # [2880,7]
    test_inputs = d['test_inputs'].astype(np.float32)  # [2785,720,7]

    # Combined series for validation window construction
    combined = np.concatenate([train, val], axis=0)  # [11520,7]

    # ---- Training samples (sliding window on train only) ----
    T = train.shape[0]
    train_samples = T - seq_len - pred_len + 1
    train_x = np.zeros((train_samples, seq_len, 7), dtype=np.float32)
    train_y = np.zeros((train_samples, pred_len, 7), dtype=np.float32)
    for i in range(train_samples):
        train_x[i] = train[i:i+seq_len]
        train_y[i] = train[i+seq_len:i+seq_len+pred_len]

    # ---- Validation windows (117 windows) ----
    val_windows = 117
    val_start_base = 8640          # first validation forecast start
    step = 24                      # each window starts 24 hours later
    val_x = np.zeros((val_windows, seq_len, 7), dtype=np.float32)
    val_y = np.zeros((val_windows, pred_len, 7), dtype=np.float32)
    for w in range(val_windows):
        start = val_start_base + step * w          # forecast start index in combined
        val_x[w] = combined[start-seq_len:start]   # history before start
        val_y[w] = combined[start:start+pred_len]  # ground truth forecast

    # ---- Test samples (last seq_len of each test_inputs[i]) ----
    test_n = test_inputs.shape[0]  # 2785
    test_x = np.zeros((test_n, seq_len, 7), dtype=np.float32)
    for i in range(test_n):
        test_x[i] = test_inputs[i, -seq_len:, :]   # use last L=seq_len steps

    # Convert to torch tensors
    return (torch.from_numpy(train_x), torch.from_numpy(train_y),
            torch.from_numpy(val_x), torch.from_numpy(val_y),
            torch.from_numpy(test_x))

# ------------------ Trainer ------------------
class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cpu')
        self.model = RLinear(configs).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=configs.learning_rate)
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.best_state = None

    def train_one_epoch(self, train_x, train_y, batch_size, epoch):
        self.model.train()
        N = train_x.size(0)
        indices = torch.randperm(N)
        total_loss = 0.0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = indices[start:end]
            x_batch = train_x[idx].to(self.device)
            y_batch = train_y[idx].to(self.device)
            self.optimizer.zero_grad()
            _, loss = self.model(x_batch, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        return total_loss / N

    def evaluate(self, x, y, batch_size):
        self.model.eval()
        N = x.size(0)
        total_loss = 0.0
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                x_batch = x[start:end].to(self.device)
                y_batch = y[start:end].to(self.device)
                _, loss = self.model(x_batch, y_batch)
                total_loss += loss.item() * x_batch.size(0)
        return total_loss / N

    def predict(self, x, batch_size):
        self.model.eval()
        N = x.size(0)
        preds = []
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                x_batch = x[start:end].to(self.device)
                # dummy y with correct shape [B, pred_len, C]
                y_dummy = torch.zeros(x_batch.size(0), self.configs.pred_len, self.configs.channel, device=self.device)
                out, _ = self.model(x_batch, y_dummy)
                preds.append(out.cpu())
        return torch.cat(preds, dim=0)

    def fit(self, train_x, train_y, val_x, val_y, epochs, batch_size, patience):
        wait = 0
        for epoch in range(1, epochs+1):
            train_loss = self.train_one_epoch(train_x, train_y, batch_size, epoch)
            val_loss = self.evaluate(val_x, val_y, batch_size)
            adjust_learning_rate(self.optimizer, epoch, self.configs)
            print(f'Epoch {epoch:02d} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f}')
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = self.model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

# ------------------ Main ------------------
def main():
    seed = int(os.environ.get('SEED', '1024'))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    smoke = os.environ.get('SMOKE') == '1'

    # Plan step: Model -> RLinear
    # Plan step: Input/output lengths -> seq_len=336, pred_len=96
    # Plan step: Number of channels -> 7
    # Plan step: Individual linear layers -> False (shared)
    # Plan step: RevIN -> True
    # Plan step: Dropout -> 0.1
    # Plan step: Optimizer -> Adam lr=0.001
    # Plan step: Batch size -> 16 (or 32 in smoke)
    # Plan step: Epochs -> 15 (or 1 in smoke)
    # Plan step: Early stopping patience -> 10
    # Plan step: LR schedule -> type1 (halve each epoch)
    # Plan step: Weight init -> PyTorch default
    # Plan step: Loss -> MSE
    # Plan step: AMP -> disabled
    # Plan step: Seed -> from env

    seq_len = 336 if not smoke else 32  # keep shapes valid in smoke test
    pred_len = 96
    configs = Configs(seq_len=seq_len, pred_len=pred_len,
                      channel=7, individual=False, rev=True, drop=0.1,
                      learning_rate=0.001)

    # Plan step: create datasets
    train_x, train_y, val_x, val_y, test_x = create_datasets(seq_len, pred_len)

    if smoke:
        train_x = train_x[:32]
        train_y = train_y[:32]
        epochs = 1
        batch_size = 32
    else:
        epochs = 15
        batch_size = 16

    trainer = Trainer(configs)
    trainer.fit(train_x, train_y, val_x, val_y, epochs, batch_size, patience=10)

    # Predict validation and test
    val_pred = trainer.predict(val_x, batch_size)   # [117,96,7]
    test_pred = trainer.predict(test_x, batch_size) # [2785,96,7]

    # Convert to numpy for writer
    val_pred_np = val_pred.numpy()
    test_pred_np = test_pred.numpy()

    import nullius_io as nio
    nio.write_predictions(test_pred_np, val_pred_np)
    print(f'Final validation MSE: {trainer.best_val_loss:.6f}')

if __name__ == '__main__':
    main()