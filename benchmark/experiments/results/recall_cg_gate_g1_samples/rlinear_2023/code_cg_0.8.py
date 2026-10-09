

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

# Plan step -> implemented in: load_data
# Plan step -> implemented in: set_seeds
# Plan step -> implemented in: CONFIG definition
# Plan step -> implemented in: RevIN class
# Plan step -> implemented in: RLinear class
# Plan step -> implemented in: create_sequences
# Plan step -> implemented in: train_model
# Plan step -> implemented in: predict_from_history
# Plan step -> implemented in: make_val_predictions
# Plan step -> implemented in: make_test_predictions
# Plan step -> implemented in: nio.write_predictions
# Plan step -> implemented in: print validation MSE

# Hyperparameters (read only from this dict)
CONFIG = {
    "lr": 0.001,
    "epochs": 15,
    "batch_size": 16,
    "dropout": 0.1,
    "patience": 10,
    "seq_len": 336,
    "pred_len": 96,
    "channel": 7,
    "individual": False,
    "rev": True,
    "weight_decay": 0.0,
}

def set_seeds():
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data():
    d = nio.load()
    train = d['train'].astype(np.float32)
    val = d['val'].astype(np.float32)
    test_inputs = d['test_inputs'].astype(np.float32)
    return train, val, test_inputs

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
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel = configs.channel
        self.individual = configs.individual
        self.drop = nn.Dropout(configs.drop)
        self.rev = RevIN(configs.channel) if configs.rev else None
        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channel)
            ])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)
    def forward(self, x, y):
        # x: [B, L, D], y: [B, pred_len, D]
        x_in = self.rev(x, 'norm') if self.rev else x
        x_in = self.drop(x_in)
        if self.individual:
            pred = torch.zeros_like(y)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x_in[:, :, idx])
        else:
            pred = self.Linear(x_in.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm') if self.rev else pred
        return pred, self.forward_loss(pred, y)

def create_sequences(data, seq_len, pred_len):
    T = data.shape[0]
    N = T - seq_len - pred_len + 1
    if N <= 0:
        return torch.empty(0), torch.empty(0)
    x = torch.zeros((N, seq_len, data.shape[1]), dtype=torch.float32)
    y = torch.zeros((N, pred_len, data.shape[1]), dtype=torch.float32)
    for i in range(N):
        x[i] = torch.from_numpy(data[i:i+seq_len]).float()
        y[i] = torch.from_numpy(data[i+seq_len:i+seq_len+pred_len]).float()
    return x, y

def adjust_learning_rate(optimizer, epoch, cfg):
    # type1 schedule: lr * 0.5^{(epoch-1)//1}
    if epoch == 0:
        lr = cfg["lr"]
    else:
        lr = cfg["lr"] * (0.5 ** ((epoch - 1) // 1))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr

def train_model(train_x, train_y, val_x, val_y, cfg):
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False, drop_last=False)
    configs = type('Configs', (), cfg)()
    model = RLinear(configs)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            if torch.cuda.is_available():
                bx = bx.cuda()
                by = by.cuda()
            optimizer.zero_grad()
            _, loss = model(bx, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bx.size(0)
        epoch_loss /= len(train_loader.dataset)
        lr = adjust_learning_rate(optimizer, epoch, cfg)
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                if torch.cuda.is_available():
                    vx = vx.cuda()
                    vy = vy.cuda()
                _, loss = model(vx, vy)
                val_loss += loss.item() * vx.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{cfg['epochs']} - train loss: {epoch_loss:.6f} - val loss: {val_loss:.6f} - lr: {lr:.6f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val

def predict_from_history(history, model, cfg):
    # history: np array [L, C]
    seq_len = cfg["seq_len"]
    if history.shape[0] < seq_len:
        # pad with zeros (should not happen)
        pad = np.zeros((seq_len - history.shape[0], history.shape[1]), dtype=history.dtype)
        history = np.concatenate([pad, history], axis=0)
    recent = history[-seq_len:]  # [seq_len, C]
    xt = torch.from_numpy(recent).float().unsqueeze(0)  # [1, seq_len, C]
    if torch.cuda.is_available():
        xt = xt.cuda()
    model.eval()
    with torch.no_grad():
        pred, _ = model(xt, torch.zeros_like(xt[:, :cfg["pred_len"], :]))
    pred = pred.squeeze(0)
    if torch.cuda.is_available():
        pred = pred.cpu()
    return pred.numpy()

def make_val_predictions(train, val, model, cfg):
    combined = np.concatenate([train, val], axis=0)  # [11520,7]
    val_preds = []
    for w in range(117):
        start = 8640 + 24 * w
        hist = combined[:start]  # exclude start point
        pred = predict_from_history(hist, model, cfg)
        val_preds.append(pred)
    return np.stack(val_preds, axis=0)  # [117,96,7]

def make_test_predictions(test_inputs, model, cfg):
    test_preds = []
    for i in range(test_inputs.shape[0]):
        hist = test_inputs[i]  # [720,7]
        pred = predict_from_history(hist, model, cfg)
        test_preds.append(pred)
    return np.stack(test_preds, axis=0)  # [2785,96,7]

def main():
    set_seeds()
    train, val, test_inputs = load_data()
    # Smoke mode handling
    smoke = os.environ.get("SMOKE") == "1"
    if smoke:
        # limit training data to at most 32 samples, 1 epoch
        tr_x, tr_y = create_sequences(train, CONFIG["seq_len"], CONFIG["pred_len"])
        max_samples = min(32, tr_x.shape[0])
        tr_x = tr_x[:max_samples]
        tr_y = tr_y[:max_samples]
        va_x, va_y = create_sequences(val, CONFIG["seq_len"], CONFIG["pred_len"])
        # use small validation set for early stopping (still need some)
        va_x = va_x[:max_samples]
        va_y = va_y[:max_samples]
        CONFIG["epochs"] = 1
        # ensure batch size not exceed samples
        if CONFIG["batch_size"] > max_samples:
            CONFIG["batch_size"] = max_samples
    else:
        tr_x, tr_y = create_sequences(train, CONFIG["seq_len"], CONFIG["pred_len"])
        va_x, va_y = create_sequences(val, CONFIG["seq_len"], CONFIG["pred_len"])
    model, best_val = train_model(tr_x, tr_y, va_x, va_y, CONFIG)
    val_pred = make_val_predictions(train, val, model, CONFIG)
    test_pred = make_test_predictions(test_inputs, model, CONFIG)
    nio.write_predictions(test_pred, val_pred)
    print(f"Best validation MSE: {best_val:.6f}")

if __name__ == "__main__":
    main()