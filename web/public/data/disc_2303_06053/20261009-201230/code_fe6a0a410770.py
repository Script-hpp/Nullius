import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import nullius_io as nio

CONFIG = {"lr": 0.0001, "epochs": 100, "batch_size": 32, "dropout": 0.9, "hidden": 512, "layers": 6, "patience": 5, "seq_len": 512}
# Plan step -> implemented in: TSMixerRevIN class
# Plan step -> implemented in: create_dataset function
# Plan step -> implemented in: TSMixerBlock class
# Plan step -> implemented in: TSMixerRevIN.head
# Plan step -> implemented in: nn.MSELoss()
# Plan step -> implemented in: optim.Adam
# Plan step -> implemented in: CONFIG["lr"]
# Plan step -> implemented in: CONFIG["batch_size"]
# Plan step -> implemented in: training loop with CONFIG["epochs"], CONFIG["patience"]
# Plan step -> implemented in: weight decay not used (default 0)
# Plan step -> implemented in: set_seeds()
# Plan step -> implemented in: fit()
# Plan step -> implemented in: predict_windows()
# Plan step -> implemented in: nio.write_predictions()

# Add any additional config keys here
CONFIG.update({
    "pred_len": 96,
    "activation": "relu",
})


def set_seeds():
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)


class RevIN(nn.Module):
    """Reversible Instance Normalization: normalize over the length dimension per feature."""
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        # x: [B, L, C]
        self.mean = x.mean(dim=1, keepdim=True)          # [B, 1, C]
        self.var = ((x - self.mean) ** 2).mean(dim=1, keepdim=True)  # [B, 1, C]
        self.std = torch.sqrt(self.var + self.eps)

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.std
        if self.affine:
            x = x * self.affine_weight.view(1, 1, -1) + self.affine_bias.view(1, 1, -1)
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias.view(1, 1, -1)
            x = x / (self.affine_weight.view(1, 1, -1) + self.eps)
        x = x * self.std
        x = x + self.mean
        return x


class TSMixerBlock(nn.Module):
    def __init__(self, seq_len, channels, ff_dim, dropout, activation):
        super().__init__()
        # Normalization over last two dimensions (L, C) as in the TF code
        self.norm1 = nn.LayerNorm([seq_len, channels])
        self.norm2 = nn.LayerNorm([seq_len, channels])
        # Temporal mixing
        self.fc_temporal = nn.Linear(seq_len, seq_len)
        # Feature mixing
        self.fc1 = nn.Linear(channels, ff_dim)
        self.fc2 = nn.Linear(ff_dim, channels)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        # Temporal mixing
        x = self.norm1(x)                     # [B, L, C]
        x = x.transpose(1, 2)                 # [B, C, L]
        x = self.fc_temporal(x)               # [B, C, L]
        x = self.activation(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)                 # [B, L, C]
        x = x + residual

        residual = x
        # Feature mixing
        x = self.norm2(x)                     # [B, L, C]
        x = self.fc1(x)                       # [B, L, ff_dim]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)                       # [B, L, C]
        x = self.dropout(x)
        x = x + residual
        return x


class TSMixerRevIN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config["seq_len"]
        self.pred_len = config["pred_len"]
        self.channels = 7
        self.revin = RevIN(self.channels, affine=False)
        self.blocks = nn.ModuleList([
            TSMixerBlock(
                seq_len=self.seq_len,
                channels=self.channels,
                ff_dim=config["hidden"],
                dropout=config["dropout"],
                activation=config["activation"]
            )
            for _ in range(config["layers"])
        ])
        self.head = nn.Linear(self.seq_len, self.pred_len)  # maps length -> pred_len

    def forward(self, x):
        # x: [B, L, C]
        x = self.revin(x, 'norm')
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2)       # [B, C, L]
        x = self.head(x)            # [B, C, pred_len]
        x = x.transpose(1, 2)       # [B, pred_len, C]
        x = self.revin(x, 'denorm')
        return x


def create_dataset(data, seq_len, pred_len):
    T, C = data.shape
    N = T - seq_len - pred_len + 1
    inputs = np.zeros((N, seq_len, C), dtype=np.float32)
    targets = np.zeros((N, pred_len, C), dtype=np.float32)
    for i in range(N):
        inputs[i] = data[i:i + seq_len]
        targets[i] = data[i + seq_len:i + seq_len + pred_len]
    return inputs, targets


def fit(model, train_loader, val_loader, config):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    best_val = float('inf')
    epochs_no_improve = 0
    best_state = None

    # Handle SMOKE mode: 1 epoch only
    max_epochs = config["epochs"]
    smoke = os.environ.get("SMOKE") == "1"
    if smoke:
        max_epochs = 1

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
            if smoke and train_batches >= 1:
                break

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
                val_batches += 1
                if smoke and val_batches >= 1:
                    break

        avg_train = train_loss / max(1, train_batches)
        avg_val = val_loss / max(1, val_batches)

        if epoch == 0 or smoke:
            print(f"Epoch {epoch+1}/{max_epochs} - train MSE: {avg_train:.6f} - val MSE: {avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["patience"]:
                if not smoke:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Full validation MSE over entire val set (not smoke-limited)
    model.eval()
    full_val_loss = 0.0
    full_batches = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            full_val_loss += loss.item()
            full_batches += 1
    final_mse = full_val_loss / max(1, full_batches)
    if not smoke:
        print(f"Final validation MSE: {final_mse:.6f}")
    return final_mse


def predict_windows(model, history, seq_len, pred_len):
    """
    history: np.ndarray [T, C] (standardized)
    Returns predictions for all possible windows: shape [N, pred_len, C]
    where N = T - seq_len (we predict the next pred_len steps after each history of length seq_len)
    """
    model.eval()
    T, C = history.shape
    N = T - seq_len
    if N <= 0:
        return np.zeros((0, pred_len, C), dtype=np.float32)
    preds = np.zeros((N, pred_len, C), dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            x = history[i:i + seq_len]          # [seq_len, C]
            xt = torch.from_numpy(x).unsqueeze(0)  # [1, seq_len, C]
            out = model(xt)                     # [1, pred_len, C]
            preds[i] = out.squeeze(0).cpu().numpy()
    return preds


def main():
    set_seeds()
    d = nio.load()
    train_data = d["train"].astype(np.float32)   # [8640, 7]
    val_data = d["val"].astype(np.float32)       # [2880, 7]
    test_inputs = d["test_inputs"].astype(np.float32)  # [2785, 720, 7]

    # --- Prepare training and validation datasets ---
    seq_len = CONFIG["seq_len"]
    pred_len = CONFIG["pred_len"]

    X_train, Y_train = create_dataset(train_data, seq_len, pred_len)
    X_val, Y_val = create_dataset(val_data, seq_len, pred_len)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # --- Build model ---
    model = TSMixerRevIN(CONFIG)

    # --- Train ---
    _ = fit(model, train_loader, val_loader, CONFIG)

    # --- Prepare full history (train+val) for validation and test window inputs ---
    full_history = np.concatenate([train_data, val_data], axis=0)  # [11520, 7]

    # Validation predictions: 117 windows, each starting at 8640 + 24*w
    val_start_base = train_data.shape[0]  # 8640
    val_windows = 117
    val_step = 24
    val_preds = np.zeros((val_windows, pred_len, 7), dtype=np.float32)
    for w in range(val_windows):
        start = val_start_base + w * val_step
        hist = full_history[start - seq_len: start]  # [seq_len, 7]
        with torch.no_grad():
            xt = torch.from_numpy(hist).unsqueeze(0)  # [1, seq_len, 7]
            out = model(xt)                           # [1, pred_len, 7]
            val_preds[w] = out.squeeze(0).cpu().numpy()

    # Test predictions: use last seq_len of each test_inputs[i]
    test_windows = test_inputs.shape[0]  # 2785
    test_preds = np.zeros((test_windows, pred_len, 7), dtype=np.float32)
    for i in range(test_windows):
        hist = test_inputs[i][-seq_len:, :]  # [seq_len, 7]
        with torch.no_grad():
            xt = torch.from_numpy(hist).unsqueeze(0)
            out = model(xt)
            test_preds[i] = out.squeeze(0).cpu().numpy()

    # --- Write outputs ---
    nio.write_predictions(test_preds, val_preds)


if __name__ == "__main__":
    main()
