

# Plan step -> implemented in: function or class name
# 1. Model: TSMixer (6 residual blocks) with reversible instance normalization (RevIN) applied per‑sample before the model and inverted after. -> TSMixerWithRevIN
# 2. Input shape: (seq_len, C) = (512, 7) -> defined via CONFIG["seq_len"] and data shape
# 3. Block hyper‑parameters: n_block=6, ff_dim=hidden=512, dropout=0.9, activation=relu, norm_type=BatchNorm -> TSMixerBlock and TSMixerCore
# 4. Output head: Dense layer projecting to pred_len=96 -> TSMixerCore.head
# 5. Loss: Mean‑squared error (MSE) on standardized values -> nn.MSELoss
# 6. Optimizer: Adam -> torch.optim.Adam
# 7. Learning rate: CONFIG["lr"]
# 8. Batch size: CONFIG["batch_size"]
# 9. Epochs: maximum CONFIG["epochs"] with early stopping if validation loss does not improve for CONFIG["patience"] epochs -> train_loop()
#10. Weight decay: 0 (uncertain) -> optimizer weight_decay=0
#11. Seeding: read SEED from environment, set numpy and torch seeds -> set_seeds()
#12. Training procedure: standard mini‑batch shuffling, validation on provided val split, checkpoint best model -> train_loop()
#13. Inference: for each window, feed the last L=512 steps through the trained model with RevIN -> predict_windows()
#14. Output format: CSV via nullius_io -> nio.write_predictions()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import nullius_io as nio

# ----------------------------------------------------------------------
CONFIG = {
    "lr": 0.0001,
    "epochs": 100,
    "batch_size": 32,
    "dropout": 0.9,
    "hidden": 512,
    "layers": 6,
    "patience": 5,
    "seq_len": 512,
}
# ----------------------------------------------------------------------


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------
class RevIN(nn.Module):
    """Reversible Instance Normalization (per-sample, over time dimension)."""
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        """
        x: [B, L, C]
        mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        # reduce over the time dimension (dim=1)
        self.mean = x.mean(dim=1, keepdim=True)          # [B, 1, C]
        self.var = ((x - self.mean) ** 2).mean(dim=1, keepdim=True)  # [B, 1, C]

    def _normalize(self, x):
        x = x - self.mean
        x = x / torch.sqrt(self.var + self.eps)
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * torch.sqrt(self.var + self.eps)
        x = x + self.mean
        return x


# ----------------------------------------------------------------------
class TSMixerBlock(nn.Module):
    """One residual block of TSMixer (temporal + feature mixing)."""
    def __init__(self, seq_len: int, num_features: int,
                 ff_dim: int, dropout: float,
                 norm_type: str, activation: str):
        super().__init__()
        self.norm_type = norm_type
        self.seq_len = seq_len
        self.num_features = num_features

        # Normalization layers
        if norm_type == 'L':
            self.norm1 = nn.LayerNorm([seq_len, num_features])
            self.norm2 = nn.LayerNorm([seq_len, num_features])
        else:  # 'B' -> BatchNorm over flattened (seq_len * num_features)
            self.norm1 = nn.BatchNorm1d(seq_len * num_features)
            self.norm2 = nn.BatchNorm1d(seq_len * num_features)

        # Activation
        self.act = nn.ReLU() if activation == 'relu' else nn.GELU()

        # Temporal dense (mix across time)
        self.temporal_dense = nn.Linear(seq_len, seq_len)

        # Feature dense (mix across features)
        self.feature_dense1 = nn.Linear(num_features, ff_dim)
        self.feature_dense2 = nn.Linear(ff_dim, num_features)

        self.dropout = nn.Dropout(dropout)

    def _apply_norm(self, norm, x):
        if self.norm_type == 'L':
            return norm(x)
        else:
            B, L, C = x.shape
            x_flat = x.view(B, L * C)
            out = self.norm1 if norm is self.norm1 else self.norm2
            out = out(x_flat)
            return out.view(B, L, C)

    def forward(self, x):
        # ----- Temporal mixing -----
        y = self._apply_norm(self.norm1, x)               # [B, L, C]
        y = y.transpose(1, 2)                             # [B, C, L]
        y = self.temporal_dense(y)                        # [B, C, L]
        y = self.act(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)                             # [B, L, C]
        y = self.dropout(y)
        y = y + x                                         # residual

        # ----- Feature mixing -----
        z = self._apply_norm(self.norm2, y)               # [B, L, C]
        z = self.feature_dense1(z)                        # [B, L, ff_dim]
        z = self.act(z)
        z = self.dropout(z)
        z = self.feature_dense2(z)                        # [B, L, C]
        z = self.dropout(z)
        return z + y                                      # residual


# ----------------------------------------------------------------------
class TSMixerCore(nn.Module):
    """TSMixer stack of residual blocks + output head."""
    def __init__(self, seq_len: int, pred_len: int,
                 num_features: int,
                 n_block: int, ff_dim: int,
                 dropout: float,
                 norm_type: str,
                 activation: str):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features

        self.blocks = nn.ModuleList([
            TSMixerBlock(seq_len, num_features,
                         ff_dim, dropout,
                         norm_type, activation)
            for _ in range(n_block)
        ])

        # Output head: map from seq_len to pred_len per channel
        self.head = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C]
        for block in self.blocks:
            x = block(x)
        # [B, L, C] -> [B, C, L]
        x = x.transpose(1, 2)
        # [B, C, L] -> [B, C, pred_len]
        x = self.head(x)
        # [B, C, pred_len] -> [B, pred_len, C]
        x = x.transpose(1, 2)
        return x


# ----------------------------------------------------------------------
class TSMixerWithRevIN(nn.Module):
    """TSMixer core wrapped with RevIN."""
    def __init__(self, seq_len: int, pred_len: int,
                 num_features: int,
                 n_block: int, ff_dim: int,
                 dropout: float,
                 norm_type: str,
                 activation: str):
        super().__init__()
        self.revin = RevIN(num_features=num_features, affine=False)
        self.core = TSMixerCore(seq_len=seq_len, pred_len=pred_len,
                                num_features=num_features,
                                n_block=n_block, ff_dim=ff_dim,
                                dropout=dropout,
                                norm_type=norm_type,
                                activation=activation)

    def forward(self, x):
        # x: [B, L, C]
        x = self.revin(x, mode='norm')
        x = self.core(x)
        x = self.revin(x, mode='denorm')
        return x


# ----------------------------------------------------------------------
def create_datasets(data: np.ndarray, seq_len: int, pred_len: int):
    """Create sliding window datasets from a 2D array (time, features)."""
    T, C = data.shape
    num_samples = T - seq_len - pred_len + 1
    if num_samples <= 0:
        return None
    inputs = np.zeros((num_samples, seq_len, C), dtype=np.float32)
    targets = np.zeros((num_samples, pred_len, C), dtype=np.float32)
    for i in range(num_samples):
        inputs[i] = data[i:i + seq_len]
        targets[i] = data[i + seq_len:i + seq_len + pred_len]
    return inputs, targets


# ----------------------------------------------------------------------
def train_loop(model, train_loader, val_loader, optimizer, criterion,
               epochs, patience, device):
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train = np.mean(train_losses) if train_losses else 0.0

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())
        avg_val = np.mean(val_losses) if val_losses else 0.0

        print(f'Epoch {epoch+1}/{epochs} | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f}')

        if avg_val < best_val_loss - 1e-8:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ----------------------------------------------------------------------
def predict_windows(model, series: torch.Tensor, starts: np.ndarray,
                    seq_len: int, pred_len: int, device):
    """
    series: torch tensor [TotalTime, C] (train+val concatenated)
    starts: array of start indices for each window
    Returns predictions [num_windows, pred_len, C]
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for start in starts:
            idx_start = int(start)
            input_seq = series[idx_start - seq_len: idx_start].unsqueeze(0)  # [1, L, C]
            input_seq = input_seq.to(device)
            out = model(input_seq)                                          # [1, P, C]
            preds.append(out.squeeze(0).cpu())
    return torch.stack(preds, dim=0)  # [W, P, C]


# ----------------------------------------------------------------------
def main():
    # ----- Seeding -----
    seed = int(os.environ.get('SEED', '0'))
    set_seeds(seed)

    # ----- Smoke mode -----
    smoke = os.environ.get('SMOKE') == '1'

    # ----- Load data -----
    d = nio.load()
    train_np = d['train'].astype(np.float32)      # [8640, 7]
    val_np   = d['val'].astype(np.float32)        # [2880, 7]
    test_np  = d['test_inputs'].astype(np.float32) # [2785, 720, 7]

    # ----- Prepare series for validation windows -----
    series_np = np.concatenate([train_np, val_np], axis=0)  # [11520, 7]
    series = torch.from_numpy(series_np)

    # ----- Datasets -----
    seq_len = CONFIG["seq_len"]
    pred_len = 96  # fixed horizon for this task

    train_data = create_datasets(train_np, seq_len, pred_len)
    val_data   = create_datasets(val_np, seq_len, pred_len)

    if train_data is None or val_data is None:
        raise ValueError("Not enough data to create sliding windows.")

    train_inputs, train_targets = train_data
    val_inputs,   val_targets   = val_data

    train_dataset = TensorDataset(torch.from_numpy(train_inputs),
                                  torch.from_numpy(train_targets))
    val_dataset   = TensorDataset(torch.from_numpy(val_inputs),
                                  torch.from_numpy(val_targets))

    # Adjust for smoke: use only first batch and fewer epochs
    if smoke:
        train_loader = DataLoader(train_dataset,
                                  batch_size=CONFIG["batch_size"],
                                  shuffle=True)
        # take only first batch
        train_loader = [next(iter(train_loader))]
        val_loader   = DataLoader(val_dataset,
                                  batch_size=CONFIG["batch_size"],
                                  shuffle=False)
        epochs_run = 1
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=CONFIG["batch_size"],
                                  shuffle=True)
        val_loader   = DataLoader(val_dataset,
                                  batch_size=CONFIG["batch_size"],
                                  shuffle=False)
        epochs_run = CONFIG["epochs"]

    # ----- Model, optimizer, loss -----
    device = torch.device('cpu')
    model = TSMixerWithRevIN(seq_len=seq_len,
                             pred_len=pred_len,
                             num_features=train_np.shape[1],
                             n_block=CONFIG["layers"],
                             ff_dim=CONFIG["hidden"],
                             dropout=CONFIG["dropout"],
                             norm_type='B',   # BatchNorm as per plan (uncertain)
                             activation='relu')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CONFIG["lr"],
                                 weight_decay=0.0)
    criterion = nn.MSELoss()

    # ----- Training -----
    model = train_loop(model, train_loader, val_loader,
                       optimizer, criterion,
                       epochs=epochs_run,
                       patience=CONFIG["patience"],
                       device=device)

    # ----- Validation predictions (117 windows) -----
    # Validation window starts at row 8640 + 24*w, w=0..116
    val_starts = 8640 + 24 * np.arange(117, dtype=np.int64)
    val_pred = predict_windows(model, series, val_starts,
                               seq_len, pred_len, device)  # [117,96,7]
    # Compute validation MSE for logging
    val_targets_list = []
    for start in val_starts:
        idx_start = int(start)
        target = series[idx_start: idx_start + pred_len]  # [96,7]
        val_targets_list.append(target)
    val_targets_tensor = torch.stack(val_targets_list, dim=0)  # [117,96,7]
    val_mse = F.mse_loss(val_pred, val_targets_tensor).item()
    print(f'Validation MSE over 117 windows: {val_mse:.6f}')

    # ----- Test predictions (2785 windows) -----
    # Use last seq_len steps of each test_inputs[i]
    test_input_seq = test_np[:, -seq_len:, :]  # [2785,512,7]
    test_input_tensor = torch.from_numpy(test_input_seq)  # [2785,512,7]
    model.eval()
    with torch.no_grad():
        test_pred = model(test_input_tensor.to(device)).cpu()  # [2785,96,7]

    # ----- Write predictions via nullius_io -----
    nio.write_predictions(test_pred.numpy(), val_pred.numpy())


if __name__ == "__main__":
    main()