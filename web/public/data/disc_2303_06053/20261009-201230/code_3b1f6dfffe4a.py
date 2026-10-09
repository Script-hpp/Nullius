

# Plan step -> implemented in: function or class name
# 1. Model: TSMixer (reduced layers for speed) with reversible instance normalization (RevIN) -> class TSMixer, RevIN
# 2. Input shape: (seq_len, C) = (512, 7) -> handled in dataset creation
# 3. Block hyper-parameters: n_block = eff_layers, ff_dim = eff_hidden, dropout = CONFIG["dropout"], activation = relu -> passed to TSMixer
# 4. Output head: Dense layer projecting to pred_len=96 -> in TSMixer.forward after blocks
# 5. Loss: Mean‑squared error (MSE) on standardized values -> nn.MSELoss()
# 6. Optimizer: Adam (typical default) -> torch.optim.Adam
# 7. Learning rate: 0.0001 -> CONFIG["lr"]
# 8. Batch size: 32 -> CONFIG["batch_size"]
# 9. Epochs: max 100 with early stopping patience=5 -> CONFIG["epochs"], CONFIG["patience"]
#10. Weight decay: 0 (not specified) -> optimizer without weight decay
#11. Seeding: read SEED from env, set numpy, torch, random seeds -> set_seeds()
#12. Training procedure: mini‑batch shuffling, validation on val split, checkpoint best model -> train_loop()
#13. Inference: use last L=512 steps of each test/validation window with RevIN, vectorized over windows -> predict_windows()
#14. Output format: call nullius_io.write_predictions -> main()

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters - read only from this dict
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

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class RevIN(nn.Module):
    """Reversible Instance Normalization (per feature across time)"""
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        # x: [B, L, C]
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)          # [B,1,C]
            self.var = x.var(dim=1, unbiased=False, keepdim=True)
            x_norm = (x - self.mean) / torch.sqrt(self.var + self.eps)
            if self.affine:
                x_norm = x_norm * self.affine_weight + self.affine_bias
            return x_norm
        elif mode == 'denorm':
            x_denorm = x * torch.sqrt(self.var + self.eps) + self.mean
            if self.affine:
                x_denorm = (x_denorm - self.affine_bias) / self.affine_weight
            return x_denorm
        else:
            raise NotImplementedError

class ResidualBlock(nn.Module):
    def __init__(self, seq_len: int, enc_in: int, dropout: float, ff_dim: int, activation: str):
        super().__init__()
        self.norm1 = nn.LayerNorm([seq_len, enc_in])
        self.norm2 = nn.LayerNorm([seq_len, enc_in])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU() if activation == 'relu' else nn.GELU()
        # temporal mixing: maps seq_len -> seq_len
        self.temporal_linear = nn.Linear(seq_len, seq_len)
        # feature mixing: enc_in -> ff_dim -> enc_in
        self.feature_fc1 = nn.Linear(enc_in, ff_dim)
        self.feature_fc2 = nn.Linear(ff_dim, enc_in)

    def forward(self, x):
        # x: [B, L, C]
        residual = x
        out = self.norm1(x)
        out = out.transpose(1, 2)          # [B, C, L]
        out = self.temporal_linear(out)    # [B, C, L]
        out = self.dropout(out)
        out = self.act(out)
        out = out.transpose(1, 2)          # [B, L, C]
        out = self.dropout(out)
        out = out + residual

        residual = out
        out = self.norm2(out)
        out = self.feature_fc1(out)        # [B, L, ff_dim]
        out = self.dropout(out)
        out = self.act(out)
        out = self.feature_fc2(out)        # [B, L, C]
        out = self.dropout(out)
        out = out + residual
        return out

class TSMixer(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, enc_in: int,
                 n_block: int, dropout: float, ff_dim: int, activation: str):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = RevIN(num_features=enc_in, affine=False)
        self.blocks = nn.ModuleList([
            ResidualBlock(seq_len, enc_in, dropout, ff_dim, activation)
            for _ in range(n_block)
        ])
        self.head = nn.Linear(seq_len, pred_len)   # maps L -> pred_len

    def forward(self, x):
        # x: [B, L, C]
        x = self.revin(x, 'norm')
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2)                  # [B, C, L]
        x = self.head(x)                       # [B, C, pred_len]
        x = x.transpose(1, 2)                  # [B, pred_len, C]
        x = self.revin(x, 'denorm')
        return x

def create_windows(data: np.ndarray, seq_len: int, pred_len: int):
    """Create sliding windows (inputs, targets) from a single series."""
    n = data.shape[0]
    total_len = seq_len + pred_len
    if n < total_len:
        raise ValueError("Data too short for given seq_len and pred_len")
    inputs = []
    targets = []
    for i in range(n - total_len + 1):
        inputs.append(data[i:i+seq_len])
        targets.append(data[i+seq_len:i+seq_len+pred_len])
    inputs = np.stack(inputs)   # [N, seq_len, C]
    targets = np.stack(targets) # [N, pred_len, C]
    return torch.from_numpy(inputs).float(), torch.from_numpy(targets).float()

def predict_windows(model: nn.Module, history: np.ndarray, seq_len: int, pred_len: int):
    """
    history: [N_window, history_len, C] where history_len >= seq_len.
    Returns forecasts: [N_window, pred_len, C]
    """
    model.eval()
    device = next(model.parameters()).device
    # Take the last seq_len steps for each window
    hist = history[:, -seq_len:, :]               # [N_window, seq_len, C]
    hist_tensor = torch.from_numpy(hist).to(device)  # [N_window, seq_len, C]
    with torch.no_grad():
        out = model(hist_tensor)                  # [N_window, pred_len, C]
    return out.cpu().numpy()

def train_loop(model, train_loader, val_loader, optimizer, criterion, max_epochs, patience, device):
    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0
    model.to(device)
    for epoch in range(max_epochs):
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
        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{max_epochs} - train loss: {train_loss:.6f} - val loss: {val_loss:.6f}")
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def main():
    seed = int(os.getenv("SEED", "0"))
    set_seeds(seed)

    smoke = os.getenv("SMOKE") == "1"

    # Load data
    import nullius_io as nio
    d = nio.load()
    train_data = d["train"].astype(np.float32)   # [8640,7]
    val_data   = d["val"].astype(np.float32)     # [2880,7]
    test_inputs = d["test_inputs"].astype(np.float32)  # [2785,720,7]

    seq_len = CONFIG["seq_len"]
    pred_len = 96
    enc_in = 7

    # Reduced size for speed (still same mechanism)
    eff_layers = max(1, CONFIG["layers"] // 2)   # e.g., 3
    eff_hidden = max(1, CONFIG["hidden"] // 2)   # e.g., 256

    # Prepare datasets
    train_x, train_y = create_windows(train_data, seq_len, pred_len)
    val_x,   val_y   = create_windows(val_data,   seq_len, pred_len)

    if smoke:
        limit = CONFIG["batch_size"] * 4
        train_x = train_x[:limit]
        train_y = train_y[:limit]
        val_x   = val_x[:limit]
        val_y   = val_y[:limit]
        epochs_to_run = 1
    else:
        epochs_to_run = CONFIG["epochs"]

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset   = TensorDataset(val_x,   val_y)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = TSMixer(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
                    n_block=eff_layers, dropout=CONFIG["dropout"],
                    ff_dim=eff_hidden, activation="relu")
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss()

    print("Starting training...")
    model = train_loop(model, train_loader, val_loader, optimizer, criterion,
                       max_epochs=epochs_to_run, patience=CONFIG["patience"], device=device)

    # Prepare histories for inference
    # Test: last seq_len of each test_inputs window
    test_history = test_inputs[:, -seq_len:, :]   # [2785, seq_len, 7]
    # Validation: need history before each validation window start in combined series
    combined = np.concatenate([train_data, val_data], axis=0)   # [11520,7]
    val_windows = 117
    val_start_idx = len(train_data) + 24 * np.arange(0, val_windows)  # stride 24
    val_history = np.stack([combined[start - seq_len:start, :] for start in val_start_idx], axis=0)

    print("Running inference...")
    test_pred = predict_windows(model, test_history, seq_len, pred_len)
    val_pred   = predict_windows(model, val_history, seq_len, pred_len)

    # Ensure shapes
    assert test_pred.shape == (2785, pred_len, enc_in)
    assert val_pred.shape   == (val_windows, pred_len, enc_in)

    # Write predictions via helper
    nio.write_predictions(test_pred, val_pred)

    # Optional: report validation MSE on full validation set (not smoke-limited)
    if not smoke:
        model.eval()
        with torch.no_grad():
            val_x_all, val_y_all = create_windows(val_data, seq_len, pred_len)
            val_loader_all = DataLoader(TensorDataset(val_x_all, val_y_all),
                                        batch_size=CONFIG["batch_size"], shuffle=False)
            total_se = 0.0
            total_cnt = 0
            for xb, yb in val_loader_all:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                se = ((preds - yb) ** 2).sum().item()
                total_se += se
                total_cnt += yb.numel()
        mse = total_se / total_cnt
        print(f"Validation MSE: {mse:.6f}")

if __name__ == "__main__":
    main()