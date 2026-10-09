# Plan step -> implemented in: function or class name
# 1. Model – NLinear: NLinear class
# 2. Input / output lengths – L=336, H=96 defined as constants
# 3. Data preparation – create_windows function and DataLoader usage
# 4. Optimizer – Adam instantiated in train function
# 5. Loss – MSE loss via torch.nn.MSELoss
# 6. Training schedule – train function with max epochs, early stopping, batch size
# 7. Validation – validate function using provided val split
# 8. Seeding – set_seed function called at start
# 9. Inference – predict_windows function for test and val predictions
#10. Output – nio.write_predictions called at end
#11. Runtime – CPU‑only loops, finishes within limits

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import nullius_io as nio

# -------------------- hyperparameters --------------------
L = 336          # history length
H = 96           # forecast horizon
LR = 0.001       # learning rate
BATCH_SIZE = 256
MAX_EPOCHS = 20
PATIENCE = 5     # early stopping patience
# ---------------------------------------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_windows(data: np.ndarray):
    """Create sliding windows (x, y) from a 2D np.ndarray [T, C].
    Returns torch tensors xs [N, L, C], ys [N, H, C]."""
    T, C = data.shape
    N = T - L - H + 1
    xs = np.zeros((N, L, C), dtype=np.float32)
    ys = np.zeros((N, H, C), dtype=np.float32)
    for i in range(N):
        xs[i] = data[i:i+L]
        ys[i] = data[i+L:i+L+H]
    return torch.from_numpy(xs), torch.from_numpy(ys)

class NLinear(nn.Module):
    """NLinear model: separate linear layer per channel mapping L -> H."""
    def __init__(self, enc_len: int, dec_len: int, channels: int):
        super().__init__()
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.channels = channels
        # one Linear per channel: weight [dec_len, enc_len], bias [dec_len]
        self.linears = nn.ModuleList([
            nn.Linear(enc_len, dec_len, bias=True) for _ in range(channels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, L, C]
        returns: [batch, H, C]
        """
        batch = x.size(0)
        outs = []
        for c in range(self.channels):
            x_c = x[:, :, c]               # [batch, L]
            y_c = self.linears[c](x_c)     # [batch, H]
            outs.append(y_c)
        # stack along channel dimension -> [batch, H, C]
        return torch.stack(outs, dim=2)

def train_model(model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0)
    criterion = nn.MSELoss()
    best_val = float('inf')
    epochs_no_improve = 0

    model.train()
    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train = epoch_loss / max(n_batches, 1)

        # validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
                n_val += 1
        avg_val = val_loss / max(n_val, 1)
        model.train()

        print(f"Epoch {epoch+1}/{MAX_EPOCHS} - train MSE: {avg_train:.6f} - val MSE: {avg_val:.6f}")

        if avg_val < best_val - 1e-8:
            best_val = avg_val
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    # restore best model
    model.load_state_dict(best_state)
    return best_val

def predict_windows(model, histories: np.ndarray, device):
    """
    histories: [N, L, C] np.ndarray (already standardized)
    returns: [N, H, C] np.ndarray predictions
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for hist in histories:
            x = torch.from_numpy(hist).unsqueeze(0).to(device)   # [1, L, C]
            out = model(x)                                      # [1, H, C]
            preds.append(out.squeeze(0).cpu().numpy())
    return np.stack(preds, axis=0)

def main():
    seed = int(os.environ.get("SEED", "0"))
    set_seed(seed)

    smoke = os.environ.get("SMOKE") == "1"

    # load data
    data = nio.load()
    train_np = data["train"].astype(np.float32)   # [8640,7]
    val_np   = data["val"].astype(np.float32)     # [2880,7]
    test_np  = data["test_inputs"].astype(np.float32)  # [2785,720,7]

    device = torch.device("cpu")

    # ----- prepare training windows -----
    x_train, y_train = create_windows(train_np)   # from train only
    if smoke:
        # limit to first 32 samples, one epoch
        x_train = x_train[:32]
        y_train = y_train[:32]

    # ----- prepare validation windows for early stopping -----
    # use the provided val split (same windowing)
    x_val, y_val = create_windows(val_np)

    train_ds = TensorDataset(x_train, y_train)
    val_ds   = TensorDataset(x_val, y_val)

    batch_size = BATCH_SIZE if not smoke else 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # ----- model -----
    model = NLinear(enc_len=L, dec_len=H, channels=train_np.shape[1]).to(device)

    # ----- train -----
    best_val_mse = train_model(model, train_loader, val_loader, device)
    print(f"Best validation MSE: {best_val_mse:.6f}")

    # ----- inference on test -----
    # take last L steps of each test_inputs[i]
    test_hist = test_np[:, -L:, :]          # [2785, L, 7]
    if smoke:
        test_hist = test_hist[:8]           # at least first 8 windows
    test_pred = predict_windows(model, test_hist, device)   # [N,96,7]

    # ----- inference on validation windows (as required for output) -----
    # Build combined series train+val
    combined = np.concatenate([train_np, val_np], axis=0)   # [11520,7]
    val_start_base = train_np.shape[0]                     # 8640
    num_val_windows = 117
    val_histories = []
    for w in range(num_val_windows):
        start_idx = val_start_base + 24 * w
        hist = combined[start_idx - L:start_idx, :]   # [L,7]
        val_histories.append(hist)
    val_histories = np.stack(val_histories, axis=0)   # [117, L, 7]
    if smoke:
        val_histories = val_histories[:8]
    val_pred = predict_windows(model, val_histories, device)   # [117,96,7]

    # ----- write predictions -----
    nio.write_predictions(test_pred, val_pred)

if __name__ == "__main__":
    main()
