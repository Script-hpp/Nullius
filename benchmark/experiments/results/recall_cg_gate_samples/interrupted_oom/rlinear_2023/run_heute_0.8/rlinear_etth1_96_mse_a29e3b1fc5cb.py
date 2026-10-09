# Plan step -> implemented in: function or class name
# 1. Read SEED from environment, seed numpy and torch. -> set_seed()
# 2. Load `/data/etth1_public.npz`; extract `train`, `val`, `test_inputs`. -> load_data()
# 3. Set input length **L=336**, horizon **H=96**, channels **C=7** (as reported). -> constants
# 4. Form training/validation samples by sliding windows: `x ∈ ℝ^{L·C}`, `y ∈ ℝ^{H·C}`. -> create_train_samples()
# 5. Model = `nn.Linear(L·C, H·C, bias=True)`  # RLinear. -> RLinear class
# 6. Loss = `nn.MSELoss()`. -> criterion variable
# 7. Optimizer = `torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)`. -> optimizer
# 8. Batch size = **256**; shuffle each epoch. -> DataLoader
# 9. Max epochs = **30**; early‑stop if validation loss does not improve for **5** epochs. -> training loop with early stopping
# 10. Train, validate each epoch, retain the best model (lowest val MSE). -> training loop
# 11. Test: for each window *i*, take the last **L** steps of `test_inputs[i]` (or the full 720 if L<720), compute `ŷ = model(x).reshape(H, C)`. -> predict_test() and predict_val()
# 12. Write predictions via `nio.write_predictions(test_pred, val_pred)`. -> main()
# 13. Total rows = 2785 × 96 × 7 = 1,871,520 (implicit in shape).
# 14. Expected runtime < 5 minutes on CPU.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import nullius_io as nio

def set_seed():
    seed = int(os.getenv("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    d = nio.load()  # same as np.load('/data/etth1_public.npz')
    train = d["train"].astype(np.float32)   # [8640, 7]
    val = d["val"].astype(np.float32)       # [2880, 7]
    test_inputs = d["test_inputs"].astype(np.float32)  # [2785, 720, 7]
    return train, val, test_inputs

# Constants
L = 336   # input length
H = 96    # horizon
C = 7     # channels
LC = L * C
HC = H * C

def create_train_samples(train):
    """Generate sliding-window samples from training data."""
    T = train.shape[0]
    xs, ys = [], []
    for i in range(T - L - H + 1):
        x = train[i:i+L].reshape(-1)          # flatten L*C
        y = train[i+L:i+L+H].reshape(-1)      # flatten H*C
        xs.append(x)
        ys.append(y)
    x_tensor = torch.from_numpy(np.stack(xs)).float()  # [N, LC]
    y_tensor = torch.from_numpy(np.stack(ys)).float()  # [N, HC]
    return TensorDataset(x_tensor, y_tensor)

class RLinear(nn.Module):
    """Simple linear mapping from flattened history to flattened future."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(LC, HC, bias=True)

    def forward(self, x):
        return self.linear(x)  # x: [batch, LC] -> [batch, HC]

def train_model(model, train_loader, val_x, val_y, device, smoke=False):
    """Train with early stopping; return best model state dict."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    max_epochs = 1 if smoke else 30
    patience = 5

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            val_pred = model(val_x)
            val_loss = criterion(val_pred, val_y).item()
        print(f"Epoch {epoch+1}/{max_epochs} - train MSE: {epoch_loss:.6f} - val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss

def prepare_validation_inputs(val_series, train_series):
    """Create validation inputs/targets using history from train+val."""
    combined = np.concatenate([train_series, val_series], axis=0)  # [11520,7]
    split = train_series.shape[0]  # 8640
    val_windows = 117
    xs, ys = [], []
    for w in range(val_windows):
        start_idx = split + 24 * w          # forecast start in combined
        x = combined[start_idx - L:start_idx].reshape(-1)   # last L steps
        y = combined[start_idx:start_idx + H].reshape(-1)   # next H steps
        xs.append(x)
        ys.append(y)
    val_x = torch.from_numpy(np.stack(xs)).float()
    val_y = torch.from_numpy(np.stack(ys)).float()
    return val_x, val_y

def predict_test(model, test_inputs, device):
    """Forecast for all test windows using last L steps."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(test_inputs.shape[0]):
            x = test_inputs[i, -L:, :].reshape(1, -1)   # [1, L*C]
            x = torch.from_numpy(x).float().to(device)
            y_hat = model(x)                            # [1, H*C]
            y_hat = y_hat.cpu().numpy().reshape(H, C)   # [96,7]
            preds.append(y_hat)
    return np.stack(preds).astype(np.float32)           # [2785,96,7]

def predict_val(model, val_series, train_series, device):
    """Forecast for validation windows."""
    val_x, _ = prepare_validation_inputs(val_series, train_series)
    model.eval()
    with torch.no_grad():
        val_x = val_x.to(device)
        val_pred = model(val_x)                     # [117, H*C]
        val_pred = val_pred.cpu().numpy().reshape(-1, H, C)  # [117,96,7]
    return val_pred.astype(np.float32)

def main():
    set_seed()
    train, val, test_inputs = load_data()
    device = torch.device("cpu")

    # Prepare training data
    train_dataset = create_train_samples(train)
    # Adjust batch size for smoke mode
    smoke = os.getenv("SMOKE") == "1"
    batch_size = 32 if smoke else 256
    # For smoke mode, use only a subset of training data (first batch)
    if smoke:
        # Take first 32 samples
        subset_indices = list(range(min(32, len(train_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Model
    model = RLinear().to(device)

    # Validation inputs/targets (full val set)
    val_x, val_y = prepare_validation_inputs(val, train)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    # Train
    model, best_val_mse = train_model(model, train_loader, val_x, val_y, device, smoke=smoke)
    print(f"Best validation MSE: {best_val_mse:.6f}")

    # Produce predictions
    test_pred = predict_test(model, test_inputs, device)   # [2785,96,7]
    val_pred = predict_val(model, val, train, device)      # [117,96,7]

    # Write outputs via helper
    nio.write_predictions(test_pred, val_pred)

if __name__ == "__main__":
    main()
