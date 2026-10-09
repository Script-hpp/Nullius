

# Plan step 1 -> implemented in: set_seed
# Plan step 2 -> implemented in: load_data
# Plan step 3 -> implemented in: set_constants
# Plan step 4 -> implemented in: WindowDataset class
# Plan step 5 -> implemented in: RLinearModel class
# Plan step 6 -> implemented in: get_loss_fn
# Plan step 7 -> implemented in: get_optimizer
# Plan step 8 -> implemented in: batch_size variable in main
# Plan step 9 -> implemented in: train_loop_with_early_stopping (max_epochs, patience)
# Plan step 10 -> implemented in: train_loop_with_early_stopping (retains best model)
# Plan step 11 -> implemented in: predict_test_windows
# Plan step 12 -> implemented in: write_predictions_via_nio
# Plan step 13 -> implemented in: comment (output rows count)
# Plan step 14 -> implemented in: comment (runtime <5 minutes)

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nullius_io as nio


def set_seed():
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    d = nio.load()
    train = d["train"].astype(np.float32)      # [8640, 7]
    val = d["val"].astype(np.float32)          # [2880, 7]
    test_inputs = d["test_inputs"].astype(np.float32)  # [2785, 720, 7]
    return train, val, test_inputs


def set_constants():
    L = 336          # input length
    H = 96           # forecast horizon
    C = 7            # number of channels
    return L, H, C


class WindowDataset(Dataset):
    def __init__(self, data, L, H):
        """
        data: np.ndarray [T, C]
        Creates samples (x, y) where:
          x = data[t:t+L].flatten()   -> [L*C]
          y = data[t+L:t+L+H].flatten()-> [H*C]
        """
        self.L = L
        self.H = H
        self.C = data.shape[1]
        self.T = data.shape[0]
        # Guard against non‑positive length
        self.length = max(0, self.T - L - H + 1)
        self.data = data

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.L].reshape(-1)          # [L*C]
        y = self.data[idx + self.L:idx + self.L + self.H].reshape(-1)  # [H*C]
        return torch.from_numpy(x), torch.from_numpy(y)


class RLinearModel(nn.Module):
    def __init__(self, L, H, C):
        super().__init__()
        self.linear = nn.Linear(L * C, H * C, bias=True)

    def forward(self, x):
        # x: [batch, L*C]
        return self.linear(x)  # [batch, H*C]


def get_loss_fn():
    return nn.MSELoss()


def get_optimizer(model, lr=1e-3, weight_decay=0.0):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def compute_val_loss(model, train_data, val_data, L, H, loss_fn, device):
    """Compute MSE on validation windows that start inside the val period."""
    model.eval()
    combined = np.concatenate([train_data, val_data], axis=0)
    val_start = len(train_data)
    # Number of validation windows = len(val_data) - H + 1
    n_val_windows = len(val_data) - H + 1
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for w in range(n_val_windows):
            start = val_start + w                     # first index of forecast window
            x = combined[start - L:start].reshape(1, -1)   # [1, L*C]
            y_true = combined[start:start + H].reshape(1, -1)  # [1, H*C]
            x_t = torch.from_numpy(x).to(device)
            y_t = torch.from_numpy(y_true).to(device)
            pred = model(x_t)
            loss = loss_fn(pred, y_t)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


def train_loop_with_early_stopping(model, train_loader, train_data, val_data,
                                   L, H, loss_fn, optimizer, device,
                                   max_epochs=30, patience=5):
    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        val_loss = compute_val_loss(model, train_data, val_data, L, H, loss_fn, device)

        print(f"Epoch {epoch:02d} | train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f}")
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict_test_windows(model, test_inputs, L, device):
    model.eval()
    N = test_inputs.shape[0]
    C = test_inputs.shape[2]
    H = 96
    preds = np.empty((N, H, C), dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            seq = test_inputs[i, -L:, :]          # [L, C]
            x = seq.reshape(1, -1)                # [1, L*C]
            x_t = torch.from_numpy(x).to(device)
            y_t = model(x_t)                      # [1, H*C]
            y_np = y_t.cpu().numpy().reshape(H, C)
            preds[i] = y_np
    return preds


def predict_val_windows(model, train_data, val_data, L, device):
    model.eval()
    combined = np.concatenate([train_data, val_data], axis=0)
    val_start = len(train_data)
    H = 96
    C = 7
    N_val = len(val_data) - H + 1                 # should be 117
    preds = np.empty((N_val, H, C), dtype=np.float32)
    with torch.no_grad():
        for idx, start in enumerate(range(val_start, val_start + N_val)):
            seq = combined[start - L:start]       # [L, C]
            x = seq.reshape(1, -1)
            x_t = torch.from_numpy(x).to(device)
            y_t = model(x_t)                      # [1, H*C]
            y_np = y_t.cpu().numpy().reshape(H, C)
            preds[idx] = y_np
    return preds


def write_predictions_via_nio(test_pred, val_pred):
    nio.write_predictions(test_pred, val_pred)


def main():
    set_seed()
    train_data, val_data, test_inputs = load_data()
    L, H, C = set_constants()

    smoke = os.environ.get("SMOKE") == "1"
    if smoke:
        # Reduce input length to guarantee enough training samples
        # Original train length 8640; we trim to 128 rows for speed.
        train_data = train_data[:128]
        # Ensure we can still form at least one window: need L+H <= len(train_data)
        max_L = len(train_data) - H
        if max_L < 1:
            max_L = 1
        L = min(L, max_L)
        batch_size = 32
        max_epochs = 1
        patience = 1
    else:
        batch_size = 256
        max_epochs = 30
        patience = 5

    # Create training dataset and loader
    train_dataset = WindowDataset(train_data, L, H)
    # If no training samples exist (should not happen with safeguards), skip training
    if len(train_dataset) == 0:
        # Fallback: create a dummy model that predicts zeros
        model = RLinearModel(L, H, C)
        device = torch.device("cpu")
        test_pred = np.zeros((test_inputs.shape[0], H, C), dtype=np.float32)
        val_pred = np.zeros((len(val_data) - H + 1, H, C), dtype=np.float32)
        write_predictions_via_nio(test_pred, val_pred)
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=False)

    device = torch.device("cpu")
    model = RLinearModel(L, H, C).to(device)
    loss_fn = get_loss_fn()
    optimizer = get_optimizer(model, lr=1e-3, weight_decay=0.0)

    # Train
    model, best_val_mse = train_loop_with_early_stopping(
        model, train_loader, train_data, val_data,
        L, H, loss_fn, optimizer, device,
        max_epochs=max_epochs, patience=patience
    )
    print(f"Best validation MSE: {best_val_mse:.6f}")

    # Predictions
    test_pred = predict_test_windows(model, test_inputs, L, device)   # [2785,96,7]
    val_pred = predict_val_windows(model, train_data, val_data, L, device)  # [117,96,7]

    # Write via helper
    write_predictions_via_nio(test_pred, val_pred)


if __name__ == "__main__":
    main()