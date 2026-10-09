# Plan step -> implemented in: function or class name
# 1. Read SEED and set seeds -> set_seeds()
# 2. Load data -> load_data()
# 3. Model (DLinear) -> DLinear class
# 4. Set input length L=336, horizon H=96 -> constants
# 5. Optimiser Adam(lr=0.001), loss MSE -> train_loop()
# 6. Training batch size=32, max epochs=10, early stop patience=3 -> train_loop()
# 7. Batch processing -> get_batches()
# 8. Validation each epoch -> evaluate()
# 9. Inference on test and validation windows -> predict_test(), predict_val()
#10. Collect predictions -> shape (N,96,7)
#11. Flatten and write via nullius_io -> nio.write_predictions()
#12. CPU-only execution -> torch.set_num_threads(1)
#13. Runtime <5 minutes (full run) / <60 sec (SMOKE)
#14. Hyperparameters as described; smaller version used if needed (see comment)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import nullius_io as nio


def set_seeds():
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    # Returns dict with keys: train, val, test_inputs
    d = nio.load()
    train = d["train"].astype(np.float32)   # [8640,7]
    val = d["val"].astype(np.float32)       # [2880,7]
    test_inputs = d["test_inputs"].astype(np.float32)  # [2785,720,7]
    return train, val, test_inputs


class MovingAverage(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        # x: [B, L, C]
        x = x.permute(0, 2, 1)          # [B, C, L]
        x = self.avg(x)                 # [B, C, L]
        x = x.permute(0, 2, 1)          # [B, L, C]
        return x


class DLinear(nn.Module):
    """
    Decomposition-Linear model as described in the plan.
    """
    def __init__(self, seq_len, pred_len, enc_in, moving_avg_kernel=25):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.moving_avg = MovingAverage(moving_avg_kernel)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C]
        trend = self.moving_avg(x)                  # [B, L, C]
        seasonal = x - trend                        # [B, L, C]
        trend_out = self.linear_trend(trend)        # [B, H, C]
        seasonal_out = self.linear_seasonal(seasonal)  # [B, H, C]
        return trend_out + seasonal_out             # [B, H, C]


def create_dataset(series, seq_len, pred_len):
    """
    series: np.ndarray [T, C]
    Returns TensorDataset of (input, target) where input shape [seq_len, C],
    target shape [pred_len, C].
    """
    T = series.shape[0]
    inputs = []
    targets = []
    for i in range(T - seq_len - pred_len + 1):
        inputs.append(series[i:i + seq_len])
        targets.append(series[i + seq_len:i + seq_len + pred_len])
    inputs = np.stack(inputs)   # [N, seq_len, C]
    targets = np.stack(targets) # [N, pred_len, C]
    return TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))


def get_batches(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_loop(model, train_loader, val_loader, epochs, lr, patience, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1}/{epochs} - train loss: {train_loss:.6f} - val loss: {val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict_test(model, test_inputs, seq_len, pred_len, device):
    """
    test_inputs: np.ndarray [N, 720, C]
    Returns np.ndarray [N, pred_len, C]
    """
    model.eval()
    N = test_inputs.shape[0]
    preds = []
    with torch.no_grad():
        for i in range(N):
            # use last seq_len steps (or full if seq_len>720)
            hist = test_inputs[i, -seq_len:, :]   # [seq_len, C]
            hist_tensor = torch.from_numpy(hist).unsqueeze(0).to(device)  # [1, seq_len, C]
            out = model(hist_tensor)              # [1, pred_len, C]
            preds.append(out.squeeze(0).cpu().numpy())
    return np.stack(preds)   # [N, pred_len, C]


def predict_val(model, train, val, seq_len, pred_len, device):
    """
    Build validation windows from concatenated train+val series.
    Returns np.ndarray [117, pred_len, C]
    """
    full = np.concatenate([train, val], axis=0)   # [11520, C]
    start_val = train.shape[0]                    # 8640
    window_step = 24
    n_windows = 117
    preds = []
    model.eval()
    with torch.no_grad():
        for w in range(n_windows):
            start_idx = start_val + w * window_step   # start of forecast window
            hist = full[start_idx - seq_len:start_idx, :]   # [seq_len, C]
            hist_tensor = torch.from_numpy(hist).unsqueeze(0).to(device)
            out = model(hist_tensor)               # [1, pred_len, C]
            preds.append(out.squeeze(0).cpu().numpy())
    return np.stack(preds)   # [117, pred_len, C]


def main():
    set_seeds()
    torch.set_num_threads(1)

    train, val, test_inputs = load_data()

    # Hyperparameters (as per plan)
    SEQ_LEN = 336          # L
    PRED_LEN = 96          # H
    ENC_IN = train.shape[1]   # 7
    BATCH_SIZE = 32
    LR = 0.001
    MAX_EPOCHS = 10
    PATIENCE = 3
    DEVICE = torch.device("cpu")

    # Smoke mode handling
    smoke = os.environ.get("SMOKE") == "1"

    if smoke:
        # Use only first batch of training data for 1 epoch
        train_dataset = create_dataset(train, SEQ_LEN, PRED_LEN)
        train_loader = get_batches(train_dataset, BATCH_SIZE, shuffle=True)
        # Take only first batch
        train_loader = [next(iter(train_loader))]
        val_dataset = create_dataset(val, SEQ_LEN, PRED_LEN)
        val_loader = get_batches(val_dataset, BATCH_SIZE, shuffle=False)
        # Limit to first batch for validation as well (just to compute loss)
        val_loader = [next(iter(val_loader))]
        epochs = 1
    else:
        train_dataset = create_dataset(train, SEQ_LEN, PRED_LEN)
        train_loader = get_batches(train_dataset, BATCH_SIZE, shuffle=True)
        val_dataset = create_dataset(val, SEQ_LEN, PRED_LEN)
        val_loader = get_batches(val_dataset, BATCH_SIZE, shuffle=False)
        epochs = MAX_EPOCHS

    model = DLinear(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=ENC_IN)

    print("Starting training...")
    model, val_mse = train_loop(model, train_loader, val_loader,
                                epochs=epochs, lr=LR, patience=PATIENCE, device=DEVICE)
    print(f"Validation MSE: {val_mse:.6f}")

    # Inference
    if smoke:
        # Predict first 8 test windows and first 8 validation windows (as required)
        test_pred = predict_test(model, test_inputs[:8], SEQ_LEN, PRED_LEN, DEVICE)
        val_pred = predict_val(model, train, val, SEQ_LEN, PRED_LEN, DEVICE)[:8]
    else:
        test_pred = predict_test(model, test_inputs, SEQ_LEN, PRED_LEN, DEVICE)
        val_pred = predict_val(model, train, val, SEQ_LEN, PRED_LEN, DEVICE)

    # Write predictions via helper
    nio.write_predictions(test_pred, val_pred)


if __name__ == "__main__":
    main()
