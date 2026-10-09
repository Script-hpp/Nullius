# Plan step 1 -> implemented in: set_seeds
# Plan step 2 -> implemented in: load_data
# Plan step 3 -> implemented in: DLinear class (with MovingAvg submodule)
# Plan step 4 -> implemented in: globals SEQ_LEN, PRED_LEN, MOVING_AVG_K
# Plan step 5 -> implemented in: train_loop (optimizer Adam, loss MSE)
# Plan step 6 -> implemented in: train_loop (batch size, max epochs, early stopping)
# Plan step 7 -> implemented in: get_batch (uses last L steps of series)
# Plan step 8 -> implemented in: validate_each_epoch
# Plan step 9 -> implemented in: predict_test_and_val (uses last L steps)
# Plan step 10-> implemented in: collect_predictions (shape [N,96,7])
# Plan step 11-> implemented in: nio.write_predictions (called at end)
# Plan step 12-> implemented in: torch.set_num_threads(1) and CPU tensors
# Plan step 13-> expected runtime <5 minutes (commented)
# Plan step 14-> hyperparameters as described; can be adjusted if needed

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import nullius_io as nio

def set_seeds():
    seed = int(os.environ.get('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    d = nio.load()
    train = d['train'].astype(np.float32)      # [8640,7]
    val = d['val'].astype(np.float32)          # [2880,7]
    test_inputs = d['test_inputs'].astype(np.float32)  # [2785,720,7]
    return train, val, test_inputs

class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1)//2

    def forward(self, x):
        # x: [B, L, C]
        x = x.permute(0, 2, 1)               # [B, C, L]
        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = F.avg_pool1d(x, self.kernel_size, stride=1, padding=0, count_include_pad=False)
        x = x.permute(0, 2, 1)               # [B, L, C]
        return x

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, moving_avg_kernel):
        super().__init__()
        self.moving_avg = MovingAvg(moving_avg_kernel)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C]
        trend = self.moving_avg(x)                     # [B, L, C]
        seasonal = x - trend                           # [B, L, C]

        B, L, C = x.shape
        # reshape to [B*C, L] for linear layers (shared weights across channels)
        trend = trend.reshape(B * C, L)
        seasonal = seasonal.reshape(B * C, L)

        trend_pred = self.linear_trend(trend)          # [B*C, H]
        seasonal_pred = self.linear_seasonal(seasonal) # [B*C, H]
        H = self.linear_trend.out_features

        trend_pred = trend_pred.view(B, C, H)
        seasonal_pred = seasonal_pred.view(B, C, H)
        pred = trend_pred + seasonal_pred              # [B, C, H]
        return pred.permute(0, 2, 1)                   # [B, H, C]

def create_dataset(series, seq_len, pred_len):
    """series: np.ndarray [T, C]; returns inputs [N, seq_len, C], targets [N, pred_len, C]"""
    T, C = series.shape
    N = T - seq_len - pred_len + 1
    if N <= 0:
        return np.empty((0, seq_len, C), dtype=np.float32), np.empty((0, pred_len, C), dtype=np.float32)
    inputs = np.zeros((N, seq_len, C), dtype=np.float32)
    targets = np.zeros((N, pred_len, C), dtype=np.float32)
    for i in range(N):
        inputs[i] = series[i:i+seq_len]
        targets[i] = series[i+seq_len:i+seq_len+pred_len]
    return inputs, targets

def train_loop(model, train_loader, val_loader, device, epochs, patience):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val = float('inf')
    best_state = None
    wait = 0
    model.to(device)
    for ep in range(epochs):
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
            if os.environ.get('SMOKE') == '1':
                break  # only one batch in smoke mode
        train_loss = np.mean(train_losses) if train_losses else 0.0

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())
                if os.environ.get('SMOKE') == '1':
                    break
        val_loss = np.mean(val_losses) if val_losses else 0.0
        print(f'Epoch {ep+1}: train loss {train_loss:.6f}, val loss {val_loss:.6f}')

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = model.state_dict().copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping')
                break

        if os.environ.get('SMOKE') == '1':
            break  # only one epoch in smoke mode

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict(model, inputs, device, batch_size=256):
    """inputs: np.ndarray [N, seq_len, C]; returns preds [N, pred_len, C]"""
    model.eval()
    dataset = tud.TensorDataset(torch.from_numpy(inputs))
    loader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds_list = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            pred = model(xb)               # [B, H, C]
            preds_list.append(pred.cpu().numpy())
    return np.concatenate(preds_list, axis=0)

def main():
    set_seeds()
    torch.set_num_threads(1)

    train_series, val_series, test_inputs_raw = load_data()

    # concatenate train+val for validation window creation (as per description)
    train_val = np.concatenate([train_series, val_series], axis=0)  # [11520,7]

    SEQ_LEN = 336
    PRED_LEN = 96
    MOVING_AVG_K = 25

    # --- training set ---
    train_inputs, train_targets = create_dataset(train_series, SEQ_LEN, PRED_LEN)
    # --- validation set (using train+val) ---
    val_inputs_all, val_targets_all = create_dataset(train_val, SEQ_LEN, PRED_LEN)
    # According to spec we need only 117 validation windows starting at 8640+24*w
    # The above creates many more; we will slice to the required windows.
    # Compute start indices for the 117 windows:
    val_start_base = train_series.shape[0]  # 8640
    val_indices = []
    for w in range(117):
        start = val_start_base + 24 * w
        # input is the SEQ_LEN steps before start
        inp_start = start - SEQ_LEN
        if inp_start < 0 or start + PRED_LEN > train_val.shape[0]:
            continue
        val_indices.append((inp_start, start))
    # Build arrays
    val_inputs = np.stack([train_val[s:e, :] for s, e in [(i[0], i[0]+SEQ_LEN) for i in val_indices]], axis=0)
    val_targets = np.stack([train_val[s:e, :] for s, e in [(i[1], i[1]+PRED_LEN) for i in val_indices]], axis=0)

    # In smoke mode, drastically reduce data to speed up
    if os.environ.get('SMOKE') == '1':
        # use only first 32 training samples for 1 batch
        train_inputs = train_inputs[:32]
        train_targets = train_targets[:32]
        # use first 8 validation windows
        val_inputs = val_inputs[:8]
        val_targets = val_targets[:8]

    train_dataset = tud.TensorDataset(torch.from_numpy(train_inputs), torch.from_numpy(train_targets))
    val_dataset = tud.TensorDataset(torch.from_numpy(val_inputs), torch.from_numpy(val_targets))

    batch_size = 32
    train_loader = tud.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = tud.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cpu')
    model = DLinear(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=train_series.shape[1], moving_avg_kernel=MOVING_AVG_K)

    model = train_loop(model, train_loader, val_loader, device, epochs=10, patience=3)

    # --- Test prediction ---
    # take last SEQ_LEN steps from each test_inputs_raw[i]
    if SEQ_LEN <= test_inputs_raw.shape[1]:
        test_inputs = test_inputs_raw[:, -SEQ_LEN:, :]
    else:
        test_inputs = test_inputs_raw  # fallback (should not happen)
    test_pred = predict(model, test_inputs, device, batch_size=256)  # [2785,96,7]

    # --- Validation prediction (for output) ---
    val_pred = predict(model, val_inputs, device, batch_size=256)   # [117,96,7]

    # Print validation MSE for monitoring
    val_mse = np.mean((val_pred - val_targets) ** 2)
    print(f'Validation MSE: {val_mse:.6f}')

    # Write predictions via helper
    nio.write_predictions(test_pred, val_pred)

if __name__ == '__main__':
    main()
