# Plan step -> implemented in: function or class name
# 1. Read SEED and set seeds -> set_seeds()
# 2. Load data -> load_data()
# 3. Define DLinear model -> DLinear class
# 4. Set input length L and horizon H -> variables in main()
# 5. Define optimizer and loss -> configure_optim()
# 6. Training loop with early stopping -> train_model()
# 7. Batch processing -> get_batches()
# 8. Validation -> validate()
# 9. Test inference -> predict_test()
# 10. Collect predictions -> collect_predictions()
# 11. Write predictions via helper -> nio.write_predictions()
# 12. CPU-only execution -> torch.set_num_threads(1)
# 13. Runtime <5 min -> lightweight DLinear
# 14. Hyperparameters as described

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def set_seeds():
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    import nullius_io as nio
    d = nio.load()          # keys: train, val, test_inputs
    train = d['train'].astype(np.float32)   # [8640,7]
    val = d['val'].astype(np.float32)       # [2880,7]
    test_inputs = d['test_inputs'].astype(np.float32)  # [2785,720,7]
    return train, val, test_inputs

class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)

    def forward(self, x):
        # x: [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)
        x = self.avg_pool(x)
        # [B, C, L] -> [B, L, C]
        return x.permute(0, 2, 1)

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, moving_avg_kernel):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.moving_avg = MovingAvg(moving_avg_kernel)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C]
        trend = self.moving_avg(x)
        seasonal = x - trend
        B, L, C = x.shape
        # trend & seasonal: [B, L, C] -> [B*C, L]
        trend = trend.reshape(B * C, L)
        seasonal = seasonal.reshape(B * C, L)
        trend_out = self.linear_trend(trend)      # [B*C, H]
        seasonal_out = self.linear_seasonal(seasonal)  # [B*C, H]
        out = trend_out + seasonal_out            # [B*C, H]
        out = out.reshape(B, C, self.pred_len)    # [B, C, H]
        return out.permute(0, 2, 1)               # [B, H, C]

def create_samples(data, seq_len, pred_len):
    """Generate sliding window samples from a 2D array [T, C]."""
    T, C = data.shape
    inputs = []
    targets = []
    for i in range(T - seq_len - pred_len + 1):
        inputs.append(data[i:i+seq_len])
        targets.append(data[i+seq_len:i+seq_len+pred_len])
    inputs = np.stack(inputs)   # [N, L, C]
    targets = np.stack(targets) # [N, H, C]
    return inputs, targets

class Seq2SeqDataset(TensorDataset):
    def __init__(self, inputs, targets):
        super().__init__(
            torch.from_numpy(inputs),
            torch.from_numpy(targets)
        )

def get_batches(data, seq_len, pred_len, batch_size, shuffle):
    inputs, targets = create_samples(data, seq_len, pred_len)
    dataset = Seq2SeqDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def train_model(model, train_loader, val_loader, epochs, lr, patience, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    best_state = None
    wait = 0
    for epoch in range(1, epochs+1):
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
        train_loss = np.mean(train_losses)

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

        print(f"Epoch {epoch}: train MSE {train_loss:.6f}, val MSE {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict().copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val

def predict_test(model, test_inputs, seq_len, device):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(test_inputs.shape[0]):
            hist = test_inputs[i]                     # [720, C]
            if seq_len < hist.shape[0]:
                hist = hist[-seq_len:, :]             # [L, C]
            else:
                hist = hist                           # [720, C] (if L>=720)
            hist_tensor = torch.from_numpy(hist).unsqueeze(0).to(device)  # [1, L, C]
            pred = model(hist_tensor)                 # [1, H, C]
            pred = pred.squeeze(0).cpu().numpy()      # [H, C]
            preds_list.append(pred)
    return np.stack(preds_list)                       # [N, H, C]

def predict_validation(model, train, val, seq_len, pred_len, device):
    """Create val_pred of shape [117, 96, 7] as required."""
    combined = np.concatenate([train, val], axis=0)   # [11520, 7]
    total_len = combined.shape[0]
    val_preds = []
    for w in range(117):
        start = 8640 + 24 * w                         # forecast start index
        hist_start = start - seq_len
        hist_end = start
        hist = combined[hist_start:hist_end]           # [L, 7]
        hist_tensor = torch.from_numpy(hist).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(hist_tensor)                  # [1, 96, 7]
        pred = pred.squeeze(0).cpu().numpy()
        val_preds.append(pred)
    return np.stack(val_preds)                        # [117, 96, 7]

def main():
    set_seeds()
    torch.set_num_threads(1)
    device = torch.device('cpu')

    train, val, test_inputs = load_data()

    # Hyperparameters (can be tuned)
    L = 336          # input length
    H = 96           # horizon
    moving_avg_kernel = 25
    enc_in = 7       # number of channels
    batch_size = 32
    lr = 0.001
    epochs = 20
    patience = 5

    # Prepare data loaders
    train_loader = get_batches(train, L, H, batch_size, shuffle=True)
    val_loader   = get_batches(val,   L, H, batch_size, shuffle=False)

    # Build model
    model = DLinear(seq_len=L, pred_len=H, enc_in=enc_in, moving_avg_kernel=moving_avg_kernel)

    # Smoke mode handling
    smoke = os.environ.get("SMOKE", "0") == "1"
    if smoke:
        # Limit to 1 batch for 1 epoch
        print("SMOKE mode: reduced training")
        epochs = 1
        # We'll still iterate through loader but break after first batch
        original_len = len(train_loader)
        # Wrap loader to yield only first batch
        class LimitedLoader:
            def __init__(self, loader):
                self.loader = loader
            def __iter__(self):
                self.iter_iter = iter(self.loader)
                return self
            def __next__(self):
                return next(self.iter_iter)
            def __len__(self):
                return 1
        train_loader = LimitedLoader(train_loader)
        val_loader   = LimitedLoader(val_loader)

    # Train
    model, best_val_mse = train_model(model, train_loader, val_loader,
                                      epochs=epochs, lr=lr,
                                      patience=patience, device=device)
    print(f"Best validation MSE: {best_val_mse:.6f}")

    # Produce validation predictions for submission
    val_pred = predict_validation(model, train, val, L, H, device)  # [117,96,7]

    # Produce test predictions
    test_pred = predict_test(model, test_inputs, L, device)         # [2785,96,7]

    # Write predictions via helper
    import nullius_io as nio
    nio.write_predictions(test_pred, val_pred)

if __name__ == "__main__":
    main()
