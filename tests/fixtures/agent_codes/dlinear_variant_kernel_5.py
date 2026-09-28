import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

class DLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, moving_avg: int = 5):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=moving_avg, stride=1, padding=(moving_avg - 1) // 2)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, L, C]
        # trend component
        trend = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)          # [B, L, C]
        seasonal = x - trend                                                # [B, L, C]
        # linear layers operate on the time dimension (L)
        trend_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)  # [B, pred_len, C]
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)  # [B, pred_len, C]
        return trend_out + seasonal_out

def create_windows(data: np.ndarray, seq_len: int, pred_len: int):
    n = data.shape[0]
    num_windows = n - seq_len - pred_len + 1
    inputs = np.zeros((num_windows, seq_len, data.shape[1]), dtype=np.float32)
    targets = np.zeros((num_windows, pred_len, data.shape[1]), dtype=np.float32)
    for i in range(num_windows):
        inputs[i] = data[i:i + seq_len]
        targets[i] = data[i + seq_len:i + seq_len + pred_len]
    return inputs, targets

def main():
    seed = int(os.environ.get("SEED", "0"))
    set_seed(seed)

    # Load data
    data = np.load('/data/etth1_public.npz')
    train_data = data['train'].astype(np.float32)   # [8640, 7]
    val_data = data['val'].astype(np.float32)       # [2880, 7]
    test_inputs = data['test_inputs'].astype(np.float32)  # [2785, 720, 7]

    seq_len = 336
    pred_len = 96
    moving_avg = 5
    batch_size = 32
    lr = 1e-3
    max_epochs = 20
    patience = 5

    # Prepare training and validation windows
    train_x, train_y = create_windows(train_data, seq_len, pred_len)
    val_x, val_y = create_windows(val_data, seq_len, pred_len)

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device('cpu')
    model = DLinear(seq_len, pred_len, moving_avg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
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
        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch:02d} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss - 1e-8:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation MSE (using best model)
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_losses.append(loss.item())
    final_val_mse = np.mean(val_losses)
    print(f"Final Validation MSE: {final_val_mse:.6f}")

    # Test inference
    model.eval()
    num_test = test_inputs.shape[0]
    lines = ["id,prediction"]
    with torch.no_grad():
        for w in range(num_test):
            hist = test_inputs[w, -seq_len:, :]          # [seq_len, 7]
            hist_tensor = torch.from_numpy(hist).unsqueeze(0).to(device)  # [1, seq_len, 7]
            pred = model(hist_tensor)                    # [1, pred_len, 7]
            pred_np = pred.squeeze(0).cpu().numpy()      # [pred_len, 7]
            for t in range(pred_len):
                for c in range(test_inputs.shape[2]):
                    lines.append(f"{w}_{t}_{c},{pred_np[t, c]:.6f}")

    os.makedirs('/output', exist_ok=True)
    with open('/output/predictions.csv', 'w') as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
