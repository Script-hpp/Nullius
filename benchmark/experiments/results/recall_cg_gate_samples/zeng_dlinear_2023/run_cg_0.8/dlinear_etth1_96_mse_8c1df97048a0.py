

# Plan step -> implemented in: function or class name
# Load data and set seeds -> main
# Build DLinear model -> DLinear class
# Create training/validation datasets -> create_windows function
# Training loop with early stopping and LR schedule -> train_model function
# Inference for test and validation windows -> predict_windows function
# Write predictions -> nio.write_predictions call

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Hyperparameters read only from this dict
CONFIG = {"lr": 0.005, "epochs": 10, "batch_size": 32, "dropout": 0.05, "patience": 7, "seq_len": 336}
# Additional keys required for the model/data
CONFIG["weight_decay"] = 0.0
CONFIG["pred_len"] = 96
CONFIG["enc_in"] = 7
CONFIG["kernel_size"] = 25

# ------------------ Model definition ------------------
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.seq_len = configs["seq_len"]
        self.pred_len = configs["pred_len"]
        kernel_size = configs["kernel_size"]
        self.decomposition = series_decomp(kernel_size)
        self.individual = configs.get("individual", False)
        self.channels = configs["enc_in"]
        self.dropout = nn.Dropout(p=configs["dropout"])

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = self.dropout(x)  # Apply dropout as per CONFIG["dropout"]
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


# ------------------ Utility functions ------------------
def create_windows(data, seq_len, pred_len):
    """Return (X, Y) where X.shape = (N, seq_len, C), Y.shape = (N, pred_len, C)."""
    N = data.shape[0] - seq_len - pred_len + 1
    C = data.shape[1]
    X = np.zeros((N, seq_len, C), dtype=np.float32)
    Y = np.zeros((N, pred_len, C), dtype=np.float32)
    for i in range(N):
        X[i] = data[i:i + seq_len]
        Y[i] = data[i + seq_len:i + seq_len + pred_len]
    return X, Y


def adjust_learning_rate(optimizer, epoch, args):
    """Type 1 schedule: lr * 0.5^{(epoch-1)}."""
    lr = args["lr"] * (0.5 ** ((epoch - 1) // 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f'Updating learning rate to {lr}')


def train_model(model, train_loader, val_loader, optimizer, criterion, config, device):
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    patience = config["patience"]  # read patience from CONFIG

    for epoch in range(config["epochs"]):
        # LR schedule (epoch index starts from 1)
        adjust_learning_rate(optimizer, epoch + 1, config)
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        print(f'Epoch {epoch + 1}/{config["epochs"]} | Train Loss: {np.mean(train_losses):.6f} | Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def predict_windows(model, data, seq_len, pred_len, device, indices_func):
    """
    Generic prediction wrapper.
    indices_func(i) -> slice of data to use as input (length seq_len).
    Returns predictions array of shape (len(indices), pred_len, C).
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for idx in indices_func():
            inp = data[idx]  # shape (seq_len, C)
            inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(device)  # (1, seq_len, C)
            out = model(inp_tensor)  # (1, pred_len, C)
            preds.append(out.squeeze(0).cpu().numpy())
    return np.stack(preds, axis=0)


def main():
    # Seed
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Smoke mode
    smoke = os.environ.get("SMOKE") == "1"

    # Load data
    import nullius_io as nio
    d = nio.load()
    train = d["train"].astype(np.float32)      # (8640, 7)
    val = d["val"].astype(np.float32)          # (2880, 7)
    test_inputs = d["test_inputs"].astype(np.float32)  # (2785, 720, 7)

    # Combine train+val for validation window creation
    train_val = np.concatenate([train, val], axis=0)   # (11520, 7)

    # Device
    device = torch.device("cpu")

    # ------------------ Prepare datasets for training ------------------
    seq_len = CONFIG["seq_len"]
    pred_len = CONFIG["pred_len"]

    X_train, Y_train = create_windows(train, seq_len, pred_len)
    X_val, Y_val = create_windows(val, seq_len, pred_len)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

    # In smoke mode, restrict to a single batch for speed
    if smoke:
        # Take first batch only
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
        # We'll just iterate once; but DataLoader will yield batches; we limit epochs to 1 anyway
        CONFIG["epochs"] = 1
    else:
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # ------------------ Model, optimizer, loss ------------------
    model = DLinear(CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CONFIG["lr"],
                                 weight_decay=CONFIG["weight_decay"])
    criterion = nn.MSELoss()

    # ------------------ Training ------------------
    trained_model, best_val_loss = train_model(model, train_loader, val_loader,
                                               optimizer, criterion, CONFIG, device)
    print(f'Best validation MSE: {best_val_loss:.6f}')

    # ------------------ Inference ------------------
    # Test predictions: use last seq_len of each test_inputs[i]
    test_pred_list = []
    for i in range(test_inputs.shape[0]):
        inp = test_inputs[i, -seq_len:, :]  # (seq_len, 7)
        test_pred_list.append(inp)
    test_pred_array = np.stack(test_pred_list, axis=0)  # (2785, seq_len, 7)
    test_pred = predict_windows(trained_model, test_pred_array, seq_len, pred_len,
                                device,
                                lambda: range(test_pred_array.shape[0]))

    # Validation predictions: windows w=0..N-1 forecasting from train_val
    # According to spec: validation window w forecasts 96 hours starting at row
    # train_len + 24*w, where stride = 24.
    train_len = train.shape[0]          # 8640
    val_len = val.shape[0]              # 2880
    stride = 24                         # fixed stride for validation windows
    num_val_windows = (val_len - pred_len) // stride + 1  # 117
    val_pred_list = []
    for w in range(num_val_windows):
        start_idx = train_len + stride * w  # forecast start in train_val
        inp_start = start_idx - seq_len
        inp_end = start_idx
        inp = train_val[inp_start:inp_end, :]  # (seq_len, 7)
        val_pred_list.append(inp)
    val_pred_array = np.stack(val_pred_list, axis=0)  # (num_val_windows, seq_len, 7)
    val_pred = predict_windows(trained_model, val_pred_array, seq_len, pred_len,
                               device,
                               lambda: range(val_pred_array.shape[0]))

    # Write predictions
    nio.write_predictions(test_pred, val_pred)


if __name__ == "__main__":
    main()