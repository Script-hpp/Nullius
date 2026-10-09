

# Plan step -> implemented in: load_data
# Plan step -> implemented in: DLinear class
# Plan step -> implemented in: get_criterion
# Plan step -> implemented in: get_optimizer
# Plan step -> implemented in: adjust_learning_rate
# Plan step -> implemented in: EarlyStopping class
# Plan step -> implemented in: train_one_epoch
# Plan step -> implemented in: evaluate
# Plan step -> implemented in: predict_test
# Plan step -> implemented in: predict_val
# Plan step -> implemented in: main

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

# Hyperparameters (read only from this dict)
CONFIG = {"lr": 0.005, "epochs": 10, "batch_size": 32, "dropout": 0.05, "patience": 7, "seq_len": 336}

# Other fixed constants (from paper / spec)
KERNEL_SIZE = 25          # moving average kernel size in DLinear
INDIVIDUAL = False        # DLinear shares weights across channels
PRED_LEN = 96             # forecast horizon
ENC_IN = 7                # number of channels (ETTh1)
STEP = 24                 # sampling step for creating windows (1 day)


def load_data():
    """Load train, val and test_inputs from the provided npz file via nullius_io."""
    d = nio.load()
    train = d["train"].astype(np.float32)      # shape [8640, 7]
    val = d["val"].astype(np.float32)          # shape [2880, 7]
    test_inputs = d["test_inputs"].astype(np.float32)  # shape [2785, 720, 7]
    return train, val, test_inputs


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
    Decomposition-Linear model as specified with dropout applied to decomposed components
    """
    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.seq_len = configs["seq_len"]
        self.pred_len = configs["pred_len"]
        self.channels = configs["enc_in"]

        # Decomposition Kernel Size
        kernel_size = configs.get("kernel_size", KERNEL_SIZE)
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.get("individual", INDIVIDUAL)
        self.dropout = nn.Dropout(configs.get("dropout", CONFIG["dropout"]))

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
        seasonal_init, trend_init = self.decompsition(x)
        # Apply dropout to decomposed components
        seasonal_init = self.dropout(seasonal_init)
        trend_init = self.dropout(trend_init)
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
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


def get_criterion():
    """Mean Squared Error loss."""
    return nn.MSELoss()


def get_optimizer(model):
    """Adam optimizer with betas=(0.9,0.999) and weight_decay=0."""
    return torch.optim.Adam(model.parameters(),
                            lr=CONFIG["lr"],
                            betas=(0.9, 0.999),
                            weight_decay=0)


def adjust_learning_rate(optimizer, epoch):
    """
    Type 1 learning rate adjustment:
    lr = lr_initial * (0.5 ** ((epoch-1)//1))
    """
    lr = CONFIG["lr"] * (0.5 ** ((epoch - 1) // 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f'Updating learning rate to {lr}')


class EarlyStopping:
    """Early stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)               # [B, pred_len, C]
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item() * batch_x.size(0)
    val_loss = running_loss / len(loader.dataset)
    return val_loss


def create_windows(data, seq_len, pred_len, step=STEP):
    """
    Create sliding windows (inputs, targets) from a 2D array.
    Inputs: [num_windows, seq_len, channels]
    Targets: [num_windows, pred_len, channels]
    Windows start at indices: seq_len, seq_len+step, ..., up to len(data)-pred_len
    """
    n = data.shape[0]
    starts = np.arange(seq_len, n - pred_len + 1, step)
    inputs = np.stack([data[s - seq_len:s] for s in starts], axis=0)
    targets = np.stack([data[s:s + pred_len] for s in starts], axis=0)
    return inputs, targets


def predict_test(model, test_inputs, seq_len, device):
    """Predict for all test windows using the last `seq_len` steps of each test_inputs[i]."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(test_inputs.shape[0]):
            hist = test_inputs[i, -seq_len:, :]          # [seq_len, 7]
            hist_tensor = torch.from_numpy(hist).unsqueeze(0).to(device)  # [1, seq_len, 7]
            out = model(hist_tensor)                     # [1, pred_len, 7]
            preds.append(out.squeeze(0).cpu().numpy())
    return np.stack(preds, axis=0)                       # [num_test, pred_len, 7]


def predict_val(model, train, val, seq_len, device):
    """Predict for validation windows as defined in the statement."""
    combined = np.concatenate([train, val], axis=0)      # [11520, 7]
    val_start_base = train.shape[0]                      # 8640
    num_val_windows = (val.shape[0] - PRED_LEN) // STEP + 1  # 117
    preds = []
    model.eval()
    with torch.no_grad():
        for w in range(num_val_windows):
            start_idx = val_start_base + w * STEP
            hist = combined[start_idx - seq_len:start_idx]   # [seq_len, 7]
            hist_tensor = torch.from_numpy(hist).unsqueeze(0).to(device)
            out = model(hist_tensor)                         # [1, pred_len, 7]
            preds.append(out.squeeze(0).cpu().numpy())
    return np.stack(preds, axis=0)                         # [117, pred_len, 7]


def main():
    # Seed
    seed = int(os.getenv("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = torch.device("cpu")

    # Load data
    train, val, test_inputs = load_data()

    # Optionally reduce workload in SMOKE mode
    smoke = os.getenv("SMOKE") == "1"
    if smoke:
        CONFIG["epochs"] = 1
        # Use only first batch for training (still need at least one sample)
        # We'll create windows then slice later.

    # Prepare training and validation windows
    train_inputs, train_targets = create_windows(train, CONFIG["seq_len"], PRED_LEN, step=STEP)
    val_inputs, val_targets = create_windows(val, CONFIG["seq_len"], PRED_LEN, step=STEP)

    if smoke:
        # Limit to first batch (32 samples) for speed
        limit = min(CONFIG["batch_size"], len(train_inputs))
        train_inputs = train_inputs[:limit]
        train_targets = train_targets[:limit]

    # Convert to torch tensors
    train_tensor_x = torch.from_numpy(train_inputs)
    train_tensor_y = torch.from_numpy(train_targets)
    val_tensor_x = torch.from_numpy(val_inputs)
    val_tensor_y = torch.from_numpy(val_targets)

    train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
    val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=CONFIG["batch_size"],
                                               shuffle=True,
                                               drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=CONFIG["batch_size"],
                                             shuffle=False,
                                             drop_last=False)

    # Model, loss, optimizer
    model_cfg = {
        "seq_len": CONFIG["seq_len"],
        "pred_len": PRED_LEN,
        "enc_in": ENC_IN,
        "kernel_size": KERNEL_SIZE,
        "individual": INDIVIDUAL,
        "dropout": CONFIG["dropout"]
    }
    model = DLinear(model_cfg).to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model)

    # Early stopping
    early_stopper = EarlyStopping(patience=CONFIG["patience"], verbose=True)

    # Training loop
    for epoch in range(1, CONFIG["epochs"] + 1):
        adjust_learning_rate(optimizer, epoch)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break

    # Restore best model
    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)

    # Final validation loss (using best model)
    final_val_loss = evaluate(model, val_loader, criterion, device)
    print(f'Final Validation MSE: {final_val_loss:.6f}')

    # Inference
    test_pred = predict_test(model, test_inputs, CONFIG["seq_len"], device)
    val_pred = predict_val(model, train, val, CONFIG["seq_len"], device)

    # Write predictions via nullius_io
    nio.write_predictions(test_pred, val_pred)


if __name__ == "__main__":
    main()