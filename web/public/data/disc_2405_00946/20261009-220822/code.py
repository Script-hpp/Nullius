

# Plan step -> implemented in: load_data_and_set_seed
# Plan step -> implemented in: build_dataloaders
# Plan step -> implemented in: SparseTSF
# Plan step -> implemented in: train_model
# Plan step -> implemented in: predict_windows
# Plan step -> implemented in: main

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import nullius_io as nio

# Hyperparameters to read only from this dict
CONFIG = {"lr": 0.02, "epochs": 30, "batch_size": 256, "patience": 5, "seq_len": 720}

def load_data_and_set_seed():
    """Read seed, load data, and return train, val, test_inputs."""
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = nio.load()
    train = data["train"].astype(np.float32)   # [8640,7]
    val = data["val"].astype(np.float32)       # [2880,7]
    test_inputs = data["test_inputs"].astype(np.float32)  # [2785,720,7]
    return train, val, test_inputs

class ForecastDataset(Dataset):
    """Sliding window dataset for time series forecasting."""
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.length = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return max(self.length, 0)

    def __getitem__(self, idx):
        s = self.data[idx:idx + self.seq_len]
        t = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.from_numpy(s), torch.from_numpy(t)

def build_dataloaders(train, val):
    """Create train and validation DataLoaders."""
    seq_len = CONFIG["seq_len"]
    pred_len = 96  # fixed horizon for this task
    train_set = ForecastDataset(train, seq_len, pred_len)
    val_set = ForecastDataset(val, seq_len, pred_len)

    # Adjust for SMOKE mode: use only first batch if needed
    smoke = os.environ.get("SMOKE") == "1"
    if smoke:
        # Reduce epochs to 1 and limit batches to 1 of size 32
        CONFIG["epochs"] = 1
        # We'll still create full loaders but will break after first batch in training loop
        # Optionally we could shrink datasets, but breaking after first batch is sufficient.
        pass

    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader

class SparseTSF(nn.Module):
    """Exact replication of the SparseTSF model (linear variant)."""
    def __init__(self, configs):
        super(SparseTSF, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        assert self.model_type in ['linear', 'mlp']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1,
                                kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2,
                                padding_mode="zeros", bias=False)

        if self.model_type == 'linear':
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        elif self.model_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_y)
            )

    def forward(self, x):
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)          # (B,1,C)
        x = (x - seq_mean).permute(0, 2, 1)                   # (B,C,S)

        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # sparse forecasting
        if self.model_type == 'linear':
            y = self.linear(x)                               # (B*C, W, seg_num_y)
        elif self.model_type == 'mlp':
            y = self.mlp(x)

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean                     # (B, P, C)
        return y

def train_model(model, train_loader, val_loader, device):
    """Train with early stopping on validation MSE."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CONFIG["lr"],
                                 weight_decay=0.0)  # weight decay uncertain; set to 0
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    model.to(device)
    for epoch in range(CONFIG["epochs"]):
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
        val_loss = np.mean(val_losses) if val_losses else 0.0
        train_loss = np.mean(train_losses)

        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG["patience"]:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss

def predict_windows(model, data, start_indices, seq_len, pred_len, device, use_raw_history=True):
    """Forecast for given start indices using history of length seq_len.
    If use_raw_history is False, data is assumed to be already windowed (N, seq_len, C).
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for start in start_indices:
            if use_raw_history:
                history = data[start - seq_len:start]  # (seq_len, C)
            else:
                history = data[start]                  # already (seq_len, C)
            inp = torch.from_numpy(history).unsqueeze(0).to(device)  # (1, seq_len, C)
            out = model(inp)                           # (1, pred_len, C)
            preds.append(out.squeeze(0).cpu().numpy())
    return np.stack(preds, axis=0)                     # (N, pred_len, C)

def main():
    train, val, test_inputs = load_data_and_set_seed()
    train_loader, val_loader = build_dataloaders(train, val)

    device = torch.device("cpu")
    # Model configuration
    configs = type('Configs', (), {})()
    configs.seq_len = CONFIG["seq_len"]
    configs.pred_len = 96
    configs.enc_in = 7
    configs.period_len = 24          # from scripts/SparseTSF/linear/etth1.sh
    configs.d_model = 16             # arbitrary; not used in linear mode
    configs.model_type = 'linear'

    model = SparseTSF(configs)
    model, val_mse = train_model(model, train_loader, val_loader, device)

    # Prepare test predictions (all 2785 windows)
    test_pred = predict_windows(model, test_inputs,
                                start_indices=np.arange(test_inputs.shape[0]),
                                seq_len=CONFIG["seq_len"],
                                pred_len=96,
                                device=device,
                                use_raw_history=False)   # test_inputs are already windows

    # Prepare validation predictions for the 117 specific windows
    series = np.concatenate([train, val], axis=0)  # shape [11520,7]
    val_start_indices = 8640 + 24 * np.arange(117)  # as per description
    val_pred = predict_windows(model, series,
                               start_indices=val_start_indices,
                               seq_len=CONFIG["seq_len"],
                               pred_len=96,
                               device=device,
                               use_raw_history=True)    # need to slice history

    # Output via helper
    nio.write_predictions(test_pred, val_pred)
    print(f"Final validation MSE (early stopping): {val_mse:.6f}")

if __name__ == "__main__":
    main()