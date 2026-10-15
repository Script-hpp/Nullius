

# Plan step -> implemented in: imports and seed
# Plan step -> implemented in: CONFIG dict
# Plan step -> implemented in: FITS model class definition
# Plan step -> implemented in: learning rate adjustment (type3)
# Plan step -> implemented in: train_forecaster function
# Plan step -> implemented in: predictor closure

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters from spec
CONFIG = {
    "lr": 0.0005,
    "epochs": 50,
    "batch_size": 128,
    "patience": 20,
    "seq_len": 360,
    "cut_freq": 106,
    "enc_in": 7,
    "pred_len": 96,
    "train_mode": 1
}

# Seed handling
_SEED = int(os.environ.get("SEED", 0))
np.random.seed(_SEED)
torch.manual_seed(_SEED)

# FITS model (non-individual) as per authors' code
class FITS(nn.Module):
    def __init__(self, configs):
        super(FITS, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in

        self.dominance_freq = configs.cut_freq
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat))
        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat)

    def forward(self, x):
        # RevIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq:] = 0  # LPF
        low_specx = low_specx[:, 0:self.dominance_freq, :]  # LPF

        if self.individual:
            low_specxy_ = torch.zeros([low_specx.size(0), int(self.dominance_freq * self.length_ratio), low_specx.size(2)],
                                      dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)

        low_specxy = torch.zeros([low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_.size(2)],
                                 dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_  # zero padding
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio  # energy compensation

        xy = low_xy * torch.sqrt(x_var) + x_mean
        return xy, low_xy * torch.sqrt(x_var)


def train_forecaster(data, seed, smoke):
    # Use provided seed (overrides env)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Determine effective epochs
    epochs = 1 if smoke else CONFIG["epochs"]

    # Get windowed data via nullius_io
    import nullius_io as nio
    windows = nio.forecast_data(CONFIG["seq_len"])
    train_x = windows["train_inputs"]   # [N, seq_len, 7]
    train_y = windows["train_targets"]  # [N, pred_len, 7]
    val_x_all = windows["val_inputs"]
    val_y_all = windows["val_targets"]
    # Optional calendars (may be None)
    train_cal = windows.get("train_calendar")
    val_cal_all = windows.get("val_calendar")

    # Withhold final 20% of validation for independent candidate selection
    n_val = len(val_x_all)
    n_val_visible = int(0.8 * n_val)
    val_x = val_x_all[:n_val_visible]
    val_y = val_y_all[:n_val_visible]
    if train_cal is not None:
        train_cal = train_cal[:len(train_x)]
    if val_cal_all is not None:
        val_cal = val_cal_all[:n_val_visible]
    else:
        val_cal = None

    # Convert to torch tensors
    train_x_t = torch.from_numpy(train_x).float()
    train_y_t = torch.from_numpy(train_y).float()
    val_x_t = torch.from_numpy(val_x).float()
    val_y_t = torch.from_numpy(val_y).float()

    train_ds = TensorDataset(train_x_t, train_y_t)
    val_ds = TensorDataset(val_x_t, val_y_t)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, drop_last=False)

    # If smoke, we will only use first batch each epoch
    if smoke:
        # Wrap loaders to yield only first batch
        def first_batch_only(loader):
            data_iter = iter(loader)
            yield next(data_iter)
        train_loader = first_batch_only(train_loader)
        val_loader = first_batch_only(val_loader)

    # Model, optimizer, loss
    model_cfg = type('Config', (), {
        "seq_len": CONFIG["seq_len"],
        "pred_len": CONFIG["pred_len"],
        "individual": False,
        "enc_in": CONFIG["enc_in"],
        "cut_freq": CONFIG["cut_freq"]
    })()
    model = FITS(model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss()

    best_val = float('inf')
    patience_wait = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for i, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            if CONFIG["train_mode"] == 1:
                # Train on prediction horizon only
                pred = outputs[:, -CONFIG["pred_len"]:, :]
                loss = criterion(pred, batch_y)
            else:
                # Train on full sequence (inputs + predictions)
                outputs_full = outputs  # [B, seq_len+pred_len, C]
                batch_xy = torch.cat([batch_x, batch_y], dim=1)  # [B, seq_len+pred_len, C]
                loss = criterion(outputs_full, batch_xy)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if smoke:
                break  # only first batch

        # Validation (always on prediction horizon)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                outputs, _ = model(batch_x)
                pred = outputs[:, -CONFIG["pred_len"]:, :]
                loss = criterion(pred, batch_y)
                val_losses.append(loss.item())
                if smoke:
                    break
        val_loss = np.mean(val_losses) if val_losses else 0.0

        # Learning rate adjustment (type3)
        lr = CONFIG["lr"] * (0.95 ** epoch)
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Early stopping logic
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_wait = 0
        else:
            patience_wait += 1
            if patience_wait >= CONFIG["patience"]:
                break

        if smoke:
            break  # only one epoch

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Predictor closure
    def predictor(history, *, calendar=None):
        # history: [720, 7] np.ndarray
        # calendar: optional, same length as history (ignored by FITS)
        if history.shape[0] > CONFIG["seq_len"]:
            history = history[-CONFIG["seq_len"]:, :]
            if calendar is not None:
                calendar = calendar[-CONFIG["seq_len"]:, :]
        # Convert to tensor
        inp = torch.from_numpy(history).float().unsqueeze(0)  # [1, seq_len, 7]
        with torch.no_grad():
            outputs, _ = model(inp)
            forecast = outputs[:, -CONFIG["pred_len"]:, :]  # [1, pred_len, 7]
        return forecast.squeeze(0).cpu().numpy()  # [pred_len, 7]

    return predictor