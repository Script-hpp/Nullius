# Plan step -> implemented in: train_forecaster (seed numpy and torch from env SEED)
# Plan step -> implemented at top: CONFIG dict with hyperparameters
# Plan step -> implemented in: train_forecaster (call nio.forecast_data for windows)
# Plan step -> implemented in: train_forecaster (hold out final 20% of val for unbiased selection)
# Plan step -> implemented in: train_forecaster (build TensorDataset and DataLoader)
# Plan step -> implemented in: FITS Model class and instantiation in train_forecaster
# Plan step -> implemented in: train_forecaster (training loop with Adam, LR schedule type3, early stopping)
# Plan step -> implemented in: train_forecaster (save best checkpoint to /tmp/forecast-agent)
# Plan step -> implemented in: train_forecaster (load best model and set to eval)
# Plan step -> implemented in: predictor function inside train_forecaster (crop history, run model, slice forecast)
# Plan step -> implemented in: predictor function (accepts optional calendar argument, does not use it)

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


class FITS(nn.Module):
    """FITS model as per the authors' code (non‑individual version)."""
    def __init__(self, configs):
        super(FITS, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = getattr(configs, "individual", False)
        self.channels = configs.enc_in

        self.dominance_freq = configs.cut_freq
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(
                    nn.Linear(self.dominance_freq,
                              int(self.dominance_freq * self.length_ratio))
                )
        else:
            self.freq_upsampler = nn.Linear(
                self.dominance_freq,
                int(self.dominance_freq * self.length_ratio)
            )

    def forward(self, x):
        # x: [B, L, C]
        # RevIN normalization
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        # FFT and low‑pass filter
        low_specx = torch.fft.rfft(x, dim=1)          # [B, L//2+1, C]
        low_specx[:, self.dominance_freq:, :] = 0     # zero out high frequencies
        low_specx = low_specx[:, :self.dominance_freq, :]  # keep only low frequencies

        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0),
                 int(self.dominance_freq * self.length_ratio),
                 low_specx.size(2)],
                dtype=low_specx.dtype,
                device=low_specx.device
            )
            for i in range(self.channels):
                # Linear expects [..., dominance_freq]; permute to [B, C, D]
                low_specxy_[:, :, i] = self.freq_upsampler[i](
                    low_specx[:, :, i].permute(0, 1)
                ).permute(0, 1)
        else:
            # Permute to [B, C, D] for linear, then back
            low_specxy_ = self.freq_upsampler(
                low_specx.permute(0, 2, 1)
            ).permute(0, 2, 1)

        # Zero‑pad to the full frequency length of (seq_len+pred_len)
        low_specxy = torch.zeros(
            [low_specxy_.size(0),
             int((self.seq_len + self.pred_len) / 2 + 1),
             low_specxy_.size(2)],
            dtype=low_specxy_.dtype,
            device=low_specxy_.device
        )
        low_specxy[:, :low_specxy_.size(1), :] = low_specxy_

        # Inverse FFT to get the upsampled series in time domain
        low_xy = torch.fft.irfft(low_specxy, dim=1)      # [B, seq_len+pred_len, C]
        low_xy = low_xy * self.length_ratio              # energy compensation

        # De‑RevIN
        xy = low_xy * torch.sqrt(x_var) + x_mean
        # Return forecast and the low‑pass component (scaled) as in original code
        return xy, low_xy * torch.sqrt(x_var)


def train_forecaster(data, seed, smoke=False):
    # --- seeding ---
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- prepare windows via nullius_io ---
    import nullius_io as nio
    windows = nio.forecast_data(CONFIG["seq_len"])
    train_x = windows["train_inputs"]   # [N, seq_len, 7]
    train_y = windows["train_targets"]  # [N, pred_len, 7]
    val_x = windows["val_inputs"]
    val_y = windows["val_targets"]

    # --- hold out final 20% of validation for unbiased selection ---
    n_val = val_x.shape[0]
    split_idx = int(0.8 * n_val)
    val_x_vis = val_x[:split_idx]
    val_y_vis = val_y[:split_idx]
    # the remaining part is ignored for training/early stopping

    # --- torch datasets ---
    train_dataset = TensorDataset(
        torch.from_numpy(train_x).float(),
        torch.from_numpy(train_y).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_x_vis).float(),
        torch.from_numpy(val_y_vis).float()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        drop_last=False
    )

    # --- model ---
    class Configs:
        pass
    cfg = Configs()
    cfg.seq_len = CONFIG["seq_len"]
    cfg.pred_len = CONFIG["pred_len"]
    cfg.enc_in = CONFIG["enc_in"]
    cfg.cut_freq = CONFIG["cut_freq"]
    cfg.individual = False   # non‑individual as used in the spec

    model = FITS(cfg)
    model.float()

    # --- optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=0)

    # --- learning rate schedule (type3) ---
    def adjust_lr(optimizer, epoch_idx, base_lr):
        lr = base_lr * (0.95 ** ((epoch_idx - 1) // 1))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    # --- early stopping ---
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    checkpoint_dir = "/tmp/forecast-agent"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

    max_epochs = 1 if smoke else CONFIG["epochs"]
    train_limit_batches = 1 if smoke else None
    val_limit_batches = 1 if smoke else None

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for i, (batch_x, batch_y) in enumerate(train_loader):
            if train_limit_batches is not None and i >= train_limit_batches:
                break
            batch_x = batch_x.to(torch.get_default_device())
            batch_y = batch_y.to(torch.get_default_device())

            optimizer.zero_grad()
            output, _ = model(batch_x)               # output: [B, seq_len+pred_len, 7]
            forecast = output[:, -CONFIG["pred_len"]:, :]  # take last pred_len steps
            loss = F.mse_loss(forecast, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        adjust_lr(optimizer, epoch, CONFIG["lr"])

        # --- validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                if val_limit_batches is not None and i >= val_limit_batches:
                    break
                batch_x = batch_x.to(torch.get_default_device())
                batch_y = batch_y.to(torch.get_default_device())
                output, _ = model(batch_x)
                forecast = output[:, -CONFIG["pred_len"]:, :]
                loss = F.mse_loss(forecast, batch_y)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')

        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            torch.save(best_state, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG["patience"]:
                break

    # --- load best model ---
    if best_state is None:
        best_state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(best_state)
    model.eval()

    # --- predictor function ---
    def predictor(history, *, calendar=None):
        """
        history: np.ndarray of shape [T, 7] (T >= seq_len)
        calendar: optional, not used by FITS
        Returns: np.ndarray of shape [pred_len, 7]
        """
        # crop to the last seq_len steps
        hist = history[-CONFIG["seq_len"]:, :]          # [seq_len, 7]
        tensor = torch.from_numpy(hist).unsqueeze(0).float()  # [1, seq_len, 7]
        with torch.no_grad():
            output, _ = model(tensor)                   # [1, seq_len+pred_len, 7] or [1, ..., 7]
            forecast = output[:, -CONFIG["pred_len"]:, :]  # [1, pred_len, 7]
        return forecast.squeeze(0).cpu().numpy()        # [pred_len, 7]

    return predictor
</|assistant|>