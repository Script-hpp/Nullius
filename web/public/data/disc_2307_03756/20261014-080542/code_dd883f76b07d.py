

# Plan step -> implemented in: imports and seed setting
# Plan step -> implemented in: CONFIG dict
# Plan step -> implemented in: FITS model class
# Plan step -> implemented in: adjust_learning_rate and EarlyStopping
# Plan step -> implemented in: train_forecaster function
# Plan step -> implemented in: predictor closure

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import nullius_io as nio

CONFIG = {"lr": 0.0005, "epochs": 50, "batch_size": 128, "patience": 20, "seq_len": 360, "cut_freq": 106, "enc_in": 7, "pred_len": 96, "train_mode": 1}


class FITS(nn.Module):
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
                    nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(
                        torch.cfloat
                    )
                )
        else:
            self.freq_upsampler = nn.Linear(
                self.dominance_freq,
                int(self.dominance_freq * self.length_ratio),
            ).to(torch.cfloat)

    def forward(self, x):
        # RevIN normalization
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        # FFT and low-pass
        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq:, :] = 0
        low_specx = low_specx[:, 0 : self.dominance_freq, :]

        if self.individual:
            low_specxy_ = torch.zeros(
                [
                    low_specx.size(0),
                    int(self.dominance_freq * self.length_ratio),
                    low_specx.size(2),
                ],
                dtype=low_specx.dtype,
                device=low_specx.device,
            )
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](
                    low_specx[:, :, i].permute(0, 1)
                ).permute(0, 1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)

        # Zero-pad to full length for IFFT
        low_specxy = torch.zeros(
            [
                low_specxy_.size(0),
                int((self.seq_len + self.pred_len) / 2 + 1),
                low_specxy_.size(2),
            ],
            dtype=low_specxy_.dtype,
            device=low_specxy_.device,
        )
        low_specxy[:, 0 : low_specxy_.size(1), :] = low_specxy_

        # IFFT and scaling
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio

        # De-normalize
        xy = low_xy * torch.sqrt(x_var) + x_mean
        return xy  # we ignore the second return as per training code


def adjust_learning_rate(optimizer, epoch, args):
    # type3: lr * 0.95^{(epoch-1)}
    lr = args.learning_rate * (0.95 ** ((epoch - 1) // 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f"Updating learning rate to {lr}")


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/checkpoint.pth")
        self.val_loss_min = val_loss


def train_forecaster(data, seed, smoke=False):
    # Seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = torch.device("cpu")

    # Unpack data
    train_arr = data["train"]  # [8640, 7]
    val_arr = data["val"]      # [2880, 7]

    # Prepare windows via nullius_io
    length = CONFIG["seq_len"]
    windows = nio.forecast_data(length)
    train_x = windows["train_inputs"]  # [N, length, 7]
    train_y = windows["train_targets"]  # [N, pred_len, 7]
    val_x = windows["val_inputs"]
    val_y = windows["val_targets"]

    # Independent candidate selection: hide last 20% of validation for unbiased evaluation
    n_val = len(val_x)
    n_visible = int(0.8 * n_val)
    val_x_vis = val_x[:n_visible]
    val_y_vis = val_y[:n_visible]

    # Convert to torch tensors
    train_x_tensor = torch.from_numpy(train_x).float()
    train_y_tensor = torch.from_numpy(train_y).float()
    val_x_vis_tensor = torch.from_numpy(val_x_vis).float()
    val_y_vis_tensor = torch.from_numpy(val_y_vis).float()

    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    val_vis_dataset = TensorDataset(val_x_vis_tensor, val_y_vis_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_vis_loader = DataLoader(
        val_vis_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    # Model configuration
    class Config:
        pass

    cfg = Config()
    cfg.seq_len = CONFIG["seq_len"]
    cfg.pred_len = CONFIG["pred_len"]
    cfg.individual = False
    cfg.enc_in = CONFIG["enc_in"]
    cfg.cut_freq = CONFIG["cut_freq"]

    model = FITS(cfg).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=0.0)
    criterion = nn.MSELoss()

    # Learning rate schedule arguments
    class LRArgs:
        pass

    lr_args = LRArgs()
    lr_args.lradj = "type3"
    lr_args.learning_rate = CONFIG["lr"]

    # Early stopping
    ckpt_dir = "/tmp/forecast-agent"
    os.makedirs(ckpt_dir, exist_ok=True)
    early_stopping = EarlyStopping(patience=CONFIG["patience"], verbose=True)

    # Determine number of epochs and possible smoke truncation
    epochs = CONFIG["epochs"]
    if smoke:
        epochs = 1

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for i, (xb, yb) in enumerate(train_loader):
            if smoke and i >= 1:
                break  # only one batch in smoke mode
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)  # [B, seq_len+pred_len, 7]

            if CONFIG["train_mode"] == 1:
                # train on y only (forecast part)
                preds = outputs[:, -CONFIG["pred_len"] :, :]
                target = yb[:, -CONFIG["pred_len"] :, :]
                loss = criterion(preds, target)
            else:
                # train on reconstruction (input + forecast)
                target_xy = torch.cat([xb, yb], dim=1)  # [B, seq_len+pred_len, 7]
                loss = criterion(outputs, target_xy)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Adjust learning rate after epoch (epoch index starts at 0)
        adjust_learning_rate(optimizer, epoch + 1, lr_args)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i, (xb, yb) in enumerate(val_vis_loader):
                if smoke and i >= 1:
                    break
                xb = xb.to(device)
                yb = yb.to(device)
                outputs = model(xb)

                if CONFIG["train_mode"] == 1:
                    preds = outputs[:, -CONFIG["pred_len"] :, :]
                    target = yb[:, -CONFIG["pred_len"] :, :]
                    loss = criterion(preds, target)
                else:
                    target_xy = torch.cat([xb, yb], dim=1)
                    loss = criterion(outputs, target_xy)

                val_losses.append(loss.item())
        val_loss = np.mean(val_losses) if val_losses else 0.0

        early_stopping(val_loss, model, ckpt_dir)
        if early_stopping.early_stop:
            if early_stopping.verbose:
                print("Early stopping triggered")
            break

    # Load best model
    best_path = os.path.join(ckpt_dir, "checkpoint.pth")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # Predictor function
    def predictor(history, *, calendar=None):
        # history: np.ndarray [T, 7] where T >= seq_len (or we assume)
        if history.shape[0] < CONFIG["seq_len"]:
            raise ValueError(
                f"History length {history.shape[0]} is shorter than required seq_len {CONFIG['seq_len']}"
            )
        hist = history[-CONFIG["seq_len"] :].copy()
        # calendar cropping is not used by FITS, but we accept the argument
        if calendar is not None:
            _ = calendar[-CONFIG["seq_len"] :]  # no-op, just to honor the signature
        tensor = torch.from_numpy(hist).unsqueeze(0).to(device)  # [1, seq_len, 7]
        with torch.no_grad():
            out = model(tensor)
            forecast = out[:, -CONFIG["pred_len"] :, :].cpu().numpy().squeeze(0)  # [pred_len, 7]
        return forecast

    return predictor