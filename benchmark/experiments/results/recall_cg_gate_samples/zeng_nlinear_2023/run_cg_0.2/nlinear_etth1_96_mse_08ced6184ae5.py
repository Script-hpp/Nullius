# Plan step -> Model: class NLinear
# Plan step -> Input / output lengths: seq_len and pred_len from CONFIG
# Plan step -> Channels: derived from data shape
# Plan step -> Optimizer: Adam with lr and weight_decay from CONFIG
# Plan step -> Learning rate: CONFIG["lr"]
# Plan step -> Learning‑rate schedule: adjust_learning_rate function
# Plan step -> Batch size: CONFIG["batch_size"]
# Plan step -> Weight decay: CONFIG["weight_decay"]
# Plan step -> Dropout: none (model has no dropout)
# Plan step -> Epochs: CONFIG["epochs"]
# Plan step -> Early stopping patience: CONFIG["patience"]
# Plan step -> Loss: Mean Squared Error (F.mse_loss)
# Plan step -> Pre‑processing inside model: subtraction of last timestep in forward
# Plan step -> Validation: sliding windows on val for early stopping
# Plan step -> Testing: forecast using last seq_len of each test_inputs[i]
# Plan step -> Validation forecast production: specific stride‑24 windows on train+val

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import nullius_io as nio

# Seed handling
seed = int(os.getenv("SEED", "0"))
np.random.seed(seed)
torch.manual_seed(seed)

# Hyperparameters (read only from this dict)
CONFIG = {
    "lr": 0.005,
    "batch_size": 32,
    "seq_len": 336,
    "pred_len": 96,
    "weight_decay": 0.0,
    "epochs": 20,
    "patience": 5,
    "lradj": "type1"
}

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')

class NLinear(nn.Module):
    def __init__(self, configs):
        super(NLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]

def create_dataset(data, seq_len, pred_len):
    # data: [T, C]
    T = data.shape[0]
    indices = np.arange(T - seq_len - pred_len + 1)
    inputs = np.stack([data[i:i + seq_len] for i in indices], axis=0)  # [N, seq_len, C]
    targets = np.stack([data[i + seq_len:i + seq_len + pred_len] for i in indices], axis=0)  # [N, pred_len, C]
    return torch.from_numpy(inputs).float(), torch.from_numpy(targets).float()

def main():
    smoke = os.getenv("SMOKE") == "1"

    # Load data
    d = nio.load()
    train_data = d['train'].astype(np.float32)      # [8640, 7]
    val_data = d['val'].astype(np.float32)          # [2880, 7]
    test_inputs_raw = d['test_inputs'].astype(np.float32)  # [2785, 720, 7]

    C = train_data.shape[1]
    seq_len = CONFIG["seq_len"]
    pred_len = CONFIG["pred_len"]

    # Datasets for training and validation (early stopping)
    train_inputs, train_targets = create_dataset(train_data, seq_len, pred_len)
    val_inputs, val_targets = create_dataset(val_data, seq_len, pred_len)

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(val_inputs, val_targets),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        drop_last=False
    )

    # Model configuration
    class Config:
        pass
    cfg = Config()
    cfg.seq_len = seq_len
    cfg.pred_len = pred_len
    cfg.enc_in = C
    cfg.individual = False  # shared linear across channels

    model = NLinear(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    args = type('Args', (), {'lradj': CONFIG["lradj"], 'learning_rate': CONFIG["lr"]})()

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    # Training loop
    max_epochs = 1 if smoke else CONFIG["epochs"]
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        nb_batches = 0
        for batch_x, batch_y in train_loader:
            if smoke and nb_batches >= 1:
                break
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = F.mse_loss(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb_batches += 1
        # Adjust LR after each epoch (using epoch index starting from 1 as in original code)
        adjust_learning_rate(optimizer, epoch + 1, args)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = F.mse_loss(preds, batch_y)
                val_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_loss / max(val_batches, 1)
        print(f'Epoch {epoch+1}/{max_epochs}, Train Loss: {epoch_loss/max(nb_batches,1):.6f}, Val Loss: {avg_val_loss:.6f}')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG["patience"]:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Inference on test windows
    N_test = test_inputs_raw.shape[0]
    test_pred = np.zeros((N_test, pred_len, C), dtype=np.float32)
    limit_test = N_test if not smoke else min(8, N_test)
    for i in range(limit_test):
        # take last seq_len steps
        inp = test_inputs_raw[i, -seq_len:, :]  # [seq_len, C]
        inp_tensor = torch.from_numpy(inp).unsqueeze(0).float()  # [1, seq_len, C]
        with torch.no_grad():
            out = model(inp_tensor)  # [1, pred_len, C]
        test_pred[i] = out.squeeze(0).cpu().numpy()
    # If smoke and we didn't fill all, leave zeros (helper will ignore beyond first 8)

    # Inference on validation windows (specific stride-24 windows)
    # Build combined series: train + val
    series = np.concatenate([train_data, val_data], axis=0)  # [11520, 7]
    N_val_win = 117  # as per specification
    val_pred = np.zeros((N_val_win, pred_len, C), dtype=np.float32)
    limit_val = N_val_win if not smoke else min(8, N_val_win)
    start_base = train_data.shape[0]  # 8640
    for w in range(limit_val):
        start_idx = start_base + 24 * w
        hist_start = start_idx - seq_len
        hist_end = start_idx
        inp = series[hist_start:hist_end, :]  # [seq_len, C]
        inp_tensor = torch.from_numpy(inp).unsqueeze(0).float()
        with torch.no_grad():
            out = model(inp_tensor)
        val_pred[w] = out.squeeze(0).cpu().numpy()

    # Write predictions
    nio.write_predictions(test_pred, val_pred)

    # Print final validation MSE (best observed)
    print(f'Best validation MSE: {best_val_loss:.6f}')

if __name__ == '__main__':
    main()
