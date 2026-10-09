# Plan step -> implemented in: function or class name
# Load data -> load_data
# Create training/validation datasets -> create_datasets
# Build NLinear model -> NLinear class
# Training loop with early stopping and LR schedule -> train_model
# Validation/test inference -> predict
# Write predictions CSV -> write_predictions
# Main driver -> main

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Read seed and set deterministic behavior
SEED = int(os.environ.get("SEED", "0"))
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameters (only these keys may be used as literals elsewhere)
CONFIG = {
    "lr": 0.005,
    "batch_size": 32,
    "patience": 7,
    "seq_len": 336,
    # Additional defaults (chosen as per plan)
    "weight_decay": 0.0,
    "max_epochs": 20,
}

# --------------------------------------------------------------
def load_data():
    data = np.load("/data/etth1_public.npz")
    train = data["train"].astype(np.float32)      # [8640,7]
    val = data["val"].astype(np.float32)          # [2880,7]
    test_inputs = data["test_inputs"].astype(np.float32)  # [2785,720,7]
    return train, val, test_inputs

# --------------------------------------------------------------
def create_datasets(train, val, test_inputs, seq_len, pred_len, smoke=False):
    """
    Returns:
        train_loader: DataLoader for training (shuffled)
        val_inputs, val_targets: torch tensors for validation
        test_inputs_tensor: torch tensor for test (full or smoke)
        N_test, N_val: number of windows for test/validation
    """
    # ---------- Training ----------
    # Sliding windows over train
    total_train = len(train)
    num_train = total_train - seq_len - pred_len + 1  # 8209
    if smoke:
        # Use at most one batch (32 samples)
        num_train = min(num_train, CONFIG["batch_size"])
        start_idx = 0
    else:
        start_idx = 0
    train_in = np.zeros((num_train, seq_len, train.shape[1]), dtype=np.float32)
    train_tgt = np.zeros((num_train, pred_len, train.shape[1]), dtype=np.float32)
    for i in range(num_train):
        s = start_idx + i
        train_in[i] = train[s:s+seq_len]
        train_tgt[i] = train[s+seq_len:s+seq_len+pred_len]
    train_in_t = torch.from_numpy(train_in)
    train_tgt_t = torch.from_numpy(train_tgt)
    train_dataset = TensorDataset(train_in_t, train_tgt_t)
    train_loader = DataLoader(train_dataset,
                              batch_size=CONFIG["batch_size"],
                              shuffle=True,
                              drop_last=False)

    # ---------- Validation ----------
    # Combined series train+val
    combined = np.concatenate([train, val], axis=0)  # [11520,7]
    # Validation windows: start at 8640 + 24*w, w=0..116
    N_val_full = 117
    N_val = N_val_full if not smoke else 8
    val_in = np.zeros((N_val, seq_len, combined.shape[1]), dtype=np.float32)
    val_tgt = np.zeros((N_val, pred_len, combined.shape[1]), dtype=np.float32)
    for w in range(N_val):
        start = 8640 + 24 * w          # index in combined where forecast starts
        val_in[w] = combined[start - seq_len:start]
        val_tgt[w] = combined[start:start+pred_len]
    val_in_t = torch.from_numpy(val_in)
    val_tgt_t = torch.from_numpy(val_tgt)

    # ---------- Test ----------
    N_test_full = test_inputs.shape[0]  # 2785
    N_test = N_test_full if not smoke else 8
    # Use last seq_len steps of each test_inputs[i]
    if smoke:
        test_in = test_inputs[:N_test, -seq_len:, :]
    else:
        test_in = test_inputs[:, -seq_len:, :]
    test_in_t = torch.from_numpy(test_in)  # [N_test, seq_len, 7]

    return (train_loader,
            val_in_t, val_tgt_t,
            test_in_t,
            N_test, N_val)

# --------------------------------------------------------------
class NLinear(nn.Module):
    """
    Normalization-Linear model as per the authors' code.
    """
    def __init__(self, seq_len, pred_len, enc_in, individual=False):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()          # [B,1,C]
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)],
                                 dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)   # [B, C, T] -> Linear -> [B, C, pred_len] -> permute back
        x = x + seq_last
        return x   # [B, Output length, Channel]

# --------------------------------------------------------------
def adjust_learning_rate(optimizer, epoch, args):
    # lr_adjust dictionary as per authors' utils/tools.py
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')

# --------------------------------------------------------------
def train_model(model, train_loader, val_in, val_tgt, config):
    device = torch.device('cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["lr"],
                                 weight_decay=config.get("weight_decay", 0.0))
    # Arguments for LR scheduler
    class Args: pass
    args = Args()
    args.learning_rate = config["lr"]
    args.lradj = 'type1'

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        nb_batches = 0
        adjust_learning_rate(optimizer, epoch, args)
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = F.mse_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb_batches += 1
        avg_train_loss = epoch_loss / max(nb_batches,1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_in.to(device))
            val_loss = F.mse_loss(val_pred, val_tgt.to(device)).item()

        print(f'Epoch {epoch:02d} | Train MSE: {avg_train_loss:.6f} | Val MSE: {val_loss:.6f}')

        # Early stopping check
        if val_loss < best_val_loss - 1e-12:  # min_delta = 0
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f'Early stopping triggered after epoch {epoch}')
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss

# --------------------------------------------------------------
def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        preds = model(inputs)
    return preds.cpu().numpy()

# --------------------------------------------------------------
def write_predictions(preds, path, start_window=0):
    """
    preds: numpy array shape (N, 96, 7)
    writes to path with header id,prediction
    """
    N, steps, channels = preds.shape
    with open(path, 'w') as f:
        f.write('id,prediction\n')
        for w in range(N):
            base_id = w + start_window
            for s in range(steps):
                for c in range(channels):
                    pid = f'{base_id}_{s}_{c}'
                    val = preds[w, s, c]
                    f.write(f'{pid},{val:.6f}\n')

# --------------------------------------------------------------
def main():
    smoke = os.environ.get('SMOKE') == '1'
    train, val, test_inputs = load_data()
    seq_len = CONFIG["seq_len"]
    pred_len = 96
    enc_in = 7

    (train_loader,
     val_in, val_tgt,
     test_in,
     N_test, N_val) = create_datasets(train, val, test_inputs, seq_len, pred_len, smoke=smoke)

    model = NLinear(seq_len=seq_len,
                    pred_len=pred_len,
                    enc_in=enc_in,
                    individual=False)  # shared linear layer as per plan

    trained_model, val_mse = train_model(model,
                                         train_loader,
                                         val_in,
                                         val_tgt,
                                         CONFIG)
    print(f'Validation MSE: {val_mse:.6f}')

    # Predict validation (for val_predictions.csv)
    val_preds = predict(trained_model, val_in)   # [N_val,96,7]
    write_predictions(val_preds, '/output/val_predictions.csv',
                      start_window=0)

    # Predict test
    test_preds = predict(trained_model, test_in)  # [N_test,96,7]
    write_predictions(test_preds, '/output/predictions.csv',
                      start_window=0)

if __name__ == '__main__':
    main()
