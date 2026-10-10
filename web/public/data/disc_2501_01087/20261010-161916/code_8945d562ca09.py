# Plan step -> implemented in: imports and CONFIG
# Plan step -> implemented in: RevIN class
# Plan step -> implemented in: GLinear model class
# Plan step -> implemented in: adjust_learning_rate function
# Plan step -> implemented in: dataset creation helpers
# Plan step -> implemented in: training loop with early stopping
# Plan step -> implemented in: prediction generation
# Plan step -> implemented in: main driver and I/O

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import nullius_io as nio

# Hyperparameters to read only from this dict
CONFIG = {"lr": 0.001, "patience": 7, "seq_len": 336}


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x


class GLinear(nn.Module):
    def __init__(self, configs):
        super(GLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_num_features = configs.enc_in

        self.Linear = nn.Linear(self.seq_len, self.seq_len)
        self.GeLU = nn.GELU()
        self.Hidden1 = nn.Linear(self.seq_len, self.pred_len)

        self.revin_layer = RevIN(self.in_num_features)

    def forward(self, x):
        # x: [Batch, Seq_len, Channel]
        x = self.revin_layer(x, "norm")
        x = x.permute(0, 2, 1)  # [Batch, Channel, Seq_len]
        x = self.Linear(x)
        x = self.GeLU(x)
        x = self.Hidden1(x)  # [Batch, Channel, Pred_len]
        x = x.permute(0, 2, 1)  # [Batch, Pred_len, Channel]
        x = self.revin_layer(x, "denorm")
        return x


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
    elif args.lradj == "3":
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == "4":
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == "5":
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == "6":
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")


def create_sliding_window(data, seq_len, pred_len):
    """data: np.ndarray [T, C]; returns inputs [N, seq_len, C], targets [N, pred_len, C]"""
    T = data.shape[0]
    N = T - seq_len - pred_len + 1
    inputs = np.zeros((N, seq_len, data.shape[1]), dtype=np.float32)
    targets = np.zeros((N, pred_len, data.shape[1]), dtype=np.float32)
    for i in range(N):
        inputs[i] = data[i : i + seq_len]
        targets[i] = data[i + seq_len : i + seq_len + pred_len]
    return inputs, targets


def main():
    # Seed
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Smoke mode
    smoke = os.environ.get("SMOKE") == "1"

    # Load data
    d = nio.load()
    train = d["train"].astype(np.float32)  # [8640,7]
    val = d["val"].astype(np.float32)      # [2880,7]
    test_inputs = d["test_inputs"].astype(np.float32)  # [2785,720,7]

    seq_len = CONFIG["seq_len"]
    pred_len = 96
    enc_in = 7

    # Combined series for validation history
    combined = np.concatenate([train, val], axis=0)  # [11520,7]

    # Prepare training dataset
    train_inputs, train_targets = create_sliding_window(train, seq_len, pred_len)
    train_inputs_t = torch.from_numpy(train_inputs)
    train_targets_t = torch.from_numpy(train_targets)

    # Prepare validation dataset (stride 24)
    val_starts = 8640 + 24 * np.arange(0, 117)  # 0..116
    val_inputs_list = []
    val_targets_list = []
    for start in val_starts:
        val_inputs_list.append(combined[start - seq_len : start])
        val_targets_list.append(combined[start : start + pred_len])
    val_inputs = np.stack(val_inputs_list, axis=0)  # [117, seq_len, 7]
    val_targets = np.stack(val_targets_list, axis=0)  # [117, pred_len, 7]
    val_inputs_t = torch.from_numpy(val_inputs)
    val_targets_t = torch.from_numpy(val_targets)

    # Prepare test dataset (use last seq_len of each 720-length window)
    test_inputs_seq = test_inputs[:, -seq_len:, :]  # [2785, seq_len, 7]
    test_inputs_t = torch.from_numpy(test_inputs_seq)

    # DataLoaders
    batch_size = 64 if not smoke else 32
    train_loader = DataLoader(
        TensorDataset(train_inputs_t, train_targets_t),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_inputs_t, val_targets_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Model, optimizer, loss
    class Configs:
        pass

    configs = Configs()
    configs.seq_len = seq_len
    configs.pred_len = pred_len
    configs.enc_in = enc_in

    model = GLinear(configs)
    model.float()  # CPU

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss()

    # Learning‑rate schedule args
    class Args:
        pass

    args = Args()
    args.learning_rate = CONFIG["lr"]
    args.lradj = "type1"  # default choice as not specified

    # Early stopping
    patience = CONFIG["patience"]
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    max_epochs = 100 if not smoke else 1

    for epoch in range(max_epochs):
        adjust_learning_rate(optimizer, epoch, args)
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item() * xb.size(0))
        val_loss = np.sum(val_losses) / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{max_epochs} - train loss: {epoch_loss:.6f} - val loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation loss (print to stdout)
    model.eval()
    with torch.no_grad():
        val_preds_list = []
        val_targets_list = []
        for xb, yb in val_loader:
            preds = model(xb)
            val_preds_list.append(preds.numpy())
            val_targets_list.append(yb.numpy())
    val_preds = np.concatenate(val_preds_list, axis=0)  # [117,96,7]
    # Compute MSE for reporting
    mse_val = np.mean((val_preds - np.concatenate(val_targets_list, axis=0)) ** 2)
    print(f"Validation MSE: {mse_val:.6f}")

    # Test predictions
    model.eval()
    test_preds_list = []
    with torch.no_grad():
        # Process in batches to avoid memory issues
        test_loader = DataLoader(
            TensorDataset(test_inputs_t),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        for (xb,) in test_loader:
            preds = model(xb)
            test_preds_list.append(preds.numpy())
    test_preds = np.concatenate(test_preds_list, axis=0)  # [2785,96,7]

    # Write predictions via helper
    nio.write_predictions(test_preds, val_preds)


if __name__ == "__main__":
    main()
