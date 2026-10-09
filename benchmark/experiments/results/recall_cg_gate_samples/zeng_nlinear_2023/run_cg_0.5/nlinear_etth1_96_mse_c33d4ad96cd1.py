# Plan step -> implemented in: NLinear class
# Plan step -> implemented in: seq_len and pred_len variables from CONFIG
# Plan step -> implemented in: enc_in variable set to 7
# Plan step -> implemented in: Adam optimizer
# Plan step -> implemented in: learning_rate variable from CONFIG
# Plan step -> implemented in: weight_decay variable from CONFIG
# Plan step -> implemented in: dropout = 0.0 (no dropout layer)
# Plan step -> implemented in: epochs variable from CONFIG
# Plan step -> implemented in: patience variable from CONFIG
# Plan step -> implemented in: MSELoss
# Plan step -> implemented in: preprocessing inside NLinear.forward (subtract last timestep)
# Plan step -> implemented in: validation loop computing MSE on spaced windows
# Plan step -> implemented in: test prediction loop using last seq_len of test_inputs

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters read only from this dict
CONFIG = {
    "lr": 0.005,
    "batch_size": 32,
    "seq_len": 336,
    "weight_decay": 0.0,
    "epochs": 20,
    "patience": 5,
}

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class NLinear(nn.Module):
    """
    Normalization-Linear model as in the authors' code.
    """
    def __init__(self, configs):
        super(NLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual  # False -> shared linear
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

def create_dataset(data: np.ndarray, seq_len: int, pred_len: int):
    """Create sliding window dataset for training."""
    L = data.shape[0]
    inputs = []
    targets = []
    for i in range(0, L - seq_len - pred_len + 1):
        inputs.append(data[i:i + seq_len])
        targets.append(data[i + seq_len:i + seq_len + pred_len])
    inputs = np.stack(inputs)  # [N, seq_len, C]
    targets = np.stack(targets)  # [N, pred_len, C]
    return inputs, targets

def main():
    seed = int(os.environ.get("SEED", "0"))
    set_seed(seed)

    smoke = os.environ.get("SMOKE") == "1"

    import nullius_io as nio
    d = nio.load()
    train_data = d["train"].astype(np.float32)      # [8640, 7]
    val_data = d["val"].astype(np.float32)          # [2880, 7]
    test_inputs = d["test_inputs"].astype(np.float32)  # [2785, 720, 7]

    # Combine train+val for history when predicting validation/test
    series_full = np.concatenate([train_data, val_data], axis=0)  # [11520, 7]

    seq_len = CONFIG["seq_len"]
    pred_len = 96
    enc_in = 7

    # Prepare training data
    train_inputs, train_targets = create_dataset(train_data, seq_len, pred_len)
    train_inputs = torch.from_numpy(train_inputs)   # [N, seq_len, 7]
    train_targets = torch.from_numpy(train_targets) # [N, pred_len, 7]

    train_dataset = TensorDataset(train_inputs, train_targets)
    loader = DataLoader(train_dataset,
                        batch_size=CONFIG["batch_size"],
                        shuffle=True,
                        drop_last=False)

    device = torch.device("cpu")
    model_cfg = type('Config', (), {})()
    model_cfg.seq_len = seq_len
    model_cfg.pred_len = pred_len
    model_cfg.enc_in = enc_in
    model_cfg.individual = False  # shared linear across channels
    model = NLinear(model_cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CONFIG["lr"],
                                 weight_decay=CONFIG["weight_decay"])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    max_epochs = CONFIG["epochs"]
    patience = CONFIG["patience"]

    # If smoke mode, limit to one epoch and one batch
    if smoke:
        max_epochs = 1
        # we will only take first batch from loader later

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            batches += batch_x.size(0)
            if smoke:
                break  # only one batch
        epoch_loss /= max(batches, 1)

        # Validation loss (spaced windows as per spec)
        model.eval()
        val_losses = []
        with torch.no_grad():
            num_val_windows = 117  # as derived
            for w in range(num_val_windows):
                start_idx = 8640 + 24 * w  # start of forecast in series_full
                history = series_full[:start_idx]  # all available history
                # take last seq_len steps
                if history.shape[0] < seq_len:
                    # should not happen given start_idx >= 8640 >> seq_len
                    hist = np.zeros((seq_len, enc_in), dtype=np.float32)
                    hist[-history.shape[0]:] = history[-history.shape[0]:]
                else:
                    hist = history[-seq_len:]
                hist_tensor = torch.from_numpy(hist).unsqueeze(0).to(device)  # [1, seq_len, 7]
                pred = model(hist_tensor)  # [1, 96, 7]
                true = series_full[start_idx:start_idx + pred_len]  # [96, 7]
                true_tensor = torch.from_numpy(true).unsqueeze(0).to(device)
                loss = criterion(pred, true_tensor)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses) if val_losses else 0.0

        print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {epoch_loss:.6f} - Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Produce validation predictions (117 windows)
    model.eval()
    val_pred_list = []
    with torch.no_grad():
        for w in range(117):
            start_idx = 8640 + 24 * w
            history = series_full[:start_idx]
            if history.shape[0] < seq_len:
                hist = np.zeros((seq_len, enc_in), dtype=np.float32)
                hist[-history.shape[0]:] = history[-history.shape[0]:]
            else:
                hist = history[-seq_len:]
            hist_tensor = torch.from_numpy(hist).unsqueeze(0).to(device)
            pred = model(hist_tensor)  # [1, 96, 7]
            val_pred_list.append(pred.squeeze(0).cpu().numpy())
    val_pred = np.stack(val_pred_list, axis=0)  # [117, 96, 7]

    # Produce test predictions (2785 windows)
    test_pred_list = []
    with torch.no_grad():
        for i in range(test_inputs.shape[0]):
            # use last seq_len of the provided 720-step history
            hist = test_inputs[i, -seq_len:, :]  # [seq_len, 7]
            hist_tensor = torch.from_numpy(hist).unsqueeze(0).to(device)
            pred = model(hist_tensor)  # [1, 96, 7]
            test_pred_list.append(pred.squeeze(0).cpu().numpy())
    test_pred = np.stack(test_pred_list, axis=0)  # [2785, 96, 7]

    # Write predictions via helper
    nio.write_predictions(test_pred, val_pred)

    # Print final validation MSE (the one from best model)
    print(f"Final Validation MSE: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
