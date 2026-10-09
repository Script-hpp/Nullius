

# Plan step -> implemented in: NLinear class
# Plan step -> implemented in: main() (L=336, H=96)
# Plan step -> implemented in: create_windows() and DataLoader
# Plan step -> implemented in: train_model() (Adam optimizer)
# Plan step -> implemented in: train_model() (nn.MSELoss)
# Plan step -> implemented in: train_model() (training loop with early stopping)
# Plan step -> implemented in: validate()
# Plan step -> implemented in: set_seeds()
# Plan step -> implemented in: predict_test() and predict_val()
# Plan step -> implemented in: nio.write_predictions()
# Plan step -> implemented in: CPU-only settings, early stopping to keep runtime <5 min

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import nullius_io as nio


def set_seeds(seed: int):
    """Step 8 -> implemented in: set_seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_windows(data: np.ndarray, L: int, H: int):
    """
    Step 3 -> implemented in: create_windows
    Returns sliding windows (x, y) where:
        x[t] = data[t : t+L]
        y[t] = data[t+L : t+L+H]
    """
    T, C = data.shape
    num = T - L - H + 1
    if num <= 0:
        raise ValueError("Not enough data to create windows with given L and H.")
    # Use sliding_window_view and copy to ensure writable arrays
    x = np.lib.stride_tricks.sliding_window_view(data, window_shape=(L, C))[:num].copy()
    y = np.lib.stride_tricks.sliding_window_view(data, window_shape=(H, C))[L:L + num].copy()
    return x, y


class NLinear(nn.Module):
    """
    Step 1 -> implemented in: NLinear class
    Maps last L steps of each variate to next H steps using a shared linear layer.
    """
    def __init__(self, seq_len: int, pred_len: int, enc_in: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

    def forward(self, x):
        # Accept [B, L, C] or [B, C, L] and convert to [B, L, C];
        # also tolerate accidental singleton dimensions.
        if x.dim() != 3:
            # Remove singleton dimensions until we have 3D
            while x.dim() > 3:
                # find a dimension of size 1 to squeeze
                squeeze_dims = [i for i, s in enumerate(x.shape) if s == 1]
                if not squeeze_dims:
                    break
                x = x.squeeze(squeeze_dims[0])
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input after squeezing singletons, got {x.dim()}D with shape {x.shape}")
        B = x.size(0)
        if x.size(1) == self.seq_len and x.size(2) == self.enc_in:
            # already [B, L, C]
            pass
        elif x.size(2) == self.seq_len and x.size(1) == self.enc_in:
            # [B, C, L] -> permute to [B, L, C]
            x = x.permute(0, 2, 1)
        else:
            raise ValueError(
                f"Input shape {x.shape} does not match expected seq_len={self.seq_len}, enc_in={self.enc_in}"
            )
        # Now x is [B, L, C]
        x = x.permute(0, 2, 1).reshape(B * self.enc_in, self.seq_len)  # [B*C, L]
        out = self.linear(x)                                          # [B*C, H]
        out = out.view(B, self.enc_in, self.pred_len)                # [B, C, H]
        out = out.permute(0, 2, 1)                                   # [B, H, C]
        return out


def train_model(model, train_loader, device, max_epochs=20, patience=5, lr=0.001):
    """
    Steps 4-6 -> implemented in: train_model
    Returns the trained model.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)
    criterion = nn.MSELoss()

    best_val = float('inf')
    epochs_no_improve = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # validation loss (using provided val split)
        val_loss = validate(model, device)
        print(f"Epoch {epoch+1}/{max_epochs} - Train MSE: {train_loss:.6f} - Val MSE: {val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def validate(model, device):
    """
    Step 7 -> implemented in: validate
    Computes MSE on the validation windows defined in the statement.
    """
    model.eval()
    d = nio.load()
    train = d['train'].astype(np.float32)
    val = d['val'].astype(np.float32)
    series = np.concatenate([train, val], axis=0)   # shape [11520, 7]
    L = model.seq_len
    H = model.pred_len
    C = model.enc_in

    total_se = 0.0
    total_cnt = 0
    with torch.no_grad():
        for w in range(117):  # validation windows count
            start = len(train) + 24 * w          # start index in series
            hist_start = start - L
            hist_end = start
            x_np = series[hist_start:hist_end]   # [L, 7]
            y_true = series[start:start+H]       # [H, 7]

            x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(device)   # [1, L, 7]
            y_tensor = torch.from_numpy(y_true).unsqueeze(0).to(device) # [1, H, 7]

            preds = model(x_tensor)
            se = torch.sum((preds - y_tensor) ** 2).item()
            total_se += se
            total_cnt += y_tensor.numel()

    mse = total_se / total_cnt
    return mse


def predict_test(model, device):
    """
    Step 9 -> implemented in: predict_test
    Returns forecasts for all test windows.
    """
    d = nio.load()
    test_inputs = d['test_inputs'].astype(np.float32)   # [2785, 720, 7]
    L = model.seq_len
    H = model.pred_len
    C = model.enc_in

    model.eval()
    preds_list = []
    with torch.no_grad():
        for i in range(test_inputs.shape[0]):
            x_np = test_inputs[i, -L:, :]               # [L, 7]
            x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(device)  # [1, L, 7]
            pred = model(x_tensor)                       # [1, H, 7]
            preds_list.append(pred.squeeze(0).cpu().numpy())  # [H, 7]
    test_pred = np.stack(preds_list, axis=0)            # [2785, H, 7]
    return test_pred


def predict_val(model, device):
    """
    Produces val_pred as required: 117 validation windows.
    Uses same logic as validate but returns predictions.
    """
    d = nio.load()
    train = d['train'].astype(np.float32)
    val = d['val'].astype(np.float32)
    series = np.concatenate([train, val], axis=0)       # [11520, 7]
    L = model.seq_len
    H = model.pred_len
    C = model.enc_in

    model.eval()
    preds_list = []
    with torch.no_grad():
        for w in range(117):
            start = len(train) + 24 * w
            hist_start = start - L
            hist_end = start
            x_np = series[hist_start:hist_end]          # [L, 7]
            x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(device)
            pred = model(x_tensor)                      # [1, H, 7]
            preds_list.append(pred.squeeze(0).cpu().numpy())
    val_pred = np.stack(preds_list, axis=0)             # [117, H, 7]
    return val_pred


def main():
    # Step 8 -> implemented in: set_seeds
    seed_str = os.environ.get('SEED', '0')
    try:
        seed = int(seed_str)
    except ValueError:
        seed = 0
    set_seeds(seed)

    # Hyperparameters (from plan)
    L = 336          # history length
    H = 96           # forecast horizon
    batch_size = 256
    max_epochs = 20
    patience = 5
    lr = 0.001

    # Smoke mode handling
    smoke = os.environ.get('SMOKE') == '1'
    if smoke:
        max_epochs = 1
        batch_size = 32
        print("SMOKE mode enabled: reduced epochs and batch size.")

    # Load data
    d = nio.load()
    train = d['train'].astype(np.float32)   # [8640, 7]

    # Prepare training windows
    x_train, y_train = create_windows(train, L, H)   # x: [N, L, 7], y: [N, H, 7]
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Instantiate model
    model = NLinear(seq_len=L, pred_len=H, enc_in=train.shape[1])

    # Train
    model = train_model(model, train_loader,
                        device=torch.device('cpu'),
                        max_epochs=max_epochs,
                        patience=patience,
                        lr=lr)

    # Final validation MSE (print to stdout)
    final_val_mse = validate(model, torch.device('cpu'))
    print(f"Final Validation MSE: {final_val_mse:.6f}")

    # Inference
    test_pred = predict_test(model, torch.device('cpu'))   # [2785, 96, 7]
    val_pred = predict_val(model, torch.device('cpu'))     # [117, 96, 7]

    # Write predictions via helper
    nio.write_predictions(test_pred, val_pred)


if __name__ == "__main__":
    main()