

# Plan step 1 -> implemented in: set_seed
# Plan step 2 -> implemented in: load_data
# Plan step 3 -> implemented in: hyperparameters (L, H, C)
# Plan step 4 -> implemented in: create_training_samples
# Plan step 5 -> implemented in: LinearForecaster (nn.Linear)
# Plan step 6 -> implemented in: loss_fn (MSELoss)
# Plan step 7 -> implemented in: configure_optimizer (Adam)
# Plan step 8 -> implemented in: training_loop (batch size, shuffling)
# Plan step 9 -> implemented in: training_loop (early stopping)
# Plan step 10 -> implemented in: training_loop (best model retention)
# Plan step 11 -> implemented in: predict_test_and_val
# Plan step 12 -> implemented in: nio.write_predictions (handled externally)

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import nullius_io as nio


def set_seed(seed: int):
    """Set random seeds for numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    """Load the NPZ file via nullius_io."""
    d = nio.load()
    return d['train'], d['val'], d['test_inputs']  # each as np.ndarray float32


def create_training_samples(data: np.ndarray, L: int, H: int):
    """
    Create sliding-window samples (x, y) from a single series.
    Returns torch tensors of shape (N, L*C) and (N, H*C).
    """
    C = data.shape[1]
    window_len = L + H
    # sliding_window_view yields shape (N, window_len, C)
    windows = np.lib.stride_tricks.sliding_window_view(
        data, window_shape=(window_len, C)
    )  # shape (N, window_len, C)
    # Take history part and forecast part, then flatten
    x = windows[:, :L, :].astype(np.float32).reshape(-1, L * C)
    y = windows[:, L:, :].astype(np.float32).reshape(-1, H * C)
    return torch.from_numpy(x), torch.from_numpy(y)


class LinearForecaster(nn.Module):
    """Simple linear mapping from flattened history to flattened forecast."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        return self.linear(x)


def configure_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.0):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def compute_val_loss(model: nn.Module, combined: np.ndarray, L: int, H: int,
                     device: torch.device) -> float:
    """
    Compute MSE loss on the 117 validation windows as defined in the statement.
    combined = train + val concatenated.
    """
    model.eval()
    total_se = 0.0
    total_elems = 0
    with torch.no_grad():
        for w in range(117):  # 0..116 inclusive
            start_idx = 8640 + 24 * w  # start of forecast in combined series
            end_idx = start_idx + H
            # input is the L steps before start_idx
            x_np = combined[start_idx - L:start_idx, :]  # shape (L, C)
            y_true_np = combined[start_idx:end_idx, :]   # shape (H, C)
            x = torch.from_numpy(x_np).reshape(1, -1).to(device)
            y_true = torch.from_numpy(y_true_np).reshape(1, -1).to(device)
            y_pred = model(x)
            se = torch.sum((y_pred - y_true) ** 2).item()
            total_se += se
            total_elems += y_true.numel()
    mse = total_se / total_elems if total_elems > 0 else 0.0
    return mse


def predict_test_and_val(model: nn.Module, test_inputs: np.ndarray,
                         train: np.ndarray, val: np.ndarray,
                         L: int, H: int, device: torch.device):
    """
    Produce forecasts for all test windows (2785) and validation windows (117).
    Returns numpy arrays test_pred [2785,96,7] and val_pred [117,96,7].
    """
    model.eval()
    C = train.shape[1]
    combined = np.concatenate([train, val], axis=0)  # shape (11520, C)

    # ---- Test predictions ----
    n_test = test_inputs.shape[0]  # 2785
    test_pred = np.empty((n_test, H, C), dtype=np.float32)
    with torch.no_grad():
        for i in range(n_test):
            # take last L steps (or whole if L>720, but L<=720 here)
            hist = test_inputs[i, -L:, :]  # (L, C)
            x = torch.from_numpy(hist).reshape(1, -1).to(device)
            y = model(x).cpu().numpy().reshape(H, C)
            test_pred[i] = y.astype(np.float32)

    # ---- Validation predictions (117 windows) ----
    n_val_win = 117
    val_pred = np.empty((n_val_win, H, C), dtype=np.float32)
    with torch.no_grad():
        for w in range(n_val_win):
            start_idx = 8640 + 24 * w
            hist = combined[start_idx - L:start_idx, :]  # (L, C)
            x = torch.from_numpy(hist).reshape(1, -1).to(device)
            y = model(x).cpu().numpy().reshape(H, C)
            val_pred[w] = y.astype(np.float32)

    return test_pred, val_pred


def main():
    # ----- Step 1: Seed -----
    seed_str = os.environ.get('SEED', '0')
    set_seed(int(seed_str))

    # ----- Step 2: Load data -----
    train_np, val_np, test_inputs_np = load_data()

    # ----- Step 3: Hyperparameters -----
    L, H, C = 336, 96, 7
    # Adjust for SMOKE mode
    smoke = os.environ.get('SMOKE') == '1'
    if smoke:
        batch_size = 32
        max_epochs = 1
        patience = 1
        # reduce training data to a single batch for speed
        fast_mode = True
    else:
        batch_size = 256
        max_epochs = 30
        patience = 5
        fast_mode = False

    lr = 1e-3
    weight_decay = 0.0

    device = torch.device('cpu')

    # ----- Step 4: Create training samples -----
    if fast_mode:
        # Use only first batch worth of samples
        x_all, y_all = create_training_samples(train_np, L, H)
        # Take first batch_size samples
        x_all = x_all[:batch_size]
        y_all = y_all[:batch_size]
    else:
        x_all, y_all = create_training_samples(train_np, L, H)

    train_dataset = TensorDataset(x_all, y_all)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=False)

    # ----- Step 5: Model -----
    model = LinearForecaster(L * C, H * C).to(device)

    # ----- Step 6 & 7: Loss & Optimizer -----
    loss_fn = nn.MSELoss()
    optimizer = configure_optimizer(model, lr, weight_decay)

    # ----- Steps 8-10: Training loop with early stopping -----
    best_val_mse = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            n_batches += xb.size(0)
        avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0.0

        # Validation loss on the 117 defined windows
        combined_np = np.concatenate([train_np, val_np], axis=0)
        val_mse = compute_val_loss(model, combined_np, L, H, device)

        print(f"Epoch {epoch+1}/{max_epochs} - train MSE: {avg_train_loss:.6f} - val MSE: {val_mse:.6f}")

        if val_mse < best_val_mse - 1e-8:
            best_val_mse = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ----- Step 11: Produce predictions -----
    test_pred, val_pred = predict_test_and_val(
        model, test_inputs_np, train_np, val_np, L, H, device
    )

    # ----- Step 12: Write predictions via helper -----
    nio.write_predictions(test_pred, val_pred)

    # Final validation MSE (best) to stdout
    print(f"Best validation MSE: {best_val_mse:.6f}")


if __name__ == "__main__":
    main()