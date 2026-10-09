# Plan step 1 -> implemented in: set_seeds()
# Plan step 2 -> implemented in: load_data()
# Plan step 3 -> implemented in: set_hyperparameters()
# Plan step 4 -> implemented in: create_window_dataset()
# Plan step 5 -> implemented in: initialize_model()
# Plan step 6 -> implemented in: train_model()
# Plan step 7 -> implemented in: train_model() (best model selection)
# Plan step 8 -> implemented in: predict_test() and predict_val()
# Plan step 9 -> implemented in: write_predictions()

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def set_seeds():
    seed = int(os.environ.get('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    data = np.load('/data/etth1_public.npz')
    train = data['train'].astype(np.float32)   # [8640, 7]
    val = data['val'].astype(np.float32)       # [2880, 7]
    test_inputs = data['test_inputs'].astype(np.float32)  # [2785, 720, 7]
    return train, val, test_inputs

def set_hyperparameters():
    L = 336   # input length
    H = 96    # horizon
    C = 7     # channels
    return L, H, C

def create_window_dataset(series, L, H):
    """
    series: 1D np.ndarray length T
    Returns X (N, L), y (N, H) where N = T - L - H + 1
    """
    T = series.shape[0]
    N = T - L - H + 1
    X = np.lib.stride_tricks.sliding_window_view(series, L)[0:N]  # (N, L)
    y = np.lib.stride_tricks.sliding_window_view(series, H)[L:L+N]  # (N, H)
    return X, y

def initialize_model(L, H):
    model = nn.Linear(L, H, bias=True)
    return model

def train_model(X_train, y_train, X_val, y_val, L, H, smoke=False):
    """
    Trains a single-channel Linear model.
    Returns the best model (state_dict) and the best validation MSE.
    """
    # Tensors
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    batch_size = 32 if smoke else 256
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = initialize_model(L, H)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_epochs = 1 if smoke else 20
    patience = 5 if not smoke else max_epochs + 1  # effectively disable early stop in smoke
    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val

def predict_test(model, test_inputs, L, H, smoke=False):
    """
    Returns predictions for all test windows and this channel.
    Shape: (num_windows, H)
    """
    if smoke:
        windows = np.arange(0, 8)  # first 8 windows
    else:
        windows = np.arange(test_inputs.shape[0])  # 0..2784
    # Take last L steps for each window
    seq = test_inputs[windows, -L:, :]  # (num_windows, L, C)
    # We'll predict per channel outside; this function expects already sliced channel
    # So caller will pass seq_c of shape (num_windows, L)
    raise NotImplementedError  # placeholder

def predict_channel(model, seq_L):
    """
    seq_L: torch tensor (N, L)
    Returns: np.ndarray (N, H)
    """
    model.eval()
    with torch.no_grad():
        pred = model(seq_L)
    return pred.cpu().numpy()

def write_predictions(pred_array, out_path, start_window=0):
    """
    pred_array: shape (num_windows, H, C)
    Writes CSV with header id,prediction.
    """
    num_windows, H, C = pred_array.shape
    with open(out_path, 'w') as f:
        f.write('id,prediction\n')
        for w in range(num_windows):
            wi = start_window + w
            for step in range(H):
                for ch in range(C):
                    val = pred_array[w, step, ch]
                    f.write(f'{wi}_{step}_{ch},{val}\n')

def main():
    set_seeds()
    train, val, test_inputs = load_data()
    L, H, C = set_hyperparameters()
    smoke = os.environ.get('SMOKE', '0') == '1'

    # Prepare per-channel containers
    test_preds = np.zeros((test_inputs.shape[0] if not smoke else 8, H, C), dtype=np.float32)
    val_preds = None
    val_mse_total = 0.0
    val_count = 0

    # For validation predictions we need combined series
    combined = np.concatenate([train, val], axis=0)  # [11520, 7]
    val_start_idx = train.shape[0]  # 8640
    if smoke:
        val_windows = np.arange(0, 8)
    else:
        # number of validation windows as per description: 117 windows (0..116)
        # compute max w such that start+96 <= len(combined)
        max_w = (combined.shape[0] - val_start_idx - H) // 24 + 1
        val_windows = np.arange(0, max_w)  # should be 117

    val_preds = np.zeros((len(val_windows), H, C), dtype=np.float32)

    # Loop over channels
    for c in range(C):
        print(f'Processing channel {c}')
        # --- training data ---
        series_train = train[:, c]
        X_train, y_train = create_window_dataset(series_train, L, H)
        series_val = val[:, c]
        X_val, y_val = create_window_dataset(series_val, L, H)

        # --- maybe subsample for smoke ---
        if smoke:
            # use only first 32 samples
            if X_train.shape[0] > 32:
                X_train = X_train[:32]
                y_train = y_train[:32]

        # --- train model ---
        model, best_val_mse = train_model(X_train, y_train, X_val, y_val, L, H, smoke=smoke)
        print(f'  Best val MSE for channel {c}: {best_val_mse:.6f}')

        # --- test prediction ---
        if smoke:
            test_seq = test_inputs[:8, -L:, c]  # (8, L)
        else:
            test_seq = test_inputs[:, -L:, c]   # (2785, L)
        test_seq_t = torch.from_numpy(test_seq).float()
        pred_test_c = predict_channel(model, test_seq_t)  # (num_windows, H)
        test_preds[:, :, c] = pred_test_c

        # --- validation prediction ---
        # Build history windows for each validation window
        val_seq_list = []
        for w in val_windows:
            start = val_start_idx + 24 * w
            hist = combined[start - L:start, c]  # (L,)
            val_seq_list.append(hist)
        val_seq = np.stack(val_seq_list, axis=0)  # (num_val_windows, L)
        val_seq_t = torch.from_numpy(val_seq).float()
        pred_val_c = predict_channel(model, val_seq_t)  # (num_val_windows, H)
        val_preds[:, :, c] = pred_val_c

        # Accumulate validation MSE for reporting (compare to true future values)
        # True future values: combined[start:start+H, c]
        true_list = []
        for w in val_windows:
            start = val_start_idx + 24 * w
            true_list.append(combined[start:start+H, c])
        true_val = np.stack(true_list, axis=0)  # (num_val_windows, H)
        mse_c = np.mean((pred_val_c - true_val) ** 2)
        val_mse_total += mse_c
        val_count += 1

    avg_val_mse = val_mse_total / val_count if val_count > 0 else 0.0
    print(f'Average validation MSE over {val_count} channels: {avg_val_mse:.6f}')

    # --- write outputs ---
    os.makedirs('/output', exist_ok=True)
    write_predictions(test_preds, '/output/predictions.csv',
                      start_window=0 if not smoke else 0)
    write_predictions(val_preds, '/output/val_predictions.csv',
                      start_window=0 if not smoke else 0)

if __name__ == '__main__':
    main()
