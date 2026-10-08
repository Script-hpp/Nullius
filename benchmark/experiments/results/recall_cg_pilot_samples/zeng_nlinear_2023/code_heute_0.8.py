# Plan step -> implemented in: set_seeds
# Plan step -> implemented in: load_data
# Plan step -> implemented in: configure_hyperparams
# Plan step -> implemented in: make_dataset
# Plan step -> implemented in: LinearForecaster
# Plan step -> implemented in: train_channel
# Plan step -> implemented in: train_channel (best model kept)
# Plan step -> implemented in: predict_test, predict_val
# Plan step -> implemented in: save_predictions, save_val_predictions

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def set_seeds():
    seed = int(os.environ.get('SEED', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    data = np.load('/data/etth1_public.npz')
    train = data['train'].astype(np.float32)   # [8640,7]
    val = data['val'].astype(np.float32)       # [2880,7]
    test_inputs = data['test_inputs'].astype(np.float32)  # [2785,720,7]
    return train, val, test_inputs

def configure_hyperparams():
    L = 336          # input length
    H = 96           # horizon
    C = 7            # channels
    batch_size = 256
    lr = 0.001
    max_epochs = 20
    patience = 5
    return L, H, C, batch_size, lr, max_epochs, patience

def make_dataset(series, L, H):
    """Create sliding window dataset from 1D series."""
    n = len(series)
    n_samples = n - L - H + 1
    X = np.zeros((n_samples, L), dtype=np.float32)
    y = np.zeros((n_samples, H), dtype=np.float32)
    for i in range(n_samples):
        X[i] = series[i:i+L]
        y[i] = series[i+L:i+L+H]
    return X, y

class LinearForecaster(nn.Module):
    def __init__(self, L, H):
        super().__init__()
        self.linear = nn.Linear(L, H, bias=True)

    def forward(self, x):
        return self.linear(x)

def train_channel(X_train, y_train, X_val, y_val, L, H, batch_size, lr, max_epochs, patience, smoke=False):
    device = torch.device('cpu')
    model = LinearForecaster(L, H).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        if smoke:
            # only one epoch in smoke mode
            break

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_losses.append(criterion(preds, yb).item())
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss

def predict_test(model, test_inputs, L, C, smoke=False):
    """Return predictions for all test windows, shape (N_test, H, C)."""
    N_test = test_inputs.shape[0]
    if smoke:
        N_test = 8  # windows 0..7
    preds = np.zeros((N_test, 96, C), dtype=np.float32)
    for i in range(N_test):
        seq = test_inputs[i, -L:, :]  # (L,7)
        for c in range(C):
            x = torch.from_numpy(seq[:, c]).unsqueeze(0).float()  # (1,L)
            with torch.no_grad():
                out = model(x).squeeze(0).numpy()  # (H,)
            preds[i, :, c] = out
    return preds

def predict_val(model, combined, L, C, smoke=False):
    """Return predictions for validation windows, shape (N_val_win, H, C)."""
    total_len = combined.shape[0]  # train+val
    start_idx = 8640  # first validation hour in combined
    step = 24
    N_val = 117
    if smoke:
        N_val = 8
    preds = np.zeros((N_val, 96, C), dtype=np.float32)
    for w in range(N_val):
        start = start_idx + step * w
        seq = combined[start-L:start, :]  # (L,7)
        for c in range(C):
            x = torch.from_numpy(seq[:, c]).unsqueeze(0).float()
            with torch.no_grad():
                out = model(x).squeeze(0).numpy()
            preds[w, :, c] = out
    return preds

def save_predictions(preds, path):
    """Save predictions to CSV with header id,prediction."""
    N, H, C = preds.shape
    lines = ["id,prediction"]
    idx = 0
    for i in range(N):
        for step in range(H):
            for c in range(C):
                lines.append(f"{i}_{step}_{c},{preds[i, step, c]:.6f}")
    with open(path, 'w') as f:
        f.write("\n".join(lines))

def save_val_predictions(preds, path):
    N, H, C = preds.shape
    lines = ["id,prediction"]
    for w in range(N):
        for step in range(H):
            for c in range(C):
                lines.append(f"{w}_{step}_{c},{preds[w, step, c]:.6f}")
    with open(path, 'w') as f:
        f.write("\n".join(lines))

def main():
    set_seeds()
    train, val, test_inputs = load_data()
    L, H, C, batch_size, lr, max_epochs, patience = configure_hyperparams()
    smoke = os.environ.get('SMOKE') == '1'

    models = []
    val_losses = []

    # combined series for validation inputs
    combined = np.concatenate([train, val], axis=0)  # [11520,7]

    for c in range(C):
        series_train = train[:, c]
        series_val = val[:, c]
        Xtr, ytr = make_dataset(series_train, L, H)
        Xval, yval = make_dataset(series_val, L, H)

        if smoke:
            # use only first 32 samples for one epoch
            Xtr = Xtr[:32]
            ytr = ytr[:32]
            # keep validation as is (small)
        model, best_val = train_channel(
            Xtr, ytr, Xval, yval,
            L, H, batch_size, lr, max_epochs, patience, smoke
        )
        models.append(model)
        val_losses.append(best_val)

    avg_val_loss = np.mean(val_losses)
    print(f"Validation MSE: {avg_val_loss:.6f}")

    # Test predictions
    test_preds = predict_test(models[0], test_inputs, L, C, smoke)  # placeholder shape
    # Actually need to loop over channels and fill; predict_test already does that
    test_preds = np.zeros((test_inputs.shape[0] if not smoke else 8, 96, C), dtype=np.float32)
    for c in range(C):
        model = models[c]
        for i in range(test_preds.shape[0]):
            seq = test_inputs[i, -L:, :] if not smoke else test_inputs[i, -L:, :]
            x = torch.from_numpy(seq[:, c]).unsqueeze(0).float()
            with torch.no_grad():
                out = model(x).squeeze(0).numpy()
            test_preds[i, :, c] = out

    os.makedirs('/output', exist_ok=True)
    save_predictions(test_preds, '/output/predictions.csv')

    # Validation predictions
    val_preds = predict_val(models[0], combined, L, C, smoke)
    val_preds = np.zeros((117 if not smoke else 8, 96, C), dtype=np.float32)
    for c in range(C):
        model = models[c]
        for w in range(val_preds.shape[0]):
            start = 8640 + 24 * w
            seq = combined[start-L:start, :]
            x = torch.from_numpy(seq[:, c]).unsqueeze(0).float()
            with torch.no_grad():
                out = model(x).squeeze(0).numpy()
            val_preds[w, :, c] = out
    save_val_predictions(val_preds, '/output/val_predictions.csv')

if __name__ == '__main__':
    main()
