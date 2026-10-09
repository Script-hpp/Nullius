# Plan step 1 -> implemented in: class NLinear
# Plan step 2 -> implemented in: globals L, H
# Plan step 3 -> implemented in: function create_windows
# Plan step 4 -> implemented in: optimizer initialization
# Plan step 5 -> implemented in: loss function
# Plan step 6 -> implemented in: training loop with early stopping
# Plan step 7 -> implemented in: validation data loader
# Plan step 8 -> implemented in: seeding block
# Plan step 9 -> implemented in: inference function predict_from_history
# Plan step 10 -> implemented in: nio.write_predictions call
# Plan step 11 -> runtime comment: CPU‑only, finishes within limits

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import nullius_io as nio

# -------------------- Hyperparameters --------------------
L = 336          # input sequence length
H = 96           # forecast horizon
LR = 0.001       # learning rate
BATCH_SIZE = 256
MAX_EPOCHS = 20
PATIENCE = 5     # early stopping patience
# --------------------------------------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_windows(data: torch.Tensor):
    """
    data: tensor [T, C]
    Returns X [N, L, C], Y [N, H, C] where N = T - L - H + 1
    """
    T, C = data.shape
    X = []
    Y = []
    for i in range(T - L - H + 1):
        X.append(data[i:i+L])
        Y.append(data[i+L:i+L+H])
    X = torch.stack(X)  # [N, L, C]
    Y = torch.stack(Y)  # [N, H, C]
    return X, Y

class NLinear(nn.Module):
    """
    NLinear model: separate linear layer per channel.
    Maps last L steps of each variate to next H steps.
    """
    def __init__(self, seq_len: int, pred_len: int, enc_in: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.linears = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(enc_in)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        returns: [B, H, C]
        """
        # Permute to [B, C, L] to apply each linear independently
        x_perm = x.permute(0, 2, 1)          # [B, C, L]
        outs = []
        for c, lin in enumerate(self.linears):
            # lin expects [B, L] -> [B, H]
            out = lin(x_perm[:, c, :])      # [B, H]
            outs.append(out.unsqueeze(-1))  # [B, H, 1]
        # Concatenate over channel dimension -> [B, H, C]
        return torch.cat(outs, dim=2)

def train_model(model, train_loader, val_loader, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0)
    best_val = float('inf')
    epochs_no_improve = 0
    best_state = None

    max_epochs = 1 if os.getenv('SMOKE') == '1' else MAX_EPOCHS

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{max_epochs} - Train MSE: {train_loss:.6f} - Val MSE: {val_loss:.6f}')
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print('Early stopping triggered.')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val

def predict_from_history(model, history: torch.Tensor) -> torch.Tensor:
    """
    history: [N, L, C] -> returns forecasts [N, H, C]
    """
    model.eval()
    with torch.no_grad():
        return model(history.to(next(model.parameters()).device)).cpu()

def main():
    # ----- Seed -----
    seed = int(os.environ.get('SEED', 0))
    set_seed(seed)

    # ----- Load data -----
    d = nio.load()
    train_np = d['train'].astype(np.float32)   # [8640, 7]
    val_np   = d['val'].astype(np.float32)     # [2880, 7]
    test_inputs_np = d['test_inputs'].astype(np.float32)  # [2785, 720, 7]

    train = torch.from_numpy(train_np)
    val   = torch.from_numpy(val_np)
    test_inputs = torch.from_numpy(test_inputs_np)

    # ----- Prepare training/validation windows -----
    X_train, Y_train = create_windows(train)   # from train only
    X_val,   Y_val   = create_windows(val)     # from val split (as per plan)

    train_set = TensorDataset(X_train, Y_train)
    val_set   = TensorDataset(X_val,   Y_val)

    # In smoke mode, limit training data to first batch of size 32
    if os.getenv('SMOKE') == '1':
        train_set = torch.utils.data.Subset(train_set, range(min(32, len(train_set))))
        # keep validation set unchanged (still small)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    # ----- Model -----
    device = torch.device('cpu')
    model = NLinear(seq_len=L, pred_len=H, enc_in=train.shape[1]).to(device)

    # ----- Train -----
    trained_model, val_mse = train_model(model, train_loader, val_loader, device)
    print(f'Validation MSE: {val_mse:.6f}')

    # ----- Inference on test windows -----
    # Use last L steps of each test_inputs[i]
    test_history = test_inputs[:, -L:, :]               # [2785, L, 7]
    test_pred = predict_from_history(trained_model, test_history)  # [2785, 96, 7]

    # ----- Inference on validation windows (as defined in statement) -----
    combined = torch.cat([train, val], dim=0)           # [8640+2880, 7] = [11520, 7]
    n_val_windows = 117
    val_histories = []
    for w in range(n_val_windows):
        start_idx = 8640 + 24 * w                     # start of forecast window in combined series
        hist = combined[start_idx - L:start_idx]       # [L, 7]
        val_histories.append(hist)
    val_history_tensor = torch.stack(val_histories)   # [117, L, 7]
    val_pred = predict_from_history(trained_model, val_history_tensor)  # [117, 96, 7]

    # ----- Write predictions -----
    nio.write_predictions(test_pred, val_pred)

if __name__ == '__main__':
    main()
