import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def main():
    # Seed
    seed = int(os.getenv('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    data_npz = np.load('/data/etth1_public.npz')
    train_data = data_npz['train'].astype(np.float32)
    val_data = data_npz['val'].astype(np.float32)
    # Try to get test_inputs; fallback to 'test' if needed
    if 'test_inputs' in data_npz:
        test_inputs = data_npz['test_inputs'].astype(np.float32)
    else:
        test_inputs = data_npz['test'].astype(np.float32)

    L, H, C = 336, 96, 7  # input length, horizon, channels

    # Dataset definition
    class Seq2SeqDataset(Dataset):
        def __init__(self, data, L, H):
            self.data = data
            self.L = L
            self.H = H
            self.length = data.shape[0] - L - H + 1

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            x = self.data[idx:idx + self.L]          # (L, C)
            y = self.data[idx + self.L:idx + self.L + self.H]  # (H, C)
            return x.reshape(-1), y.reshape(-1)      # flattened

    train_set = Seq2SeqDataset(train_data, L, H)
    val_set   = Seq2SeqDataset(val_data, L, H)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=0)

    # Model
    model = nn.Linear(L * C, H * C, bias=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device('cpu')
    model.to(device)

    best_val = float('inf')
    best_state = None
    patience = 10
    min_delta = 1e-5
    epochs_no_improve = 0
    max_epochs = 100

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best validation MSE: {best_val:.6f}")

    # Inference on test windows
    model.eval()
    N = test_inputs.shape[0]
    preds = np.empty((N, H, C), dtype=np.float32)
    with torch.no_grad():
        for i in range(N):
            x = test_inputs[i, -L:, :]          # (L, C)
            xt = torch.from_numpy(x.reshape(-1)).to(device).unsqueeze(0)  # (1, L*C)
            out = model(xt)                     # (1, H*C)
            preds[i] = out.cpu().numpy().reshape(H, C)

    # Write predictions.csv efficiently
    out_path = '/output/predictions.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('id,prediction\n')
        for i in range(N):
            for step in range(H):
                base = f"{i}_{step}"
                for ch in range(C):
                    fid = f"{base}_{ch}"
                    val = preds[i, step, ch]
                    f.write(f"{fid},{val:.6f}\n")

if __name__ == '__main__':
    main()
