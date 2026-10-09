

# Plan step -> implemented in: NLinear class
# Plan step -> implemented in: adjust_learning_rate function
# Plan step -> implemented in: create_dataset function
# Plan step -> implemented in: train_model function
# Plan step -> implemented in: predict_windows function
# Plan step -> implemented in: main function under if __name__ == "__main__":

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import nullius_io as nio

# Literal dict as required; read hyperparameters only from this dict
CONFIG = {"lr": 0.005, "batch_size": 32, "seq_len": 336}
# Additional configuration keys (values chosen as per plan)
CONFIG.update({
    "pred_len": 96,
    "enc_in": 7,
    "individual": False,
    "weight_decay": 0.0,
    "epochs": 20,
    "patience": 5,
    "lradj": "type1"
})

def adjust_learning_rate(optimizer, epoch, args):
    """Adapted from utils/tools.py (type1 schedule)."""
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    else:
        lr_adjust = {}
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')

class NLinear(nn.Module):
    """
    Normalization-Linear model as per the authors' code.
    """
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
    """Create sliding window dataset (step=1). Handles insufficient data."""
    L = data.shape[0]
    max_start = L - seq_len - pred_len
    if max_start < 0:
        empty_inputs = torch.empty((0, seq_len, data.shape[1]), dtype=torch.float32)
        empty_targets = torch.empty((0, pred_len, data.shape[1]), dtype=torch.float32)
        return empty_inputs, empty_targets
    indices = np.arange(0, max_start + 1)
    inputs = np.stack([data[i:i+seq_len] for i in indices])
    targets = np.stack([data[i+seq_len:i+seq_len+pred_len] for i in indices])
    return torch.from_numpy(inputs).float(), torch.from_numpy(targets).float()

def predict_windows(model, data, seq_len, pred_len, start_indices):
    """
    Generate forecasts for given start indices.
    data: full series (train+val or test_inputs processed separately)
    start_indices: list/array of forecast start positions in data.
    Returns np.ndarray of shape [len(start_indices), pred_len, channels].
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for start in start_indices:
            inp = data[start - seq_len:start]  # [seq_len, C]
            inp_tensor = torch.from_numpy(inp).unsqueeze(0).float()  # [1, seq_len, C]
            out = model(inp_tensor)  # [1, pred_len, C]
            preds.append(out.squeeze(0).cpu().numpy())
    return np.stack(preds, axis=0)

def train_model(train_data, val_data, config):
    """Train NLinear with early stopping."""
    # Build a config object with attribute access from the dict
    cfg = type('Cfg', (), config)()
    
    # Datasets
    train_x, train_y = create_dataset(train_data, cfg.seq_len, cfg.pred_len)
    val_x, val_y = create_dataset(val_data, cfg.seq_len, cfg.pred_len)
    train_loader = DataLoader(TensorDataset(train_x, train_y),
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(TensorDataset(val_x, val_y),
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            drop_last=False)

    device = torch.device("cpu")
    model = NLinear(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    args = type('Args', (), {
        "learning_rate": cfg.lr,
        "lradj": cfg.lradj
    })()

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    max_epochs = cfg.epochs
    # SMOKE mode reduces epochs and batches
    if os.getenv('SMOKE') == '1':
        max_epochs = 1

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        batches_processed = 0
        for batch_x, batch_y in train_loader:
            batches_processed += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            if os.getenv('SMOKE') == '1':
                break  # only one batch
        denom = batches_processed * cfg.batch_size if os.getenv('SMOKE') == '1' else len(train_loader.dataset)
        train_loss /= denom if denom > 0 else 1.0

        adjust_learning_rate(optimizer, epoch, args)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                val_batches += 1
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                if os.getenv('SMOKE') == '1':
                    break
        denom_val = val_batches * cfg.batch_size if os.getenv('SMOKE') == '1' else len(val_loader.dataset)
        val_loss /= denom_val if denom_val > 0 else 1.0

        print(f'Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val

def main():
    # Seed
    seed = int(os.getenv("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    d = nio.load()
    train = d["train"]  # [8640,7] float32
    val = d["val"]      # [2880,7]
    test_inputs = d["test_inputs"]  # [2785,720,7]

    # Optionally subsample for smoke
    if os.getenv('SMOKE') == '1':
        # Use tiny subsets to speed up
        train = train[:500]
        val = val[:200]
        test_inputs = test_inputs[:10]

    # Train
    model, val_mse = train_model(train, val, CONFIG)
    print(f'Validation MSE: {val_mse:.6f}')

    # Prepare series for validation window forecasts (train+val)
    series = np.concatenate([train, val], axis=0)  # [11520,7]
    val_starts = 8640 + 24 * np.arange(0, 117)  # 117 windows
    val_pred = predict_windows(model, series, CONFIG["seq_len"], CONFIG["pred_len"], val_starts)

    # Test forecasts: use last seq_len of each test_inputs window
    test_pred = np.empty((test_inputs.shape[0],
                          CONFIG["pred_len"],
                          CONFIG["enc_in"]), dtype=np.float32)
    for i in range(test_inputs.shape[0]):
        inp = test_inputs[i, -CONFIG["seq_len"]:, :]  # [seq_len,7]
        inp_t = torch.from_numpy(inp).unsqueeze(0).float()
        with torch.no_grad():
            out = model(inp_t)
        test_pred[i] = out.squeeze(0).cpu().numpy()

    # Write predictions
    nio.write_predictions(test_pred, val_pred)

if __name__ == "__main__":
    main()