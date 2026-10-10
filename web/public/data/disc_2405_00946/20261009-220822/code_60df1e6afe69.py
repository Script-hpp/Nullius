

# Plan step 1 -> implemented in: class SparseTSFModel
# Plan step 2 -> implemented in: class SparseTSFModel
# Plan step 3 -> implemented in: CONFIG dict and optimizer setup
# Plan step 4 -> implemented in: training loop with early stopping
# Plan step 5 -> implemented in: inference function predict_all

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters from spec
CONFIG = {"lr": 0.02, "epochs": 30, "batch_size": 256, "patience": 5, "seq_len": 720}

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SparseTSFModel(nn.Module):
    def __init__(self, configs):
        super(SparseTSFModel, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        assert self.model_type == 'linear'

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2,
            padding_mode="zeros",
            bias=False
        )

        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)

    def forward(self, x):
        # x: [B, seq_len, enc_in]
        batch_size = x.shape[0]
        seq_mean = torch.mean(x, dim=1, keepdim=True)  # [B,1]
        x_norm = (x - seq_mean).permute(0, 2, 1)       # [B, enc_in, seq_len]

        # 1D convolution per channel
        x_conv = self.conv1d(x_norm.reshape(-1, 1, self.seq_len))\
                     .reshape(-1, self.enc_in, self.seq_len)
        x_conv = x_conv + x_norm                       # residual

        # downsampling: [B*enc_in, seg_num_x, period_len] -> [B*enc_in, period_len, seg_num_x]
        x_reshaped = x_conv.reshape(-1, self.seg_num_x, self.period_len)\
                              .permute(0, 2, 1)

        # sparse forecasting (linear)
        y = self.linear(x_reshaped)                    # [B*enc_in, period_len, seg_num_y]

        # upsampling
        y = y.permute(0, 2, 1)\
              .reshape(batch_size, self.enc_in, self.pred_len)  # [B, enc_in, pred_len]

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean               # [B, pred_len, enc_in]
        return y

def create_window_data(data, seq_len, pred_len):
    n = len(data) - seq_len - pred_len + 1
    inputs = np.zeros((n, seq_len, data.shape[1]), dtype=np.float32)
    targets = np.zeros((n, pred_len, data.shape[1]), dtype=np.float32)
    for i in range(n):
        inputs[i] = data[i:i+seq_len]
        targets[i] = data[i+seq_len:i+seq_len+pred_len]
    return inputs, targets

def prepare_validation_inputs(train_data, val_data, seq_len, pred_len):
    combined = np.concatenate([train_data, val_data], axis=0)  # [11520,7]
    n_val = 117
    inputs = np.zeros((n_val, seq_len, combined.shape[1]), dtype=np.float32)
    targets = np.zeros((n_val, pred_len, combined.shape[1]), dtype=np.float32)
    for w in range(n_val):
        start_idx = 8640 + 24 * w          # forecast start in combined series
        end_idx = start_idx + pred_len
        input_start = start_idx - seq_len
        input_end = start_idx
        inputs[w] = combined[input_start:input_end]
        targets[w] = combined[end_idx-pred_len:end_idx]  # same as combined[start_idx:end_idx]
    return inputs, targets

def predict_all(model, data_np, device):
    model.eval()
    tensor = torch.from_numpy(data_np).to(device)
    with torch.no_grad():
        preds = model(tensor)
    return preds.cpu().numpy()

def main():
    seed = int(os.getenv("SEED", "0"))
    seed_everything(seed)

    smoke = os.getenv("SMOKE", "0") == "1"

    import nullius_io as nio
    d = nio.load()
    train_data = d['train'].astype(np.float32)
    val_data = d['val'].astype(np.float32)
    test_inputs = d['test_inputs'].astype(np.float32)

    seq_len = CONFIG["seq_len"]
    pred_len = 96
    enc_in = 7
    period_len = 24
    d_model = 128  # arbitrary, unused in linear mode

    # Prepare datasets
    train_inputs, train_targets = create_window_data(train_data, seq_len, pred_len)
    val_inputs, val_targets = prepare_validation_inputs(train_data, val_data, seq_len, pred_len)

    if smoke:
        # limit to first 32 training samples, 1 epoch
        train_inputs = train_inputs[:32]
        train_targets = train_targets[:32]

    train_dataset = TensorDataset(torch.from_numpy(train_inputs),
                                  torch.from_numpy(train_targets))
    loader = DataLoader(train_dataset,
                        batch_size=CONFIG["batch_size"],
                        shuffle=True,
                        drop_last=False)

    device = torch.device('cpu')
    configs = type('Configs', (), {
        'seq_len': seq_len,
        'pred_len': pred_len,
        'enc_in': enc_in,
        'period_len': period_len,
        'd_model': d_model,
        'model_type': 'linear'
    })
    model = SparseTSFModel(configs).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=0.0)
    criterion = nn.MSELoss()

    best_val = float('inf')
    patience_cnt = 0
    best_state = None

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_loss = np.mean(epoch_losses)

        model.eval()
        with torch.no_grad():
            val_preds = model(torch.from_numpy(val_inputs).to(device))
            val_loss = criterion(val_preds, torch.from_numpy(val_targets).to(device)).item()
        print(f'Epoch {epoch+1}: train loss {train_loss:.6f}, val loss {val_loss:.6f}')

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_cnt = 0
            best_state = model.state_dict()
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["patience"]:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation MSE (using best model)
    model.eval()
    with torch.no_grad():
        final_val_preds = model(torch.from_numpy(val_inputs).to(device))
        final_val_loss = criterion(final_val_preds,
                                   torch.from_numpy(val_targets).to(device)).item()
    print(f'Final validation MSE: {final_val_loss:.6f}')

    # Produce predictions
    test_pred = predict_all(model, test_inputs, device)
    val_pred = predict_all(model, val_inputs, device)

    nio.write_predictions(test_pred, val_pred)

if __name__ == '__main__':
    main()