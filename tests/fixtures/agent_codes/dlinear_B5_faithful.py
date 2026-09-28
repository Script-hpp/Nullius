import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -------------------- Seed --------------------
seed = int(os.environ.get('SEED', '2021'))
np.random.seed(seed)
torch.manual_seed(seed)
# -------------------------------------------

# -------------------- Model --------------------
class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride,
                                padding=(kernel_size - 1) // 2)

    def forward(self, x):
        # x: [B, T, C]
        x = x.permute(0, 2, 1)          # [B, C, T]
        x = self.avg(x)
        x = x.permute(0, 2, 1)          # [B, T, C]
        return x


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, individual=False, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual
        self.moving_avg = MovingAvg(kernel_size, stride=1)

        if self.individual:
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
        else:
            self.Linear_Trend = nn.Linear(seq_len, pred_len)
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)

        self._initialize_weights()

    def _initialize_weights(self):
        init_val = 1.0 / self.seq_len
        if self.individual:
            for i in range(self.enc_in):
                nn.init.constant_(self.Linear_Trend[i].weight, init_val)
                nn.init.constant_(self.Linear_Seasonal[i].weight, init_val)
                if self.Linear_Trend[i].bias is not None:
                    nn.init.constant_(self.Linear_Trend[i].bias, 0.)
                    nn.init.constant_(self.Linear_Seasonal[i].bias, 0.)
        else:
            nn.init.constant_(self.Linear_Trend.weight, init_val)
            nn.init.constant_(self.Linear_Seasonal.weight, init_val)
            if self.Linear_Trend.bias is not None:
                nn.init.constant_(self.Linear_Trend.bias, 0.)
                nn.init.constant_(self.Linear_Seasonal.bias, 0.)

    def forward(self, x):
        # x: [B, T, C]
        trend = self.moving_avg(x)
        seasonal = x - trend

        if self.individual:
            B, T, C = x.shape
            trend_out = torch.zeros(B, self.pred_len, C, device=x.device)
            seasonal_out = torch.zeros(B, self.pred_len, C, device=x.device)
            for i in range(C):
                t = self.Linear_Trend[i](trend[:, :, i].permute(1, 0)).permute(1, 0)  # [B, pred_len]
                s = self.Linear_Seasonal[i](seasonal[:, :, i].permute(1, 0)).permute(1, 0)
                trend_out[:, :, i] = t
                seasonal_out[:, :, i] = s
        else:
            trend_out = self.Linear_Trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
            seasonal_out = self.Linear_Seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)

        return trend_out + seasonal_out
# ------------------------------------------------

# -------------------- Data utils --------------------
def create_windows(data, seq_len, pred_len):
    """data: [N, C] -> windows: [num, seq_len, C], targets: [num, pred_len, C]"""
    N = data.shape[0]
    num = N - seq_len - pred_len + 1
    windows = np.lib.stride_tricks.sliding_window_view(data, (seq_len, data.shape[1]))[:num, 0, :, :]
    targets = np.lib.stride_tricks.sliding_window_view(data, (pred_len, data.shape[1]))[seq_len:seq_len+num, 0, :, :]
    return windows, targets
# ---------------------------------------------------

def main():
    device = torch.device('cpu')

    # Load data
    data = np.load('/data/etth1_public.npz')
    train_data = data['train'].astype(np.float32)   # [8640,7]
    val_data   = data['val'].astype(np.float32)     # [2880,7]
    test_inputs = data['test_inputs'].astype(np.float32)  # [2785,720,7]

    seq_len = 336
    pred_len = 96

    # Build train/val datasets
    train_x, train_y = create_windows(train_data, seq_len, pred_len)
    val_x,   val_y   = create_windows(val_data,   seq_len, pred_len)

    train_x = torch.from_numpy(train_x)  # [N, seq_len, 7]
    train_y = torch.from_numpy(train_y)
    val_x   = torch.from_numpy(val_x)
    val_y   = torch.from_numpy(val_y)

    train_set = TensorDataset(train_x, train_y)
    val_set   = TensorDataset(val_x,   val_y)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=0)

    # Model, loss, optimizer
    model = DLinear(seq_len=seq_len, pred_len=pred_len, enc_in=7,
                    individual=False, kernel_size=25).to(device)
    criterion = nn.MSELoss()
    lr0 = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=lr0)

    # Training with early stopping and LR schedule type1 (halved each epoch)
    max_epochs = 10
    patience = 3
    best_val = float('inf')
    wait = 0
    best_state = None

    for epoch in range(max_epochs):
        # Adjust LR at start of epoch (epoch 0 -> lr0, epoch1 -> lr0/2, ...)
        lr = lr0 * (0.5 ** epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        print(f'Epoch {epoch+1:02d} | Train Loss: {np.mean(train_losses):.6f} | Val Loss: {val_loss:.6f}')

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping triggered.')
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Print final validation MSE (as required)
    print(f'Final Validation MSE: {best_val:.6f}')

    # -------------------- Prediction --------------------
    L = seq_len  # use last 336 steps
    preds = np.empty((test_inputs.shape[0], pred_len, 7), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(test_inputs.shape[0]):
            inp = test_inputs[i, -L:, :]               # [L,7]
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)  # [1,L,7]
            out = model(inp_t)                         # [1,96,7]
            preds[i] = out.squeeze(0).cpu().numpy()

    # Write predictions.csv
    os.makedirs('/output', exist_ok=True)
    out_path = '/output/predictions.csv'
    with open(out_path, 'w') as f:
        f.write('id,prediction\n')
        for i in range(preds.shape[0]):
            for t in range(pred_len):
                for c in range(7):
                    f.write(f'{i}_{t}_{c},{preds[i, t, c]:.6f}\n')
    # --------------------------------------------------

if __name__ == '__main__':
    main()
