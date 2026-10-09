# Plan step -> implemented in: function or class name
# Import libraries -> imports
# Set random seed -> main function
# Load data -> nio.load()
# Define CONFIG dict -> CONFIG
# Define moving_avg class -> moving_avg
# Define series_decomp class -> series_decomp
# Define Model class -> Model
# Define adjust_learning_rate function -> adjust_learning_rate
# Define ForecastDataset class -> ForecastDataset
# Training loop with early stopping -> training loop in main
# Inference for test windows -> test prediction loop
# Inference for validation windows -> val prediction loop
# Write predictions -> nio.write_predictions
# Print validation MSE -> print

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import nullius_io as nio

CONFIG = {"lr": 0.005, "epochs": 10, "batch_size": 32, "dropout": 0.05, "patience": 7, "seq_len": 336}


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class ForecastDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.length = len(data) - seq_len - pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def __len__(self):
        return self.length


def main():
    seed = int(os.getenv("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    smoke = os.getenv("SMOKE") == "1"

    d = nio.load()
    train = d['train'].astype(np.float32)          # [8640, 7]
    val = d['val'].astype(np.float32)              # [2880, 7]
    test_inputs = d['test_inputs'].astype(np.float32)  # [2785, 720, 7]

    seq_len = CONFIG["seq_len"]
    pred_len = 96
    enc_in = 7

    epochs = CONFIG["epochs"]
    if smoke:
        epochs = 1  # reduce to one epoch in smoke mode

    train_dataset = ForecastDataset(train, seq_len, pred_len)
    val_dataset = ForecastDataset(val, seq_len, pred_len)

    batch_size = CONFIG["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class Configs:
        pass
    configs = Configs()
    configs.seq_len = seq_len
    configs.pred_len = pred_len
    configs.enc_in = enc_in
    configs.individual = False
    configs.dropout = CONFIG["dropout"]

    model = Model(configs)
    device = torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=0)
    criterion = nn.MSELoss()
    args = type('Args', (), {'learning_rate': CONFIG["lr"], 'lradj': 'type1'})()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = CONFIG["patience"]
    best_state = None

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch + 1, args)
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses) if val_losses else 0.0
        train_loss = np.mean(train_losses) if train_losses else 0.0
        # Uncomment to see progress:
        # print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test predictions
    n_test = test_inputs.shape[0]
    test_pred = np.empty((n_test, pred_len, enc_in), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(n_test):
            inp = test_inputs[i, -seq_len:, :]               # [seq_len, 7]
            inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(device)  # [1, seq_len, 7]
            out = model(inp_tensor)                          # [1, pred_len, 7]
            test_pred[i] = out.squeeze(0).cpu().numpy()

    # Validation predictions (as defined in the statement)
    combined = np.concatenate([train, val], axis=0)          # [11520, 7]
    n_val_windows = 117
    val_pred = np.empty((n_val_windows, pred_len, enc_in), dtype=np.float32)
    with torch.no_grad():
        for w in range(n_val_windows):
            start = 8640 + 24 * w
            inp = combined[start - seq_len: start, :]        # [seq_len, 7]
            inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(device)
            out = model(inp_tensor)
            val_pred[w] = out.squeeze(0).cpu().numpy()

    nio.write_predictions(test_pred, val_pred)
    print(f'{best_val_loss:.6f}')


if __name__ == "__main__":
    main()
