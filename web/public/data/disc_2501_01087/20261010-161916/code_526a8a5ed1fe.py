import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import nullius_io as nio

CONFIG = {"lr": 0.001, "patience": 7, "seq_len": 336}

# Plan step -> implemented in: set_seed
# Plan step -> implemented in: RevIN
# Plan step -> implemented in: GLinear
# Plan step -> implemented in: adjust_learning_rate
# Plan step -> implemented in: EarlyStopping
# Plan step -> implemented in: prepare_train_data
# Plan step -> implemented in: prepare_val_data
# Plan step -> implemented in: predict
# Plan step -> implemented in: main

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

class GLinear(nn.Module):
    def __init__(self, configs):
        super(GLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_num_features = configs.enc_in
        self.Linear = nn.Linear(self.seq_len, self.seq_len)
        self.GeLU = nn.GELU()
        self.Hidden1 = nn.Linear(self.seq_len, self.pred_len)
        self.revin_layer = RevIN(self.in_num_features)

    def forward(self, x):
        x = self.revin_layer(x, 'norm')
        x3 = x.permute(0, 2, 1)
        x3 = self.Linear(x3)
        x3 = self.GeLU(x3)
        x3 = self.Hidden1(x3)
        x3 = x3.permute(0, 2, 1)
        x3 = self.revin_layer(x3, 'denorm')
        return x3

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
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

def prepare_train_data(train, seq_len, pred_len):
    train_len = train.shape[0]
    X = []
    Y = []
    for i in range(seq_len, train_len - pred_len + 1):
        X.append(train[i-seq_len:i])
        Y.append(train[i:i+pred_len])
    X = np.stack(X)
    Y = np.stack(Y)
    return X, Y

def prepare_val_data(train, val, seq_len, pred_len):
    full = np.concatenate([train, val], axis=0)
    val_start = train.shape[0]
    X = []
    Y = []
    for w in range(117):  # 0..116 inclusive
        start = val_start + 24*w
        X.append(full[start-seq_len:start])
        Y.append(val[24*w:24*w+pred_len])
    X = np.stack(X)
    Y = np.stack(Y)
    return X, Y

def predict(model, data, seq_len, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(data.shape[0]):
            inp = data[i, -seq_len:, :]
            inp_t = torch.from_numpy(inp).unsqueeze(0).float().to(device)
            pred = model(inp_t)
            pred = pred.squeeze(0).cpu().numpy()
            preds.append(pred)
    return np.stack(preds)

def main():
    seed = int(os.getenv("SEED", "0"))
    set_seed(seed)

    smoke = os.getenv("SMOKE") == "1"

    d = nio.load()
    train = d['train']  # [8640,7] float32
    val = d['val']      # [2880,7]
    test_inputs = d['test_inputs']  # [2785,720,7]

    seq_len = CONFIG["seq_len"]
    pred_len = 96
    enc_in = train.shape[1]

    # Prepare datasets
    X_train, Y_train = prepare_train_data(train, seq_len, pred_len)
    if smoke:
        X_train = X_train[:32]
        Y_train = Y_train[:32]

    X_val, Y_val = prepare_val_data(train, val, seq_len, pred_len)
    if smoke:
        X_val = X_val[:8]
        Y_val = Y_val[:8]

    train_dataset = TensorDataset(torch.from_numpy(X_train).float(),
                                  torch.from_numpy(Y_train).float())
    val_dataset   = TensorDataset(torch.from_numpy(X_val).float(),
                                  torch.from_numpy(Y_val).float())

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cpu')

    class Args:
        def __init__(self):
            self.learning_rate = CONFIG["lr"]
            self.lradj = 'type1'
            self.use_amp = False
            self.train_epochs = 100
            self.patience = CONFIG["patience"]
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.enc_in = enc_in

    args = Args()

    model = GLinear(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    best_val_loss = np.inf
    for epoch in range(args.train_epochs):
        adjust_learning_rate(optimizer, epoch, args)
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        print(f'Epoch {epoch+1}, Validation MSE: {val_loss:.6f}')
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # Load best model state if saved
    if hasattr(early_stopping, 'best_state'):
        model.load_state_dict(early_stopping.best_state)

    # Produce predictions
    test_pred = predict(model, test_inputs, seq_len, device)
    # For val_pred we need to predict using the same scheme as validation windows
    full_series = np.concatenate([train, val], axis=0)
    val_start = train.shape[0]
    val_pred_list = []
    for w in range(117):
        start = val_start + 24*w
        inp = full_series[start-seq_len:start]
        inp_t = torch.from_numpy(inp).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred = model(inp_t)
            pred = pred.squeeze(0).cpu().numpy()
        val_pred_list.append(pred)
    val_pred = np.stack(val_pred_list)

    if smoke:
        test_pred = test_pred[:8]
        val_pred = val_pred[:8]

    nio.write_predictions(test_pred, val_pred)
    print(f'Best validation MSE: {best_val_loss:.6f}')

if __name__ == '__main__':
    main()
