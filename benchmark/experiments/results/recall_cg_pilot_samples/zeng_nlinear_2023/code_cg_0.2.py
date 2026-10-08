# Plan step -> implemented in: set_seeds
# Plan step -> implemented in: load_data
# Plan step -> implemented in: get_dataloaders
# Plan step -> implemented in: NLinear class
# Plan step -> implemented in: configure_optimizers
# Plan step -> implemented in: train_model
# Plan step -> implemented in: evaluate
# Plan step -> implemented in: predict_test
# Plan step -> implemented in: predict_val
# Plan step -> implemented in: write_predictions

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Hyperparameters from spec
CONFIG = {"lr": 0.005, "batch_size": 32, "patience": 7, "seq_len": 336}
# Additional fixed hyperparameters (chosen as per plan)
CONFIG["weight_decay"] = 0.0
CONFIG["epochs"] = 20
CONFIG["pred_len"] = 96
CONFIG["enc_in"] = 7
CONFIG["lradj"] = "type1"


def set_seeds():
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    data = np.load("/data/etth1_public.npz")
    train = data["train"].astype(np.float32)
    val = data["val"].astype(np.float32)
    test_inputs = data["test_inputs"].astype(np.float32)
    return train, val, test_inputs


class ForecastDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.length = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)


def get_dataloaders(train, val):
    train_set = ForecastDataset(train, CONFIG["seq_len"], CONFIG["pred_len"])
    val_set = ForecastDataset(val, CONFIG["seq_len"], CONFIG["pred_len"])
    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader


class NLinear(nn.Module):
    def __init__(self, configs):
        super(NLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = getattr(configs, "individual", False)
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
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)],
                                 dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x


def adjust_learning_rate(optimizer, epoch, args):
    # args.lradj and args.learning_rate expected
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
        print(f'Updating learning rate to {lr}')


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0


def configure_optimizers(model):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CONFIG["lr"],
                                 weight_decay=CONFIG["weight_decay"])
    # args object for lr scheduler
    class Args:
        def __init__(self):
            self.lradj = CONFIG["lradj"]
            self.learning_rate = CONFIG["lr"]
    args = Args()
    return optimizer, args


def train_model(model, train_loader, val_loader, optimizer, args):
    early_stopping = EarlyStopping(patience=CONFIG["patience"])
    criterion = nn.MSELoss()
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        adjust_learning_rate(optimizer, epoch, args)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        print(f'Epoch {epoch+1}: Train Loss {np.mean(train_losses):.6f}, Val Loss {val_loss:.6f}')
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    # restore best model
    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)
    return model


def evaluate(model, loader):
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            losses.append(loss.item())
    return np.mean(losses)


def predict_test(model, test_inputs):
    model.eval()
    seq_len = CONFIG["seq_len"]
    pred_len = CONFIG["pred_len"]
    enc_in = CONFIG["enc_in"]
    num_windows = test_inputs.shape[0]
    preds = np.zeros((num_windows, pred_len, enc_in), dtype=np.float32)
    with torch.no_grad():
        for i in range(num_windows):
            x = test_inputs[i, -seq_len:, :]  # [seq_len, enc_in]
            xt = torch.from_numpy(x).unsqueeze(0)  # [1, seq_len, enc_in]
            out = model(xt)  # [1, pred_len, enc_in]
            preds[i] = out.squeeze(0).cpu().numpy()
    return preds


def predict_val(model, train, val):
    model.eval()
    seq_len = CONFIG["seq_len"]
    pred_len = CONFIG["pred_len"]
    enc_in = CONFIG["enc_in"]
    combined = np.concatenate([train, val], axis=0)  # [11520, 7]
    num_windows = 117  # as per spec
    preds = np.zeros((num_windows, pred_len, enc_in), dtype=np.float32)
    with torch.no_grad():
        for w in range(num_windows):
            start = 8640 + 24 * w
            x = combined[start - seq_len: start, :]  # [seq_len, enc_in]
            xt = torch.from_numpy(x).unsqueeze(0)
            out = model(xt)
            preds[w] = out.squeeze(0).cpu().numpy()
    return preds


def write_predictions(preds, path, start_window=0):
    # preds shape: (num_windows, pred_len, enc_in)
    num_windows, pred_len, enc_in = preds.shape
    with open(path, "w") as f:
        f.write("id,prediction\n")
        for w in range(num_windows):
            win_id = w + start_window
            for t in range(pred_len):
                for c in range(enc_in):
                    fid = f"{win_id}_{t}_{c}"
                    val = preds[w, t, c]
                    f.write(f"{fid},{val:.6f}\n")


def main():
    set_seeds()
    train, val, test_inputs = load_data()
    train_loader, val_loader = get_dataloaders(train, val)

    # quick smoke handling
    smoke = os.environ.get("SMOKE") == "1"
    if smoke:
        # limit to one batch for one epoch
        # we will just take first batch from train_loader and val_loader
        train_data_iter = iter(train_loader)
        val_data_iter = iter(val_loader)
        try:
            batch_x, batch_y = next(train_data_iter)
        except StopIteration:
            batch_x, batch_y = None, None
        if batch_x is None:
            raise RuntimeError("No training data for smoke test")
        # create tiny loaders that yield only this batch
        class TinyLoader:
            def __init__(self, x, y):
                self.x = x
                self.y = y
            def __iter__(self):
                yield self.x, self.y
        train_loader = TinyLoader(batch_x, batch_y)
        val_loader = TinyLoader(batch_x, batch_y)  # reuse for simplicity
        CONFIG["epochs"] = 1

    # Model config object
    class ModelConfig:
        def __init__(self):
            self.seq_len = CONFIG["seq_len"]
            self.pred_len = CONFIG["pred_len"]
            self.enc_in = CONFIG["enc_in"]
            self.individual = False  # shared linear layer as per plan
    configs = ModelConfig()

    model = NLinear(configs)
    optimizer, args = configure_optimizers(model)

    model = train_model(model, train_loader, val_loader, optimizer, args)

    # final validation MSE (on the validation loader used for training)
    val_mse = evaluate(model, val_loader)
    print(f"Final Validation MSE: {val_mse:.6f}")

    # test predictions
    test_preds = predict_test(model, test_inputs)
    if smoke:
        test_preds = test_preds[:8]  # only first 8 windows
    write_predictions(test_preds, "/output/predictions.csv",
                      start_window=0)

    # validation predictions for output file
    val_preds = predict_val(model, train, val)
    if smoke:
        val_preds = val_preds[:8]
    write_predictions(val_preds, "/output/val_predictions.csv",
                      start_window=0)


if __name__ == "__main__":
    main()
