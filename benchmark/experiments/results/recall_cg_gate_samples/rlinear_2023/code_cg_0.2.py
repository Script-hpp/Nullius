# Plan step -> implemented in: function or class name
# Model: RLinear -> class RLinear
# Input / output lengths: seq_len = 336, pred_len = 96 -> variables SEQ_LEN, PRED_LEN
# Number of channels: C = 7 -> argument channel=7
# Individual linear layers: individual = False -> configs.individual
# RevIN: rev = True -> configs.rev
# Dropout: drop = 0.1 -> configs.drop
# Optimizer: Adam lr=0.001 -> optimizer creation
# Batch size: B = 16 -> DataLoader batch_size
# Epochs: train_epochs = 15 -> variable TRAIN_EPOCHS
# Early stopping: patience = 10 -> EarlyStopping class
# Learning‑rate schedule: type1 (halve each epoch) -> adjust_lr function
# Weight initialization: PyTorch default -> nn.Linear default
# Loss: MSE -> nn.MSELoss inside model.forward_loss
# AMP: disabled -> not used
# Random seed: read from SEED -> set_seed function
# Inference: last SEQ_LEN of each test_inputs[i] -> get_last_seq function
# Validation windows: specific 117 windows -> val_predict function
# SMOKE mode: quick run -> conditional block at start

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# -------------------- Helper functions --------------------
def set_seed(seed: int):
    random_seed = seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def adjust_lr(optimizer, epoch, args):
    """type1: lr *= 0.5 each epoch"""
    lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    print(f'Updating learning rate to {lr}')

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
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
        self.best_state = model.state_dict()
        self.val_loss_min = val_loss

# -------------------- Model components --------------------
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

class RLinear(nn.Module):
    def __init__(self, configs):
        super(RLinear, self).__init__()
        self.Linear = nn.ModuleList([
            nn.Linear(configs.seq_len, configs.pred_len) for _ in range(configs.channel)
        ]) if configs.individual else nn.Linear(configs.seq_len, configs.pred_len)
        self.dropout = nn.Dropout(configs.drop)
        self.rev = RevIN(configs.channel) if configs.rev else None
        self.individual = configs.individual

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y):
        # x: [B, L, D], y: [B, pred_len, D]
        if self.rev:
            x = self.rev(x, 'norm')
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros_like(y)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        if self.rev:
            pred = self.rev(pred, 'denorm')
        return pred, self.forward_loss(pred, y)

# -------------------- Configuration class --------------------
class Configs:
    def __init__(self):
        self.seq_len = SEQ_LEN
        self.pred_len = PRED_LEN
        self.channel = CHANNEL
        self.individual = INDIVIDUAL
        self.rev = REVIN
        self.drop = DROPOUT

# -------------------- Main script --------------------
if __name__ == '__main__':
    # ----- Hyperparameters (from plan) -----
    SEQ_LEN = 336          # L
    PRED_LEN = 96
    CHANNEL = 7
    INDIVIDUAL = False     # shared linear layer
    REVIN = True           # use RevIN
    DROPOUT = 0.1
    TRAIN_EPOCHS = 15
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    PATIENCE = 10
    # ----------------------------------------

    # ----- Seed -----
    seed_val = int(os.environ.get('SEED', '0'))
    set_seed(seed_val)

    # ----- Smoke mode detection -----
    SMOKE_MODE = os.environ.get('SMOKE', '0') == '1'

    # ----- Load data -----
    import nullius_io as nio
    data = nio.load()
    train_np = data['train'].astype(np.float32)      # [8640,7]
    val_np   = data['val'].astype(np.float32)       # [2880,7]
    test_np  = data['test_inputs'].astype(np.float32) # [2785,720,7]

    # ----- Prepare datasets (sliding windows) -----
    def create_windows(arr):
        L = arr.shape[0]
        X_list, Y_list = [], []
        for i in range(0, L - SEQ_LEN - PRED_LEN + 1):
            X_list.append(arr[i:i+SEQ_LEN])
            Y_list.append(arr[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN])
        X = np.stack(X_list) if X_list else np.empty((0, SEQ_LEN, CHANNEL), dtype=np.float32)
        Y = np.stack(Y_list) if Y_list else np.empty((0, PRED_LEN, CHANNEL), dtype=np.float32)
        return X, Y

    X_train, Y_train = create_windows(train_np)
    X_val,   Y_val   = create_windows(val_np)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_dataset   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(Y_val))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ----- Model, optimizer, early stopping -----
    configs = Configs()
    model = RLinear(configs)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    # ----- Training loop -----
    model.train()
    for epoch in range(TRAIN_EPOCHS):
        epoch_losses = []
        for bx, by in train_loader:
            bx = bx.to(torch.float32)
            by = by.to(torch.float32)
            optimizer.zero_grad()
            _, loss = model(bx, by)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(torch.float32)
                vy = vy.to(torch.float32)
                _, loss = model(vx, vy)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        model.train()

        print(f'Epoch {epoch+1}/{TRAIN_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')

        # LR schedule
        adjust_lr(optimizer, epoch+1, type('Args', (), {'learning_rate': LEARNING_RATE}))

        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Restore best model
    model.load_state_dict(early_stopping.best_state)
    model.eval()

    # ----- Inference function -----
    def predict_from_history(history):
        """history: np.ndarray [SEQ_LEN, CHANNEL] -> returns [PRED_LEN, CHANNEL]"""
        inp = torch.from_numpy(history).unsqueeze(0).float()  # [1, SEQ_LEN, CH]
        dummy_target = torch.zeros((1, PRED_LEN, CHANNEL))    # dummy, not used
        with torch.no_grad():
            pred, _ = model(inp, dummy_target)
        return pred.squeeze(0).cpu().numpy()  # [PRED_LEN, CH]

    # ----- Validation predictions (specific 117 windows) -----
    if SMOKE_MODE:
        # only first 8 windows
        val_start_indices = [8640 + 24*w for w in range(8)]
    else:
        val_start_indices = [8640 + 24*w for w in range(117)]  # 0..116
    combined = np.concatenate([train_np, val_np], axis=0)  # length 8640+2880=11520
    val_preds = []
    for start in val_start_indices:
        hist = combined[start - SEQ_LEN:start]  # [SEQ_LEN, CH]
        val_preds.append(predict_from_history(hist))
    val_pred = np.stack(val_preds)  # [N, 96, 7]

    # ----- Test predictions -----
    if SMOKE_MODE:
        test_range = range(8)  # first 8 windows
    else:
        test_range = range(test_np.shape[0])  # 0..2784
    test_preds = []
    for i in test_range:
        hist = test_np[i, -SEQ_LEN:, :]  # last SEQ_LEN steps
        test_preds.append(predict_from_history(hist))
    test_pred = np.stack(test_preds)  # [N, 96, 7]

    # ----- Write predictions -----
    nio.write_predictions(test_pred, val_pred)
    # Print final validation MSE for reference
    print(f'Final validation MSE (on sliding windows): {avg_val_loss:.6f}')
