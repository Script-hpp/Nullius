import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------- Configuration --------------------
SEED = int(os.environ.get('SEED', 42))
SMOKE = os.environ.get('SMOKE', '0') == '1'

# Determinism
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Paths
DATA_PATH = '/data/etth1_public.npz'
OUT_PATH = '/output/predictions.csv'

# Hyperparameters from paper/config
LOOKBACK = 672                 # approx 4 weeks * 24h
PRED_LEN = 96                  # forecast horizon
CHANNELS = 7
HUBER_DELTA = 2.0
LR_MAX = 3.2e-3
LR_MIN = 1.2e-4
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.95)
EPS = 1e-10
WARMUP_PORTION = 0.1
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-6
BATCH_SIZE = 256 if not SMOKE else 32
MAX_EPOCHS = 20 if not SMOKE else 1

# -------------------- Data Loading --------------------
data = np.load(DATA_PATH)
train = data['train'].astype(np.float32)      # [8640, 7]
val   = data['val'].astype(np.float32)       # [2880, 7]
test_inputs = data['test_inputs'].astype(np.float32)  # [2785, 720, 7]

# If smoke mode, reduce data
if SMOKE:
    train = train[:32 * (LOOKBACK + PRED_LEN)]  # ensure enough for windows
    val   = val[:32 * (LOOKBACK + PRED_LEN)]
    test_inputs = test_inputs[:8]               # only first 8 windows

# -------------------- Dataset Definition --------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, data, lookback, pred_len):
        self.data = torch.from_numpy(data)  # [T, C]
        self.lookback = lookback
        self.pred_len = pred_len
        self.C = data.shape[1]
        self.length = len(data) - lookback - pred_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.lookback]          # [L, C]
        y = self.data[idx + self.lookback:
                      idx + self.lookback + self.pred_len]  # [P, C]
        return x.view(-1), y.view(-1)  # flatten

train_set = Seq2SeqDataset(train, LOOKBACK, PRED_LEN)
val_set   = Seq2SeqDataset(val,   LOOKBACK, PRED_LEN)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=False)

# -------------------- Model --------------------
class SimpleForecaster(nn.Module):
    def __init__(self, lookback, pred_len, channels):
        super().__init__()
        self.linear = nn.Linear(lookback * channels,
                                pred_len * channels,
                                bias=False)

    def forward(self, x):
        return self.linear(x)

model = SimpleForecaster(LOOKBACK, PRED_LEN, CHANNELS)

# -------------------- Loss, Optimizer, Scheduler --------------------
criterion = nn.HuberLoss(delta=HUBER_DELTA, reduction='mean')
optimizer = optim.AdamW(model.parameters(),
                        lr=LR_MAX,
                        betas=BETAS,
                        eps=EPS,
                        weight_decay=WEIGHT_DECAY,
                        fused='fused' in optim.AdamW.__init__.__code__.co_varnames)

# Compute total steps for scheduler
steps_per_epoch = len(train_loader)
max_steps = steps_per_epoch * MAX_EPOCHS
warmup_steps = int(max_steps * WARMUP_PORTION)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    # cosine decay to LR_MIN/LR_MAX
    progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return max(LR_MIN / LR_MAX,
               0.5 * (1.0 + np.cos(np.pi * progress)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -------------------- Training Loop --------------------
best_val = float('inf')
epochs_no_improve = 0

model.train()
for epoch in range(MAX_EPOCHS):
    epoch_loss = 0.0
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_loader.dataset)
    print(f'Epoch {epoch+1}/{MAX_EPOCHS} - train loss: {epoch_loss:.6f} - val loss: {val_loss:.6f}',
          flush=True)

    # Early stopping check
    if val_loss + EARLY_STOP_MIN_DELTA < best_val:
        best_val = val_loss
        epochs_no_improve = 0
        best_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f'Early stopping triggered after {epoch+1} epochs.', flush=True)
            break

# Load best model
model.load_state_dict(best_state)

# -------------------- Final Validation MSE (non-Huber) --------------------
model.eval()
mse_sum = 0.0
count = 0
with torch.no_grad():
    for xb, yb in val_loader:
        preds = model(xb)
        mse = ((preds - yb) ** 2).mean()
        mse_sum += mse.item() * xb.size(0)
        count += xb.size(0)
final_mse = mse_sum / count
print(f'Validation MSE: {final_mse:.6f}', flush=True)

# -------------------- Prediction on Test Windows --------------------
W = test_inputs.shape[0]
history_len = test_inputs.shape[1]  # 720
use_len = min(LOOKBACK, history_len)  # we can only use what we have
preds_all = np.empty((W, PRED_LEN, CHANNELS), dtype=np.float32)

model.eval()
with torch.no_grad():
    for i in range(W):
        # take last `use_len` steps
        hist = test_inputs[i, -use_len:, :]  # [use_len, C]
        if use_len < LOOKBACK:
            # pad with zeros at the beginning (could also repeat, but zero is safe)
            pad = np.zeros((LOOKBACK - use_len, CHANNELS), dtype=np.float32)
            hist = np.concatenate([pad, hist], axis=0)
        x = torch.from_numpy(hist).view(-1)  # [L*C]
        out = model(x.unsqueeze(0))          # [1, P*C]
        out = out.squeeze(0).cpu().numpy().reshape(PRED_LEN, CHANNELS)
        preds_all[i] = out

# -------------------- Write predictions.csv --------------------
# Efficient line generation
lines = ['id,prediction']
for w in range(W):
    base = f'{w}_'
    for h in range(PRED_LEN):
        row = preds_all[w, h]
        for c in range(CHANNELS):
            lines.append(f'{base}{h}_{c},{row[c]:.6f}')
with open(OUT_PATH, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f'Written {W * PRED_LEN * CHANNELS} predictions to {OUT_PATH}', flush=True)
