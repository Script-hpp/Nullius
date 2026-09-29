import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

seed = int(os.environ.get("SEED", "0"))
np.random.seed(seed)
torch.manual_seed(seed)

SEQ_LEN, PRED_LEN = 336, 96

def moving_average(x, k=25):   # x: [B, L, C], Randwerte wiederholt, Mittel ueber k Schritte per cumsum
    pad = (k - 1) // 2
    xp = torch.cat([x[:, :1].repeat(1, pad, 1), x, x[:, -1:].repeat(1, pad, 1)], dim=1)
    c = torch.cumsum(F.pad(xp, (0, 0, 1, 0)), dim=1)
    return (c[:, k:] - c[:, :-k]) / k


class Model(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.trend_head = nn.Linear(seq_len, pred_len)
        self.season_head = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        trend = moving_average(x, 25)
        season = x - trend
        return (self.trend_head(trend.transpose(1, 2)) + self.season_head(season.transpose(1, 2))).transpose(1, 2)


def windows(x, seq_len, pred_len):
    n = x.shape[0] - seq_len - pred_len + 1
    idx = np.arange(n)[:, None] + np.arange(seq_len + pred_len)[None, :]
    w = x[idx]
    return torch.from_numpy(w[:, :seq_len]), torch.from_numpy(w[:, seq_len:])


d = np.load("/data/etth1_public.npz")
train_x, train_y = windows(d["train"], SEQ_LEN, PRED_LEN)
val_x, val_y = windows(d["val"], SEQ_LEN, PRED_LEN)
test_in = torch.from_numpy(d["test_inputs"][:, -SEQ_LEN:, :])

model = Model(SEQ_LEN, PRED_LEN)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best, best_state = float("inf"), None
for epoch in range(10):
    model.train()
    perm = torch.randperm(train_x.shape[0])
    for i in range(0, len(perm), 32):
        b = perm[i:i + 32]
        opt.zero_grad()
        loss = F.mse_loss(model(train_x[b]), train_y[b])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        v = F.mse_loss(model(val_x), val_y).item()
    if v < best:
        best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
model.load_state_dict(best_state)
print(f"val mse {best:.4f}")

model.eval()
with torch.no_grad():
    pred = torch.cat([model(test_in[i:i + 256]) for i in range(0, len(test_in), 256)]).numpy()
n_w, n_s, n_c = pred.shape
ids = [f"{w}_{s}_{c}" for w in range(n_w) for s in range(n_s) for c in range(n_c)]
os.makedirs("/output", exist_ok=True)
with open("/output/predictions.csv", "w") as f:
    f.write("id,prediction\n")
    f.writelines(f"{i},{v}\n" for i, v in zip(ids, pred.reshape(-1)))
