import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

seed = int(os.environ.get("SEED", "0"))
np.random.seed(seed)
torch.manual_seed(seed)

SEQ_LEN, PRED_LEN = 336, 96

class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [B, L, C]
        pad = (self.kernel_size - 1) // 2
        x = torch.cat([x[:, :1, :].repeat(1, pad, 1), x, x[:, -1:, :].repeat(1, pad, 1)], dim=1)
        return F.avg_pool1d(x.permute(0, 2, 1), kernel_size=self.kernel_size, stride=1).permute(0, 2, 1)


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, kernel_size=13):
        super().__init__()
        self.decomp = MovingAvg(kernel_size)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        trend = self.decomp(x)
        seasonal = x - trend
        out = self.linear_trend(trend.permute(0, 2, 1)) + self.linear_seasonal(seasonal.permute(0, 2, 1))
        return out.permute(0, 2, 1)


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
