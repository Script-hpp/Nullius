"""Referenz-Loesung (von Menschen geschrieben) fuer den ETTh1-Harness: DLinear, L=336, H=96.

Dient nur als Selbsttest der Pipeline, nie als Agenten-Ergebnis. Aufruf im Branch: SEED=0 python train.py
"""
import os

import numpy as np
import torch
import torch.nn as nn

seed = int(os.environ.get("SEED", "0"))
np.random.seed(seed)
torch.manual_seed(seed)
L, H, KERNEL = 336, 96, 25

d = np.load("/data/etth1_public.npz")
train, val, test_inputs = d["train"], d["val"], d["test_inputs"][:, -L:, :]


class DLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool1d(KERNEL, stride=1, padding=0)
        self.seasonal = nn.Linear(L, H)
        self.trend = nn.Linear(L, H)

    def forward(self, x):  # x [B, L, C]
        pad = (KERNEL - 1) // 2
        xp = torch.cat([x[:, :1].repeat(1, pad, 1), x, x[:, -1:].repeat(1, pad, 1)], 1)
        trend = self.pool(xp.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        out = self.seasonal(seasonal.transpose(1, 2)) + self.trend(trend.transpose(1, 2))
        return out.transpose(1, 2)  # [B, H, C]


def windows(arr):
    n = len(arr) - L - H + 1
    idx = np.arange(n)[:, None] + np.arange(L + H)[None, :]
    w = torch.from_numpy(arr[idx])
    return w[:, :L], w[:, L:]


xtr, ytr = windows(train)
xva, yva = windows(np.concatenate([train[-L:], val]))   # Val-Fenster mit Vorgeschichte aus Train
SMOKE = os.environ.get("SMOKE") == "1"   # Formpruefung der Pipeline: 1 Batch, 1 Epoche, 8 Testfenster
if SMOKE:
    xtr, ytr, test_inputs = xtr[:32], ytr[:32], test_inputs[:8]
model = DLinear()
opt = torch.optim.Adam(model.parameters(), lr=0.005)
best, best_state, patience = 1e9, None, 0
LR_HALVE = os.environ.get("LR_HALVE") == "1"   # Lernrate pro Epoche halbieren (Autoformer/DLinear-Repos)
for epoch in range(1 if SMOKE else 10):
    if LR_HALVE:
        for g in opt.param_groups:
            g["lr"] = 0.005 * (0.5 ** epoch)
    model.train()
    perm = torch.randperm(len(xtr))
    for i in range(0, len(perm), 32):
        b = perm[i:i + 32]
        loss = ((model(xtr[b]) - ytr[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        v = ((model(xva) - yva) ** 2).mean().item()
    print(f"epoch {epoch} val_mse {v:.4f}")
    if v < best:
        best, best_state, patience = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
    else:
        patience += 1
        if patience >= 3:
            break
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    pred = model(torch.from_numpy(test_inputs)).numpy()  # [W, H, C]

W = pred.shape[0]
w_i, h_i, c_i = np.meshgrid(np.arange(W), np.arange(H), np.arange(7), indexing="ij")
os.makedirs("/output", exist_ok=True)
with open("/output/predictions.csv", "w") as f:
    f.write("id,prediction\n")
    ids = (f"{a}_{b}_{c}" for a, b, c in zip(w_i.ravel(), h_i.ravel(), c_i.ravel()))
    f.write("\n".join(f"{i},{v:.6f}" for i, v in zip(ids, pred.ravel())))
    f.write("\n")
print("val_mse", best)
