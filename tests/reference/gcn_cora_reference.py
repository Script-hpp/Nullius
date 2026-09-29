"""Menschliche Referenzloesung Cora, Standard-Split (140/500/1000) (Kipf & Welling 2017): GCN, 2 Schichten, Hidden 16, Dropout 0,5, lr 0,01,
L2 5e-4 auf der ersten Schicht, Adam, max. 200 Epochen, Early Stopping (Validierungsloss groesser als Mittel der letzten 10).
Selbsttest der Pipeline, nie Agenten-Ergebnis. Aufruf im Branch: SEED=0 python train.py
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

seed = int(os.environ.get("SEED", "0"))
np.random.seed(seed)
torch.manual_seed(seed)

d = np.load("/data/cora_public.npz")
X = torch.from_numpy(d["features"]).float()
X = X / X.sum(1, keepdim=True).clamp(min=1)
edges = torch.from_numpy(d["edges"]).long()
y = torch.from_numpy(d["labels"]).long()
tr, va, te = (torch.from_numpy(d[k]).long() for k in ("train_idx", "val_idx", "test_idx"))
N, n_cls = X.shape[0], 7

A = torch.zeros(N, N)
A[edges[:, 0], edges[:, 1]] = 1.0
A = A + torch.eye(N)
dinv = A.sum(1).pow(-0.5)
A_hat = dinv[:, None] * A * dinv[None, :]

H, P = 16, 0.5
W1 = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(X.shape[1], H)))
W2 = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(H, n_cls)))
opt = torch.optim.Adam([{"params": [W1], "weight_decay": 5e-4}, {"params": [W2], "weight_decay": 0.0}], lr=0.01)


def forward(train):
    h = F.dropout(X, P, train)
    h = F.relu(A_hat @ (h @ W1))
    h = F.dropout(h, P, train)
    return A_hat @ (h @ W2)


hist = []
for epoch in range(200):
    opt.zero_grad()
    F.cross_entropy(forward(True)[tr], y[tr]).backward()
    opt.step()
    with torch.no_grad():
        v = F.cross_entropy(forward(False)[va], y[va]).item()
    if epoch > 10 and v > np.mean(hist[-10:]):
        break
    hist.append(v)
with torch.no_grad():
    pred = forward(False).argmax(1)
print(f"epochs {epoch + 1} val acc {(pred[va] == y[va]).float().mean():.4f}")
os.makedirs("/output", exist_ok=True)
with open("/output/predictions.csv", "w") as f:
    f.write("id,prediction\n")
    f.writelines(f"{i},{p}\n" for i, p in zip(te.tolist(), pred[te].tolist()))
