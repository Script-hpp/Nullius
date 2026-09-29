"""Menschliche Referenzloesung Cora-LCC (Shchur et al. 2018, Tabelle 4/Anhang C): GCN, Hidden 64, lr 0.01, Dropout 0.8,
L2 1e-3 auf der ersten Schicht, Adam, Early Stopping (Gesamt-Validierungsloss inkl. Regularisierung, Geduld 50, max. 10000 Epochen), beste Gewichte zuruecksetzen.
Dient nur als Selbsttest der Pipeline, nie als Agenten-Ergebnis. Aufruf im Branch: SPLIT=0 SEED=0 python train.py
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

seed = int(os.environ.get("SEED", "0"))
np.random.seed(seed)
torch.manual_seed(seed)

d = np.load("/data/cora_lcc_public.npz")
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

H, P = 64, 0.8
W1 = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(X.shape[1], H)))
W2 = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(H, n_cls)))
b1, b2 = torch.nn.Parameter(torch.zeros(H)), torch.nn.Parameter(torch.zeros(n_cls))
opt = torch.optim.Adam([{"params": [W1, b1], "weight_decay": 1e-3}, {"params": [W2, b2], "weight_decay": 0.0}], lr=0.01)


def forward(train):
    h = F.dropout(X, P, train)
    h = F.relu(A_hat @ (h @ W1) + b1)
    h = F.dropout(h, P, train)
    return A_hat @ (h @ W2) + b2


best, best_state, wait = 1e9, None, 0
for epoch in range(10000):
    opt.zero_grad()
    F.cross_entropy(forward(True)[tr], y[tr]).backward()
    opt.step()
    with torch.no_grad():
        # Anhang C: Early Stopping auf dem GESAMT-Validierungsloss (Datenloss + Regularisierungsloss, TF-l2_loss = 0,5 * lambda * ||W||^2)
        v = F.cross_entropy(forward(False)[va], y[va]).item() + 0.5 * 1e-3 * (W1 ** 2).sum().item()
    if v < best:
        best, best_state, wait = v, [p.detach().clone() for p in (W1, b1, W2, b2)], 0
    else:
        wait += 1
        if wait >= 50:
            break
with torch.no_grad():
    for p, s in zip((W1, b1, W2, b2), best_state):
        p.copy_(s)
    pred = forward(False).argmax(1)
print(f"val loss {best:.4f} val acc {(pred[va] == y[va]).float().mean():.4f}")
os.makedirs("/output", exist_ok=True)
with open("/output/predictions.csv", "w") as f:
    f.write("id,prediction\n")
    f.writelines(f"{i},{p}\n" for i, p in zip(te.tolist(), pred[te].tolist()))
