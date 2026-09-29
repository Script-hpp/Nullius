"""Menschliche Referenzloesung Cora, Standard-Split (140/500/1000) (Wu et al. 2019, SGC): S~ = D~^-1/2 (A+I) D~^-1/2, K = 2, logistische Regression auf
S~^K X, Adam lr 0,2, 100 Epochen, Weight Decay auf dem Validierungsset aus einem kleinen Log-Gitter gewaehlt (Paper: hyperopt, 60 Iterationen).
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
S = dinv[:, None] * A * dinv[None, :]
Z = S @ (S @ X)                                    # K = 2


def fit(wd):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(Z.shape[1], n_cls)
    opt = torch.optim.Adam(lin.parameters(), lr=0.2, weight_decay=wd)
    for _ in range(100):
        opt.zero_grad()
        F.cross_entropy(lin(Z[tr]), y[tr]).backward()
        opt.step()
    with torch.no_grad():
        return lin, (lin(Z[va]).argmax(1) == y[va]).float().mean().item()


best = max(((fit(wd), wd) for wd in (1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3)), key=lambda t: t[0][1])
(lin, val_acc), wd = best
with torch.no_grad():
    pred = lin(Z).argmax(1)
print(f"weight decay {wd} val acc {val_acc:.4f}")
os.makedirs("/output", exist_ok=True)
with open("/output/predictions.csv", "w") as f:
    f.write("id,prediction\n")
    f.writelines(f"{i},{p}\n" for i, p in zip(te.tolist(), pred[te].tolist()))
