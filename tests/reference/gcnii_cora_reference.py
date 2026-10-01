"""Referenz-Loesung (von Menschen geschrieben) fuer den Cora-Harness (Planetoid-Split 140/500/1000): GCNII (Chen et al. 2020, arXiv 2007.02133), 64 Schichten.

Modell wie `model.py` im offiziellen Repo (github.com/chennnM/GCNII): Dropout, Linear 1433 -> 64 + ReLU (= H0), 64 Graphfaltungen
H(l+1) = ReLU(((1 - beta_l) * Z + beta_l * Z @ W_l)) mit Z = (1 - alpha) * A_hat H(l) + alpha * H0, beta_l = log(lambda / l + 1), Dropout vor jeder Schicht,
Linear 64 -> 7, Log-Softmax. A_hat = D^-1/2 (A + I) D^-1/2, Merkmale zeilennormiert. Hyperparameter Cora aus Tabelle 6 (Anhang) des Papers und `semi.sh`:
alpha 0,1, lambda 0,5, Hidden 64, Dropout 0,6, lr 0,01, Weight Decay 0,01 auf den Faltungsgewichten und 5e-4 auf den Dense-Schichten, Adam, bis 1500 Epochen,
Early Stopping mit Geduld 100 auf dem Validierungsloss, Test mit den Gewichten der Epoche mit dem kleinsten Validierungsloss. Initialisierung der Faltungsgewichte
uniform(-1/sqrt(64), 1/sqrt(64)) wie im Repo. Nichts auf den Zielwert getunt. Dient nur als Selbsttest der Pipeline, nie als Agenten-Ergebnis.
Aufruf im Branch: SEED=0 python train.py
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
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
A = ((A + A.T) > 0).float() + torch.eye(N)
dinv = A.sum(1).pow(-0.5)
A_hat = (dinv[:, None] * A * dinv[None, :]).to_sparse()

LAYERS, HID, DROP, ALPHA, LAMDA, LR = 64, 64, 0.6, 0.1, 0.5, 0.01
WD_CONV, WD_DENSE, EPOCHS, PATIENCE = 0.01, 5e-4, 1500, 100


class GCNII(nn.Module):
    def __init__(self):
        super().__init__()
        stdv = 1.0 / math.sqrt(HID)
        self.W = nn.ParameterList([nn.Parameter(torch.empty(HID, HID).uniform_(-stdv, stdv)) for _ in range(LAYERS)])
        self.fc_in, self.fc_out = nn.Linear(X.shape[1], HID), nn.Linear(HID, n_cls)

    def forward(self, x):
        x = F.dropout(x, DROP, self.training)
        h = h0 = F.relu(self.fc_in(x))
        for i, w in enumerate(self.W, start=1):
            h = F.dropout(h, DROP, self.training)
            beta = math.log(LAMDA / i + 1)
            z = (1 - ALPHA) * torch.sparse.mm(A_hat, h) + ALPHA * h0
            h = F.relu(beta * (z @ w) + (1 - beta) * z)
        h = F.dropout(h, DROP, self.training)
        return F.log_softmax(self.fc_out(h), dim=1)


model = GCNII()
opt = torch.optim.Adam([
    {"params": list(model.W.parameters()), "weight_decay": WD_CONV},
    {"params": list(model.fc_in.parameters()) + list(model.fc_out.parameters()), "weight_decay": WD_DENSE},
], lr=LR)
SMOKE = os.environ.get("SMOKE") == "1"   # Formpruefung: 1 Epoche, 20 Testknoten
best, best_state, bad, epochs = 1e9, None, 0, 1 if SMOKE else EPOCHS
for epoch in range(epochs):
    model.train()
    opt.zero_grad()
    F.nll_loss(model(X)[tr], y[tr]).backward()
    opt.step()
    model.eval()
    with torch.no_grad():
        v = F.nll_loss(model(X)[va], y[va]).item()
    if v < best:
        best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
    else:
        bad += 1
        if bad == PATIENCE:
            break
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    pred = model(X).argmax(1)
print(f"epochs {epoch + 1} best val loss {best:.4f} val acc {(pred[va] == y[va]).float().mean():.4f}")
ids = te.tolist()[:20] if SMOKE else te.tolist()
os.makedirs("/output", exist_ok=True)
with open("/output/predictions.csv", "w") as f:
    f.write("id,prediction\n")
    f.writelines(f"{i},{pred[i].item()}\n" for i in ids)
