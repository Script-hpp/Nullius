"""Referenz-Loesung (von Menschen geschrieben) fuer den ETTh1-Harness: RLinear (Li et al. 2023, arXiv 2305.10721, "Revisiting Long-term Time Series Forecasting"), L=336, H=96.

Modell wie `models/RLinear.py` im offiziellen Repo (github.com/plumprc/RTSF): RevIN mit lernbaren Affine-Parametern (eps 1e-5), Dropout 0,1 auf der normierten
Eingabe, EIN Linear L -> H geteilt ueber alle Kanaele, Rueckrechnung mit RevIN. Training wie `script.md` (ETTh1): Batch 128, Adam lr 0,005, Standardwerte aus `run.py`:
bis 15 Epochen, Geduld 10, MSE, lradj type1 (Lernrate nach Epoche n gleich 0,005 * 0,5^(n-1)), Trainings-Batches mit drop_last. Beste Epoche nach Validierungs-MSE.
Hinweis: Der Test-Loader des Repos nutzt drop_last=True (data_factory.py); dieser Harness wertet alle Testfenster aus, die Referenz also ohne den Fehler.
Nichts auf den Zielwert getunt. Dient nur als Selbsttest der Pipeline, nie als Agenten-Ergebnis.
Aufruf im Branch: SEED=0 python train.py
"""
import os

import numpy as np
import torch
import torch.nn as nn

seed = int(os.environ.get("SEED", "0"))
np.random.seed(seed)
torch.manual_seed(seed)
L, H, BATCH, LR, DROP = 336, 96, 128, 0.005, 0.1

d = np.load("/data/etth1_public.npz")
train, val, test_inputs = d["train"], d["val"], d["test_inputs"][:, -L:, :]
C = train.shape[1]


class RevIN(nn.Module):
    def __init__(self, n, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w, self.b = nn.Parameter(torch.ones(n)), nn.Parameter(torch.zeros(n))

    def norm(self, x):
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        return (x - self.mean) / self.std * self.w + self.b

    def denorm(self, x):
        return (x - self.b) / (self.w + self.eps * self.eps) * self.std + self.mean


class RLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(L, H)
        self.drop = nn.Dropout(DROP)
        self.rev = RevIN(C)

    def forward(self, x):  # x [B, L, C]
        x = self.drop(self.rev.norm(x))
        pred = self.linear(x.transpose(1, 2)).transpose(1, 2)
        return self.rev.denorm(pred)


def windows(arr):
    n = len(arr) - L - H + 1
    idx = np.arange(n)[:, None] + np.arange(L + H)[None, :]
    w = torch.from_numpy(arr[idx])
    return w[:, :L], w[:, L:]


xtr, ytr = windows(train)
xva, yva = windows(np.concatenate([train[-L:], val]))
SMOKE = os.environ.get("SMOKE") == "1"   # Formpruefung: 1 Batch, 1 Epoche, 8 Testfenster
if SMOKE:
    xtr, ytr, test_inputs = xtr[:BATCH], ytr[:BATCH], test_inputs[:8]
model = RLinear()
opt = torch.optim.Adam(model.parameters(), lr=LR)
best, best_state, bad = 1e9, None, 0
for epoch in range(1 if SMOKE else 15):
    model.train()
    perm = torch.randperm(len(xtr))
    for i in range(0, len(perm) - BATCH + 1, BATCH):          # drop_last im Training wie im Repo
        b = perm[i:i + BATCH]
        loss = ((model(xtr[b]) - ytr[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        v = ((model(xva) - yva) ** 2).mean().item()
    print(f"epoch {epoch} val_mse {v:.4f}")
    if v < best:
        best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= 10:
            break
    for g in opt.param_groups:                             # lradj type1 nach Epoche n = epoch + 1
        g["lr"] = LR * (0.5 ** epoch)
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
