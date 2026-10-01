"""Referenz-Loesung (von Menschen geschrieben) fuer den ETTh1-Harness: SparseTSF (Lin et al. 2024, arXiv 2405.00946), Modelltyp linear, L=720, H=96.

Modell wie `models/SparseTSF.py` im offiziellen Repo (github.com/lss-1138/SparseTSF): Mittelwert der Eingabe abziehen, 1D-Faltung (Kernel 1 + 2*(w//2), zero padding,
ohne Bias) mit Residuum, Zerlegung in Perioden der Laenge w=24 (Eingabe 720/24 = 30 Segmente), ein Linear ohne Bias 30 -> 4 ueber die Segmente, Mittelwert addieren.
Training wie `scripts/SparseTSF/linear/etth1.sh` und `run_longExp.py`: Adam lr 0,02, Batch 256, bis 30 Epochen, Geduld 5, MSE, lradj type3 (Lernrate konstant fuer
die ersten drei Epochen, danach pro Epoche mal 0,8, `utils/tools.py`), Trainings-Batches mit drop_last (nur Training, nicht Test). Beste Epoche nach Validierungs-MSE.
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
L, H, W, BATCH, LR = 720, 96, 24, 256, 0.02

d = np.load("/data/etth1_public.npz")
train, val, test_inputs = d["train"], d["val"], d["test_inputs"][:, -L:, :]
C = train.shape[1]


class SparseTSF(nn.Module):
    def __init__(self):
        super().__init__()
        self.nx, self.ny = L // W, H // W
        self.conv = nn.Conv1d(1, 1, kernel_size=1 + 2 * (W // 2), stride=1, padding=W // 2, padding_mode="zeros", bias=False)
        self.linear = nn.Linear(self.nx, self.ny, bias=False)

    def forward(self, x):  # x [B, L, C]
        b = x.shape[0]
        mean = x.mean(dim=1, keepdim=True)
        x = (x - mean).permute(0, 2, 1)                                   # [B, C, L]
        x = self.conv(x.reshape(-1, 1, L)).reshape(-1, C, L) + x
        x = x.reshape(-1, self.nx, W).permute(0, 2, 1)                    # [B*C, W, nx]
        y = self.linear(x)                                                # [B*C, W, ny]
        y = y.permute(0, 2, 1).reshape(b, C, H)
        return y.permute(0, 2, 1) + mean                                  # [B, H, C]


def windows(arr):
    n = len(arr) - L - H + 1
    idx = np.arange(n)[:, None] + np.arange(L + H)[None, :]
    w = torch.from_numpy(arr[idx])
    return w[:, :L], w[:, L:]


xtr, ytr = windows(train)
xva, yva = windows(np.concatenate([train[-L:], val]))   # Val-Fenster mit Vorgeschichte aus Train
SMOKE = os.environ.get("SMOKE") == "1"                  # Formpruefung: 1 Batch, 1 Epoche, 8 Testfenster
if SMOKE:
    xtr, ytr, test_inputs = xtr[:32], ytr[:32], test_inputs[:8]
model = SparseTSF()
opt = torch.optim.Adam(model.parameters(), lr=LR)
best, best_state, bad = 1e9, None, 0
for epoch in range(1 if SMOKE else 30):
    model.train()
    perm = torch.randperm(len(xtr))
    for i in range(0, len(perm) - BATCH + 1 if not SMOKE else 1, BATCH):          # drop_last im Training wie im Repo
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
        if bad >= 5:
            break
    n = epoch + 1                                        # lradj type3 nach Epoche n (1-basiert): lr * 0.8^(n-3) ab n = 3
    for g in opt.param_groups:
        g["lr"] = LR if n < 3 else LR * (0.8 ** (n - 3))
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    pred = model(torch.from_numpy(test_inputs)).numpy()  # [W, H, C]

Wn = pred.shape[0]
w_i, h_i, c_i = np.meshgrid(np.arange(Wn), np.arange(H), np.arange(7), indexing="ij")
os.makedirs("/output", exist_ok=True)
with open("/output/predictions.csv", "w") as f:
    f.write("id,prediction\n")
    ids = (f"{a}_{b}_{c}" for a, b, c in zip(w_i.ravel(), h_i.ravel(), c_i.ravel()))
    f.write("\n".join(f"{i},{v:.6f}" for i, v in zip(ids, pred.ravel())))
    f.write("\n")
print("val_mse", best)
