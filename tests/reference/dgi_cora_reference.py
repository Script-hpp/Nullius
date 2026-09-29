"""Menschliche Referenzloesung Cora, Standard-Split (140/500/1000): DGI (Velickovic et al. 2019), Ein-Schicht-GCN-Encoder mit 512 Einheiten
und PReLU, Korruption durch zeilenweises Mischen der Merkmale, Readout Mittel + Sigmoid, bilinearer Diskriminator, Adam lr 0,001,
Early Stopping auf dem Trainingsloss (Geduld 20), danach Merkmale ueber die Trainingsknoten standardisieren und logistische Regression
auf den 140 Trainingsknoten. Selbsttest der Pipeline, nie Agenten-Ergebnis. Aufruf im Branch: SEED=0 python train.py
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

seed = int(os.environ.get("SEED", "0"))
np.random.seed(seed)
torch.manual_seed(seed)
t0 = time.time()

d = np.load("/data/cora_public.npz")
X = torch.from_numpy(d["features"]).float()
X = X / X.sum(1, keepdim=True).clamp(min=1)
edges = torch.from_numpy(d["edges"]).long()
y = torch.from_numpy(d["labels"]).long()
tr, te = torch.from_numpy(d["train_idx"]).long(), torch.from_numpy(d["test_idx"]).long()
N, n_cls, H = X.shape[0], 7, 512

loops = torch.arange(N)
rows, cols = torch.cat([edges[:, 0], loops]), torch.cat([edges[:, 1], loops])   # A + I
deg = torch.bincount(rows, minlength=N).float()
vals = deg[rows].pow(-0.5) * deg[cols].pow(-0.5)
A_hat = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (N, N)).coalesce()


class DGI(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(X.shape[1], H, bias=False)
        self.act = torch.nn.PReLU()
        self.W = torch.nn.Parameter(torch.empty(H, H))
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.W)

    def encode(self, x):
        return self.act(torch.sparse.mm(A_hat, self.lin(x)))

    def loss(self):
        pos = self.encode(X)
        neg = self.encode(X[torch.randperm(N)])          # Korruption: Zeilen mischen
        s = torch.sigmoid(pos.mean(0))
        logits = torch.cat([pos @ (self.W @ s), neg @ (self.W @ s)])
        labels = torch.cat([torch.ones(N), torch.zeros(N)])
        return F.binary_cross_entropy_with_logits(logits, labels)


model = DGI()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
best, best_state, wait, epochs = 1e9, None, 0, 0
for epochs in range(1, 2001):
    opt.zero_grad()
    loss = model.loss()
    loss.backward()
    opt.step()
    if loss.item() < best:
        best, best_state, wait = loss.item(), {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        wait += 1
        if wait >= 20:
            break
model.load_state_dict(best_state)
with torch.no_grad():
    emb = model.encode(X)
mu, sd = emb[tr].mean(0), emb[tr].std(0).clamp(min=1e-6)
emb = (emb - mu) / sd
clf = torch.nn.Linear(H, n_cls)
copt = torch.optim.Adam(clf.parameters(), lr=0.01)
for _ in range(300):
    copt.zero_grad()
    F.cross_entropy(clf(emb[tr]), y[tr]).backward()
    copt.step()
with torch.no_grad():
    pred = clf(emb).argmax(1)
print(f"epochs {epochs} loss {best:.4f} seconds {time.time() - t0:.0f}")
os.makedirs("/output", exist_ok=True)
with open("/output/predictions.csv", "w") as f:
    f.write("id,prediction\n")
    f.writelines(f"{i},{p}\n" for i, p in zip(te.tolist(), pred[te].tolist()))
