"""Menschliche Referenzloesung Cora-LCC (Shchur et al. 2018): GAT mit den Hyperparametern aus Tabelle 4 (Anhang C): 8 Koepfe x 8 Merkmale
(Hidden 64), lr 0,01, Dropout 0,6 auf Eingaben und 0,3 auf Attention-Koeffizienten, L2 0,01 (TF-l2_loss = 0,5 * lambda * ||W||^2 auf allen Gewichten),
Adam, Early Stopping auf dem GESAMT-Validierungsloss (Datenloss + Regularisierung), Geduld 50, beste Gewichte zuruecksetzen, Glorot-Init.
Nach dem Original-Code des Papers (github.com/shchur/gnn-benchmark, gnnbench/models/gat.py und config/optimized/gat.conf.yaml): Merkmale NICHT zeilennormalisiert
(config/train.conf.yaml: normalize_features false), zusaetzliches Dropout (0,6) auf den transformierten Merkmalen nach der Attention-Berechnung, L2 auf allen Gewichten der
Schichten ausser Bias. Kantenweise (duenn besetzt) gerechnet, Architektur sonst wie tests/reference/gat_cora_reference.py. Selbsttest der Pipeline, nie Agenten-Ergebnis.
Aufruf im Branch: SPLIT=0 SEED=0 python train.py
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

d = np.load("/data/cora_lcc_public.npz")
X = torch.from_numpy(d["features"]).float()
edges = torch.from_numpy(d["edges"]).long()
y = torch.from_numpy(d["labels"]).long()
tr, va, te = (torch.from_numpy(d[k]).long() for k in ("train_idx", "val_idx", "test_idx"))
N, n_cls = X.shape[0], 7
loops = torch.arange(N)
src = torch.cat([edges[:, 0], loops])      # Nachbar j (Quelle der Nachricht)
dst = torch.cat([edges[:, 1], loops])      # Knoten i (Ziel), inklusive Selbstschleife
P, P_ATT, HEADS, HID, LR, L2, PATIENCE = 0.6, 0.3, 8, 8, 0.01, 1e-2, 50


class GATLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, heads):
        super().__init__()
        self.heads, self.n_out = heads, n_out
        self.W = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(n_in, heads * n_out)))
        self.a_src = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(heads, n_out)))
        self.a_dst = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(heads, n_out)))
        self.bias = torch.nn.Parameter(torch.zeros(heads * n_out))

    def forward(self, x, train):
        x = F.dropout(x, P, train)
        h = (x @ self.W).view(N, self.heads, self.n_out)
        e = F.leaky_relu((h * self.a_src).sum(-1)[src] + (h * self.a_dst).sum(-1)[dst], 0.2)   # [E, heads]
        emax = torch.full((N, self.heads), -1e30).scatter_reduce(0, dst[:, None].expand_as(e), e, "amax", include_self=True)
        w = torch.exp(e - emax[dst])
        denom = torch.zeros(N, self.heads).index_add_(0, dst, w)
        alpha = F.dropout(w / denom[dst], P_ATT, train)                                            # Dropout auf Attention-Koeffizienten
        h = F.dropout(h, P, train)                                                             # zusaetzliches Dropout auf den transformierten Merkmalen (gnn-benchmark gat.py, "used in the official implementation")
        out = torch.zeros(N, self.heads, self.n_out).index_add_(0, dst, alpha[:, :, None] * h[src])
        return out.reshape(N, -1) + self.bias


l1, l2 = GATLayer(X.shape[1], HID, HEADS), GATLayer(HID * HEADS, n_cls, 1)
params = [p for m in (l1, l2) for p in m.parameters()]
opt = torch.optim.Adam(params, lr=LR)


def forward(train):
    return l2(F.elu(l1(X, train)), train)


def state():
    return [p.detach().clone() for p in params]


best, best_state, wait, epochs = 1e9, None, 0, 0
for epochs in range(1, 10001):
    opt.zero_grad()
    reg = 0.5 * L2 * sum((p ** 2).sum() for m in (l1, l2) for n, p in m.named_parameters() if n != "bias")
    (F.cross_entropy(forward(True)[tr], y[tr]) + reg).backward()
    opt.step()
    with torch.no_grad():
        reg = 0.5 * L2 * sum((p ** 2).sum() for m in (l1, l2) for n, p in m.named_parameters() if n != "bias")
        v = (F.cross_entropy(forward(False)[va], y[va]) + reg).item()
    if v < best:
        best, best_state, wait = v, state(), 0
    else:
        wait += 1
        if wait >= PATIENCE:
            break
with torch.no_grad():
    for p, s in zip(params, best_state):
        p.copy_(s)
    pred = forward(False).argmax(1)
print(f"epochs {epochs} best val loss {best:.4f} seconds {time.time() - t0:.0f}")
os.makedirs("/output", exist_ok=True)
with open("/output/predictions.csv", "w") as f:
    f.write("id,prediction\n")
    f.writelines(f"{i},{p}\n" for i, p in zip(te.tolist(), pred[te].tolist()))
