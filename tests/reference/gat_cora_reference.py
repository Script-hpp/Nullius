"""Menschliche Referenzloesung Cora, Standard-Split (140/500/1000): GAT (Velickovic et al. 2018) nach Abschnitt 3.3 des Papers und dem
Original-Code (execute_cora.py): 2 Schichten, erste Schicht 8 Koepfe x 8 Merkmale mit ELU, zweite Schicht 1 Kopf mit 7 Ausgaengen, Dropout 0,6 auf
den Eingaben jeder Schicht und den normalisierten Attention-Koeffizienten, L2 5e-4 (TF-l2_loss = Summe der Quadrate / 2) auf allen Gewichten,
Adam lr 0,005, Early Stopping auf Validierungsloss UND -accuracy mit Geduld 100 (Checkpoint, wenn beide sich verbessern), Glorot-Init.
Kantenweise (duenn besetzt) gerechnet. Selbsttest der Pipeline, nie Agenten-Ergebnis. Aufruf im Branch: SEED=0 python train.py
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
tr, va, te = (torch.from_numpy(d[k]).long() for k in ("train_idx", "val_idx", "test_idx"))
N, n_cls = X.shape[0], 7
loops = torch.arange(N)
src = torch.cat([edges[:, 0], loops])      # Nachbar j (Quelle der Nachricht)
dst = torch.cat([edges[:, 1], loops])      # Knoten i (Ziel), inklusive Selbstschleife
P, HEADS, HID, LR, L2, PATIENCE = 0.6, 8, 8, 0.005, 5e-4, 100


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
        alpha = F.dropout(w / denom[dst], P, train)                                            # Dropout auf Attention-Koeffizienten
        out = torch.zeros(N, self.heads, self.n_out).index_add_(0, dst, alpha[:, :, None] * h[src])
        return out.reshape(N, -1) + self.bias


l1, l2 = GATLayer(X.shape[1], HID, HEADS), GATLayer(HID * HEADS, n_cls, 1)
params = [p for m in (l1, l2) for p in m.parameters()]
opt = torch.optim.Adam(params, lr=LR)


def forward(train):
    return l2(F.elu(l1(X, train)), train)


def state():
    return [p.detach().clone() for p in params]


vacc_mx, vlss_mn, curr, best_state, epochs = 0.0, 1e9, 0, None, 0
for epochs in range(1, 5001):
    opt.zero_grad()
    reg = L2 * sum((p ** 2).sum() / 2 for m in (l1, l2) for n, p in m.named_parameters() if n != "bias")
    (F.cross_entropy(forward(True)[tr], y[tr]) + reg).backward()
    opt.step()
    with torch.no_grad():
        out = forward(False)
        v_loss = (F.cross_entropy(out[va], y[va]) + reg).item()
        v_acc = (out[va].argmax(1) == y[va]).float().mean().item()
    if v_acc >= vacc_mx or v_loss <= vlss_mn:
        if v_acc >= vacc_mx and v_loss <= vlss_mn:
            best_state = state()
        vacc_mx, vlss_mn, curr = max(v_acc, vacc_mx), min(v_loss, vlss_mn), 0
    else:
        curr += 1
        if curr == PATIENCE:
            break
with torch.no_grad():
    for p, s in zip(params, best_state):
        p.copy_(s)
    pred = forward(False).argmax(1)
print(f"epochs {epochs} best val acc {vacc_mx:.4f} seconds {time.time() - t0:.0f}")
os.makedirs("/output", exist_ok=True)
with open("/output/predictions.csv", "w") as f:
    f.write("id,prediction\n")
    f.writelines(f"{i},{p}\n" for i, p in zip(te.tolist(), pred[te].tolist()))
