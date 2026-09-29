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
train_idx, val_idx, test_idx = (torch.from_numpy(d[k]).long() for k in ("train_idx", "val_idx", "test_idx"))
N = X.shape[0]

loops = torch.arange(N)
rows = torch.cat([edges[:, 0], loops])
cols = torch.cat([edges[:, 1], loops])          # Selbstschleifen
deg = torch.bincount(rows, minlength=N).float()
vals = deg[rows].pow(-0.5) * deg[cols].pow(-0.5)
A_hat = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (N, N)).coalesce()


class Net(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super().__init__()
        self.l1 = nn.Linear(n_in, n_hid)
        self.l2 = nn.Linear(n_hid, n_out)

    def forward(self, x):
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(torch.sparse.mm(A_hat, self.l1(x)))
        x = F.dropout(x, 0.5, self.training)
        return torch.sparse.mm(A_hat, self.l2(x))


model = Net(X.shape[1], 16, 7)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
best, best_state = 0.0, None
for epoch in range(200):
    model.train()
    opt.zero_grad()
    out = model(X)
    F.cross_entropy(out[train_idx], y[train_idx]).backward()
    opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(X)[val_idx].argmax(1) == y[val_idx]).float().mean().item()
    if acc >= best:
        best, best_state = acc, {k: t.clone() for k, t in model.state_dict().items()}
model.load_state_dict(best_state)
print(f"val acc {best:.4f}")
model.eval()
with torch.no_grad():
    pred = model(X)[test_idx].argmax(1).numpy()
os.makedirs("/output", exist_ok=True)
with open("/output/predictions.csv", "w") as f:
    f.write("id,prediction\n")
    f.writelines(f"{i},{p}\n" for i, p in zip(test_idx.tolist(), pred))
