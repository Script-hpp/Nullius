import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_

def main():
    # ----- environment -----
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    smoke = os.environ.get("SMOKE") == "1"

    # ----- load data -----
    data = np.load("/data/cora_lcc_public.npz", allow_pickle=True)
    features = torch.from_numpy(data["features"]).float()          # [N, 1433]
    edges = torch.from_numpy(data["edges"]).long()                # [E, 2] directed pairs
    labels = torch.from_numpy(data["labels"]).long()              # [N]
    train_idx = torch.from_numpy(data["train_idx"]).long()
    val_idx = torch.from_numpy(data["val_idx"]).long()
    test_idx = torch.from_numpy(data["test_idx"]).long()

    N = features.shape[0]
    num_classes = int(labels.max().item()) + 1   # 7

    # ----- build normalized adjacency (A+I) * D^{-1/2} (A+I) D^{-1/2} -----
    # add self-loops to edge list
    self_loop = torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(1, 2)
    self_loop = self_loop[:, [0, 0]]  # (i,i)
    edge_all = torch.cat([edges, self_loop], dim=0)   # [E+N, 2]
    # remove possible duplicates (coalesce later)
    idx = edge_all.t()  # [2, E+N]
    vals = torch.ones(idx.shape[1], dtype=torch.float32)

    # degree
    deg = torch.zeros(N, dtype=torch.float32)
    deg.index_add_(0, idx[0], vals)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # normalize
    norm_vals = deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
    adj_norm = torch.sparse_coo_tensor(idx, norm_vals, size=(N, N)).coalesce()

    # ----- model hyperparameters -----
    depths = [2, 4, 8, 16, 32, 64, 128]
    hidden_dim = 64
    lr = 0.01
    weight_decay = 5e-4
    max_epochs = 1000
    patience = 100
    dt = 1.0
    alpha = 0.1
    gamma = 0.1

    best_val_acc = 0.0
    best_state = None
    best_depth = None

    # ----- train each depth -----
    for depth in depths:
        model = SLGCN(
            num_features=features.shape[1],
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            depth=depth,
            adj_norm=adj_norm,
            dt=dt,
            alpha=alpha,
            gamma=gamma,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # smoke mode: only 1 epoch
        epochs = 1 if smoke else max_epochs
        wait = 0
        best_local_acc = 0.0
        best_local_state = None

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(features)
            loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()

            # validation
            model.eval()
            with torch.no_grad():
                val_logits = model(features)
                val_pred = val_logits[val_idx].argmax(dim=1)
                val_acc = (val_pred == labels[val_idx]).float().mean().item()

            if val_acc > best_local_acc:
                best_local_acc = val_acc
                best_local_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_local_acc > best_val_acc:
            best_val_acc = best_local_acc
            best_state = best_local_state
            best_depth = depth

    # print final validation accuracy
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # ----- load best model and predict test -----
    model = SLGCN(
        num_features=features.shape[1],
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        depth=best_depth,
        adj_norm=adj_norm,
        dt=dt,
        alpha=alpha,
        gamma=gamma,
    )
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(features)
        test_pred = test_logits[test_idx].argmax(dim=1)

    # ----- write predictions -----
    os.makedirs("/output", exist_ok=True)
    with open("/output/predictions.csv", "w") as f:
        f.write("id,prediction\n")
        if smoke:
            # only first 20 test nodes
            for idx, pred in zip(test_idx[:20], test_pred[:20]):
                f.write(f"{idx.item()},{pred.item()}\n")
        else:
            for idx, pred in zip(test_idx, test_pred):
                f.write(f"{idx.item()},{pred.item()}\n")

class SLGCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, depth, adj_norm,
                 dt=1.0, alpha=0.1, gamma=0.1):
        super().__init__()
        self.adj_norm = adj_norm
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.alpha = alpha
        self.gamma = gamma

        self.lin0 = nn.Linear(num_features, hidden_dim, bias=False)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(hidden_dim, hidden_dim)) for _ in range(depth)
        ])
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.lin0.weight)
        for w in self.weights:
            xavier_uniform_(w)
        xavier_uniform_(self.classifier.weight)
        zeros_(self.classifier.bias)

    def forward(self, x):
        x = self.lin0(x)  # [N, hidden_dim]
        Y = torch.zeros_like(x)
        for i in range(self.depth):
            W = self.weights[i]
            support = torch.sparse.mm(self.adj_norm, x)   # [N, hidden_dim]
            F = support @ W                              # [N, hidden_dim]
            F = torch.relu(F)
            Y = Y + self.dt * F - self.gamma * x - self.alpha * Y
            x = x + self.dt * Y
        logits = self.classifier(x)
        return logits

if __name__ == "__main__":
    main()
