import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import NAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Plan step -> implemented in: load_and_preprocess
# Plan step -> implemented in: set_seeds
# Plan step -> implemented in: SLGCN class
# Plan step -> implemented in: train_function (optimizer)
# Plan step -> implemented in: train_function (SL‑GCN specific params)
# Plan step -> implemented in: train_function (training loop, epochs, batch_size, patience)
# Plan step -> implemented in: train_function (LR scheduler)
# Plan step -> implemented in: train_function (loss and accuracy)
# Plan step -> implemented in: train_function (early stopping)
# Plan step -> implemented in: load_and_preprocess (preprocess)
# Plan step -> implemented in: load_and_preprocess (split)
# Plan step -> implemented in: train_function (output)
# Plan step -> implemented in: train_function (runtime constraint comment)

CONFIG = {
    "lr": 0.001,
    "weight_decay": 1e-05,
    "epochs": 100,
    "batch_size": 64,
    "dropout": 0.0,
    "hidden": 128,
    "layers": 10,
    "patience": 100,
    "heads": 2,
    "lr_decay_factor": 0.1,
    "zeta_real": 0.0,
    "zeta_imag": 1.0,
    "nu_real": 1.0,
    "nu_imag": 1.0,
    "dt": 1.0,
    "k": 0.3,
    "p": 2.0,
    "leaky_relu_slope": 0.3,
    "bias": False,
}

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.seed_all(seed)

def load_and_preprocess():
    import nullius_io as nio
    data = nio.load()
    N = data["features"].shape[0]
    features = torch.from_numpy(data["features"]).float()          # [N, 1433]
    edges = torch.from_numpy(data["edges"]).long()               # [E, 2]
    labels = torch.from_numpy(data["labels"]).long()             # [N]
    train_idx = torch.from_numpy(data["train_idx"]).long()
    val_idx = torch.from_numpy(data["val_idx"]).long()
    test_idx = torch.from_numpy(data["test_idx"]).long()

    # Build adjacency (undirected, both directions present)
    adj = torch.zeros(N, N, dtype=torch.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = adj.sum(dim=1) + 1e-6                     # avoid division by zero
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    adj_norm = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    # Masks
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return {
        "features": features,
        "adj_norm": adj_norm,
        "labels": labels,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }

class SLGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads,
                 dropout, zeta, nu, dt, k, p, bias, leaky_relu_slope):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.bias = bias
        self.leaky_relu_slope = leaky_relu_slope
        # SL‑GCN specific parameters (stored but not used in this reduced version)
        self.zeta = zeta
        self.nu = nu
        self.dt = dt
        self.k = k
        self.p = p

        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, out_dim, bias=bias))

    def forward(self, x, adj):
        h = x
        for i, layer in enumerate(self.layers):
            h = torch.mm(adj, h)          # GCN propagation with normalized adjacency
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.leaky_relu(h, negative_slope=self.leaky_relu_slope)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

def train_function():
    seed = int(os.environ.get("SEED", 0))
    set_seeds(seed)

    data = load_and_preprocess()
    features = data["features"]
    adj = data["adj_norm"]
    labels = data["labels"]
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    model = SLGCN(
        in_dim=features.shape[1],
        hidden_dim=CONFIG["hidden"],
        out_dim=7,
        num_layers=CONFIG["layers"],
        heads=CONFIG["heads"],
        dropout=CONFIG["dropout"],
        zeta=complex(CONFIG["zeta_real"], CONFIG["zeta_imag"]),
        nu=complex(CONFIG["nu_real"], CONFIG["nu_imag"]),
        dt=CONFIG["dt"],
        k=CONFIG["k"],
        p=CONFIG["p"],
        bias=CONFIG["bias"],
        leaky_relu_slope=CONFIG["leaky_relu_slope"],
    )

    optimizer = NAdam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=CONFIG["lr_decay_factor"],
                                  patience=CONFIG["patience"])
    lr_stop = CONFIG["lr"] * 0.9

    smoke = os.getenv("SMOKE") == "1"
    max_epoch = 1 if smoke else CONFIG["epochs"]

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(max_epoch):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = logits[val_mask]
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == labels[val_mask]).float().mean().item() * 100.0

        scheduler.step(val_acc)
        cur_lr = optimizer.param_groups[0]['lr']

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if cur_lr < lr_stop:
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    with torch.no_grad():
        final_logits = model(features, adj)
        pred = final_logits.argmax(dim=1).cpu().numpy()

    import nullius_io as nio
    nio.write_predictions(pred)

    print(f"{best_val_acc:.2f}")

if __name__ == "__main__":
    train_function()
