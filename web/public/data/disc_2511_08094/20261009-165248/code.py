

# Plan step -> implemented in: function or class name
# Load data and set seeds -> lines after imports
# Build adjacency matrix -> build_adjacency function
# Define SL-GCN approximation (standard GCN with LeakyReLU) -> SLGCN class
# Training loop with optimizer, scheduler, early stopping -> train function
# Evaluation and prediction -> evaluate and predict blocks
# Write predictions via nullius_io -> nio.write_predictions call

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

# Base hyperparameters as required
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
}
# Additional hyperparameters (allowed to extend CONFIG)
CONFIG["leaky_slope"] = 0.3
CONFIG["lr_decay_factor"] = 0.1
CONFIG["lr_stop_factor"] = 0.9

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_adjacency(edges, num_nodes):
    # edges: np.ndarray [E, 2] (directed pairs, includes both directions)
    idx = torch.from_numpy(edges.T).long()  # shape [2, E]
    vals = torch.ones(edges.shape[0], dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, vals, size=(num_nodes, num_nodes), dtype=torch.float32)
    return adj.coalesce()

class SLGCN(nn.Module):
    """
    Simplified SL-GCN mechanism: multi-head graph convolution with LeakyReLU.
    This is a smaller version of the full Stuart-Landau GCN, preserving the
    core idea of layered feature propagation with non-linearity.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, heads, leaky_slope):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.head_dim = hidden_dim // heads  # assume divisible
        self.dropout = nn.Dropout(dropout)
        self.leaky_slope = leaky_slope
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            if l == 0:
                in_ch = in_dim
            else:
                in_ch = hidden_dim
            if l == num_layers - 1:
                # final layer to logits
                self.layers.append(nn.Linear(in_ch, out_dim))
            else:
                # multi-head linear projections
                head_linears = nn.ModuleList([
                    nn.Linear(in_ch, self.head_dim) for _ in range(heads)
                ])
                self.layers.append(head_linears)

    def forward(self, x, adj):
        h = x
        for i, layer in enumerate(self.layers):
            # graph convolution: aggregate neighbors
            h = torch.sparse.mm(adj, h)  # [N, in_ch]
            if i < self.num_layers - 1:
                # apply multi-head linear layers
                head_outs = [lin(h) for lin in layer]  # list of [N, head_dim]
                h = torch.cat(head_outs, dim=-1)       # [N, hidden_dim]
                h = F.leaky_relu(h, negative_slope=self.leaky_slope)
                h = self.dropout(h)
            else:
                # final logits layer
                h = layer(h)  # [N, out_dim]
        return h

def train():
    seed = int(os.environ.get("SEED", 0))
    set_seed(seed)

    d = nio.load()
    features = torch.from_numpy(d["features"]).float()
    edges = d["edges"].astype(np.int64)
    labels = torch.from_numpy(d["labels"]).long()
    train_idx = torch.from_numpy(d["train_idx"]).long()
    val_idx = torch.from_numpy(d["val_idx"]).long()
    test_idx = torch.from_numpy(d["test_idx"]).long()
    num_nodes = features.shape[0]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    adj = build_adjacency(edges, num_nodes)

    model = SLGCN(
        in_dim=features.shape[1],
        hidden_dim=CONFIG["hidden"],
        out_dim=7,
        num_layers=CONFIG["layers"],
        dropout=CONFIG["dropout"],
        heads=CONFIG["heads"],
        leaky_slope=CONFIG["leaky_slope"],
    )

    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=CONFIG["lr_decay_factor"],
        patience=CONFIG["patience"]
    )

    smoke = os.environ.get("SMOKE") == "1"
    max_epoch = 1 if smoke else CONFIG["epochs"]

    best_val_acc = 0.0
    best_state = None
    lr_initial = CONFIG["lr"]
    lr_stop = lr_initial * CONFIG["lr_stop_factor"]

    for epoch in range(max_epoch):
        model.train()
        # shuffle training indices for mini-batch order
        perm = train_idx[torch.randperm(train_idx.size(0))]
        for start in range(0, perm.size(0), CONFIG["batch_size"]):
            batch_idx = perm[start:start+CONFIG["batch_size"]]
            optimizer.zero_grad()
            logits = model(features, adj)
            loss = F.cross_entropy(logits[batch_idx], labels[batch_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval = model(features, adj)
            pred = logits_eval.argmax(dim=1)

            def accuracy(mask):
                correct = (pred[mask] == labels[mask]).sum().item()
                total = mask.sum().item()
                return correct / total * 100.0 if total > 0 else 0.0

            train_acc = accuracy(train_mask)
            val_acc = accuracy(val_mask)
            test_acc = accuracy(test_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < lr_stop:
            break

        if epoch % 10 == 0 or epoch == max_epoch - 1:
            print(
                f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                f"Train {train_acc:.2f}% | Val {val_acc:.2f}% | Test {test_acc:.2f}% | LR {current_lr:.1e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        final_logits = model(features, adj)
        final_pred = final_logits.argmax(dim=1).cpu().numpy()

    nio.write_predictions(final_pred)
    print(f"Best validation accuracy: {best_val_acc:.2f}")

if __name__ == "__main__":
    train()