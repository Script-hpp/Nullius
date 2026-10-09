

# Plan step -> implemented in: load_data
# Plan step -> implemented in: build_adjacency
# Plan step -> implemented in: aug_norm_adj
# Plan step -> implemented in: propagate_features
# Plan step -> implemented in: row_normalize_features
# Plan step -> implemented in: SGC class
# Plan step -> implemented in: train_model
# Plan step -> implemented in: main

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {"lr": 0.2, "weight_decay": 5e-06, "epochs": 100, "dropout": 0, "hidden": 0}
CONFIG["degree"] = 2  # additional configurable parameter


def load_data():
    """Load data via nullius_io."""
    d = nio.load()
    features = d["features"]          # np.float32 [N, F]
    edges = d["edges"]                # np.int64 [E, 2] (directed, both directions)
    labels = d["labels"]              # np.int64 [N] (-1 for test)
    train_idx = d["train_idx"]
    val_idx = d["val_idx"]
    test_idx = d["test_idx"]
    return features, edges, labels, train_idx, val_idx, test_idx


def build_adjacency(edges, num_nodes):
    """Build sparse adjacency matrix with self-loops."""
    idx = torch.from_numpy(edges.T).long()          # [2, E]
    vals = torch.ones(idx.shape[1], dtype=torch.float32)
    self_idx = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)  # [2, N]
    self_vals = torch.ones(num_nodes, dtype=torch.float32)
    all_idx = torch.cat([idx, self_idx], dim=1)
    all_vals = torch.cat([vals, self_vals])
    adj = torch.sparse_coo_tensor(all_idx, all_vals, size=(num_nodes, num_nodes))
    adj = adj.coalesce()
    return adj


def aug_norm_adj(adj):
    """Compute symmetrically normalized adjacency with self-loops: D^{-1/2} (A+I) D^{-1/2}."""
    adj = adj.coalesce()
    row = adj.indices()[0]
    col = adj.indices()[1]
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    norm_vals = adj.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    norm_adj = torch.sparse_coo_tensor(adj.indices(), norm_vals, size=adj.shape)
    return norm_adj.coalesce()


def propagate_features(features, adj_norm, degree):
    """Compute \hat{A}^k X via repeated spmm."""
    X = torch.from_numpy(features).float()
    for _ in range(degree):
        X = torch.spmm(adj_norm, X)
    return X


def row_normalize_features(features):
    """Row-normalize features so each row sums to 1."""
    row_sum = features.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1.0
    return features / row_sum


class SGC(nn.Module):
    """Simple Linear model (SGC) assuming pre‑propagated features."""
    def __init__(self, nfeat, nclass, dropout):
        super().__init__()
        self.W = nn.Linear(nfeat, nclass)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.W(x)
        x = self.dropout(x)
        return x


def train_model(model, features, labels, train_idx, val_idx):
    """Train model with full‑batch Adam, return best model state based on val accuracy."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    best_val_acc = 0.0
    best_state = None

    epochs = 1 if os.environ.get("SMOKE") == "1" else CONFIG["epochs"]
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features)
            val_pred = val_logits[val_idx].argmax(dim=1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return model


def main():
    seed = int(os.environ.get("SEED", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    features_np, edges_np, labels_np, train_idx, val_idx, test_idx = load_data()
    num_nodes = labels_np.shape[0]
    num_classes = int(np.max(labels_np[labels_np >= 0])) + 1

    adj = build_adjacency(edges_np, num_nodes)
    adj_norm = aug_norm_adj(adj)
    propagated = propagate_features(features_np, adj_norm, CONFIG["degree"])
    propagated = row_normalize_features(propagated)

    assert CONFIG["hidden"] == 0, "Hidden units must be zero for SGC"
    model = SGC(nfeat=propagated.shape[1], nclass=num_classes, dropout=CONFIG["dropout"])

    labels_tensor = torch.from_numpy(labels_np).long()
    train_idx_t = torch.from_numpy(train_idx).long()
    val_idx_t = torch.from_numpy(val_idx).long()
    trained_model = train_model(
        model,
        propagated,
        labels_tensor,
        train_idx_t,
        val_idx_t
    )

    trained_model.eval()
    with torch.no_grad():
        all_logits = trained_model(propagated)
        all_preds = all_logits.argmax(dim=1).cpu().numpy()

    nio.write_predictions(all_preds)


if __name__ == "__main__":
    main()