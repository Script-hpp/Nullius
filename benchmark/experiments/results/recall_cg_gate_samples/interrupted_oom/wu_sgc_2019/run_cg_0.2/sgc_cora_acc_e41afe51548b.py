

# Plan step -> implemented in: class SGC
# Plan step -> implemented in: function preprocess_features
# Plan step -> implemented in: reading CONFIG
# Plan step -> implemented in: function train_model
# Plan step -> implemented in: main block calling nio.write_predictions

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

# Exact literal dict as required
CONFIG = {
    "lr": 0.2,
    "weight_decay": 5e-06,
    "epochs": 100,
    "dropout": 0,
    "hidden": 0,
}
# Additional key needed for propagation (not part of the required literals)
CONFIG["degree"] = 2

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

class SGC(nn.Module):
    """Simple linear model for SGC (no hidden units)."""
    def __init__(self, nfeat, nclass):
        super().__init__()
        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)

def preprocess_features(features, edges, degree):
    """
    AugNormAdj normalization: \hat{A}=D^{-1/2}(A+I)D^{-1/2}
    Propagation: \tilde{X} = \hat{A}^k X
    Row-normalize \tilde{X}.
    """
    N = features.shape[0]
    device = features.device

    # Build adjacency (COO) from edges (already contains both directions)
    idx = edges.t()  # shape [2, E]
    vals = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(idx, vals, (N, N), device=device).coalesce()

    # Add self-loops
    i = torch.arange(N, device=device)
    self_loop_idx = torch.stack([i, i], dim=0)
    self_loop_vals = torch.ones(N, dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(
        torch.cat([idx, self_loop_idx], dim=1),
        torch.cat([vals, self_loop_vals]),
        (N, N),
        device=device,
    ).coalesce()

    # Compute degree vector of (A+I)
    deg = torch.sparse.sum(adj, dim=1).to_dense()  # [N]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Normalize: D^{-1/2} A D^{-1/2}
    col_weights = deg_inv_sqrt
    row_weights = deg_inv_sqrt
    idx = adj.indices()
    vals = adj.values()
    vals = vals * col_weights[idx[1]] * row_weights[idx[0]]
    adj_norm = torch.sparse_coo_tensor(idx, vals, (N, N), device=device).coalesce()

    # Propagation: \hat{A}^k X
    prop_feat = features
    for _ in range(degree):
        prop_feat = torch.spmm(adj_norm, prop_feat)

    # Row-normalize propagated features (each row sums to 1)
    row_sums = prop_feat.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    prop_feat = prop_feat / row_sums

    return prop_feat

def train_model(model, feat, labels, train_idx, val_idx, config):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    best_val_acc = -1.0
    best_state = None

    epochs = config["epochs"]
    # Smoke mode override
    if os.getenv("SMOKE") == "1":
        epochs = 1

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(feat[train_idx])
        loss = F.cross_entropy(output, labels[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(feat[val_idx])
            val_acc = (val_output.argmax(dim=1) == labels[val_idx]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_acc

def main():
    # Seed
    seed = int(os.getenv("SEED", "42"))
    set_seed(seed)

    # Load data via nullius_io
    data = nio.load()
    features_np = data["features"]          # [2708, 1433] float32
    edges_np = data["edges"]                # [E, 2] int64
    labels_np = data["labels"]              # [2708] int64, -1 for test
    train_idx_np = data["train_idx"]
    val_idx_np = data["val_idx"]
    test_idx_np = data["test_idx"]

    # Convert to torch tensors
    device = torch.device("cpu")
    features = torch.from_numpy(features_np).to(device)
    edges = torch.from_numpy(edges_np).to(device)
    labels = torch.from_numpy(labels_np).to(device)
    train_idx = torch.from_numpy(train_idx_np).to(device)
    val_idx = torch.from_numpy(val_idx_np).to(device)
    test_idx = torch.from_numpy(test_idx_np).to(device)

    # Preprocess features (normalization + propagation + row-normalization)
    feat_processed = preprocess_features(features, edges, CONFIG["degree"])

    # Model
    nfeat = features.shape[1]
    # Determine number of classes ignoring unlabeled test nodes (-1)
    valid_labels = labels[labels != -1]
    nclass = int(valid_labels.max().item()) + 1
    model = SGC(nfeat=nfeat, nclass=nclass).to(device)

    # Train
    val_acc = train_model(model, feat_processed, labels, train_idx, val_idx, CONFIG)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Predict on all nodes
    model.eval()
    with torch.no_grad():
        logits = model(feat_processed)          # [N, C]
        preds = logits.argmax(dim=1)            # [N]

    # Write predictions (helper expects predictions for all nodes)
    nio.write_predictions(preds.cpu().numpy())

if __name__ == "__main__":
    main()