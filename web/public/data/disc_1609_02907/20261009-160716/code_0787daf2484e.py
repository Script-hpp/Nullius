

# Plan step 1 -> implemented in: set_seeds
# Plan step 2 -> implemented in: load_and_preprocess
# Plan step 3 -> implemented in: row_normalize_features
# Plan step 4 -> implemented in: normalize_adjacency
# Plan step 5 -> implemented in: build_sparse_tensors
# Plan step 6 -> implemented in: GCN
# Plan step 7 -> implemented in: training_loop
# Plan step 8 -> implemented in: training_loop
# Plan step 9 -> implemented in: training_loop
# Plan step 10 -> implemented in: training_loop
# Plan step 11 -> implemented in: training_loop
# Plan step 12 -> implemented in: main (after training)
# Plan step 13 -> implemented in: main (call to nio.write_predictions)
# Plan step 14 -> implemented in: training_loop (full‑batch)
# Plan step 15 -> implemented in: GCN (two layers)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {
    "lr": 0.01,
    "epochs": 200,
    "dropout": 0.5,
    "hidden": 16,
    "patience": 10,
    "weight_decay": 5e-4,
}

def set_seeds():
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_and_preprocess():
    d = nio.load()
    features = d["features"].astype(np.float32)          # (N, F)
    edges = d["edges"].astype(np.int64)                  # (E, 2)
    labels = d["labels"].astype(np.int64)                # (N,)
    train_idx = d["train_idx"]
    val_idx = d["val_idx"]
    test_idx = d["test_idx"]
    return features, edges, labels, train_idx, val_idx, test_idx

def row_normalize_features(feats):
    row_sum = feats.sum(axis=1, keepdims=True)
    row_inv = np.where(row_sum > 0, 1.0 / row_sum, 0.0)
    return feats * row_inv

def normalize_adjacency(edges, num_nodes):
    src = torch.from_numpy(edges[:, 0])
    dst = torch.from_numpy(edges[:, 1])
    idx = torch.stack([src, dst], dim=0)
    vals = torch.ones_like(src, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, vals, size=(num_nodes, num_nodes)).coalesce()
    # Each undirected edge appears twice; after coalesce duplicates are summed -> weight 2.
    # Divide by 2 to get weight 1 for each undirected edge.
    if adj._nnz() > 0:
        adj_vals = adj.values() / 2.0
        adj = torch.sparse_coo_tensor(adj.indices(), adj_vals, size=(num_nodes, num_nodes)).coalesce()
    # Add self-loops
    self_loop_idx = torch.arange(num_nodes, dtype=torch.long)
    self_loop_vals = torch.ones(num_nodes, dtype=torch.float32)
    idx = torch.cat([adj.indices(), self_loop_idx.unsqueeze(0).repeat(2, 1)], dim=1)
    vals = torch.cat([adj.values(), self_loop_vals])
    adj_plus_I = torch.sparse_coo_tensor(idx, vals, size=(num_nodes, num_nodes)).coalesce()
    # Symmetric normalization: D^{-1/2} (A+I) D^{-1/2}
    deg = torch.sparse.sum(adj_plus_I, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    row = adj_plus_I.indices()[0]
    col = adj_plus_I.indices()[1]
    norm_vals = deg_inv_sqrt[row] * adj_plus_I.values() * deg_inv_sqrt[col]
    adj_norm = torch.sparse_coo_tensor(adj_plus_I.indices(), norm_vals, size=(num_nodes, num_nodes)).coalesce()
    return adj_norm

def build_sparse_tensors(features_norm, adj_norm):
    X = torch.from_numpy(features_norm)  # dense (N, F)
    return X, adj_norm

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.W0 = nn.Parameter(torch.Tensor(in_dim, hidden_dim))
        self.W1 = nn.Parameter(torch.Tensor(hidden_dim, out_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W0)
        nn.init.xavier_uniform_(self.W1)

    def forward(self, X, adj):
        # First graph convolution
        x = torch.sparse.mm(adj, X)          # (N, in_dim)
        x = x @ self.W0                      # (N, hidden)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Second graph convolution
        x = torch.sparse.mm(adj, x)          # (N, hidden)
        logits = x @ self.W1                 # (N, out_dim)
        return logits

def masked_cross_entropy(logits, labels, mask):
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    return F.cross_entropy(logits[mask], labels[mask])

def accuracy(logits, labels, mask):
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    selected_logits = logits[mask]
    selected_labels = labels[mask]
    pred = selected_logits.argmax(dim=1)
    correct = (pred == selected_labels).float().sum()
    return correct / mask.numel()

def training_loop(X, adj_norm, labels, train_idx, val_idx):
    # Determine number of classes from labeled nodes only
    labeled_labels = labels[labels >= 0]
    num_classes = int(labeled_labels.max().item() + 1)
    model = GCN(
        in_dim=X.shape[1],
        hidden_dim=CONFIG["hidden"],
        out_dim=num_classes,
        dropout=CONFIG["dropout"]
    )
    # Apply weight decay only to W0 as in the original code
    optimizer = torch.optim.Adam([
        {"params": model.W0, "weight_decay": CONFIG["weight_decay"]},
        {"params": model.W1, "weight_decay": 0.0}
    ], lr=CONFIG["lr"])

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    max_epochs = 1 if os.environ.get("SMOKE") == "1" else CONFIG["epochs"]

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X, adj_norm)
        loss = masked_cross_entropy(logits, labels, train_idx)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X, adj_norm)
            val_loss = masked_cross_entropy(val_logits, labels, val_idx)
            val_acc = accuracy(val_logits, labels, val_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= CONFIG["patience"]:
            break

    # restore best model
    model.load_state_dict(best_state)
    return model

def main():
    set_seeds()
    features, edges, labels, train_idx, val_idx, test_idx = load_and_preprocess()
    features_norm = row_normalize_features(features)
    adj_norm = normalize_adjacency(edges, features.shape[0])
    X, adj_norm = build_sparse_tensors(features_norm, adj_norm)
    labels_tensor = torch.from_numpy(labels)

    model = training_loop(X, adj_norm, labels_tensor, train_idx, val_idx)

    model.eval()
    with torch.no_grad():
        all_logits = model(X, adj_norm)
        # Optional: print validation accuracy
        val_logits = all_logits[val_idx]
        val_acc = accuracy(val_logits, labels_tensor, val_idx)
        print(f"Validation accuracy: {val_acc.item():.5f}")

    # Provide logits for all nodes to the helper (it will select test_idx)
    preds_np = all_logits.detach().cpu().numpy()
    nio.write_predictions(preds_np)

if __name__ == "__main__":
    main()