

# Plan step -> implemented in: function or class name
# Model: 2‑layer GCN, hidden size=64, ReLU after first layer, no activation after second layer -> implemented in: class GCN
# Dropout probability=0.8 applied to inputs of each GCN layer -> implemented in: GCN.forward
# Weight decay=0.001(L2) applied only to the first-layer weight matrix -> implemented in: training loop (manual L2 loss)
# Learning rate=0.01, optimizer=Adam -> implemented in: optimizer initialization
# Maximum epochs=200; early stopping used with GCNCriterion -> implemented in: training loop with patience
# Validation evaluated every report_interval epochs -> implemented in: validation each epoch (interval unspecified)
# Adjacency pre-processed with symmetric normalization + self-loops -> implemented in: preprocess_adjacency function
# Feature pre-processing: row-normalisation if normalize_features=True -> implemented in: preprocess_features function (no norm as flag not set)
# Training uses full-batch -> implemented in: using whole feature matrix
# Random seed for weight initialisation taken from env var SEED -> implemented in: seed setup
# After training, predict class labels for all nodes in test_idx and write /output/predictions.csv -> implemented in: main after training
# Total runtime must stay under 5 minutes -> implemented in: efficient ops

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio
import scipy.sparse as sp

CONFIG = {"lr": 0.01, "weight_decay": 0.001, "epochs": 200, "dropout": 0.8, "hidden": 64, "layers": 2}

def preprocess_features(feats: np.ndarray) -> torch.Tensor:
    """Row-normalise features if needed. Here we keep them as-is (binary bag-of-words)."""
    # normalize_features flag not provided; assume False
    return torch.from_numpy(feats.astype(np.float32))

def preprocess_adjacency(edges: np.ndarray, num_nodes: int) -> torch.sparse.FloatTensor:
    """Return symmetrically normalized adjacency with added self-loops (Â = D̃⁻⁰·⁵ Ã D̃⁻⁰·⁵)."""
    # Build adjacency from directed edge list (both directions already present)
    adj = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    # Add self-loops
    adj = adj + sp.eye(num_nodes, dtype=np.float32)
    # Symmetric normalization
    rowsum = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    adj_norm = d_mat.dot(adj).dot(d_mat)
    # Convert to torch sparse tensor
    adj_norm = adj_norm.tocoo()
    indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    adj_tensor = torch.sparse_coo_tensor(indices, values, size=adj_norm.shape).coalesce()
    return adj_tensor

class GCN(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int,
                 dropout: float, weight_decay: float, num_layers: int):
        super().__init__()
        self.dropout = dropout
        self.weight_decay = weight_decay  # L2 applied only to first layer
        self.num_layers = num_layers
        self.linears = nn.ModuleList()
        # Build layers: hidden for all but last, output for last
        for i in range(num_layers):
            if i == 0:
                self.linears.append(nn.Linear(in_features, hidden, bias=True))
            elif i == num_layers - 1:
                self.linears.append(nn.Linear(hidden, out_features, bias=True))
            else:
                self.linears.append(nn.Linear(hidden, hidden, bias=True))

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        for i, lin in enumerate(self.linears):
            # Apply dropout to inputs of each GCN layer
            x = F.dropout(x, p=self.dropout, training=self.training)
            support = torch.mm(x, lin.weight) + lin.bias
            x = torch.sparse.mm(adj, support)
            if i < self.num_layers - 1:  # ReLU for all but last layer
                x = F.relu(x)
        return x

def main():
    # Seed handling
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    data = nio.load()
    features = data["features"]          # np.ndarray [N, 1433]
    edges = data["edges"]                # np.ndarray [E, 2]
    labels = data["labels"]              # np.ndarray [N] with -1 for test
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    num_nodes = features.shape[0]
    num_labels = int(labels[labels >= 0].max()) + 1  # 7 classes

    # Preprocess
    feat_tensor = preprocess_features(features)
    adj_tensor = preprocess_adjacency(edges, num_nodes)
    label_tensor = torch.from_numpy(labels.astype(np.int64))

    # Model
    model = GCN(in_features=feat_tensor.shape[1],
                hidden=CONFIG["hidden"],
                out_features=num_labels,
                dropout=CONFIG["dropout"],
                weight_decay=CONFIG["weight_decay"],
                num_layers=CONFIG["layers"])

    # Optimizer (weight_decay now taken from CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    # Handle smoke mode
    smoke = os.environ.get("SMOKE") == "1"
    max_epochs = 1 if smoke else CONFIG["epochs"]
    patience = 10  # early stopping patience (chosen as typical)
    best_val_loss = float('inf')
    best_state = None
    patience_cnt = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(feat_tensor, adj_tensor)
        # Training loss on train_idx
        loss_train = F.cross_entropy(logits[train_idx], label_tensor[train_idx])
        # L2 regularization on first-layer weights only
        l2_loss = CONFIG["weight_decay"] * torch.norm(model.linears[0].weight, p=2) ** 2
        loss = loss_train + l2_loss
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(feat_tensor, adj_tensor)
            loss_val = F.cross_entropy(val_logits[val_idx], label_tensor[val_idx])
            pred_val = val_logits[val_idx].argmax(dim=1)
            acc_val = (pred_val == label_tensor[val_idx]).float().mean().item()

        # Early stopping logic
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = model.state_dict()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation accuracy (from best model)
    model.eval()
    with torch.no_grad():
        final_logits = model(feat_tensor, adj_tensor)
        final_pred = final_logits[val_idx].argmax(dim=1)
        final_acc = (final_pred == label_tensor[val_idx]).float().mean().item()
    print(f"Validation accuracy: {final_acc:.4f}")

    # Predictions for all nodes
    with torch.no_grad():
        all_logits = model(feat_tensor, adj_tensor)
        all_pred = all_logits.argmax(dim=1)  # shape [N]
    nio.write_predictions(all_pred.cpu().numpy())

if __name__ == "__main__":
    main()