

# Plan step -> implemented in: CONFIG dict
# Plan step -> implemented in: seed_setting function
# Plan step -> implemented in: feature_normalize function
# Plan step -> implemented in: adj_normalize function
# Plan step -> implemented in: weight_init function
# Plan step -> implemented in: GCNLayer class
# Plan step -> implemented in: Net class
# Plan step -> implemented in: training_loop function (full-batch)
# Plan step -> implemented in: compute_loss function (weight decay on first layer)
# Plan step -> implemented in: compute_accuracy function
# Plan step -> implemented in: training_loop function (early stopping)
# Plan step -> implemented in: training_loop function (validation masking)
# Plan step -> implemented in: main function (predictions for all nodes)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

# Hyperparameters from spec (read only from this dict)
CONFIG = {
    "lr": 0.01,
    "epochs": 200,
    "dropout": 0.5,
    "hidden": 16,
    "patience": 10,
    "weight_decay": 5e-4,
}

# ----- Seed handling -----
def seed_setting():
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

# ----- Feature preprocessing -----
def feature_normalize(feats: np.ndarray) -> np.ndarray:
    """Row-normalize feature matrix (each row sums to 1)."""
    row_sum = feats.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0  # avoid division by zero for isolated nodes
    return feats / row_sum

# ----- Adjacency preprocessing -----
def adj_normalize(edges: np.ndarray, num_nodes: int) -> torch.Tensor:
    """Add self-loops and symmetrically normalize adjacency."""
    # Build COO matrix
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    # Add self-loops
    adj = adj + sp.eye(num_nodes, dtype=np.float32)
    # Degree vector
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    # Normalize: D^{-1/2} A D^{-1/2}
    adj_norm = sp.diags(deg_inv_sqrt) @ adj @ sp.diags(deg_inv_sqrt)
    # Convert to dense torch tensor
    return torch.from_numpy(adj_norm.toarray()).float()

# ----- Weight initialization (Glorot & Bengio) -----
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ----- Model definition -----
class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)
        support = torch.mm(adj, x)  # (N, N) @ (N, F) -> (N, F)
        out = self.linear(support)
        return out  # activation applied outside if needed


class Net(nn.Module):
    def __init__(self, nfeat: int, nhidden: int, nclass: int, dropout: float):
        super().__init__()
        self.gcn1 = GCNLayer(nfeat, nhidden, dropout=dropout)
        self.gcn2 = GCNLayer(nhidden, nclass, dropout=dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gcn1(x, adj))
        x = self.gcn2(x, adj)
        return x

# ----- Loss and accuracy helpers -----
def compute_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, model: Net, weight_decay: float) -> torch.Tensor:
    """Masked softmax cross-entropy + L2 on first-layer weights."""
    loss = F.cross_entropy(logits[mask], labels[mask], reduction="mean")
    # L2 regularization on first layer weights only (no 0.5 factor)
    l2_reg = torch.sum(model.gcn1.linear.weight ** 2)
    loss = loss + weight_decay * l2_reg
    return loss


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Classification accuracy under mask."""
    preds = logits[mask].argmax(dim=1)
    correct = (preds == labels[mask]).float()
    return correct.mean()

# ----- Training loop with early stopping -----
def training_loop(
    model: Net,
    features: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    best_val_loss = float("inf")
    wait = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = compute_loss(logits, labels, train_mask, model, weight_decay)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj)
            val_loss = compute_loss(val_logits, labels, val_mask, model, weight_decay)
            val_acc = compute_accuracy(val_logits, labels, val_mask)
        model.train()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc.item():.4f}"
            )

    # Return best validation accuracy (from last evaluated epoch)
    model.eval()
    with torch.no_grad():
        final_logits = model(features, adj)
        final_val_acc = compute_accuracy(final_logits, labels, val_mask)
    return final_val_acc.item()

def main():
    seed_setting()

    # Load data via nullius_io
    import nullius_io as nio
    data = nio.load()
    features_np = data["features"]          # (3327, 3703) float32
    edges = data["edges"]                   # (E, 2) int64
    labels_np = data["labels"]              # (3327,) int64, -1 for test
    train_idx = data["train_idx"]           # (120,)
    val_idx = data["val_idx"]               # (500,)
    test_idx = data["test_idx"]             # (1000,)

    N = features_np.shape[0]

    # Preprocess features
    features_norm = feature_normalize(features_np)
    features_tensor = torch.from_numpy(features_norm).float()

    # Preprocess adjacency
    adj_tensor = adj_normalize(edges, N)

    # Labels tensor (keep -1 for test; they will be masked out)
    labels_tensor = torch.from_numpy(labels_np).long()

    # Masks
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Model
    model = Net(
        nfeat=features_norm.shape[1],
        nhidden=CONFIG["hidden"],
        nclass=int(labels_np[labels_np >= 0].max() + 1),  # 6 classes
        dropout=CONFIG["dropout"],
    )
    model.apply(weight_init)

    # Determine number of epochs (SMOKE mode overrides)
    smoke_mode = os.environ.get("SMOKE") == "1"
    epochs = 1 if smoke_mode else CONFIG["epochs"]

    # Train
    val_acc = training_loop(
        model=model,
        features=features_tensor,
        adj=adj_tensor,
        labels=labels_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        epochs=epochs,
        patience=CONFIG["patience"],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Print final validation accuracy as required
    print(f"{val_acc:.5f}")

    # Predictions for all nodes (logits)
    model.eval()
    with torch.no_grad():
        all_logits = model(features_tensor, adj_tensor)
    # Convert to numpy for nullius_io
    preds_np = all_logits.cpu().numpy()
    nio.write_predictions(preds_np)

if __name__ == "__main__":
    main()