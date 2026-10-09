# Plan step -> implemented in: function or class name
# Load data / build symmetric adjacency A, add self-loops, compute Â = D⁻¹ᐟ² (A+I) D⁻¹ᐟ²; convert to sparse torch tensor -> normalize_adjacency
# Seed numpy & torch with env var SEED -> set_seed
# Model: two-layer GCN with dropout -> class GCN
# Loss: cross-entropy on train_idx -> training loop
# Optimizer: Adam(lr=0.01, weight_decay=5e-4) -> optimizer
# Train ≤200 epochs; early stopping patience 100 -> train_model
# Inference: argmax(Z[test_idx]) -> after training
# Write predictions via nullius_io -> nio.write_predictions
# Print final validation accuracy -> printed after training

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio
from scipy import sparse as sp


def set_seed(seed: int):
    """Seed numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_adjacency(edges: np.ndarray, num_nodes: int):
    """
    Build symmetric adjacency matrix from edge list (already bidirectional),
    add self-loops, compute symmetrically normalized Â = D^{-1/2} (A+I) D^{-1/2},
    and return as a torch sparse COO tensor.
    """
    # adjacency without self-loops
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0], dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    # add self-loops
    adj = adj + sp.eye(num_nodes, dtype=np.float32)

    # degree vector
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == np.inf] = 0.0

    # create normalized adjacency
    row, col = adj.nonzero()
    data = deg_inv_sqrt[row] * deg_inv_sqrt[col] * adj.data
    norm_adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    # convert to torch sparse tensor
    indices = torch.from_numpy(
        np.vstack([norm_adj.row, norm_adj.col]).astype(np.int64)
    )
    values = torch.from_numpy(norm_adj.data.astype(np.float32))
    shape = torch.Size(norm_adj.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


class GCN(nn.Module):
    """Two-layer Graph Convolutional Network as described in the plan."""

    def __init__(self, in_features: int, hidden: int, out_features: int, dropout: float = 0.5):
        super().__init__()
        self.W0 = nn.Parameter(torch.Tensor(in_features, hidden))
        self.W1 = nn.Parameter(torch.Tensor(hidden, out_features))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W0)
        nn.init.xavier_uniform_(self.W1)

    def forward(self, x: torch.Tensor, adj: torch.sparse.Tensor) -> torch.Tensor:
        """
        x: [N, in_features] feature matrix
        adj: [N, N] normalized adjacency (sparse)
        Returns: [N, out_features] logits
        """
        # First layer: H0 = ReLU(Â * X * W0)
        support0 = torch.mm(x, self.W0)          # [N, hidden]
        support0 = torch.sparse.mm(adj, support0)  # Â * (XW0)
        h = F.relu(support0)
        h = self.dropout(h)

        # Second layer: Z = Â * H0 * W1
        support1 = torch.mm(h, self.W1)          # [N, out_features]
        logits = torch.sparse.mm(adj, support1)  # Â * (H0W1)
        return logits


def train_model(
    model: nn.Module,
    features: torch.Tensor,
    adj: torch.sparse.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    smoke: bool,
):
    """Train the GCN with early stopping (unless in smoke mode)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val_loss = float("inf")
    patience_counter = 0
    max_epochs = 1 if smoke else 200
    patience = 100 if not smoke else None  # not used in smoke

    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj)
            val_loss = F.cross_entropy(val_logits[val_idx], labels[val_idx])
            val_pred = val_logits[val_idx].argmax(dim=1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()
        model.train()

        # Early stopping check (only in full run)
        if not smoke:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Optionally could save best model state here
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if epoch % 20 == 0 or epoch == max_epochs - 1:
            print(
                f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Val Loss {val_loss.item():.4f} | Val Acc {val_acc:.4f}"
            )

    # Return validation accuracy of the final model (or best if early stopped)
    model.eval()
    with torch.no_grad():
        final_logits = model(features, adj)
        final_pred = final_logits[val_idx].argmax(dim=1)
        final_acc = (final_pred == labels[val_idx]).float().mean().item()
    return final_acc


def main():
    # ----- Seed -----
    seed = int(os.getenv("SEED", "0"))
    set_seed(seed)

    # ----- Load data -----
    d = nio.load()  # returns dict with keys: features, edges, labels, train_idx, val_idx, test_idx
    features = torch.from_numpy(d["features"]).to(torch.float32)          # [2708, 1433]
    edges = d["edges"].astype(np.int64)                                   # [E, 2]
    labels = torch.from_numpy(d["labels"]).to(torch.long)                 # [2708] with -1 for test
    train_idx = torch.from_numpy(d["train_idx"]).to(torch.long)
    val_idx = torch.from_numpy(d["val_idx"]).to(torch.long)
    test_idx = torch.from_numpy(d["test_idx"]).to(torch.long)

    num_nodes = features.shape[0]

    # ----- Build normalized adjacency -----
    adj_norm = normalize_adjacency(edges, num_nodes)  # torch sparse tensor

    # ----- Model -----
    model = GCN(in_features=features.shape[1], hidden=16, out_features=7, dropout=0.5)

    # ----- Smoke flag -----
    smoke = os.getenv("SMOKE") == "1"

    # ----- Train -----
    val_acc = train_model(
        model=model,
        features=features,
        adj=adj_norm,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        smoke=smoke,
    )
    print(f"Final validation accuracy: {val_acc:.4f}")

    # ----- Inference (predict for all nodes) -----
    model.eval()
    with torch.no_grad():
        logits_all = model(features, adj_norm)  # [N, 7]
        # For writing predictions we can pass logits; helper will handle test/val selection.
        predictions = logits_all.argmax(dim=1).detach().cpu().numpy()  # shape [N]

    # ----- Write predictions via helper -----
    nio.write_predictions(predictions)


if __name__ == "__main__":
    main()
