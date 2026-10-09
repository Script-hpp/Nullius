# Plan step -> implemented in: function or class name
# 1. Model: two-layer GCN -> class GCN
# 2. Features: row-normalized -> function row_normalize_features
# 3. Adjacency: renormalization trick -> function normalize_adjacency
# 4. Weight initialization: Glorot uniform -> in GraphConvolution.reset_parameters and GCN.__init__
# 5. Optimizer: Adam, lr=0.01, weight_decay=5e-4 applied to first GCN layer -> optimizer and manual L2 loss
# 6. Dropout: rate=0.5 applied to hidden-layer inputs -> F.dropout in GCN.forward
# 7. Epochs: max 200, early stopping patience=10 -> training loop with early stopping
# 8. Validation set: 500 nodes used only for early-stopping -> val_idx used for validation loss
# 9. Training loss: computed only on train_idx -> loss uses train_idx mask
# 10. Batch size: full-batch -> processing all nodes each epoch
# 11. Number of layers: 2; no attention heads -> GCN with two GraphConvolution layers
# 12. Random seed: read from SEED -> set_seeds function
# 13. After training, restore model with lowest validation loss -> best_state loading
# 14. Output: /output/predictions.csv -> call nio.write_predictions(logits)
# 15. Expected test accuracy ~81.5% -> printed validation accuracy

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

import nullius_io as nio

# --------------------- Configuration ---------------------
CONFIG = {
    "lr": 0.01,
    "epochs": 200,
    "dropout": 0.5,
    "hidden": 16,
    "patience": 10,
    "weight_decay": 5e-4,  # L2 penalty on first GCN layer
}
# -------------------------------------------------------


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def row_normalize_features(mat: np.ndarray) -> torch.Tensor:
    """Row-normalize feature matrix (each row divided by its sum)."""
    rowsum = np.array(mat.sum(axis=1)).flatten()
    r_inv = np.power(rowsum, -1.0)
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    normalized = r_mat_inv.dot(mat)
    return torch.from_numpy(normalized).float()


def normalize_adjacency(edges: np.ndarray, num_nodes: int) -> torch.sparse.FloatTensor:
    """Adjacency preprocessing: Â = A + I, then ĤÂ = D̃^{-1/2} Â D̃^{-1/2}."""
    # Build adjacency matrix (COO)
    adj = sp.coo_matrix((np.ones(edges.shape[0], dtype=np.float32), (edges[:, 0], edges[:, 1])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    # Add self-loops
    adj = adj + sp.eye(num_nodes)
    # Degree matrix
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    deg_inv_sqrt_mat = sp.diags(deg_inv_sqrt)
    # Normalized adjacency
    adj_norm = deg_inv_sqrt_mat.dot(adj).dot(deg_inv_sqrt_mat)
    # Convert to torch sparse tensor
    adj_norm_coo = adj_norm.tocoo()
    indices = torch.from_numpy(
        np.vstack((adj_norm_coo.row, adj_norm_coo.col)).astype(np.int64)
    )
    values = torch.from_numpy(adj_norm_coo.data).float()
    shape = adj_norm_coo.shape
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32).coalesce()


class GraphConvolution(nn.Module):
    """
    Simple GCN layer: supports sparse adjacency matrix multiplication.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        support = torch.sparse.mm(adj, x)  # [N, in_features]
        output = support @ self.weight     # [N, out_features]
        if self.bias is not None:
            output += self.bias
        return output


class GCN(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(in_features, hidden, bias=False)
        self.gc2 = GraphConvolution(hidden, out_features, bias=False)
        # Glorot initialization already done in GraphConvolution.reset_parameters

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


def main():
    # ----- Seed -----
    seed = int(os.environ.get("SEED", "0"))
    set_seeds(seed)

    # ----- Load data -----
    data = nio.load()
    features_np = data["features"]          # [2708, 1433] float32
    edges = data["edges"]                   # [E, 2] int64
    labels_np = data["labels"]              # [2708] int64, -1 for test
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    num_nodes = features_np.shape[0]
    num_classes = int(labels_np[labels_np >= 0].max()) + 1  # 7 for Cora

    # ----- Preprocess -----
    features = row_normalize_features(features_np)               # torch [N, F]
    labels = torch.from_numpy(labels_np).long()                  # torch [N]
    adj = normalize_adjacency(edges, num_nodes)                  # torch sparse [N, N]

    # ----- Model -----
    model = GCN(
        in_features=features.shape[1],
        hidden=CONFIG["hidden"],
        out_features=num_classes,
        dropout=CONFIG["dropout"],
    )

    # ----- Optimizer -----
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # ----- Smoke mode -----
    smoke = os.environ.get("SMOKE") == "1"
    max_epochs = 1 if smoke else CONFIG["epochs"]

    # ----- Training -----
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)                     # [N, C]
        loss_train = F.cross_entropy(logits[train_idx], labels[train_idx])
        # L2 penalty on first GCN layer weights
        l2_loss = torch.sum(model.gc1.weight ** 2)
        loss = loss_train + CONFIG["weight_decay"] * l2_loss
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_val = model(features, adj)
            loss_val = F.cross_entropy(logits_val[val_idx], labels[val_idx])
            preds_val = logits_val[val_idx].argmax(dim=1)
            acc_val = (preds_val == labels[val_idx]).float().mean().item()

        # Update best model
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_val_acc = acc_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= CONFIG["patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:03d} | "
                f"Train loss: {loss_train.item():.4f} | "
                f"Val loss: {loss_val.item():.4f} | "
                f"Val acc: {acc_val:.4f}"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # ----- Final inference -----
    model.eval()
    with torch.no_grad():
        final_logits = model(features, adj)   # [N, C]

    # ----- Output -----
    nio.write_predictions(final_logits)

    # ----- Print validation accuracy -----
    print(f"Validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
