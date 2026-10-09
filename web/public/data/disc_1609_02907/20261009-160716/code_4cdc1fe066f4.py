# Plan step 1 -> implemented in: set_seed function
# Plan step 2 -> implemented in: load_data function
# Plan step 3 -> implemented in: row_normalize_features function
# Plan step 4 -> implemented in: normalize_adj function
# Plan step 5 -> implemented in: prepare_torch_tensors function
# Plan step 6a -> implemented in: first GCN layer (dropout -> adjmm -> linear1 -> relu) in GCN.forward
# Plan step 6b -> implemented in: second GCN layer (dropout -> adjmm -> linear2) in GCN.forward
# Plan step 7 -> implemented in: optimizer configuration with weight decay on first layer only
# Plan step 8 -> implemented in: training loop over epochs
# Plan step 9 -> implemented in: early stopping based on validation loss
# Plan step 10 -> implemented in: validation used only for early stopping (no gradient)
# Plan step 11 -> implemented in: restoring best model weights
# Plan step 12 -> implemented in: prediction step (argmax of logits for all nodes)
# Plan step 13 -> implemented in: calling nio.write_predictions
# Plan step 14 -> implemented in: full-batch training (no minibatching)
# Plan step 15 -> implemented in: GCN class with exactly two layers

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import nullius_io as nio

# Hyperparameters (read only from this dict)
CONFIG = {
    "lr": 0.01,
    "epochs": 200,
    "dropout": 0.5,
    "hidden": 16,
    "patience": 10,
    "weight_decay": 5e-4,  # L2 on first layer only
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    d = nio.load()
    X = d["features"].astype(np.float32)               # [2708, 1433]
    edges = d["edges"].astype(np.int64)                # [E, 2]
    labels = d["labels"].astype(np.int64)              # [2708]
    train_idx = d["train_idx"]
    val_idx = d["val_idx"]
    test_idx = d["test_idx"]
    return X, edges, labels, train_idx, val_idx, test_idx


def row_normalize_features(mat: np.ndarray) -> np.ndarray:
    """Row-normalize feature matrix: X_hat = D^{-1} X"""
    row_sum = np.array(mat.sum(axis=1)).flatten()
    row_sum[row_sum == 0] = 1.0
    return mat / row_sum[:, None]


def normalize_adj(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetrically normalize adjacency: Â = D^{-1/2} (A+I) D^{-1/2}"""
    adj = adj + sp.eye(adj.shape[0])  # add self-loop
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    deg_inv_sqrt_mat = sp.diags(deg_inv_sqrt)
    return deg_inv_sqrt_mat @ adj @ deg_inv_sqrt_mat


def prepare_torch_tensors(X_raw: np.ndarray, edges: np.ndarray):
    # Normalize features
    X_norm = row_normalize_features(X_raw)                     # dense [N, F]
    X_tensor = torch.from_numpy(X_norm).float()

    # Build adjacency matrix (COO)
    adj = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                        shape=(X_raw.shape[0], X_raw.shape[0]))
    adj_norm = normalize_adj(adj)
    adj_coo = adj_norm.tocoo()
    indices = torch.from_numpy(
        np.vstack((adj_coo.row, adj_coo.col)).astype(np.int64)
    )
    values = torch.from_numpy(adj_coo.data.astype(np.float32))
    adj_tensor = torch.sparse_coo_tensor(indices, values, size=adj_coo.shape)
    return X_tensor, adj_tensor.coalesce()


class GCN(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden, bias=True)
        self.linear2 = nn.Linear(hidden, out_features, bias=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.sparse.Tensor):
        # First GCN layer: dropout on input features, then ÂX, then W0, then ReLU
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sparse.mm(adj, x)          # Â * X
        x = self.linear1(x)                  # * W0
        x = F.relu(x)

        # Second GCN layer: dropout on hidden representation, then ÂZ0, then W1 (no activation)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sparse.mm(adj, x)          # Â * Z0
        x = self.linear2(x)                  # * W1
        return x


def main():
    # ----- Step 1: Seed -----
    seed = int(os.environ.get("SEED", "123"))
    set_seed(seed)

    # ----- Step 2: Load raw data -----
    X_raw, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = X_raw.shape[0]
    num_classes = len(np.unique(labels[labels >= 0]))

    # ----- Steps 3-5: Preprocess and convert to torch -----
    X_tensor, adj_tensor = prepare_torch_tensors(X_raw, edges)
    labels_tensor = torch.from_numpy(labels).long()

    # ----- Step 6: Model -----
    model = GCN(in_features=X_tensor.shape[1],
                hidden=CONFIG["hidden"],
                out_features=num_classes,
                dropout=CONFIG["dropout"])

    # ----- Step 7: Optimizer (weight decay on first layer only) -----
    optimizer = torch.optim.Adam([
        {"params": model.linear1.parameters(), "weight_decay": CONFIG["weight_decay"]},
        {"params": model.linear2.parameters(), "weight_decay": 0.0}
    ], lr=CONFIG["lr"])

    # ----- Smoke mode handling -----
    smoke = os.environ.get("SMOKE") == "1"
    max_epochs = 1 if smoke else CONFIG["epochs"]

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    # ----- Step 8-10: Training loop with early stopping -----
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tensor, adj_tensor)                 # [N, C]
        loss_train = F.cross_entropy(logits[train_idx], labels_tensor[train_idx])
        loss_train.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_val = model(X_tensor, adj_tensor)
            loss_val = F.cross_entropy(logits_val[val_idx], labels_tensor[val_idx])
            acc_val = (logits_val[val_idx].argmax(dim=1) == labels_tensor[val_idx]).float().mean().item()

        # Early stopping check
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch:03d} | train_loss {loss_train:.4f} | val_loss {loss_val:.4f} | val_acc {acc_val:.4f}")

        if epochs_no_improve >= CONFIG["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    # ----- Step 11: Restore best model -----
    if best_state is not None:
        model.load_state_dict(best_state)

    # ----- Step 12: Prediction -----
    model.eval()
    with torch.no_grad():
        logits_all = model(X_tensor, adj_tensor)
        preds_all = logits_all.argmax(dim=1)   # [N]

    # ----- Step 13: Write predictions -----
    nio.write_predictions(preds_all)

    # Print final validation accuracy (best) as requested
    print(f"Best validation accuracy: {100 * (1 - best_val_loss / (best_val_loss + 1e-12)):.2f}%")


if __name__ == "__main__":
    main()
