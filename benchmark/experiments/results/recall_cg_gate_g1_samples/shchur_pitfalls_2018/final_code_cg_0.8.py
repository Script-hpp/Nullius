import os
import numpy as np
import torch
import nullius_io as nio
import scipy.sparse as sp
import torch.nn.functional as F

# Plan step 1 -> implemented in: load_data
# Plan step 2 -> implemented in: preprocess_adj, preprocess_features
# Plan step 3 -> implemented in: GCN class
# Plan step 4 -> implemented in: train_model
# Plan step 5 -> implemented in: get_predictions
# Plan step 6 -> implemented in: nio.write_predictions

CONFIG = {"lr": 0.01, "weight_decay": 0.001, "epochs": 100000, "dropout": 0.8, "hidden": 64, "layers": 2, "patience": 50}
CONFIG["normalize_features"] = False  # from train.conf.yaml


def load_data():
    d = nio.load()
    features = d["features"].astype(np.float32)          # [N, 1433]
    edges = d["edges"].astype(np.int64)                  # [E, 2] (both directions)
    labels = d["labels"].astype(np.int64)                # [N] with -1 for test
    train_idx = d["train_idx"]
    val_idx = d["val_idx"]
    test_idx = d["test_idx"]
    return features, edges, labels, train_idx, val_idx, test_idx


def preprocess_adj(edges, num_nodes):
    # Build adjacency matrix (undirected, both directions already present)
    adj = sp.csr_matrix((np.ones(len(edges), dtype=np.float32),
                         (edges[:, 0], edges[:, 1])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    # Add self-loops
    adj = adj + sp.identity(num_nodes, dtype=np.float32)
    # Normalize: Â = D̃^{-0.5} Ã D̃^{-0.5}
    degs = np.array(adj.sum(axis=1)).flatten()
    inv_sqrt_deg = np.power(degs, -0.5)
    inv_sqrt_deg[np.isinf(inv_sqrt_deg)] = 0.0
    inv_sqrt_deg_mat = sp.diags(inv_sqrt_deg)
    adj_norm = inv_sqrt_deg_mat @ adj @ inv_sqrt_deg_mat
    # Convert to torch sparse tensor
    adj_norm = adj_norm.tocoo()
    indices = torch.from_numpy(
        np.vstack([adj_norm.row, adj_norm.col]).astype(np.int64)
    )
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    shape = torch.Size(adj_norm.shape)
    adj_torch = torch.sparse_coo_tensor(indices, values, shape).coalesce()
    return adj_torch


def preprocess_features(features, normalize):
    if normalize:
        # Row-normalize to L2 norm = 1
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        features = features / norms
    return torch.from_numpy(features.astype(np.float32))


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes,
                 dropout, weight_decay, num_layers):
        super().__init__()
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.num_layers = num_layers

        # Build linear layers
        self.linears = torch.nn.ModuleList()
        if num_layers == 1:
            self.linears.append(torch.nn.Linear(num_features, num_classes))
        else:
            # first layer
            self.linears.append(torch.nn.Linear(num_features, hidden))
            # hidden layers (none for layers==2)
            for _ in range(num_layers - 2):
                self.linears.append(torch.nn.Linear(hidden, hidden))
            # final layer
            self.linears.append(torch.nn.Linear(hidden, num_classes))

        # Initialize weights with Glorot uniform, biases zero
        for lin in self.linears:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)

    def forward(self, x, adj):
        # x: dense [N, F]
        for i, lin in enumerate(self.linears):
            # Apply dropout to inputs if training
            if self.training:
                x = F.dropout(x, p=self.dropout, training=True)
            # Linear transformation
            support = torch.add(torch.mm(x, lin.weight.t()), lin.bias)
            # Graph convolution (Â * support)
            x = torch.sparse.mm(adj, support)
            # Activation: ReLU for all but last layer
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x


def train_model(model, features, adj, labels_tensor,
                train_idx, val_idx, optimizer, device):
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(CONFIG["epochs"]):
        model.train()
        optimizer.zero_grad()
        logits = model(features.to(device), adj.to(device))
        loss_train = F.cross_entropy(
            logits[train_idx], labels_tensor[train_idx]
        )
        # L2 regularization on first-layer weights only
        l2_reg = CONFIG["weight_decay"] * torch.sum(
            model.linears[0].weight.pow(2)
        )
        loss = loss_train + l2_reg
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_v = model(features.to(device), adj.to(device))
            loss_val = F.cross_entropy(
                logits_v[val_idx], labels_tensor[val_idx]
            )
        # Early stopping (gnnbench criterion: stop if no improvement for patience epochs)
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG["patience"]:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final validation accuracy (optional, just for info)
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device), adj.to(device))
        pred = logits[val_idx].argmax(dim=1)
        acc = (pred == labels_tensor[val_idx]).float().mean().item()
    print(f"Validation accuracy: {acc:.4f}")
    return acc


def get_predictions(model, features, adj, device):
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device), adj.to(device))
    return logits.argmax(dim=1).cpu().numpy()


def main():
    # Seed
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")

    # Load data
    features_np, edges, labels_np, train_idx, val_idx, test_idx = load_data()
    num_nodes = features_np.shape[0]
    num_classes = int(labels_np[labels_np >= 0].max() + 1)  # 7

    # Preprocess
    adj = preprocess_adj(edges, num_nodes)
    features = preprocess_features(features_np, CONFIG["normalize_features"])
    labels_tensor = torch.from_numpy(labels_np)  # keep -1 for test, but we mask

    # Model
    model = GCN(num_features=features.shape[1],
                hidden=CONFIG["hidden"],
                num_classes=num_classes,
                dropout=CONFIG["dropout"],
                weight_decay=CONFIG["weight_decay"],
                num_layers=CONFIG["layers"]).to(device)

    # Optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # Smoke mode: exactly 1 epoch
    if os.environ.get("SMOKE") == "1":
        CONFIG["epochs"] = 1

    # Train
    train_model(model, features, adj, labels_tensor,
                train_idx, val_idx, optimizer, device)

    # Predictions for all nodes
    preds = get_predictions(model, features, adj, device)
    nio.write_predictions(preds)


if __name__ == "__main__":
    main()
