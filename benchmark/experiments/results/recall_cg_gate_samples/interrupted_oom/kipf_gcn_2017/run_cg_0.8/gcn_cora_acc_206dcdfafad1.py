

# Plan step -> implemented in: preprocess_features
# Plan step -> implemented in: preprocess_adj
# Plan step -> implemented in: GCN class
# Plan step -> implemented in: dropout layers in GCN.forward
# Plan step -> implemented in: weight_decay loss added in training loop
# Plan step -> implemented in: Adam optimizer initialization
# Plan step -> implemented in: training loop using whole dataset
# Plan step -> implemented in: training loop with early stopping
# Plan step -> implemented in: validation evaluation inside training loop
# Plan step -> implemented in: final evaluation after loading best model
# Plan step -> implemented in: seed setting at start
# Plan step -> implemented in: optimizer without scheduler, only weight decay loss
# Plan step -> implemented in: overall model structure mirroring TF GCN
# Plan step -> implemented in: hardcoded full batch, 2-layer GCN, no attention
# Plan step -> implemented in: reading from CONFIG dict
# Plan step -> implemented in: call to nio.write_predictions

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {"lr": 0.01, "weight_decay": 0.0005, "epochs": 200, "dropout": 0.5, "hidden": 16, "patience": 10}

def preprocess_features(feat):
    """Row-normalize feature matrix."""
    rowsum = feat.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1
    return feat / rowsum

def preprocess_adj(edges, num_nodes):
    """Return symmetrically normalized adjacency with self-loops (dense)."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    # add self-loops
    adj += np.eye(num_nodes, dtype=np.float32)
    # degree matrix
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    # normalized adjacency
    adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
    return adj_norm

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        # x: [N, input_dim], adj: [N, N] (precomputed normalized)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.mm(adj, x)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.mm(adj, x)
        x = self.linear2(x)
        return x

def main():
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    smoke = os.environ.get("SMOKE") == "1"

    data = nio.load()
    features = data["features"].astype(np.float32)          # [2708, 1433]
    edges = data["edges"].astype(np.int64)                  # [E, 2]
    labels = data["labels"].astype(np.int64)                # [2708]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    N = features.shape[0]

    # Preprocess
    features_norm = preprocess_features(features)
    adj_norm = preprocess_adj(edges, N)

    # Torch tensors
    features_tensor = torch.from_numpy(features_norm)       # [N, 1433] float32
    adj_tensor = torch.from_numpy(adj_norm)                 # [N, N] float32
    labels_tensor = torch.from_numpy(labels)                # [N] int64

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    model = GCN(input_dim=features.shape[1],
                hidden_dim=CONFIG["hidden"],
                output_dim=7,
                dropout=CONFIG["dropout"])

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    epochs = 1 if smoke else CONFIG["epochs"]
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features_tensor, adj_tensor)
        loss_train = F.cross_entropy(logits[train_mask], labels_tensor[train_mask])
        # L2 weight decay on first layer only
        l2_reg = torch.tensor(0.0)
        for param in model.linear1.parameters():
            l2_reg += torch.norm(param, p=2) ** 2
        loss_train = loss_train + CONFIG["weight_decay"] * l2_reg
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features_tensor, adj_tensor)
            loss_val = F.cross_entropy(val_logits[val_mask], labels_tensor[val_mask])
            pred_val = val_logits[val_mask].argmax(dim=1)
            acc_val = (pred_val == labels_tensor[val_mask]).float().mean().item()

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= CONFIG["patience"]:
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation accuracy (using best model)
    model.eval()
    with torch.no_grad():
        final_logits = model(features_tensor, adj_tensor)
        final_pred = final_logits[val_mask].argmax(dim=1)
        final_acc = (final_pred == labels_tensor[val_mask]).float().mean().item()
    print(f"Validation accuracy: {final_acc:.4f}")

    # Predictions for all nodes
    model.eval()
    with torch.no_grad():
        all_logits = model(features_tensor, adj_tensor)
        all_pred = all_logits.argmax(dim=1).cpu().numpy()
    nio.write_predictions(all_pred)

if __name__ == "__main__":
    main()