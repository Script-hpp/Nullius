

# Plan step -> implemented in: load_data
# Plan step -> implemented in: preprocess
# Plan step -> implemented in: class GCN
# Plan step -> implemented in: train_model
# Plan step -> implemented in: predict_and_write
# Plan step -> implemented in: smoke_mode

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {"lr": 0.01, "epochs": 100000, "layers": 2, "patience": 50}
CONFIG["hidden_size"] = 64
CONFIG["dropout"] = 0.8
CONFIG["weight_decay"] = 0.001

def load_data():
    d = nio.load()
    features = d['features']          # np.ndarray [N, 1433] float32
    edges = d['edges']                # np.ndarray [E, 2] int64
    labels = d['labels']              # np.ndarray [N] int64, -1 for test
    train_idx = d['train_idx']
    val_idx = d['val_idx']
    test_idx = d['test_idx']
    return features, edges, labels, train_idx, val_idx, test_idx

def preprocess(features, edges, num_nodes):
    # Build adjacency matrix (COO) from directed edge list (both directions present)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    # Add self-loops
    adj += np.eye(num_nodes, dtype=np.float32)
    # Degree vector
    deg = np.array(adj.sum(axis=1))
    # Inverse sqrt degree, handle zeros safely
    deg_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
    mask = deg > 0
    deg_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    # Normalize: D^{-1/2} A D^{-1/2}
    adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
    # Convert to torch sparse tensor (coalesced)
    adj_coo = np.where(adj_norm != 0)
    indices = torch.from_numpy(np.vstack([adj_coo[0], adj_coo[1]]).astype(np.int64))
    values = torch.from_numpy(adj_norm[adj_coo].astype(np.float32))
    adj_t = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
    # Features as dense tensor (no row normalization)
    feat_t = torch.from_numpy(features).float()
    return adj_t, feat_t

class GCN(nn.Module):
    def __init__(self, adj, feat_dim, hidden_dim, out_dim, dropout, weight_decay, num_layers):
        super().__init__()
        self.adj = adj
        self.dropout = dropout
        self.num_layers = num_layers

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        in_dim = feat_dim
        for i in range(num_layers):
            out_dim_layer = hidden_dim if i < num_layers - 1 else out_dim
            self.weights.append(nn.Parameter(torch.empty(in_dim, out_dim_layer)))
            self.biases.append(nn.Parameter(torch.zeros(out_dim_layer)))
            in_dim = out_dim_layer

        # Initialize weights with Glorot uniform
        for w in self.weights:
            nn.init.xavier_uniform_(w)
        # Biases already zero

    def forward(self, x):
        # x: dense feature tensor [N, feat_dim]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # Apply dropout to inputs if training
            if self.training and self.dropout > 0.0:
                x = F.dropout(x, p=self.dropout, training=True)
            # Graph convolution: Â * (x * W) + b
            support = torch.mm(x, W)          # [N, out_dim]
            x = torch.sparse.mm(self.adj, support) + b   # [N, out_dim]
            if i < self.num_layers - 1:       # activation for all but last layer
                x = F.relu(x)
        return x

def load_data_and_preprocess():
    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features.shape[0]
    adj_t, feat_t = preprocess(features, edges, num_nodes)
    labels_tensor = torch.from_numpy(labels).long()
    train_idx_t = torch.from_numpy(train_idx).long()
    val_idx_t = torch.from_numpy(val_idx).long()
    test_idx_t = torch.from_numpy(test_idx).long()
    return adj_t, feat_t, labels_tensor, train_idx_t, val_idx_t, test_idx_t, features.shape[1]

def train_model():
    adj_t, feat_t, labels_tensor, train_idx_t, val_idx_t, test_idx_t, in_dim = load_data_and_preprocess()
    # Determine number of classes (ignore -1)
    valid_labels = labels_tensor[labels_tensor >= 0]
    num_classes = int(valid_labels.max().item()) + 1

    num_layers = CONFIG["layers"]
    hidden_dim = CONFIG["hidden_size"]
    dropout = CONFIG["dropout"]
    weight_decay = CONFIG["weight_decay"]

    model = GCN(adj_t, in_dim, hidden_dim, num_classes,
                dropout, weight_decay, num_layers)

    # Optimizer: weight decay only on first layer weights
    optimizer = torch.optim.Adam([
        {"params": [model.weights[0]], "weight_decay": weight_decay},
        {"params": [model.biases[0]], "weight_decay": 0.0},
        {"params": list(model.weights[1:]) + list(model.biases[1:]), "weight_decay": 0.0}
    ], lr=CONFIG["lr"])

    best_val_loss = float('inf')
    best_state = None
    epochs_without_improve = 0

    # Smoke mode: exactly one epoch
    smoke = os.environ.get("SMOKE") == "1"
    max_epochs = 1 if smoke else CONFIG["epochs"]

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(feat_t)
        loss = F.cross_entropy(logits[train_idx_t], labels_tensor[train_idx_t])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(feat_t)
            val_loss = F.cross_entropy(val_logits[val_idx_t], labels_tensor[val_idx_t])
            _, preds = torch.max(val_logits[val_idx_t], dim=1)
            val_acc = (preds == labels_tensor[val_idx_t]).float().mean().item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= CONFIG["patience"]:
            break

    # Restore best validation state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on validation set (optional print)
    model.eval()
    with torch.no_grad():
        final_logits = model(feat_t)
        final_preds = torch.argmax(final_logits[val_idx_t], dim=1)
        final_acc = (final_preds == labels_tensor[val_idx_t]).float().mean().item()
    print(f"Validation accuracy: {final_acc:.4f}")

    # Return logits for all nodes
    with torch.no_grad():
        all_logits = model(feat_t)
        return all_logits.cpu()

def predict_and_write():
    logits_or_preds = train_model()
    # nio.write_predictions accepts either logits [N, C] or class indices [N]
    nio.write_predictions(logits_or_preds.numpy())

if __name__ == "__main__":
    # Seed numpy and torch from environment variable SEED
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    predict_and_write()