

# Plan step -> implemented in: load_data
# Plan step -> implemented in: build_adj_norm
# Plan step -> implemented in: propagate_features
# Plan step -> implemented in: row_normalize
# Plan step -> implemented in: SGC class
# Plan step -> implemented in: train_model
# Plan step -> implemented in: evaluate
# Plan step -> implemented in: predict_and_output

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters as specified
CONFIG = {"lr": 0.2, "weight_decay": 5e-06, "epochs": 100, "dropout": 0, "hidden": 0}

def load_data():
    import nullius_io as nio
    d = nio.load()
    features = d['features'].astype(np.float32)          # [N, 1433]
    labels = d['labels'].astype(np.int64)               # [N], -1 for test
    train_idx = d['train_idx'].astype(np.int64)
    val_idx = d['val_idx'].astype(np.int64)
    test_idx = d['test_idx'].astype(np.int64)
    edges = d['edges'].astype(np.int64)                 # [E, 2]
    return features, labels, train_idx, val_idx, test_idx, edges

def build_adj_norm(edges, num_nodes):
    # Build adjacency (undirected, unweighted) from edge list
    idx = edges.T                                      # [2, E]
    vals = torch.ones(idx.shape[1], dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes)).coalesce()
    # Ensure binary entries (some edges may appear twice)
    adj_vals = adj._values()
    adj_vals[:] = (adj_vals > 0).float()
    adj = adj.coalesce()
    # Add self-loops
    self_idx = torch.stack([torch.arange(num_nodes, dtype=torch.long),
                            torch.arange(num_nodes, dtype=torch.long)])
    self_vals = torch.ones(num_nodes, dtype=torch.float32)
    adj_self = torch.sparse_coo_tensor(self_idx, self_vals, (num_nodes, num_nodes))
    adj = adj + adj_self
    adj = adj.coalesce()
    # Symmetrize (should already be symmetric, but ensure)
    adj_sym = adj + adj.t()
    adj_sym_vals = adj_sym._values()
    adj_sym_vals[:] = (adj_sym_vals > 0).float()
    adj = adj_sym.coalesce()
    # Compute D^{-1/2}
    degrees = torch.sparse.sum(adj, dim=1).to_dense() + 1e-12   # avoid zeros
    deg_inv_sqrt = degrees.pow(-0.5)
    deg_inv_sqrt_idx = torch.stack([torch.arange(num_nodes, dtype=torch.long),
                                    torch.arange(num_nodes, dtype=torch.long)])
    deg_inv_sqrt_mat = torch.sparse_coo_tensor(
        deg_inv_sqrt_idx, deg_inv_sqrt, (num_nodes, num_nodes))
    # Normalized adjacency: D^{-1/2} A D^{-1/2}
    left = torch.sparse.mm(deg_inv_sqrt_mat, adj)
    adj_norm = torch.sparse.mm(left, deg_inv_sqrt_mat)
    return adj_norm.coalesce()

def propagate_features(features_tensor, adj_norm, degree):
    x = features_tensor
    for _ in range(degree):
        x = torch.spmm(adj_norm, x)
    return x

def row_normalize(tensor):
    row_sum = tensor.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1.0
    return tensor / row_sum

class SGC(nn.Module):
    def __init__(self, nfeat, nclass, hidden=0, dropout=0):
        super(SGC, self).__init__()
        self.hidden = hidden
        self.dropout = dropout
        if hidden > 0:
            self.W1 = nn.Linear(nfeat, hidden)
            self.W2 = nn.Linear(hidden, nclass)
        else:
            self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        if self.hidden > 0:
            x = self.W1(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.W2(x)
        else:
            x = self.W(x)
        return x

def train_model(model, features, labels, train_idx, val_idx, epochs, lr, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features)
        loss = F.cross_entropy(output[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(features)
            val_pred = val_output[val_idx].argmax(dim=1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc

def evaluate(model, features, labels, idx):
    model.eval()
    with torch.no_grad():
        output = model(features)
        pred = output[idx].argmax(dim=1)
        acc = (pred == labels[idx]).float().mean().item()
    return acc, pred

def predict_and_output(model, features, num_nodes):
    model.eval()
    with torch.no_grad():
        logits = model(features)               # [num_nodes, nclass]
        preds = logits.argmax(dim=1)           # [num_nodes]
    return preds

def main():
    # Seed
    seed = int(os.environ.get('SEED', '42'))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Smoke mode
    smoke = os.environ.get('SMOKE') == '1'

    # Load data
    features_np, labels_np, train_idx, val_idx, test_idx, edges = load_data()
    num_nodes = features_np.shape[0]
    nfeat = features_np.shape[1]
    nclass = int(labels_np.max()) + 1  # 7

    # Torch tensors
    features = torch.from_numpy(features_np)
    labels = torch.from_numpy(labels_np)
    train_idx = torch.from_numpy(train_idx)
    val_idx = torch.from_numpy(val_idx)
    test_idx = torch.from_numpy(test_idx)
    edges_tensor = torch.from_numpy(edges)

    # Build normalized adjacency
    adj_norm = build_adj_norm(edges_tensor, num_nodes)

    # Propagate features (degree default 2)
    degree = 2
    propagated = propagate_features(features, adj_norm, degree)
    propagated = row_normalize(propagated)

    # Model
    model = SGC(nfeat=nfeat, nclass=nclass,
                hidden=CONFIG["hidden"], dropout=CONFIG["dropout"])

    # Training
    epochs = 1 if smoke else CONFIG["epochs"]
    model, val_acc = train_model(
        model, propagated, labels, train_idx, val_idx,
        epochs=epochs,
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    # Final validation accuracy (from best model)
    final_val_acc, _ = evaluate(model, propagated, labels, val_idx)
    print(f"Validation accuracy: {final_val_acc:.4f}")

    # Predictions for all nodes
    all_preds = predict_and_output(model, propagated, num_nodes)
    # Convert to numpy for helper
    import nullius_io as nio
    nio.write_predictions(all_preds.numpy())

if __name__ == "__main__":
    main()