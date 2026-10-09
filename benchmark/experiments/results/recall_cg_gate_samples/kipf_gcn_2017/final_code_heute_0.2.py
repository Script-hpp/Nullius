# Plan step -> implemented in: function or class name
# Load data -> load_data
# Build symmetric adjacency with self-loops and normalize -> build_normalized_adj
# Seed numpy & torch -> set_seed
# Model definition -> GCN
# Training loop with early stopping -> train_model
# Validation accuracy printing -> print_validation_accuracy
# Write predictions -> nio.write_predictions

import os
import numpy as np
import torch
import torch.nn.functional as F
import nullius_io as nio

def load_data():
    """Load Cora data from nullius_io."""
    d = nio.load()
    features = torch.from_numpy(d['features'])          # [N, F] float32
    edges = torch.from_numpy(d['edges'])                # [E, 2] int64
    labels = torch.from_numpy(d['labels'])              # [N] int64, -1 for test
    train_idx = torch.from_numpy(d['train_idx'])        # [140]
    val_idx   = torch.from_numpy(d['val_idx'])          # [500]
    test_idx  = torch.from_numpy(d['test_idx'])         # [1000]
    return features, edges, labels, train_idx, val_idx, test_idx

def set_seed(seed):
    """Seed numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_normalized_adj(edge_index, num_nodes):
    """
    edge_index: Tensor [2, E] (directed edges, already contains both directions)
    Returns normalized adjacency Â = D^{-1/2} (A+I) D^{-1/2} as a sparse torch tensor.
    """
    # Add self-loops
    self_loop = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
    self_loop_edges = torch.stack([self_loop, self_loop], dim=0)  # [2, N]
    edge_all = torch.cat([edge_index, self_loop_edges], dim=1)    # [2, E+N]
    vals = torch.ones(edge_all.size(1), dtype=torch.float32, device=edge_index.device)

    # Sparse adjacency with self-loops
    A_plus_I = torch.sparse_coo_tensor(edge_all, vals, (num_nodes, num_nodes)).coalesce()

    # Degree vector
    deg = torch.sparse.sum(A_plus_I, dim=1).to_dense()  # [N]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Normalize: Â = D^{-1/2} (A+I) D^{-1/2}
    row = A_plus_I.indices()[0]
    col = A_plus_I.indices()[1]
    norm_vals = A_plus_I.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    A_hat = torch.sparse_coo_tensor(A_plus_I.indices(), norm_vals, (num_nodes, num_nodes)).coalesce()
    return A_hat

class GCN(torch.nn.Module):
    """Two-layer GCN as described in the plan."""
    def __init__(self, in_features, hidden, out_classes, dropout=0.5):
        super().__init__()
        self.W0 = torch.nn.Parameter(torch.empty(in_features, hidden))
        self.W1 = torch.nn.Parameter(torch.empty(hidden, out_classes))
        self.dropout = torch.nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W0)
        torch.nn.init.xavier_uniform_(self.W1)

    def forward(self, x, adj_hat):
        # Layer 0: ReLU(Â X W0)
        h = torch.mm(x, self.W0)          # [N, hidden]
        h = torch.spmm(adj_hat, h)        # [N, hidden]
        h = torch.relu(h)
        h = self.dropout(h)

        # Layer 1: Â H W1 (logits)
        out = torch.mm(h, self.W1)        # [N, classes]
        out = torch.spmm(adj_hat, out)    # [N, classes]
        return out

def train_model(model, features, adj_hat, labels, train_idx, val_idx, smoke=False):
    """Train the GCN with early stopping; return trained model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    max_epochs = 1 if smoke else 200
    patience = 100 if not smoke else None
    best_val_loss = float('inf')
    epochs_no_improve = 0

    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        logits = model(features, adj_hat)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj_hat)
            val_loss = F.cross_entropy(val_logits[val_idx], labels[val_idx])
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= patience:
                break

    # Restore best model
    model.load_state_dict(best_state)
    return model

def print_validation_accuracy(model, features, adj_hat, labels, val_idx):
    """Compute and print validation accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_hat)
        preds = logits[val_idx].argmax(dim=1)
        acc = (preds == labels[val_idx]).float().mean().item()
    print(f"Validation accuracy: {acc:.4f}")

def main():
    # Seed
    seed = int(os.getenv('SEED', '0'))
    set_seed(seed)

    # Load data
    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features.shape[0]

    # Build normalized adjacency
    edge_index = edges.t()  # [2, E]
    adj_hat = build_normalized_adj(edge_index, num_nodes)

    # Model
    model = GCN(in_features=features.shape[1], hidden=16, out_classes=7, dropout=0.5)

    # Smoke mode?
    smoke = getattr(nio, 'SMOKE', False)
    model = train_model(model, features, adj_hat, labels, train_idx, val_idx, smoke=smoke)

    # Validation accuracy
    print_validation_accuracy(model, features, adj_hat, labels, val_idx)

    # Predictions for all nodes
    model.eval()
    with torch.no_grad():
        logits_all = model(features, adj_hat)
        preds_all = logits_all.argmax(dim=1).cpu().numpy()  # shape [N], int64

    # Write predictions (helper handles test/val selection)
    nio.write_predictions(preds_all)

if __name__ == '__main__':
    main()
