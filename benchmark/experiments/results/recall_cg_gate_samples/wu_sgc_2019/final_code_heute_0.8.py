# Plan step -> implemented in: function or class name
# 1. Read SEED from os.environ, set seeds -> set_seeds()
# 2. Load data via nullius_io -> load_data()
# 3. Build adjacency matrix A (add self-loops), compute degree, normalize -> build_normalized_adj()
# 4. Choose propagation steps K=2; compute propagated features H = Âᵏ·features -> propagate_features()
# 5. Initialize a single linear layer W -> LinearClassifier
# 6. Loss: CrossEntropyLoss -> criterion
# 7. Optimizer: Adam with lr=0.2, weight_decay=5e-4 -> configure_optimizer()
# 8. Training loop with early stopping -> train_model()
# 9. Predict logits for all nodes, take argmax -> get_predictions()
#10. Write predictions via nullius_io -> nio.write_predictions()
#11. Print final validation accuracy -> printed in train_model()

import os
import numpy as np
import torch
import nullius_io as nio

def set_seeds():
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    d = nio.load()  # dict with keys: features, edges, labels, train_idx, val_idx, test_idx
    features = d['features'].astype(np.float32)          # [2708, 1433]
    edges = d['edges'].astype(np.int64)                  # [E, 2]
    labels = d['labels'].astype(np.int64)                # [2708]
    train_idx = d['train_idx']
    val_idx = d['val_idx']
    test_idx = d['test_idx']
    return features, edges, labels, train_idx, val_idx, test_idx

def build_normalized_adj(edges, num_nodes):
    """
    Returns normalized adjacency Â = D^{-1/2} (A + I) D^{-1/2} as a sparse torch tensor.
    """
    # self-loop indices
    self_idx = torch.arange(num_nodes, dtype=torch.long)
    self_idx = self_idx.repeat(2, 1)  # shape [2, N]

    # original edges (already contain both directions)
    edge_idx = torch.from_numpy(edges).t()  # shape [2, E]

    # combine
    idx = torch.cat([edge_idx, self_idx], dim=1)  # [2, E+N]
    vals = torch.ones(idx.size(1), dtype=torch.float32)

    # sparse adjacency with self-loops
    adj = torch.sparse_coo_tensor(idx, vals, size=(num_nodes, num_nodes)).coalesce()

    # degree vector (sum of rows)
    deg = torch.sparse.sum(adj, dim=1).to_dense()  # [N]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0  # isolated nodes -> 0

    return adj, deg_inv_sqrt

def propagate_features(features, adj, deg_inv_sqrt, K):
    """
    Compute H = Â^K * features using sparse-dense multiplications.
    """
    X = torch.from_numpy(features).float()  # [N, F]
    for _ in range(K):
        # right multiplication by D^{-1/2}
        X = X * deg_inv_sqrt[:, None]
        # Â * X = A~ * (D^{-1/2} X)  (we already have A~ in adj without extra D^{-1/2})
        X = torch.sparse.mm(adj, X)
        # left multiplication by D^{-1/2}
        X = X * deg_inv_sqrt[:, None]
    return X

class LinearClassifier(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)

def configure_optimizer(model, lr=0.2, weight_decay=5e-4):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def train_model(model, features, labels, train_idx, val_idx, optimizer, max_epochs, patience):
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features)
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred[val_idx] == labels[val_idx]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_acc

def get_predictions(model, features):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        preds = logits.argmax(dim=1)
    return preds

def main():
    set_seeds()
    features_np, edges_np, labels_np, train_idx, val_idx, test_idx = load_data()
    num_nodes = features_np.shape[0]
    num_labels = int(labels_np.max()) + 1  # assuming labels 0..6

    # Build normalized adjacency
    adj, deg_inv_sqrt = build_normalized_adj(edges_np, num_nodes)

    # Propagate features (K=2)
    K = 2
    propagated = propagate_features(features_np, adj, deg_inv_sqrt, K)  # torch tensor [N, F]

    # Model
    model = LinearClassifier(propagated.shape[1], num_labels)

    # Optimizer
    optimizer = configure_optimizer(model, lr=0.2, weight_decay=5e-4)

    # Determine epochs based on smoke mode
    smoke = os.getenv("SMOKE") == "1"
    max_epochs = 1 if smoke else 200
    patience = 50 if not smoke else 1  # irrelevant for smoke but kept

    # Train
    val_acc = train_model(model, propagated,
                          torch.from_numpy(labels_np).long(),
                          train_idx, val_idx,
                          optimizer, max_epochs, patience)

    # Get predictions for all nodes
    preds = get_predictions(model, propagated)  # torch tensor [N]

    # Output via helper (expects numpy array or torch tensor)
    nio.write_predictions(preds)

    # Print final validation accuracy
    print(f"Validation accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()
