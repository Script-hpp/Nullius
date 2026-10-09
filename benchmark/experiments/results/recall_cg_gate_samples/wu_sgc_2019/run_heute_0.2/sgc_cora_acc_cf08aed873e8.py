# Plan step -> implemented in: function or class name
# Read SEED from os.environ, set np.random.seed(seed) and torch.manual_seed(seed) -> set_seeds
# Load /data/cora_public.npz; obtain features, edges, labels, and splits -> load_data
# Build adjacency matrix A (add self-loops → Ã = A + I), compute degree vector d, and normalize: Â = D⁻¹ᐟ² Ã D⁻¹ᐟ² (sparse) -> build_normalized_adjacency
# Choose propagation steps K = 2; compute propagated features H = Âᵏ · features (use sparse‑dense multiplication) -> propagate_features
# Initialize a single linear layer W: torch.nn.Linear(H.shape[1], num_classes, bias=False) -> LogisticRegression
# Loss: torch.nn.CrossEntropyLoss() -> criterion
# Optimizer: Adam with lr = 0.2, weight_decay = 5e‑4 -> configure_optimizer
# Training loop: max 200 epochs; after each epoch evaluate accuracy on val_idx -> train_model
# Early stopping: stop if validation accuracy does not improve for 50 epochs; restore best weights -> early stopping logic inside train_model
# After training, predict logits for all nodes, take argmax to get class labels -> get_predictions
# Write /output/predictions.csv with header id,prediction; rows correspond to test_idx in order -> nio.write_predictions (called externally)

import os
import numpy as np
import torch
import torch.nn as nn
import nullius_io as nio

def set_seeds():
    seed = int(os.environ.get('SEED', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    d = nio.load()  # returns dict-like with numpy arrays
    features = torch.from_numpy(d['features']).float()          # [N, F]
    edges = torch.from_numpy(d['edges']).long()                 # [E, 2]
    labels = torch.from_numpy(d['labels']).long()               # [N]
    train_idx = torch.from_numpy(d['train_idx']).long()
    val_idx = torch.from_numpy(d['val_idx']).long()
    test_idx = torch.from_numpy(d['test_idx']).long()
    return features, edges, labels, train_idx, val_idx, test_idx

def build_normalized_adjacency(edges, num_nodes):
    # Add self-loops
    self_loop = torch.arange(num_nodes, device=edges.device).unsqueeze(1).repeat(1, 2)
    all_edges = torch.cat([edges, self_loop], dim=0)            # [E+ N, 2]
    # Remove possible duplicates? Keep as is; assume edges already contain both directions.
    idx = all_edges.t()                                         # [2, E+ N]
    values = torch.ones(idx.size(1), dtype=torch.float32, device=edges.device)
    # Compute degree
    deg = torch.sparse_coo_tensor(idx, values, (num_nodes, num_nodes)).sum(dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    # Normalize values
    row, col = idx[0], idx[1]
    norm_values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    # Build normalized adjacency sparse tensor
    adj_norm = torch.sparse_coo_tensor(idx, norm_values, (num_nodes, num_nodes))
    return adj_norm.coalesce()

def propagate_features(adj_norm, features, K=2):
    H = features
    for _ in range(K):
        H = torch.sparse.mm(adj_norm, H)
    return H

class LogisticRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x):
        # x: [N, in_features]; weight: [out, in]
        return x @ self.weight.t()   # [N, out]

def configure_optimizer(model, lr=0.2, weight_decay=5e-4):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def train_model(model, H, labels, train_idx, val_idx, max_epochs, patience):
    criterion = nn.CrossEntropyLoss()
    optimizer = configure_optimizer(model)
    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(H[train_idx])
        loss = criterion(logits, labels[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(H[val_idx])
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc

def get_predictions(model, H):
    model.eval()
    with torch.no_grad():
        logits = model(H)          # [N, num_classes]
        pred = logits.argmax(dim=1)
    return pred

def main():
    set_seeds()
    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features.size(0)
    num_classes = int(labels.max().item()) + 1  # labels are 0..6, -1 for test but max ignores -1? safer: 7
    # However test labels are -1, but we know there are 7 classes.
    num_classes = 7

    adj_norm = build_normalized_adjacency(edges, num_nodes)
    H = propagate_features(adj_norm, features, K=2)

    model = LogisticRegression(H.size(1), num_classes)

    smoke = os.environ.get('SMOKE') == '1'
    max_epochs = 1 if smoke else 200
    patience = 50 if not smoke else 1  # irrelevant in smoke

    val_acc = train_model(model, H, labels, train_idx, val_idx, max_epochs, patience)
    print(f'Validation accuracy: {val_acc:.4f}')

    preds = get_predictions(model, H)  # tensor of shape [N]
    # nio.write_predictions expects predictions for ALL nodes (logits or labels)
    nio.write_predictions(preds)

if __name__ == '__main__':
    main()
