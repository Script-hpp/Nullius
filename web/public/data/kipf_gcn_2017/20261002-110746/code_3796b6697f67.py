

# Plan step -> implemented in: function or class name
# 1. Model -> GCN class
# 2. Layers -> GCN._build_layers (implicit in __init__)
# 3. Loss -> training loop uses torch.nn.functional.cross_entropy with mask
# 4. Optimizer -> training loop uses torch.optim.Adam with weight decay on first layer
# 5. Weight init -> Glorot uniform via torch.nn.init.xavier_uniform_
# 6. Feature preprocessing -> row_normalize function (L1 norm)
# 7. Adjacency -> normalize_adj function (symmetrically normalized with self-loops)
# 8. Training -> train_model function with early stopping
# 9. Validation -> validation step inside train_model (uses val_idx for early stopping)
#10. Test -> predict function (logits[test_idx].argmax)
#11. Dropout -> dropout applied in GCN.forward (sparse dropout simulated via torch.dropout on dense)
#12. Metric -> masked_accuracy function (used for logging)
#13. Hyperparameters -> constants defined at top (lr=0.01, hidden=16, dropout=0.5, weight_decay=5e-4, epochs=200, patience=10)
#14. Random seed -> set_seeds function
#15. Output -> write_predictions function (handles smoke mode for val set)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def row_normalize(mx: np.ndarray) -> np.ndarray:
    """Row-normalize (L1) matrix."""
    rowsum = mx.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1e-12
    return mx / rowsum

def normalize_adj(adj: np.ndarray) -> np.ndarray:
    """Symmetrically normalize adjacency with self-loops: Â = D^{-1/2} (A+I) D^{-1/2}."""
    # Add self-loops
    adj = adj + np.eye(adj.shape[0], dtype=adj.shape[0] and np.float32)
    # Compute degree matrix
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
    # Create D^{-1/2}
    deg_inv_sqrt_mat = np.diag(deg_inv_sqrt)
    # Normalize
    return deg_inv_sqrt_mat @ adj @ deg_inv_sqrt_mat

class GCN(nn.Module):
    def __init__(self, nfeat, nhidden, nclass, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        # Layer 0: nfeat -> nhidden
        self.W0 = nn.Parameter(torch.Tensor(nfeat, nhidden))
        self.b0 = nn.Parameter(torch.Tensor(nhidden))
        # Layer 1: nhidden -> nclass
        self.W1 = nn.Parameter(torch.Tensor(nhidden, nclass))
        self.b1 = nn.Parameter(torch.Tensor(nclass))
        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.W0)
        zeros_(self.b0)
        xavier_uniform_(self.W1)
        zeros_(self.b1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # First layer: dropout on input, linear, adj* , ReLU
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.matmul(x, self.W0) + self.b0
        x = torch.matmul(adj, x)
        x = F.relu(x)
        # Second layer: dropout on hidden, linear, adj* , no activation
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.matmul(x, self.W1) + self.b1
        x = torch.matmul(adj, x)
        return x  # logits

def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: np.ndarray) -> float:
    """Accuracy on subset indicated by mask."""
    preds = logits.argmax(dim=1)
    correct = (preds[mask] == labels[mask]).float().sum()
    return correct.item() / mask.sum() * 100.0

def train_model(features, adj, labels, train_idx, val_idx, test_idx,
                lr=0.01, weight_decay=5e-4, epochs=200, patience=10, smoke=False):
    device = torch.device('cpu')
    features = torch.FloatTensor(features).to(device)
    adj = torch.FloatTensor(adj).to(device)
    labels = torch.LongTensor(labels).to(device)

    model = GCN(nfeat=features.shape[1], nhidden=16, nclass=int(labels.max().item())+1, dropout=0.5).to(device)
    # Apply weight decay only to first layer weights and biases
    optimizer = torch.optim.Adam([
        {'params': [model.W0, model.b0], 'weight_decay': weight_decay},
        {'params': [model.W1, model.b1], 'weight_decay': 0.0}
    ], lr=lr)

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    max_epochs = 1 if smoke else epochs

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)
        loss_train = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features, adj)
            loss_val = F.cross_entropy(logits[val_idx], labels[val_idx])

        if loss_val < best_val_loss - 1e-4:
            best_val_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        val_acc = masked_accuracy(logits, labels, val_idx)
    print(f"Validation accuracy: {val_acc:.2f}%")
    return model, logits.detach().cpu().numpy()

def write_predictions(path, ids, preds):
    """Write CSV with header id,prediction for given ids and predictions."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write("id,prediction\n")
        for i, p in zip(ids, preds):
            f.write(f"{int(i)},{int(p)}\n")

def main():
    seed = int(os.getenv('SEED', '0'))
    set_seeds(seed)

    data = np.load('/data/cora_public.npz')
    features = data['features'].astype(np.float32)          # (2708, 1433)
    edges = data['edges'].astype(np.int64)                  # (E, 2)
    labels = data['labels'].astype(np.int64)                # (2708,)
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    # Preprocess features: row-normalize (L1)
    features = row_normalize(features)

    # Build adjacency matrix (dense for simplicity)
    N = features.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    # Ensure symmetry (edges already contain both directions, but we make sure)
    adj = np.maximum(adj, adj.T)

    # Normalize adjacency with self-loops
    adj_norm = normalize_adj(adj)

    smoke = os.getenv('SMOKE', '0') == '1'

    model, logits = train_model(
        features, adj_norm, labels, train_idx, val_idx, test_idx,
        smoke=smoke
    )

    # Predictions for all nodes
    preds_all = logits.argmax(axis=1)

    # Test predictions (always full test set)
    test_pred = preds_all[test_idx]
    write_predictions('/output/predictions.csv', test_idx, test_pred)

    # Validation predictions
    if smoke:
        # In smoke mode output only first 20 ids of sorted val_idx
        sorted_val = np.sort(val_idx)
        first20 = sorted_val[:20]
        val_pred = preds_all[first20]
        write_predictions('/output/val_predictions.csv', first20, val_pred)
    else:
        val_pred = preds_all[val_idx]
        write_predictions('/output/val_predictions.csv', val_idx, val_pred)

if __name__ == '__main__':
    main()