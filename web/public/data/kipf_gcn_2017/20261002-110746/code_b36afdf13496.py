# Plan step -> implemented in: function or class name
# 1. Model: 2-layer GCN -> class GCN
# 2. Layers: (1) 1433→16, ReLU, dropout=0.5; (2) 16→7, linear, dropout=0.5 -> GCN.__init__, GCN.forward
# 3. Loss: Masked cross-entropy + L2 on first-layer weights -> train_loop
# 4. Optimizer: Adam lr=0.01 full-batch -> optimizer initialization
# 5. Weight init: Glorot uniform, zeros bias -> GCN.reset_parameters
# 6. Feature preprocessing: Row-normalize (L1) -> preprocess_features
# 7. Adjacency: add self-loops, compute Â -> preprocess_adj
# 8. Training: max 200 epochs, early stopping patience=10 -> train_loop
# 9. Validation: val_idx for early stopping (no training) -> validation loss/compute
# 10. Test: predict on test_idx -> get_predictions
# 11. Dropout: applied to inputs of each layer (rate 0.5) -> dropout in GCN.forward
# 12. Metric: masked accuracy -> compute_accuracy
# 13. Hyperparameters: hidden=16, dropout=0.5, lr=0.01, L2=5e-4, epochs=200, early_stop=10 -> constants
# 14. Random seed: read SEED env var -> set_seed
# 15. Output: write /output/predictions.csv and /output/val_predictions.csv -> save_predictions

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def preprocess_features(feats: np.ndarray) -> torch.Tensor:
    """Row-normalize features (L1 norm)."""
    # feats shape [N, F]
    row_sums = feats.sum(axis=1, keepdims=True)
    # Avoid division by zero (should not happen as features are binary bag-of-words)
    row_sums[row_sums == 0] = 1.0
    normalized = feats / row_sums
    return torch.from_numpy(normalized).float()

def preprocess_adj(edges: np.ndarray, num_nodes: int) -> torch.Tensor:
    """Add self-loops and return symmetrically normalized adjacency matrix Â."""
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    # add self-loops
    idx = torch.arange(num_nodes, dtype=torch.long)
    adj[idx, idx] = 1.0
    # degree
    deg = adj.sum(dim=1)  # [N]
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    # normalize: Â = D^{-1/2} A D^{-1/2}
    adj_norm = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    return adj_norm

class GCN(nn.Module):
    def __init__(self, nfeat: int, nhidden: int, nclass: int, dropout: float = 0.5):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(nfeat, nhidden))
        self.b1 = nn.Parameter(torch.zeros(nhidden))
        self.W2 = nn.Parameter(torch.empty(nhidden, nclass))
        self.b2 = nn.Parameter(torch.zeros(nclass))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        # biases already zero

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [N, nfeat] feature matrix
        adj: [N, N] normalized adjacency Â
        Returns logits [N, nclass]
        """
        # First layer: Â * (X * W1)
        z1 = torch.mm(x, self.W1)          # [N, nhidden]
        z1 = torch.mm(adj, z1)             # Â * (XW1)
        z1 = F.relu(z1)
        if self.training:
            z1 = F.dropout(z1, p=self.dropout, training=True)
        # Second layer: Â * (Z1 * W2)
        z2 = torch.mm(z1, self.W2)         # [N, nclass]
        z2 = torch.mm(adj, z2)             # Â * (Z1W2)
        return z2

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """Masked accuracy."""
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        correct = (preds == labels) & mask
        acc = correct.sum().float() / mask.sum().float()
        return acc.item()

def train_loop(
    model: GCN,
    features: torch.Tensor,
    adj: torch.Tensor,
    labels: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    lr: float = 0.01,
    l2_lambda: float = 5e-4,
    max_epochs: int = 200,
    patience: int = 10,
) -> tuple[list[float], list[float], GCN]:
    """Returns train_losses, val_losses, best_model (according to val loss)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)  # we add L2 manually
    train_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)  # [N, C]
        loss_per_node = F.cross_entropy(logits, labels, reduction='none')
        loss_train = loss_per_node[train_mask].mean()
        # L2 regularization on first-layer weights only
        l2_reg = l2_lambda * torch.norm(model.W1, p=2) ** 2
        loss = loss_train + l2_reg
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj)
            val_loss_per_node = F.cross_entropy(val_logits, labels, reduction='none')
            val_loss = val_loss_per_node[val_mask].mean()
        train_losses.append(loss_train.item())
        val_losses.append(val_loss.item())

        if val_loss < best_val_loss - 1e-6:  # improvement
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return train_losses, val_losses, model

def get_predictions(model: GCN, features: torch.Tensor, adj: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
    return preds.cpu().numpy()

def save_predictions(
    preds: np.ndarray,
    indices: np.ndarray,
    filepath: str,
    limit: int | None = None,
):
    """Save predictions CSV with header id,prediction.
    If limit is provided, only first `limit` indices (in given order) are written."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write('id,prediction\n')
        selected = indices[:limit] if limit is not None else indices
        for idx in selected:
            f.write(f'{int(idx)},{int(preds[idx])}\n')

def main():
    seed = int(os.environ.get('SEED', '0'))
    set_seed(seed)

    data = np.load('/data/cora_public.npz')
    features = data['features']          # [2708, 1433] float32
    edges = data['edges']                # [E, 2] int64
    labels = data['labels']              # [2708] int64, -1 for test
    train_idx = data['train_idx']        # [140]
    val_idx = data['val_idx']            # [500]
    test_idx = data['test_idx']          # [1000]

    # Preprocess
    features_tensor = preprocess_features(features)
    adj_tensor = preprocess_adj(edges, features.shape[0])
    labels_tensor = torch.from_numpy(labels).long()

    # Hyperparameters from paper
    nhidden = 16
    dropout = 0.5
    lr = 0.01
    l2_lambda = 5e-4
    max_epochs = 200
    patience = 10

    model = GCN(
        nfeat=features.shape[1],
        nhidden=nhidden,
        nclass=int(labels.max() + 1),  # 7 classes
        dropout=dropout,
    )

    smoke = os.environ.get('SMOKE') == '1'
    if smoke:
        # Smoke mode: 1 epoch, only first 20 test/val nodes
        max_epochs = 1
        limit = 20
    else:
        limit = None

    # Train
    train_loop(
        model=model,
        features=features_tensor,
        adj=adj_tensor,
        labels=labels_tensor,
        train_idx=train_idx,
        val_idx=val_idx,
        lr=lr,
        l2_lambda=l2_lambda,
        max_epochs=max_epochs,
        patience=patience,
    )

    # Get predictions for all nodes
    all_preds = get_predictions(model, features_tensor, adj_tensor)

    # Compute and print validation accuracy
    val_acc = compute_accuracy(
        torch.from_numpy(all_preds).long(),
        labels_tensor,
        torch.from_numpy(np.isin(np.arange(labels.shape[0]), val_idx)),
    )
    print(f'Validation accuracy: {val_acc * 100:.2f}%')

    # Save test predictions
    save_predictions(
        preds=all_preds,
        indices=test_idx,
        filepath='/output/predictions.csv',
        limit=limit,
    )
    # Save validation predictions
    save_predictions(
        preds=all_preds,
        indices=val_idx,
        filepath='/output/val_predictions.csv',
        limit=limit,
    )

if __name__ == '__main__':
    main()
