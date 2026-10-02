
# Plan step -> implemented in: function or class name
# 1. Model: 2-layer GCN -> class GCN
# 2. Weight init: Glorot uniform -> function glorot_init
# 3. Features: Row-normalize X -> function row_normalize
# 4. Dropout: 0.5 on inputs of both layers -> method GCN.forward uses F.dropout
# 5. Loss: Masked cross-entropy on train_idx -> function masked_cross_entropy_loss
# 6. Regularization: L2=5e-4 on W0 -> added in training loop
# 7. Optimizer: Adam lr=0.01, full-batch -> torch.optim.Adam
# 8. Training: Max 200 epochs; early stop if val loss doesn't decrease for 10 epochs -> function train_model
# 9. Validation: val_idx used for early stopping & model selection -> same
# 10. Test prediction: Argmax softmax on test_idx -> function predict
# 11. Seed: Read SEED env var -> lines setting numpy and torch seeds
# 12. Single run with given seed -> script executes one training/run

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def row_normalize(features):
    """Row-wise L1 normalization."""
    row_sums = features.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    return features / row_sums

def glorot_init(tensor):
    """Glorot uniform initialization."""
    nn.init.xavier_uniform_(tensor)

def build_adj_norm(edge_index, num_nodes):
    """
    edge_index: torch.LongTensor [2, E] (already contains both directions)
    Returns normalized adjacency (A_hat) as a sparse torch tensor.
    """
    # Add self-loops
    self_loop_idx = torch.arange(num_nodes, dtype=torch.long)
    self_loop_idx = self_loop_idx.repeat(2, 1)  # [2, N]
    edge_with_self = torch.cat([edge_index, self_loop_idx], dim=1)  # [2, E+N]
    # Values all ones
    vals = torch.ones(edge_with_self.size(1), dtype=torch.float32)
    # Compute degree
    row, col = edge_with_self
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.index_add_(0, row, vals)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    # Normalize values
    norm_vals = deg_inv_sqrt[row] * vals * deg_inv_sqrt[col]
    # Create sparse tensor and coalesce
    adj_norm = torch.sparse_coo_tensor(edge_with_self, norm_vals, size=(num_nodes, num_nodes))
    adj_norm = adj_norm.coalesce()
    return adj_norm

def masked_cross_entropy_loss(logits, labels, mask):
    """Cross-entropy loss applied only on masked nodes."""
    # Use ignore_index=-1 to avoid error with test labels (-1)
    loss = F.cross_entropy(logits, labels, reduction='none', ignore_index=-1)
    loss = loss * mask.float()
    return loss.sum() / mask.float().sum()

def masked_accuracy(logits, labels, mask):
    """Accuracy on masked nodes."""
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).float() * mask.float()
    return correct.sum() / mask.float().sum()

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, adj_norm, dropout=0.5):
        super().__init__()
        self.adj_norm = adj_norm
        self.dropout = dropout
        self.W0 = nn.Parameter(torch.empty(in_features, hidden_features))
        self.b0 = nn.Parameter(torch.zeros(hidden_features))
        self.W1 = nn.Parameter(torch.empty(hidden_features, out_features))
        self.b1 = nn.Parameter(torch.zeros(out_features))
        glorot_init(self.W0)
        glorot_init(self.W1)

    def forward(self, x):
        # Dropout on input features
        x = F.dropout(x, p=self.dropout, training=self.training)
        # First graph conv: Â * X
        x = torch.sparse.mm(self.adj_norm, x)
        x = x @ self.W0 + self.b0
        x = F.relu(x)
        # Dropout on hidden representation
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Second graph conv: Â * H
        x = torch.sparse.mm(self.adj_norm, x)
        x = x @ self.W1 + self.b1  # logits
        return x

def train_model(model, features, labels, train_idx, val_idx, optimizer, lambda_l2, max_epochs=200, patience=10, smoke=False):
    """Train with early stopping on validation loss."""
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    train_mask = torch.zeros(labels.size(0), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(labels.size(0), dtype=torch.bool)
    val_mask[val_idx] = True

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features)
        loss = masked_cross_entropy_loss(logits, labels, train_mask)
        # L2 regularization on W0 only
        l2_reg = lambda_l2 * torch.norm(model.W0, p=2) ** 2
        loss = loss + l2_reg
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features)
            val_loss = masked_cross_entropy_loss(val_logits, labels, val_mask)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if smoke and epoch >= 0:  # smoke mode: exactly 1 epoch
            break
        if epochs_no_improve >= patience:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute validation accuracy with best model
    model.eval()
    with torch.no_grad():
        val_logits = model(features)
        val_acc = masked_accuracy(val_logits, labels, val_mask).item()
    return val_acc

def predict(model, features, idx):
    """Return predicted class indices for given node indices."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
    return pred[idx].cpu().numpy()

def main():
    seed = int(os.environ.get('SEED', 0))
    set_seed(seed)

    smoke = os.environ.get('SMOKE', '0') == '1'

    data = np.load('/data/cora_public.npz', allow_pickle=False)
    features = torch.from_numpy(data['features']).float()          # [N, F]
    edges = torch.from_numpy(data['edges']).long()                # [E, 2] -> need [2, E]
    edges = edges.t()                                              # now [2, E]
    labels = torch.from_numpy(data['labels']).long()              # [N]
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx = torch.from_numpy(data['val_idx']).long()
    test_idx = torch.from_numpy(data['test_idx']).long()

    N = features.size(0)

    # Step 3: Row-normalize features
    features = torch.from_numpy(row_normalize(features.numpy())).float()

    # Step 1 & 4: Build normalized adjacency and model
    adj_norm = build_adj_norm(edges, N)
    hidden = 16
    n_classes = int(labels.max().item()) + 1  # labels are 0..6, ignoring -1 test nodes
    model = GCN(in_features=features.size(1),
                hidden_features=hidden,
                out_features=n_classes,
                adj_norm=adj_norm,
                dropout=0.5)

    # Step 7: Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Step 6: L2 regularization factor
    lambda_l2 = 5e-4

    # Step 8 & 9: Train
    val_acc = train_model(model, features, labels, train_idx, val_idx,
                          optimizer, lambda_l2,
                          max_epochs=200, patience=10,
                          smoke=smoke)

    # Print final validation accuracy (as percentage)
    print(f"{val_acc * 100:.2f}")

    # Step 10: Predict on test
    test_pred = predict(model, features, test_idx)
    # Step 12: Write predictions.csv
    os.makedirs('/output', exist_ok=True)
    with open('/output/predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        if smoke:
            # only first 20 nodes of sorted test_idx
            test_ids_sorted = test_idx[:20].cpu().numpy()
            preds_subset = test_pred[:20]
            for idx, pred in zip(test_ids_sorted, preds_subset):
                f.write(f'{int(idx)},{int(pred)}\n')
        else:
            for idx, pred in zip(test_idx.cpu().numpy(), test_pred):
                f.write(f'{int(idx)},{int(pred)}\n')

    # Also write validation predictions (not scored)
    val_pred = predict(model, features, val_idx)
    with open('/output/val_predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        if smoke:
            val_ids_sorted = val_idx[:20].cpu().numpy()
            preds_subset = val_pred[:20]
            for idx, pred in zip(val_ids_sorted, preds_subset):
                f.write(f'{int(idx)},{int(pred)}\n')
        else:
            for idx, pred in zip(val_idx.cpu().numpy(), val_pred):
                f.write(f'{int(idx)},{int(pred)}\n')

if __name__ == '__main__':
    main()