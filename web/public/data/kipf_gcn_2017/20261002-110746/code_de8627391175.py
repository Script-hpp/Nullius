# Plan step -> implemented in: function or class name
# 1. Model -> GCN class
# 2. Weight init -> _init_weights method
# 3. Features -> row_normalize function
# 4. Dropout -> nn.Dropout in forward
# 5. Loss -> masked loss computed in train loop
# 6. Regularization -> L2 on W0
# 7. Optimizer -> Adam
# 8. Training -> train function with early stopping
# 9. Validation -> used for early stopping & model selection
# 10. Test prediction -> predict function
# 11. Seed -> set_seeds function
# 12. Single run -> main

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def row_normalize(mat: np.ndarray) -> np.ndarray:
    """L1 normalize each row."""
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return mat / row_sum

def load_cora(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    features = data['features'].astype(np.float32)          # [N, 1433]
    edges = data['edges'].astype(np.int64)                  # [E, 2] (both directions)
    labels = data['labels'].astype(np.int64)                # [N]
    train_idx = data['train_idx'].astype(np.int64)
    val_idx = data['val_idx'].astype(np.int64)
    test_idx = data['test_idx'].astype(np.int64)
    return features, edges, labels, train_idx, val_idx, test_idx

def build_norm_adj(num_nodes: int, edges: np.ndarray):
    """Return symmetrically normalized adjacency with self-loops as a sparse torch tensor."""
    # edge index as [2, E]
    edge_idx = torch.from_numpy(edges).t()                  # [2, E]
    # add self-loops
    self_loop = torch.arange(num_nodes, dtype=torch.long)
    self_loop = torch.stack([self_loop, self_loop], dim=0)  # [2, N]
    edge_idx = torch.cat([edge_idx, self_loop], dim=1)      # [2, E+N]
    # values all ones
    vals = torch.ones(edge_idx.size(1), dtype=torch.float32)
    # create sparse matrix and coalesce
    adj = torch.sparse_coo_tensor(edge_idx, vals, size=(num_nodes, num_nodes))
    adj = adj.coalesce()
    # compute degree
    deg = torch.sparse.sum(adj, dim=1).to_dense()           # [N]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    # normalize
    row = edge_idx[0]
    col = edge_idx[1]
    norm_vals = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj_norm = torch.sparse_coo_tensor(edge_idx, norm_vals, size=(num_nodes, num_nodes))
    adj_norm = adj_norm.coalesce()
    return adj_norm

class GCN(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int, dropout_p: float = 0.5):
        super().__init__()
        self.W0 = nn.Parameter(torch.empty(in_features, hidden))
        self.b0 = nn.Parameter(torch.zeros(hidden))
        self.W1 = nn.Parameter(torch.empty(hidden, out_features))
        self.b1 = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(p=dropout_p)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W0)
        nn.init.xavier_uniform_(self.W1)
        # biases already zero

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor):
        """
        x: [N, in_features] dense
        adj: [N, N] sparse (already normalized)
        """
        # first layer
        x = self.dropout(x)
        support0 = x @ self.W0                     # [N, hidden]
        h = torch.sparse.mm(adj, support0) + self.b0
        h = F.relu(h)

        # second layer
        h = self.dropout(h)
        support1 = h @ self.W1                     # [N, out_features]
        out = torch.sparse.mm(adj, support1) + self.b1   # logits
        return out

def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, mask: np.ndarray):
    """Compute CE loss only on indices where mask is True."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    loss = F.cross_entropy(logits[mask], labels[mask], reduction='mean')
    return loss

def train_model(features, adj, labels, train_idx, val_idx, seed, smoke=False):
    set_seeds(seed)
    device = torch.device('cpu')
    features_t = torch.from_numpy(features).to(device)
    labels_t = torch.from_numpy(labels).to(device)
    train_mask = np.zeros_like(labels, dtype=bool)
    train_mask[train_idx] = True
    val_mask = np.zeros_like(labels, dtype=bool)
    val_mask[val_idx] = True

    model = GCN(in_features=features.shape[1], hidden=16, out_features=7, dropout_p=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)  # we add L2 manually
    l2_lambda = 5e-4

    best_val_loss = float('inf')
    best_state = None
    patience = 0
    max_epochs = 1 if smoke else 200

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features_t, adj)
        loss_train = masked_cross_entropy(logits, labels_t, train_mask)
        # L2 on W0 only
        l2_loss = l2_lambda * torch.norm(model.W0) ** 2
        loss = loss_train + l2_loss
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features_t, adj)
            loss_val = masked_cross_entropy(val_logits, labels_t, val_mask)

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # final validation accuracy
    model.eval()
    with torch.no_grad():
        final_logits = model(features_t, adj)
        preds = final_logits.argmax(dim=1)
        correct = (preds[val_idx] == labels_t[val_idx]).float().mean()
        val_acc = correct.item() * 100.0
    print(f'Validation accuracy: {val_acc:.2f}%')

    return model, val_acc

def predict(model, features, adj):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
    return preds.cpu().numpy()

def write_csv(path, ids, preds):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('id,prediction\n')
        for idx, p in zip(ids, preds):
            f.write(f'{int(idx)},{int(p)}\n')

def main():
    npz_path = '/data/cora_public.npz'
    features_raw, edges, labels, train_idx, val_idx, test_idx = load_cora(npz_path)
    features = row_normalize(features_raw)

    num_nodes = features.shape[0]
    adj = build_norm_adj(num_nodes, edges)

    seed = int(os.environ.get('SEED', '0'))
    smoke = os.environ.get('SMOKE', '0') == '1'

    model, _ = train_model(features, adj, labels, train_idx, val_idx, seed, smoke)

    # predictions for test
    test_preds = predict(model, torch.from_numpy(features).float(), adj)
    if smoke:
        test_ids = sorted(test_idx)[:20]
        test_preds = test_preds[np.isin(np.arange(len(test_idx)), test_ids)]
    else:
        test_ids = sorted(test_idx)
    write_csv('/output/predictions.csv', test_ids, test_preds)

    # predictions for validation (to be saved as requested)
    val_preds = predict(model, torch.from_numpy(features).float(), adj)
    if smoke:
        val_ids = sorted(val_idx)[:20]
        val_preds = val_preds[np.isin(np.arange(len(val_idx)), val_ids)]
    else:
        val_ids = sorted(val_idx)
    write_csv('/output/val_predictions.csv', val_ids, val_preds)

if __name__ == '__main__':
    main()
