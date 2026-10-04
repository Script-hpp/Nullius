# Plan step -> implemented in: function or class name
# 1. Model -> GCN class
# 2. Hyperparameters -> defined in main()
# 3. Input processing -> load_data(), build_adjacency()
# 4. Labels -> use labels from npz
# 5. Training -> train_model()
# 6. Early Stopping -> early stopping condition in train_model()
# 7. Model Selection -> restore best weights based on validation loss
# 8. Seeding -> set_seeds()
# 9. Output -> write_predictions()

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    features = torch.from_numpy(data['features']).float()          # [N, F]
    edges = torch.from_numpy(data['edges']).long()                # [E, 2]
    labels = torch.from_numpy(data['labels']).long()              # [N]
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx   = torch.from_numpy(data['val_idx']).long()
    test_idx  = torch.from_numpy(data['test_idx']).long()
    return features, edges, labels, train_idx, val_idx, test_idx

def build_adjacency(edge_list: torch.Tensor, num_nodes: int):
    """
    edge_list: [E, 2] directed pairs (both directions present).
    Returns symmetrically normalized adjacency with self-loops (GCN norm).
    """
    # Remove self-loops if any
    mask = edge_list[:, 0] != edge_list[:, 1]
    edge_list = edge_list[mask]
    # Make undirected (ensure both directions)
    edge_list = torch.cat([edge_list, edge_list.flip(1)], dim=0)
    # Coalesce to remove duplicates
    idx = edge_list.t()
    vals = torch.ones(idx.size(1), dtype=torch.float32)
    A = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes)).coalesce()
    # Add self-loops
    loop_idx = torch.arange(num_nodes, dtype=torch.long)
    loop = torch.stack([loop_idx, loop_idx], dim=0)
    idx_with_loop = torch.cat([idx, loop], dim=1)
    vals_with_loop = torch.cat([vals, torch.ones(num_nodes, dtype=torch.float32)], dim=0)
    A = torch.sparse_coo_tensor(idx_with_loop, vals_with_loop, (num_nodes, num_nodes)).coalesce()
    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = torch.sparse.sum(A, dim=1).to_dense() + 1e-12   # avoid zero
    deg_inv_sqrt = deg.pow(-0.5)
    inv_sqrt_idx = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
    inv_sqrt_vals = deg_inv_sqrt
    D_inv_sqrt = torch.sparse_coo_tensor(inv_sqrt_idx, inv_sqrt_vals, (num_nodes, num_nodes))
    A_hat = torch.sparse.mm(D_inv_sqrt, torch.sparse.mm(A, D_inv_sqrt))
    return A_hat.coalesce()

class GCN(nn.Module):
    def __init__(self, in_features, hidden_size, out_classes, dropout_p):
        super().__init__()
        self.W0 = nn.Parameter(torch.FloatTensor(in_features, hidden_size))
        self.W1 = nn.Parameter(torch.FloatTensor(hidden_size, out_classes))
        self.dropout = nn.Dropout(dropout_p)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W0)
        nn.init.xavier_uniform_(self.W1)

    def forward(self, x, adj_norm):
        # Layer 0
        support0 = torch.sparse.mm(adj_norm, x)          # [N, in_features]
        h0 = support0 @ self.W0                          # [N, hidden]
        h0 = F.relu(h0)
        h0 = self.dropout(h0)
        # Layer 1
        support1 = torch.sparse.mm(adj_norm, h0)         # [N, hidden]
        out = support1 @ self.W1                         # [N, out_classes]
        return out

def train_model(model, features, adj_norm, labels, train_idx, val_idx, smoke=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    best_val_loss = float('inf')
    best_state = None
    val_losses = []
    max_epochs = 1 if smoke else 200

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj_norm)
        loss_train = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features, adj_norm)
            loss_val = F.cross_entropy(logits[val_idx], labels[val_idx]).item()
            val_losses.append(loss_val)

            # Early stopping condition (GCN specific)
            if epoch >= 10 and loss_val > np.mean(val_losses[-10:]):
                if smoke:
                    # In smoke mode we still break after 1 epoch anyway
                    pass
                break

            if loss_val < best_val_loss:
                best_val_loss = loss_val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation accuracy
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_norm)
        preds = logits[val_idx].argmax(dim=1)
        acc = (preds == labels[val_idx]).float().mean().item()
    print(f"Validation accuracy: {acc*100:.2f}%")
    return model

def write_predictions(csv_path, ids, preds):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write("id,prediction\n")
        for idx, p in zip(ids, preds):
            f.write(f"{int(idx)},{int(p)}\n")

def main():
    seed = int(os.getenv('SEED', '0'))
    smoke_mode = os.getenv('SMOKE', '0') == '1'
    set_seeds(seed)

    npz_path = '/data/cora_lcc_public.npz'
    features, edges, labels, train_idx, val_idx, test_idx = load_data(npz_path)
    N = features.size(0)

    adj_norm = build_adjacency(edges, N)

    # Model hyperparameters (optimized config from Table 4)
    hidden_size = 64
    dropout_p = 0.8
    n_classes = int(labels.max().item()) + 1  # labels are 0..6, -1 for test but max ignores -1
    model = GCN(features.size(1), hidden_size, n_classes, dropout_p)

    model = train_model(model, features, adj_norm, labels, train_idx, val_idx, smoke=smoke_mode)

    # Predictions for validation set (always required)
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_norm)
        all_preds = logits.argmax(dim=1)

    # Validation output
    if smoke_mode:
        val_ids_sorted = torch.sort(val_idx)[0][:20]
        val_preds = all_preds[val_ids_sorted]
    else:
        val_ids_sorted = val_idx
        val_preds = all_preds[val_idx]
    write_predictions('/output/val_predictions.csv', val_ids_sorted.cpu().numpy(), val_preds.cpu().numpy())

    # Test output
    if smoke_mode:
        test_ids_sorted = torch.sort(test_idx)[0][:20]
        test_preds = all_preds[test_ids_sorted]
    else:
        test_ids_sorted = test_idx
        test_preds = all_preds[test_idx]
    write_predictions('/output/predictions.csv', test_ids_sorted.cpu().numpy(), test_preds.cpu().numpy())

if __name__ == '__main__':
    main()
