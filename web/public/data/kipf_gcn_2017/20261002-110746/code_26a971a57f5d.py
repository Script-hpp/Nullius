# Plan step -> implemented in: class GCN
# Plan step -> implemented in: GCN.__init__ (weight init) and reset_parameters
# Plan step -> implemented in: normalize_features
# Plan step -> implemented in: F.dropout calls in GCN.forward
# Plan step -> implemented in: training loop loss computation
# Plan step -> implemented in: l2_reg term
# Plan step -> implemented in: torch.optim.Adam optimizer
# Plan step -> implemented in: training loop with patience/early stopping
# Plan step -> implemented in: validation loss computation for model selection
# Plan step -> implemented in: get_predictions function for test/val
# Plan step -> implemented in: seed_everything function
# Plan step -> implemented in: main training loop (single run)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import csv

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def normalize_features(mx: sp.csr_matrix) -> np.ndarray:
    """Row-normalize (L1) feature matrix."""
    rowsum = np.array(mx.sum(axis=1)).flatten()
    # avoid division by zero
    rowsum[rowsum == 0] = 1.0
    mx_normalized = mx.multiply(1 / rowsum[:, None])
    return mx_normalized.toarray().astype(np.float32)

def build_adjacency(edges: np.ndarray, num_nodes: int) -> sp.csr_matrix:
    """Build symmetric adjacency with self-loops."""
    # edges already contain both directions
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj + sp.identity(num_nodes)  # add self-loops
    return adj.tocsr()

def normalize_adjacency(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Â = D^{-1/2} (A+I) D^{-1/2}"""
    deg = np.array(adj.sum(axis=1)).flatten()
    inv_sqrt_deg = 1.0 / np.sqrt(deg)
    inv_sqrt_deg[np.isinf(inv_sqrt_deg)] = 0.0
    D_inv_sqrt = sp.diags(inv_sqrt_deg)
    return D_inv_sqrt @ adj @ D_inv_sqrt

def sparse_mx_to_torch_sparse(smx: sp.csr_matrix) -> torch.sparse.FloatTensor:
    """Convert scipy CSR to torch sparse COO tensor."""
    smx_coo = smx.tocoo()
    indices = torch.from_numpy(
        np.vstack((smx_coo.row, smx_coo.col)).astype(np.int64)
    )
    values = torch.from_numpy(smx_coo.data.astype(np.float32))
    shape = smx_coo.shape
    return torch.sparse_coo_tensor(indices, values, size=shape).coalesce()

class GCN(nn.Module):
    def __init__(self, nfeat: int, nhidden: int, nclass: int, dropout: float = 0.5):
        super().__init__()
        self.W0 = nn.Parameter(torch.Tensor(nfeat, nhidden))
        self.b0 = nn.Parameter(torch.Tensor(nhidden))
        self.W1 = nn.Parameter(torch.Tensor(nhidden, nclass))
        self.b1 = nn.Parameter(torch.Tensor(nclass))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W0)
        nn.init.xavier_uniform_(self.W1)
        nn.init.zeros_(self.b0)
        nn.init.zeros_(self.b1)

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        x: [N, nfeat] dense feature matrix
        adj: [N, N] sparse normalized adjacency matrix
        """
        # First layer
        x_drop = F.dropout(x, p=self.dropout, training=self.training)
        support0 = torch.mm(x_drop, self.W0) + self.b0          # [N, nhidden]
        h = torch.sparse.mm(adj, support0)                      # [N, nhidden]
        h = F.relu(h)

        # Second layer
        h_drop = F.dropout(h, p=self.dropout, training=self.training)
        support1 = torch.mm(h_drop, self.W1) + self.b1          # [N, nclass]
        out = torch.sparse.mm(adj, support1)                    # [N, nclass]
        return out

def get_predictions(logits: torch.Tensor, idx: np.ndarray) -> np.ndarray:
    """Return class predictions (as numpy) for given node indices."""
    return logits[idx].argmax(dim=1).cpu().numpy()

def main():
    seed = int(os.environ.get('SEED', '0'))
    seed_everything(seed)

    smoke = os.environ.get('SMOKE', '0') == '1'

    # Load data
    data = np.load('/data/cora_public.npz')
    features = data['features']               # [2708, 1433] float32
    edges = data['edges']                     # [E, 2] int64
    labels = data['labels']                   # [2708] int64, -1 for test
    train_idx = data['train_idx']             # [140]
    val_idx = data['val_idx']                 # [500]
    test_idx = data['test_idx']               # [1000]

    N = features.shape[0]

    # Feature normalization (L1 per row)
    features_norm = normalize_features(sp.csr_matrix(features))
    features_tensor = torch.from_numpy(features_norm)   # [N, 1433] float32
    labels_tensor = torch.from_numpy(labels).long()     # [N]

    # Build and normalize adjacency
    adj_raw = build_adjacency(edges, N)
    adj_norm = normalize_adjacency(adj_raw)
    adj_torch = sparse_mx_to_torch_sparse(adj_norm)     # [N, N] sparse

    # Model dimensions
    nfeat = features_tensor.shape[1]
    nhidden = 16
    nclass = int(labels.max() + 1)  # 7 classes (0..6)

    model = GCN(nfeat=nfeat, nhidden=nhidden, nclass=nclass, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lambda_l2 = 5e-4

    # Prepare masks as torch tensors for convenience
    train_mask = torch.from_numpy(train_idx).long()
    val_mask = torch.from_numpy(val_idx).long()
    test_mask = torch.from_numpy(test_idx).long()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    max_epochs = 200 if not smoke else 1
    patience = 10

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features_tensor, adj_torch)
        loss_train = F.cross_entropy(logits[train_mask], labels_tensor[train_mask])
        l2_reg = lambda_l2 * torch.norm(model.W0, p=2) ** 2
        loss = loss_train + l2_reg
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_val = model(features_tensor, adj_torch)
            loss_val = F.cross_entropy(logits_val[val_mask], labels_tensor[val_mask])
            acc_val = (logits_val[val_mask].argmax(dim=1) == labels_tensor[val_mask]).float().mean()

        # Early stopping logic
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation accuracy (to print)
    model.eval()
    with torch.no_grad():
        logits_final = model(features_tensor, adj_torch)
        acc_final = (logits_final[val_mask].argmax(dim=1) == labels_tensor[val_mask]).float().mean()
    print(f"{acc_final.item()*100:.2f}")

    # Prepare predictions
    with torch.no_grad():
        logits_all = model(features_tensor, adj_torch)

    # Function to write CSV
    def write_csv(id_array: np.ndarray, pred_array: np.ndarray, path: str):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'prediction'])
            for idx, pred in zip(id_array, pred_array):
                writer.writerow([int(idx), int(pred)])

    if smoke:
        # Use first 20 sorted indices
        test_sub = np.sort(test_idx)[:20]
        val_sub = np.sort(val_idx)[:20]
        test_pred = get_predictions(logits_all, test_sub)
        val_pred = get_predictions(logits_all, val_sub)
        write_csv(test_sub, test_pred, '/output/predictions.csv')
        write_csv(val_sub, val_pred, '/output/val_predictions.csv')
    else:
        test_pred = get_predictions(logits_all, test_idx)
        val_pred = get_predictions(logits_all, val_idx)
        write_csv(test_idx, test_pred, '/output/predictions.csv')
        write_csv(val_idx, val_pred, '/output/val_predictions.csv')

if __name__ == '__main__':
    main()
