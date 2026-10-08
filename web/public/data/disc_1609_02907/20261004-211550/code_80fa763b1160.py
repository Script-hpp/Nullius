# Plan step -> implemented in: function or class name
# 1. Load `/data/cora_public.npz`; row‑normalize features (L1) -> load_and_preprocess()
# 2. Build symmetric binary adjacency from edges (add self‑loops); compute symmetrically normalized Â -> build_adjacency()
# 3. Model: 2‑layer GCN – Layer1: GraphConvolution(1433→16, ReLU, dropout=0.5, sparse_inputs=True); Layer2: GraphConvolution(16→7, linear, dropout=0.5) -> GCN class and GraphConvolution class
# 4. Weight init: Xavier (Glorot) for weights, zeros for bias -> GraphConvolution.reset_parameters()
# 5. Loss: masked softmax cross‑entropy on train_idx -> masked_cross_entropy()
# 6. L2 regularization: 5·10⁻⁴ on first‑layer weights only -> applied in training loop
# 7. Optimizer: Adam, learning rate 0.01 -> torch.optim.Adam
# 8. Train: max 200 epochs, early stopping on validation loss (patience=10) using val_idx -> train()
# 9. Seed NumPy and PyTorch from SEED env var -> set_seeds()
#10. Predict on test_idx; write /output/predictions.csv with header `id,prediction` -> save_predictions()
#    Also write /output/val_predictions.csv similarly.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

def set_seeds():
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_preprocess(npz_path="/data/cora_public.npz"):
    data = np.load(npz_path)
    features = data["features"].astype(np.float32)          # [N, 1433] binary bag-of-words
    # L1 normalization (row-wise)
    row_sums = features.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    features = features / row_sums
    edges = data["edges"].astype(np.int64)                  # [E, 2]
    labels = data["labels"].astype(np.int64)                # [N]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    return features, edges, labels, train_idx, val_idx, test_idx

def build_adjacency(edges, num_nodes):
    # symmetric binary adjacency
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    u, v = edges[:, 0], edges[:, 1]
    adj[u, v] = 1.0
    adj[v, u] = 1.0  # ensure symmetry (edges already contain both directions, but safe)
    # add self-loops
    adj += np.eye(num_nodes, dtype=np.float32)
    # symmetric normalization Â = D^{-1/2} A D^{-1/2}
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
    return torch.from_numpy(adj_norm)  # dense tensor [N, N]

def sparse_mx_to_torch_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col))
    values = sparse_mx.data
    shape = sparse_mx.shape
    return torch.sparse_coo_tensor(
        torch.LongTensor(coords),
        torch.FloatTensor(values),
        torch.Size(shape),
    )

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://github.com/tkipf/gcn
    """
    def __init__(self, in_features, out_features, dropout=0., act=F.relu, sparse_inputs=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        if self.sparse_inputs:
            support = torch.sparse.mm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        support = F.dropout(support, self.dropout, training=self.training)
        output = torch.mm(adj, support) + self.bias
        if self.act is not None:
            output = self.act(output)
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, dropout=dropout, act=F.relu, sparse_inputs=True)
        self.gc2 = GraphConvolution(nhid, nclass, dropout=dropout, act=lambda x: x, sparse_inputs=False)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        return x

def masked_cross_entropy(logits, labels, mask):
    """Compute cross entropy loss only on masked entries."""
    loss = F.cross_entropy(logits[mask], labels[mask])
    return loss

def accuracy(logits, labels, mask):
    preds = logits[mask].argmax(dim=-1)
    correct = (preds == labels[mask]).float().sum()
    return correct / mask.sum()

def train(model, features, adj, labels, train_idx, val_idx, optimizer, lambda_l2, max_epochs=200, patience=10, smoke=False):
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)
        loss_train = masked_cross_entropy(logits, labels, train_idx)
        # L2 regularization on first-layer weights only
        l2_reg = lambda_l2 * torch.sum(model.gc1.weight.pow(2))
        loss = loss_train + l2_reg
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features, adj)
            loss_val = masked_cross_entropy(logits, labels, val_idx)
            acc_val = accuracy(logits, labels, val_idx)

        if smoke:
            # In smoke mode we just run one epoch and break
            break

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    # Return final validation accuracy (from best model)
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        acc_val = accuracy(logits, labels, val_idx)
    return acc_val.item()

def save_predictions(logits, idx_list, out_path, smoke=False):
    """Write predictions.csv with header id,prediction for given idx_list."""
    preds = logits.argmax(dim=-1).cpu().numpy()
    if smoke:
        # take first 20 ids after sorting for determinism
        idx_list = np.sort(idx_list)[:20]
    with open(out_path, "w") as f:
        f.write("id,prediction\n")
        for idx in idx_list:
            f.write(f"{int(idx)},{int(preds[idx])}\n")

def main():
    set_seeds()
    features_np, edges, labels_np, train_idx, val_idx, test_idx = load_and_preprocess()

    # Convert to torch tensors
    features = torch.from_numpy(features_np)  # dense after L1 norm
    # Convert features to sparse for first layer (since they are very sparse)
    features_sparse = sparse_mx_to_torch_sparse(
        # scipy csr from numpy
        __import__('scipy.sparse').csr_matrix(features_np)
    )
    labels = torch.from_numpy(labels_np)  # int64, -1 for test nodes
    adj = build_adjacency(edges, features_np.shape[0])  # dense [N,N]

    model = GCN(nfeat=features_np.shape[1], nhid=16, nclass=int(labels_np.max()+1), dropout=0.5)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.)  # we handle L2 manually
    lambda_l2 = 5e-4

    smoke_mode = os.environ.get("SMOKE", "0") == "1"
    max_epochs = 1 if smoke_mode else 200

    val_acc = train(model, features_sparse, adj, labels,
                    torch.from_numpy(train_idx), torch.from_numpy(val_idx),
                    optimizer, lambda_l2,
                    max_epochs=max_epochs, patience=10, smoke=smoke_mode)

    print(f"Validation accuracy: {val_acc:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(features_sparse, adj)

    os.makedirs("/output", exist_ok=True)
    save_predictions(logits, test_idx, "/output/predictions.csv", smoke=smoke_mode)
    save_predictions(logits, val_idx, "/output/val_predictions.csv", smoke=smoke_mode)

if __name__ == "__main__":
    main()
