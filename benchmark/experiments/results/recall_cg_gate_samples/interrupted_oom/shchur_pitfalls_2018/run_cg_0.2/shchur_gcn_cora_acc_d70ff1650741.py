

# Plan step -> implemented in: load_data
# Plan step -> implemented in: preprocess
# Plan step -> implemented in: class GCN
# Plan step -> implemented in: train_model
# Plan step -> implemented in: predict_and_write
# Plan step -> implemented in: smoke_mode

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import nullius_io as nio

CONFIG = {"lr": 0.01, "epochs": 100000, "layers": 2, "patience": 50}
CONFIG["hidden_size"] = 64
CONFIG["dropout"] = 0.8
CONFIG["weight_decay"] = 0.001

def load_data():
    d = nio.load()
    features = d['features']          # np.ndarray [N, 1433] float32
    edges = d['edges']                # np.ndarray [E, 2] int64
    labels = d['labels']              # np.ndarray [N] int64, -1 for test
    train_idx = d['train_idx']
    val_idx = d['val_idx']
    test_idx = d['test_idx']
    return features, edges, labels, train_idx, val_idx, test_idx

def preprocess(features, edges, num_nodes):
    # Build adjacency matrix (COO) from directed edge list (both directions present)
    adj = sp.coo_matrix((np.ones(edges.shape[0], dtype=np.float32),
                         (edges[:, 0], edges[:, 1])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    # Add self-loops
    adj = adj + sp.identity(num_nodes, dtype=np.float32)
    # Convert to CSR for efficient row operations
    adj = adj.tocsr()
    # Degree vector
    deg = np.array(adj.sum(axis=1)).flatten()
    # Inverse sqrt degree, handle zeros safely
    deg_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
    mask = deg > 0
    deg_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    # Normalize: D^{-1/2} A D^{-1/2}
    adj_norm = sp.diags(deg_inv_sqrt) @ adj @ sp.diags(deg_inv_sqrt)
    adj_norm = adj_norm.tocoo()  # convert to COO for torch conversion
    # Features as sparse tensor (no row normalization)
    feat_sp = sp.coo_matrix(features)
    # Convert to torch sparse tensors
    def to_torch_sparse(sp_mat):
        indices = torch.from_numpy(
            np.vstack([sp_mat.row, sp_mat.col]).astype(np.int64))
        values = torch.from_numpy(sp_mat.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, sp_mat.shape).coalesce()
    adj_t = to_torch_sparse(adj_norm)
    feat_t = to_torch_sparse(feat_sp)
    return adj_t, feat_t

class GCN(nn.Module):
    def __init__(self, adj, feat, hidden_dim, out_dim, dropout, weight_decay, num_layers):
        super().__init__()
        self.adj = adj
        self.feat = feat
        self.dropout = dropout
        self.num_layers = num_layers
        self.weight_decay = weight_decay
        # Create linear layers
        self.linears = nn.ModuleList()
        # Input dimension of first layer is feature dimension
        in_dim = feat.shape[1]
        for i in range(num_layers):
            if i < num_layers - 1:
                out_dim_layer = hidden_dim
            else:
                out_dim_layer = out_dim
            linear = nn.Linear(in_dim, out_dim_layer, bias=True)
            self.linears.append(linear)
            in_dim = out_dim_layer
        # Initialize weights with Glorot uniform
        for linear in self.linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def _dropout_sparse(self, x, p, training):
        if not training or p == 0.0:
            return x
        if x.is_sparse:
            mask = torch.rand_like(x.values) > p
            vals = x.values * mask.float() / (1.0 - p)
            return torch.sparse_coo_tensor(x.indices(), vals, x.size()).coalesce()
        else:
            return F.dropout(x, p, training)

    def forward(self):
        x = self.feat  # sparse tensor
        for i, linear in enumerate(self.linears):
            # Apply dropout to inputs of this layer
            x = self._dropout_sparse(x, self.dropout, self.training)
            # Linear transformation (weight multiplication)
            if x.is_sparse:
                support = torch.sparse.mm(x, linear.weight)
            else:
                support = torch.mm(x, linear.weight)
            # Graph convolution: Â * support
            out = torch.sparse.mm(self.adj, support) + linear.bias
            # Activation: ReLU for all but last layer
            if i < self.num_layers - 1:
                out = F.relu(out)
            x = out
        return x

def load_data_and_preprocess():
    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features.shape[0]
    adj_t, feat_t = preprocess(features, edges, num_nodes)
    labels_tensor = torch.from_numpy(labels).long()
    train_idx_t = torch.from_numpy(train_idx).long()
    val_idx_t = torch.from_numpy(val_idx).long()
    test_idx_t = torch.from_numpy(test_idx).long()
    return adj_t, feat_t, labels_tensor, train_idx_t, val_idx_t, test_idx_t, features.shape[1]

def train_model():
    adj_t, feat_t, labels_tensor, train_idx_t, val_idx_t, test_idx_t, in_dim = load_data_and_preprocess()
    # Determine number of classes (ignore -1)
    valid_labels = labels_tensor[labels_tensor >= 0]
    num_classes = int(valid_labels.max().item()) + 1

    model = GCN(adj_t, feat_t, CONFIG["hidden_size"], num_classes,
                CONFIG["dropout"], CONFIG["weight_decay"], CONFIG["layers"])
    # Optimizer: we handle weight decay manually for first layer only
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    best_val_loss = float('inf')
    best_state = None
    epochs_without_improve = 0

    # Smoke mode: exactly one epoch
    smoke = os.environ.get("SMOKE") == "1"
    max_epochs = 1 if smoke else CONFIG["epochs"]

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model()
        # Cross entropy loss on training set
        ce_loss = F.cross_entropy(logits[train_idx_t], labels_tensor[train_idx_t])
        # L2 regularization on first layer weights only
        l2_reg = torch.sum(model.linears[0].weight ** 2)
        loss = ce_loss + CONFIG["weight_decay"] * l2_reg
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model()
            val_loss = F.cross_entropy(val_logits[val_idx_t], labels_tensor[val_idx_t])
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= CONFIG["patience"]:
            break

    # Restore best validation state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on validation set (optional print)
    model.eval()
    with torch.no_grad():
        final_logits = model()
        final_preds = torch.argmax(final_logits[val_idx_t], dim=1)
        final_acc = (final_preds == labels_tensor[val_idx_t]).float().mean().item()
    print(f"Validation accuracy: {final_acc:.4f}")

    # Return logits for all nodes
    with torch.no_grad():
        all_logits = model()
        return all_logits.cpu()

def predict_and_write():
    logits_or_preds = train_model()
    # nio.write_predictions accepts either logits [N, C] or class indices [N]
    nio.write_predictions(logits_or_preds.numpy())

if __name__ == "__main__":
    # Seed numpy and torch from environment variable SEED
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    predict_and_write()