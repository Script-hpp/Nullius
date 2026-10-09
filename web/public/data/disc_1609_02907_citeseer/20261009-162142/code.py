# Plan step 1 -> implemented in: class GCN
# Plan step 2 -> implemented in: GCN.forward (applies F.dropout)
# Plan step 3 -> implemented in: optimizer weight_decay applies to all parameters
# Plan step 4 -> implemented in: optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
# Plan step 5 -> implemented in: function row_normalize_features
# Plan step 6 -> implemented in: function normalize_adjacency
# Plan step 7 -> implemented in: GraphConvolution.reset_parameters using nn.init.xavier_uniform_
# Plan step 8 -> implemented in: training loop with early stopping
# Plan step 9 -> implemented in: using train_mask for loss, val_mask for evaluation only.
# Plan step 10 -> implemented in: seed setting lines.
# Plan step 11 -> implemented in: loss calculation (cross entropy only, weight decay handled by optimizer)
# Plan step 12 -> implemented in: compute accuracy function.
# Plan step 13 -> implemented in: we process whole graph each epoch.
# Plan step 14 -> comment: single run (paper reports mean over 100).
# Plan step 15 -> comment: test set size 1000 nodes.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

CONFIG = {"lr": 0.01, "epochs": 200, "dropout": 0.5, "hidden": 16, "patience": 10, "weight_decay": 5e-4}


def row_normalize_features(feat):
    """Row-normalize feature matrix (each row sums to 1)."""
    rowsum = np.array(feat.sum(axis=1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    feat_norm = r_mat_inv.dot(feat)
    return feat_norm


def normalize_adjacency(edges, num_nodes):
    """Add self-loops and symmetrically normalize adjacency."""
    # add self-loops
    self_loop = np.arange(num_nodes)
    edges_with_self = np.vstack([edges, np.column_stack([self_loop, self_loop])])
    data = np.ones(edges_with_self.shape[0])
    row = edges_with_self[:, 0]
    col = edges_with_self[:, 1]
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)

    # degree matrix
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    deg_mat_inv_sqrt = sp.diags(deg_inv_sqrt)

    # normalized adjacency: D^{-1/2} A D^{-1/2}
    adj_norm = deg_mat_inv_sqrt.dot(adj).dot(deg_mat_inv_sqrt)
    return adj_norm


def sparse_mx_to_torch_sparse(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col))
    values = sparse_mx.data
    shape = sparse_mx.shape
    return torch.sparse_coo_tensor(
        torch.LongTensor(coords), torch.FloatTensor(values), torch.Size(shape)
    )


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.dropout = dropout
        self.act = act
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)  # Glorot & Bengio init

    def forward(self, input, adj):
        if self.training and self.dropout > 0:
            input = F.dropout(input, p=self.dropout, training=True)
        support = torch.mm(input, self.weight)  # [N, out_features]
        output = torch.sparse.mm(adj, support)  # [N, out_features]
        if self.act is not None:
            output = self.act(output)
        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhidden, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhidden, dropout=dropout, act=F.relu)
        self.gc2 = GraphConvolution(nhidden, nclass, dropout=dropout, act=lambda x: x)  # linear

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        return x


def accuracy(output, labels):
    preds = output.argmax(dim=1)
    return preds.eq(labels).double().mean()


def main():
    # Seed handling
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data via helper
    import nullius_io as nio
    data = nio.load()
    features = data["features"]          # numpy float32 [N, F]
    edges = data["edges"]                # int64 [E, 2] (both directions)
    labels = data["labels"]              # int64 [N], -1 for test
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    N = features.shape[0]
    num_classes = len(np.unique(labels[labels >= 0]))

    # Preprocess features
    features_norm = row_normalize_features(features)
    features_tensor = torch.from_numpy(features_norm).float()

    # Preprocess adjacency
    adj_norm_sp = normalize_adjacency(edges, N)
    adj_norm = sparse_mx_to_torch_sparse(adj_norm_sp)

    # Masks
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(N, dtype=torch.bool)
    val_mask[val_idx] = True

    # Label tensor (use -1 as placeholder; masked out later)
    label_tensor = torch.from_numpy(labels).long()

    # Model
    model = GCN(
        nfeat=features_tensor.shape[1],
        nhidden=CONFIG["hidden"],
        nclass=num_classes,
        dropout=CONFIG["dropout"],
    )

    # Optimizer with weight decay applied to all parameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    # Training configuration
    max_epochs = CONFIG["epochs"]
    if os.environ.get("SMOKE") == "1":
        max_epochs = 1  # smoke mode: single epoch

    patience = CONFIG["patience"]
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features_tensor, adj_norm)  # logits [N, C]

        # Training loss (only on train nodes)
        loss_train = F.cross_entropy(output[train_mask], label_tensor[train_mask])
        loss_train.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(features_tensor, adj_norm)
            val_loss = F.cross_entropy(val_output[val_mask], label_tensor[val_mask])
            val_acc = accuracy(val_output[val_mask], label_tensor[val_mask])

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:03d} | Train Loss: {loss_train.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc.item():.4f}"
            )

    # Final evaluation on validation (using best observed)
    print(f"Validation accuracy: {best_val_acc:.4f}")

    # Produce predictions for all nodes
    model.eval()
    with torch.no_grad():
        logits = model(features_tensor, adj_norm)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # Write predictions via helper
    nio.write_predictions(preds)


if __name__ == "__main__":
    main()
