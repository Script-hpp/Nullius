

# Plan step -> implemented in: GCN class
# Plan step -> implemented in: GCNLayer.forward
# Plan step -> implemented in: GCN.__init__ (first_layer) and training loop
# Plan step -> implemented in: optimizer creation
# Plan step -> implemented in: training loop with patience
# Plan step -> implemented in: validation each epoch
# Plan step -> implemented in: build_normalized_adjacency function
# Plan step -> implemented in: GCN.forward (if normalize_features)
# Plan step -> implemented in: using whole tensors (full-batch)
# Plan step -> implemented in: seeding block
# Plan step -> implemented in: prediction and call to nio.write_predictions
# Plan step -> implemented in: smoke mode condition
# Plan step -> implemented in: comment about runtime <5 minutes

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {"lr": 0.01, "weight_decay": 0.001, "epochs": 200, "dropout": 0.8, "hidden": 64, "layers": 2}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_normalized_adjacency(edges: np.ndarray, num_nodes: int):
    # adjacency with self-loops
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    u, v = edges[:, 0], edges[:, 1]
    A[u, v] = 1.0
    # add self-loops
    A[np.arange(num_nodes), np.arange(num_nodes)] = 1.0
    # degree matrix
    deg = A.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    deg_inv_sqrt[deg_inv_sqrt == np.inf] = 0.0
    # nonzero entries
    rows, cols = np.where(A != 0)
    values = deg_inv_sqrt[rows] * deg_inv_sqrt[cols]
    indices = torch.from_numpy(np.vstack((rows, cols)).astype(np.int64))
    values = torch.from_numpy(values)
    adj_sparse = torch.sparse_coo_tensor(
        indices, values, size=(num_nodes, num_nodes), dtype=torch.float32
    )
    adj_sparse.coalesce_()
    return adj_sparse


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, activation):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.dropout = dropout
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)
        support = torch.mm(x, self.weight)  # (N, out)
        output = torch.sparse.mm(adj, support) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class GCN(nn.Module):
    def __init__(self, feat_dim, hidden_dim, out_dim, num_layers=2, dropout=0.8,
                 weight_decay=0.0, normalize_features=False):
        super().__init__()
        self.normalize_features = normalize_features
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.layers = nn.ModuleList()
        # first layer
        self.layers.append(
            GCNLayer(feat_dim, hidden_dim, dropout, activation=F.relu)
        )
        # hidden layers (if any)
        for _ in range(num_layers - 2):
            self.layers.append(
                GCNLayer(hidden_dim, hidden_dim, dropout, activation=F.relu)
            )
        # output layer
        self.layers.append(
            GCNLayer(hidden_dim, out_dim, dropout, activation=None)
        )
        self.first_layer = self.layers[0]

    def forward(self, x, adj):
        if self.normalize_features:
            rowsum = x.sum(dim=1, keepdim=True)
            rowsum[rowsum == 0] = 1
            x = x / rowsum
        for layer in self.layers:
            x = layer(x, adj)
        return x


def main():
    seed = int(os.getenv("SEED", "0"))
    set_seed(seed)

    d = nio.load()
    features = torch.from_numpy(d["features"]).float()
    edges = d["edges"]
    labels = torch.from_numpy(d["labels"]).long()
    idx_train = torch.from_numpy(d["train_idx"]).long()
    idx_val = torch.from_numpy(d["val_idx"]).long()
    idx_test = torch.from_numpy(d["test_idx"]).long()

    num_nodes = features.shape[0]
    adj_norm = build_normalized_adjacency(edges, num_nodes)

    # Model
    num_classes = int(labels.max().item()) + 1  # labels are 0..6, -1 ignored
    model = GCN(
        feat_dim=features.shape[1],
        hidden_dim=CONFIG["hidden"],
        out_dim=num_classes,
        num_layers=CONFIG["layers"],
        dropout=CONFIG["dropout"],
        weight_decay=CONFIG["weight_decay"],
        normalize_features=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=0.0)

    epochs = CONFIG["epochs"]
    if os.getenv("SMOKE") == "1":
        epochs = 1

    best_val_acc = 0.0
    patience = 30
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj_norm)
        loss_train = F.cross_entropy(logits[idx_train], labels[idx_train])
        l2_loss = CONFIG["weight_decay"] * torch.norm(model.first_layer.weight, p=2) ** 2
        loss = loss_train + l2_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(features, adj_norm)
            loss_val = F.cross_entropy(logits_val[idx_val], labels[idx_val])
            pred_val = logits_val[idx_val].argmax(dim=1)
            acc_val = (pred_val == labels[idx_val]).float().mean().item()

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final predictions for all nodes
    model.eval()
    with torch.no_grad():
        logits_all = model(features, adj_norm)
        preds_all = logits_all.argmax(dim=1)

    nio.write_predictions(preds_all.numpy())
    print(f"Validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()