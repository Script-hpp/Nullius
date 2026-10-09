import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.sparse as sp
import nullius_io as nio

# Plan step -> implemented in: seed_everything
# Plan step -> implemented in: preprocess_features
# Plan step -> implemented in: normalize_adj
# Plan step -> implemented in: GraphConvolution
# Plan step -> implemented in: GCN
# Plan step -> implemented in: train_loop
# Plan step -> implemented in: early_stopping
# Plan step -> implemented in: main

CONFIG = {"lr": 0.01, "weight_decay": 0.0005, "epochs": 200, "dropout": 0.5, "hidden": 16, "patience": 10}


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess_features(feats):
    """Row-normalize feature matrix."""
    rowsum = feats.sum(axis=1, keepdims=True)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    return r_inv[:, None] * feats


def normalize_adj(mx):
    """Preprocess adjacency matrix for GCN and return dense normalized matrix."""
    mx = mx + sp.eye(mx.shape[0])  # add self-loops
    rowsum = np.array(mx.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt @ mx @ d_mat_inv_sqrt


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # (N, out)
        output = torch.mm(adj, support)         # (N, out)
        if self.bias is not None:
            return output + self.bias
        return output


class GCN(nn.Module):
    def __init__(self, in_features, hidden, out_features, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_features, hidden)
        self.gc2 = GraphConvolution(hidden, out_features)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


def train_loop(features, adj, labels, train_idx, val_idx, epochs):
    model = GCN(
        features.shape[1],
        CONFIG["hidden"],
        int(labels.max().item() + 1),
        CONFIG["dropout"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    best_val_loss = float('inf')
    best_state = None
    epochs_since_improve = 0

    labels_tensor = labels.long()
    features_tensor = features.float()
    adj_tensor = adj.float()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features_tensor, adj_tensor)
        loss_train = F.cross_entropy(output[train_idx], labels_tensor[train_idx])
        l2_reg = torch.norm(model.gc1.weight, p=2) ** 2
        loss = loss_train + CONFIG["weight_decay"] * l2_reg
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output_val = model(features_tensor, adj_tensor)
            loss_val = F.cross_entropy(output_val[val_idx], labels_tensor[val_idx])
            pred_val = output_val[val_idx].argmax(dim=1)
            acc_val = (pred_val == labels_tensor[val_idx]).float().mean().item()

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = model.state_dict()
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= CONFIG["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, acc_val


def main():
    seed = int(os.environ.get("SEED", "0"))
    seed_everything(seed)

    data = nio.load()
    features = data['features']          # numpy float32 [N, F]
    edges = data['edges']                # int64 [E, 2]
    labels = data['labels']              # int64 [N], -1 for test
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    N = features.shape[0]

    # Preprocess features
    features_norm = preprocess_features(features)

    # Build adjacency matrix
    adj = sp.csr_matrix((np.ones(edges.shape[0], dtype=np.float32),
                         (edges[:, 0], edges[:, 1])), shape=(N, N))
    adj_norm = normalize_adj(adj)

    # Convert to torch tensors
    features_tensor = torch.from_numpy(features_norm).float()
    adj_tensor = torch.from_numpy(adj_norm.toarray()).float()
    labels_tensor = torch.from_numpy(labels).long()

    # Determine number of epochs (SMOKE mode overrides)
    if os.environ.get("SMOKE") == "1":
        epochs_to_run = 1
    else:
        epochs_to_run = CONFIG["epochs"]

    # Train
    model, val_acc = train_loop(features_tensor, adj_tensor, labels_tensor,
                                train_idx, val_idx, epochs_to_run)

    # Print final validation accuracy
    print(f"Val accuracy: {val_acc:.4f}")

    # Get predictions (logits) for all nodes
    model.eval()
    with torch.no_grad():
        logits = model(features_tensor, adj_tensor)  # [N, num_classes]
    nio.write_predictions(logits.cpu().numpy())


if __name__ == "__main__":
    main()
