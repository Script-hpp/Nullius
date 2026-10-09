# Plan step -> implemented in: function or class name
# Read SEED env var, set numpy.random.seed(SEED) and torch.manual_seed(SEED) -> set_seeds
# Load /data/cora_lcc_public.npz -> load_data
# Build normalized adjacency matrix (A_hat) -> get_norm_adj
# Define 2-layer GCN model -> GCN class
# Define optimizer (Adam) and loss (CrossEntropyLoss) -> setup_training
# Training loop with early stopping on validation loss -> train_model
# Run inference on all nodes and write predictions -> predict_and_write
# Print final validation accuracy to stdout -> evaluate

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio


def set_seeds():
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    d = nio.load()  # returns dict with keys: features, edges, labels, train_idx, val_idx, test_idx
    features = torch.from_numpy(d["features"]).float()          # [N, 1433]
    edges = torch.from_numpy(d["edges"]).long()                 # [E, 2]
    labels = torch.from_numpy(d["labels"]).long()               # [N]
    train_idx = torch.from_numpy(d["train_idx"]).long()
    val_idx = torch.from_numpy(d["val_idx"]).long()
    test_idx = torch.from_numpy(d["test_idx"]).long()
    return features, edges, labels, train_idx, val_idx, test_idx


def get_norm_adj(num_nodes, edge_index):
    """
    edge_index: LongTensor [2, E] (directed edges, includes both directions)
    Returns normalized adjacency with self-loops as sparse COO tensor:
        \hat{A} = D^{-1/2} (A + I) D^{-1/2}
    """
    # add self-loops
    self_loop = torch.arange(num_nodes, device=edge_index.device)
    self_loop = self_loop.repeat(2, 1)  # [2, N]
    edge_index_with_self = torch.cat([edge_index, self_loop], dim=1)  # [2, E+N]

    # compute degrees
    deg = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float)
    deg.index_add_(0, edge_index_with_self[0], torch.ones(edge_index_with_self.size(1), device=edge_index.device))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # compute norm values for each edge: deg^{-0.5}(u) * deg^{-0.5}(v)
    u, v = edge_index_with_self
    norm = deg_inv_sqrt[u] * deg_inv_sqrt[v]

    # build sparse tensor
    adj_norm = torch.sparse_coo_tensor(
        indices=edge_index_with_self,
        values=norm,
        size=(num_nodes, num_nodes),
        device=edge_index.device
    )
    return adj_norm.coalesce()


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, out_dim)

    def forward(self, x, adj_norm):
        x = self.conv1(x, adj_norm)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj_norm)
        return x  # logits


class GraphConv(nn.Module):
    """Simple GCN layer: H' = \hat{A} X W"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj_norm):
        support = torch.mm(x, self.weight)          # [N, out]
        out = torch.sparse.mm(adj_norm, support)    # [N, out]
        return out


def setup_training(model, lr=0.01, weight_decay=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion


def train_model(model, features, adj_norm, labels, train_idx, val_idx, optimizer, criterion,
                max_epochs=200, patience=50, smoke=False):
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        logits = model(features, adj_norm)          # [N, C]
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj_norm)
            val_loss = criterion(val_logits[val_idx], labels[val_idx])
            _, pred = torch.max(val_logits[val_idx], dim=1)
            val_acc = (pred == labels[val_idx]).float().mean().item()
        model.train()

        if smoke:
            # In smoke mode we just run one epoch regardless
            break

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_and_write(model, features, adj_norm):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_norm)   # [N, 7]
    # nio.write_predictions accepts either class predictions or logits
    nio.write_predictions(logits)   # helper will handle test/val selection


def evaluate(model, features, adj_norm, labels, val_idx):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_norm)
        _, pred = torch.max(logits[val_idx], dim=1)
        acc = (pred == labels[val_idx]).float().mean().item()
    print(f"Validation accuracy: {acc*100:.2f}%")


def main():
    set_seeds()
    features, edges, labels, train_idx, val_idx, test_idx = load_data()

    num_nodes = features.shape[0]
    # edges currently [E, 2]; need [2, E] for sparse tensor
    edge_index = edges.t().contiguous()  # [2, E]
    adj_norm = get_norm_adj(num_nodes, edge_index)

    model = GCN(in_dim=features.shape[1], hidden_dim=16, out_dim=7, dropout=0.5)
    optimizer, criterion = setup_training(model, lr=0.01, weight_decay=5e-4)

    smoke = os.environ.get("SMOKE", "0") == "1"
    max_epochs = 1 if smoke else 200
    patience = 50  # still defined but irrelevant for smoke

    model = train_model(model, features, adj_norm, labels, train_idx, val_idx,
                        optimizer, criterion, max_epochs=max_epochs, patience=patience, smoke=smoke)

    predict_and_write(model, features, adj_norm)
    evaluate(model, features, adj_norm, labels, val_idx)


if __name__ == "__main__":
    main()
