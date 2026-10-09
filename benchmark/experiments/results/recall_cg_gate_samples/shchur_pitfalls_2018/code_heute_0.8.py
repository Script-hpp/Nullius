# Plan step 1 -> implemented in: set_seeds()
# Plan step 2 -> implemented in: load_data()
# Plan step 3 -> implemented in: GCN class definition
# Plan step 4 -> implemented in: configure_optimizer_and_loss()
# Plan step 5 -> implemented in: train_model()
# Plan step 6 -> implemented in: get_predictions()
# Plan step 7 -> implemented in: write_predictions()
# Plan step 8 -> implemented in: print_validation_accuracy()
# Plan step 9 -> implemented in: handle_smoke_mode()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio


def set_seeds():
    """Read SEED env var and seed numpy and torch."""
    seed = int(os.getenv('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    """Load the Cora LCC dataset via nullius_io."""
    d = nio.load()  # dict with keys: features, edges, labels, train_idx, val_idx, test_idx
    features = torch.from_numpy(d['features']).float()          # [N, F]
    edges = torch.from_numpy(d['edges']).long()                # [E, 2]
    labels = torch.from_numpy(d['labels']).long()              # [N]
    train_idx = torch.from_numpy(d['train_idx']).long()
    val_idx = torch.from_numpy(d['val_idx']).long()
    test_idx = torch.from_numpy(d['test_idx']).long()
    return features, edges, labels, train_idx, val_idx, test_idx


class GCNLayer(nn.Module):
    """Single GCN layer using pre‑computed normalized adjacency."""
    def __init__(self, in_features, out_features, adj_norm):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.adj_norm = adj_norm
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: [N, in_features]
        support = torch.matmul(x, self.weight)          # [N, out_features]
        out = torch.sparse.mm(self.adj_norm, support)   # [N, out_features]
        return out


class GCN(nn.Module):
    """2‑layer GCN with ReLU and dropout."""
    def __init__(self, adj_norm, input_dim, hidden_dim, output_dim, dropout_p=0.5):
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim, adj_norm)
        self.gcn2 = GCNLayer(hidden_dim, output_dim, adj_norm)
        self.dropout_p = dropout_p

    def forward(self, x):
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.gcn1(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.gcn2(x)
        return x


def configure_optimizer_and_loss(model):
    """Set up Adam optimizer and CrossEntropyLoss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion


def normalize_adjacency(edges, num_nodes):
    """Return symmetric normalized adjacency with self‑loops as a sparse torch tensor."""
    # Add self‑loops
    self_loop = torch.arange(num_nodes, dtype=torch.long)
    self_loop_edge = self_loop.repeat(2, 1)          # [2, N]
    edge_index = torch.cat([edges.t(), self_loop_edge], dim=1)  # [2, E+N]

    # Compute degree vector
    deg = torch.zeros(num_nodes)
    deg.index_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # Normalization values for each edge
    values = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
    adj_norm = torch.sparse_coo_tensor(edge_index, values, size=(num_nodes, num_nodes)).coalesce()
    return adj_norm


def train_model(model, features, labels, train_idx, val_idx, optimizer, criterion,
                max_epochs, patience):
    """Training loop with early stopping on validation loss."""
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(features)
        loss = criterion(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(features)
            val_loss = criterion(val_out[val_idx], labels[val_idx])
            _, pred_val = val_out[val_idx].max(dim=1)
            acc_val = (pred_val == labels[val_idx]).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d} | Val Loss: {val_loss:.4f} | Val Acc: {acc_val:.4f}')

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def get_predictions(model, features):
    """Return logits for all nodes."""
    model.eval()
    with torch.no_grad():
        logits = model(features)   # [N, num_classes]
    return logits


def write_predictions(preds):
    """Write predictions via nullius_io (expects numpy array or torch tensor)."""
    nio.write_predictions(preds)


def print_validation_accuracy(model, features, labels, val_idx):
    """Compute and print validation accuracy using the current model."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        _, pred = logits[val_idx].max(dim=1)
        acc = (pred == labels[val_idx]).float().mean().item()
    print(f'Final Validation Accuracy: {acc:.4f}')


def handle_smoke_mode():
    """Return max_epochs based on SMOKE env var."""
    smoke = os.getenv('SMOKE') == '1'
    return 1 if smoke else 200


def main():
    set_seeds()
    features, edges, labels, train_idx, val_idx, test_idx = load_data()

    num_nodes = features.shape[0]
    adj_norm = normalize_adjacency(edges, num_nodes)

    model = GCN(
        adj_norm=adj_norm,
        input_dim=features.shape[1],
        hidden_dim=16,
        output_dim=7,
        dropout_p=0.5
    )
    optimizer, criterion = configure_optimizer_and_loss(model)

    max_epochs = handle_smoke_mode()
    patience = 50 if not (os.getenv('SMOKE') == '1') else 0  # patience unused in smoke

    model, _ = train_model(
        model, features, labels, train_idx, val_idx,
        optimizer, criterion, max_epochs=max_epochs, patience=patience
    )

    logits = get_predictions(model, features)
    write_predictions(logits)  # helper will select test_idx and val_idx internally

    print_validation_accuracy(model, features, labels, val_idx)


if __name__ == '__main__':
    main()
