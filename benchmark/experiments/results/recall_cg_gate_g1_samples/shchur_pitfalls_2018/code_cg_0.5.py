# Plan step -> implemented in: load_data
# Plan step -> implemented in: preprocess_adj
# Plan step -> implemented in: GCN class
# Plan step -> implemented in: configure_optimizer
# Plan step -> implemented in: train function
# Plan step -> implemented in: predict function
# Plan step -> implemented in: main

import os
import torch
import torch.nn.functional as F
import numpy as np

# Hyperparameters to be read only from this dict
CONFIG = {
    "lr": 0.01,
    "weight_decay": 0.001,
    "epochs": 100000,
    "dropout": 0.8,
    "hidden": 64,
    "layers": 2,
    "patience": 50,
}

def load_data():
    import nullius_io as nio
    d = nio.load()
    # d contains: features, edges, labels, train_idx, val_idx, test_idx
    return d

def preprocess_adj(edges, num_nodes):
    # edges: np.ndarray [E, 2] (directed pairs, both directions present)
    # Add self-loops
    self_loop = np.arange(num_nodes)[:, None].repeat(2, axis=1)  # [N,2] (i,i)
    edge_list = np.concatenate([edges, self_loop], axis=0)      # [E+N, 2]
    # Build COO tensor
    idx = torch.from_numpy(edge_list.T).long()                  # [2, E+N]
    vals = torch.ones(idx.size(1), dtype=torch.float32)
    # Compute degree
    adj_sparse = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes))
    deg = torch.sparse.sum(adj_sparse, dim=1).to_dense()       # [N]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    # Normalize values
    row, col = idx
    norm_vals = deg_inv_sqrt[row] * vals * deg_inv_sqrt[col]
    adj_norm = torch.sparse_coo_tensor(idx, norm_vals, (num_nodes, num_nodes)).coalesce()
    return adj_norm

class GCN(torch.nn.Module):
    def __init__(self, in_features, hidden, out_features, adj_norm):
        super().__init__()
        self.adj_norm = adj_norm
        self.W0 = torch.nn.Parameter(torch.empty(in_features, hidden))
        self.b0 = torch.nn.Parameter(torch.zeros(hidden))
        self.W1 = torch.nn.Parameter(torch.empty(hidden, out_features))
        self.b1 = torch.nn.Parameter(torch.zeros(out_features))
        torch.nn.init.glorot_uniform_(self.W0)
        torch.nn.init.glorot_uniform_(self.W1)

    def forward(self, x, dropout_flag):
        # Apply dropout to input
        if dropout_flag > 0:
            x = F.dropout(x, p=dropout_flag, training=True)
        # Layer 0
        support0 = torch.sparse.mm(self.adj_norm, x)          # [N, in_features]
        out0 = torch.mm(support0, self.W0) + self.b0
        out0 = F.relu(out0)
        # Dropout after first layer (input to second layer)
        if dropout_flag > 0:
            out0 = F.dropout(out0, p=dropout_flag, training=True)
        # Layer 1
        support1 = torch.sparse.mm(self.adj_norm, out0)
        out1 = torch.mm(support1, self.W1) + self.b1
        return out1

def configure_optimizer(model, lr):
    # We will handle weight decay manually for first layer only
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

def train(model, optimizer, features, labels, train_idx, val_idx, adj_norm, max_epochs, patience, dropout_val, weight_decay):
    best_loss = float('inf')
    wait = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, dropout_flag=dropout_val)
        loss_train = F.cross_entropy(logits[train_idx], labels[train_idx])
        # L2 regularization on first layer weights only
        l2_reg = torch.norm(model.W0) ** 2
        loss = loss_train + weight_decay * l2_reg
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(features, dropout_flag=0.0)
            loss_val = F.cross_entropy(logits_val[val_idx], labels[val_idx])

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss

def predict(model, features):
    model.eval()
    with torch.no_grad():
        logits = model(features, dropout_flag=0.0)
        return torch.argmax(logits, dim=1)

def main():
    seed = int(os.environ.get('SEED', 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    smoke = os.environ.get('SMOKE') == '1'

    data = load_data()
    features_np = data['features']          # [N, 1433] float32
    edges_np = data['edges']                # [E, 2] int64
    labels_np = data['labels']              # [N] int64, -1 for test nodes
    train_idx_np = data['train_idx']        # [140]
    val_idx_np = data['val_idx']            # [210]
    test_idx_np = data['test_idx']          # [2135]

    N = features_np.shape[0]
    features = torch.from_numpy(features_np).float()
    labels = torch.from_numpy(labels_np).long()
    train_idx = torch.from_numpy(train_idx_np).long()
    val_idx = torch.from_numpy(val_idx_np).long()
    test_idx = torch.from_numpy(test_idx_np).long()

    adj_norm = preprocess_adj(edges_np, N)

    model = GCN(in_features=features.shape[1],
                hidden=CONFIG['hidden'],
                out_features=int(labels.max().item() + 1),  # 7 classes
                adj_norm=adj_norm)

    optimizer = configure_optimizer(model, lr=CONFIG['lr'])

    max_epochs = 1 if smoke else CONFIG['epochs']
    patience = CONFIG['patience'] if not smoke else 0  # patience irrelevant in smoke

    train(model, optimizer, features, labels, train_idx, val_idx, adj_norm,
          max_epochs=max_epochs,
          patience=patience,
          dropout_val=CONFIG['dropout'],
          weight_decay=CONFIG['weight_decay'])

    preds = predict(model, features)
    # Convert to numpy for nullius_io
    preds_np = preds.cpu().numpy()
    # Write predictions (helper will select test_idx and val_idx)
    import nullius_io as nio
    nio.write_predictions(preds_np)

    # Print validation accuracy
    with torch.no_grad():
        val_pred = preds[val_idx]
        val_true = labels[val_idx]
        acc = (val_pred == val_true).float().mean().item()
    print(f"Validation accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
