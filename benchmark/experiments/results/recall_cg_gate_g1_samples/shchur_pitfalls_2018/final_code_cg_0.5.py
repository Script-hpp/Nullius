

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

CONFIG = {"lr": 0.01, "weight_decay": 0.001, "epochs": 100000, "dropout": 0.8, "hidden": 64, "layers": 2, "patience": 50}

def load_data():
    import nullius_io as nio
    d = nio.load()
    return d

def preprocess_adj(edges, num_nodes):
    # edges: np.ndarray [E, 2] (directed pairs, both directions present)
    self_loop = np.arange(num_nodes)[:, None].repeat(2, axis=1)  # [N,2] (i,i)
    edge_list = np.concatenate([edges, self_loop], axis=0)      # [E+N, 2]
    idx = torch.from_numpy(edge_list.T).long()                  # [2, E+N]
    vals = torch.ones(idx.size(1), dtype=torch.float32)
    adj_sparse = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes))
    deg = torch.sparse.sum(adj_sparse, dim=1).to_dense()       # [N]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    row, col = idx
    norm_vals = deg_inv_sqrt[row] * vals * deg_inv_sqrt[col]
    adj_norm = torch.sparse_coo_tensor(idx, norm_vals, (num_nodes, num_nodes)).coalesce()
    return adj_norm

class GCN(torch.nn.Module):
    def __init__(self, in_features, hidden, out_features, adj_norm, num_layers):
        super().__init__()
        self.adj_norm = adj_norm
        self.num_layers = num_layers
        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        for i in range(num_layers):
            in_dim = in_features if i == 0 else hidden
            out_dim = hidden if i < num_layers - 1 else out_features
            w = torch.nn.Parameter(torch.empty(in_dim, out_dim))
            b = torch.nn.Parameter(torch.zeros(out_dim))
            torch.nn.init.xavier_uniform_(w)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x, dropout_flag):
        h = x
        for i in range(self.num_layers):
            if dropout_flag > 0:
                h = F.dropout(h, p=dropout_flag, training=True)
            support = torch.sparse.mm(self.adj_norm, h)
            h = torch.mm(support, self.weights[i]) + self.biases[i]
            if i < self.num_layers - 1:
                h = F.relu(h)
        return h

def configure_optimizer(model, lr):
    # No weight decay in optimizer; L2 penalty applied manually to first layer only
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

def train(model, optimizer, features, labels, train_idx, val_idx, adj_norm, max_epochs, patience, dropout_val):
    best_loss = float('inf')
    wait = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, dropout_flag=dropout_val)
        loss_train = F.cross_entropy(logits[train_idx], labels[train_idx])
        # L2 regularization on first layer weights only
        l2_reg = torch.norm(model.weights[0]) ** 2
        loss = loss_train + CONFIG["weight_decay"] * l2_reg
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
    features_np = data['features']
    edges_np = data['edges']
    labels_np = data['labels']
    train_idx_np = data['train_idx']
    val_idx_np = data['val_idx']
    test_idx_np = data['test_idx']

    N = features_np.shape[0]
    features = torch.from_numpy(features_np).float()
    labels = torch.from_numpy(labels_np).long()
    train_idx = torch.from_numpy(train_idx_np).long()
    val_idx = torch.from_numpy(val_idx_np).long()
    test_idx = torch.from_numpy(test_idx_np).long()

    adj_norm = preprocess_adj(edges_np, N)

    n_classes = int(labels.max().item() + 1)  # 7
    model = GCN(in_features=features.shape[1],
                hidden=CONFIG['hidden'],
                out_features=n_classes,
                adj_norm=adj_norm,
                num_layers=CONFIG['layers'])

    optimizer = configure_optimizer(model, lr=CONFIG['lr'])

    max_epochs = 1 if smoke else CONFIG['epochs']
    patience = CONFIG['patience'] if not smoke else 0

    train(model, optimizer, features, labels, train_idx, val_idx, adj_norm,
          max_epochs=max_epochs,
          patience=patience,
          dropout_val=CONFIG['dropout'])

    preds = predict(model, features)
    preds_np = preds.cpu().numpy()
    import nullius_io as nio
    nio.write_predictions(preds_np)

    with torch.no_grad():
        val_pred = preds[val_idx]
        val_true = labels[val_idx]
        acc = (val_pred == val_true).float().mean().item()
    print(f"Validation accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()