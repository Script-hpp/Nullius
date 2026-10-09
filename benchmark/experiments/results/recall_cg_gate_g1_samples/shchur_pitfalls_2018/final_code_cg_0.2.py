

# Plan step -> implemented in: function or class name
# 1. Data loading -> load_data()
# 2. Pre-processing -> preprocess()
# 3. Model definition -> GCN class
# 4. Optimizer & training setup -> configure_optimizer()
# 5. Training protocol with early stopping -> train()
# 6. Inference & output -> predict_and_write()
# 7. Main driver -> main()

import os
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import nullius_io as nio

CONFIG = {"lr": 0.01, "weight_decay": 0.001, "epochs": 100000, "dropout": 0.8, "hidden": 64, "layers": 2, "patience": 50}

def load_data():
    d = nio.load()
    features = d['features'].astype(np.float32)          # [N, F]
    edges = d['edges'].astype(np.int64)                  # [E, 2]
    labels = d['labels'].astype(np.int64)                # [N]
    train_idx = d['train_idx'].astype(np.int64)
    val_idx = d['val_idx'].astype(np.int64)
    test_idx = d['test_idx'].astype(np.int64)
    return features, edges, labels, train_idx, val_idx, test_idx

def preprocess(features, edges, normalize_features=False):
    N = features.shape[0]
    # Build adjacency (undirected, add self-loops)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(N, N))
    adj = adj + sp.identity(N, dtype=adj.dtype)
    # Symmetric normalization: Â = D̃^-0.5 * Ã * D̃^-0.5
    rowsum = np.array(adj.sum(axis=1)).flatten()
    r_inv_sqrt = np.power(rowsum, -0.5)
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    adj_norm = r_mat_inv_sqrt.dot(adj).dot(r_mat_inv_sqrt)
    adj_norm = adj_norm.tocoo()
    # Convert to torch sparse tensor
    indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    adj_sparse = torch.sparse_coo_tensor(indices, values, adj_norm.shape).coalesce()
    # Features
    if normalize_features:
        # row-normalize each row to L2 norm 1
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        features = features / norms
    features_tensor = torch.from_numpy(features).float()
    return adj_sparse, features_tensor

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_classes, adj_sparse, dropout, weight_decay, num_layers):
        super().__init__()
        self.adj = adj_sparse
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.linears = torch.nn.ModuleList()
        if num_layers == 1:
            self.linears.append(torch.nn.Linear(num_features, num_classes))
        else:
            # first layer
            self.linears.append(torch.nn.Linear(num_features, num_hidden))
            # hidden layers
            for _ in range(num_layers - 2):
                self.linears.append(torch.nn.Linear(num_hidden, num_hidden))
            # final layer
            self.linears.append(torch.nn.Linear(num_hidden, num_classes))
        # Initialize weights with glorot uniform and zeros bias
        for lin in self.linears:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            # apply dropout to inputs of this layer
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.sparse.mm(self.adj, x)  # Â * X
            x = lin(x)
            if i < len(self.linears) - 1:  # apply activation except last
                x = F.relu(x)
        return x

    def l2_loss(self):
        # L2 only on first-layer weights as in original TF code
        if self.num_layers >= 1:
            return self.weight_decay * torch.sum(self.linears[0].weight ** 2)
        else:
            return 0.0

def configure_optimizer(model, lr):
    # No weight decay in optimizer; we add L2 loss manually
    return torch.optim.Adam(model.parameters(), lr=lr)

def train(model, optimizer, features, labels, train_idx, val_idx, max_epochs, patience):
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features)
        loss_train = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss_train = loss_train + model.l2_loss()
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(features)
            loss_val = F.cross_entropy(logits_val[val_idx], labels[val_idx])
        # Early stopping (gnnbench style: stop if no improvement for patience epochs)
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_and_write(model, features):
    model.eval()
    with torch.no_grad():
        logits = model(features)
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    nio.write_predictions(preds)
    return preds, logits

def main():
    # Seed
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Smoke mode
    smoke = os.environ.get("SMOKE") == "1"

    # Load data
    features_np, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_classes = int(labels.max() + 1)  # labels are 0..6 for known nodes

    # Preprocess
    adj_sparse, features_tensor = preprocess(features_np, edges,
                                             normalize_features=False)  # from config

    # Build model
    model = GCN(num_features=features_tensor.shape[1],
                num_hidden=CONFIG["hidden"],
                num_classes=num_classes,
                adj_sparse=adj_sparse,
                dropout=CONFIG["dropout"],
                weight_decay=CONFIG["weight_decay"],
                num_layers=CONFIG["layers"])

    optimizer = configure_optimizer(model, CONFIG["lr"])

    # Determine epochs
    max_epochs = 1 if smoke else CONFIG["epochs"]

    # Train
    model = train(model, optimizer, features_tensor, torch.from_numpy(labels),
                  train_idx, val_idx, max_epochs, CONFIG["patience"])

    # Predict and write output
    preds, logits = predict_and_write(model, features_tensor)

    # Compute validation accuracy for logging
    model.eval()
    with torch.no_grad():
        val_logits = model(features_tensor)
    val_pred = torch.argmax(val_logits[val_idx], dim=1).cpu().numpy()
    val_true = labels[val_idx]
    val_acc = (val_pred == val_true).mean()
    print(f"Validation accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()