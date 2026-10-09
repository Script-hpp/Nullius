

# Plan step -> implemented in: load_data
# Plan step -> implemented in: preprocess_adj_and_features
# Plan step -> implemented in: GCN class
# Plan step -> implemented in: configure_optimizer
# Plan step -> implemented in: train_model
# Plan step -> implemented in: evaluate_and_predict
# Plan step -> implemented in: nio.write_predictions call

import os
import numpy as np
import torch
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {"lr": 0.01, "epochs": 100000, "layers": 2, "patience": 50}
CONFIG["hidden"] = 64
CONFIG["weight_decay"] = 0.001
CONFIG["dropout"] = 0.8

def load_data():
    d = nio.load()
    return d

def preprocess_adj_and_features(features_np, edges_np, N):
    # adjacency with self-loops
    edges = torch.from_numpy(edges_np).long()          # [E,2]
    self_loop = torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(1,2)
    self_loop[:,1] = torch.arange(N, dtype=torch.long)
    edge_index = torch.cat([edges, self_loop], dim=0)  # [E+N,2]
    adj = torch.sparse_coo_tensor(
        edge_index.t(),
        torch.ones(edge_index.size(0), dtype=torch.float32),
        (N, N)
    ).coalesce()
    # degree
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    # symmetric normalization
    indices = adj.indices()
    vals = adj.values()
    rows = indices[0,:]
    cols = indices[1,:]
    vals = vals * deg_inv_sqrt[cols]
    vals = vals * deg_inv_sqrt[rows]
    adj_norm = torch.sparse_coo_tensor(indices, vals, (N,N)).coalesce()
    # features as sparse (no normalization)
    features = torch.from_numpy(features_np)           # [N,F] dense
    nz = features.nonzero(as_tuple=False)              # [nnz,2]
    indices_f = nz.t()                                 # [2,nnz]
    values_f = features[nz[:,0], nz[:,1]]
    features_sparse = torch.sparse_coo_tensor(indices_f, values_f, features.size()).coalesce()
    return adj_norm, features_sparse

class GCN(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout, weight_decay, num_layers=2):
        super().__init__()
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        if num_layers == 2:
            self.weights.append(torch.nn.Parameter(torch.empty(in_features, hidden_features)))
            self.biases.append(torch.nn.Parameter(torch.zeros(hidden_features)))
            self.weights.append(torch.nn.Parameter(torch.empty(hidden_features, out_features)))
            self.biases.append(torch.nn.Parameter(torch.zeros(out_features)))
            torch.nn.init.xavier_uniform_(self.weights[0])
            torch.nn.init.xavier_uniform_(self.weights[1])
        else:
            raise ValueError("Only 2 layers supported in this implementation")
    def forward(self, adj_norm, features_sparse):
        x = features_sparse
        for i in range(self.num_layers):
            if i == 0:
                x = self._dropout_sparse(x) if self.training else x
                support = torch.sparse.mm(x, self.weights[i])
                out = torch.sparse.mm(adj_norm, support) + self.biases[i]
                out = F.relu(out)
                x = out
            else:
                x = F.dropout(x, p=self.dropout, training=self.training)
                support = torch.mm(x, self.weights[i])
                out = torch.sparse.mm(adj_norm, support) + self.biases[i]
                x = out
        return x
    def _dropout_sparse(self, x):
        if not x.is_coalesced():
            x = x.coalesce()
        vals = x.values()
        mask = torch.rand_like(vals) > self.dropout
        vals = vals * mask.float() / (1.0 - self.dropout)
        return torch.sparse_coo_tensor(x.indices(), vals, x.size())
    def l2_loss(self):
        # L2 only on first layer as per original implementation
        return self.weight_decay * torch.norm(self.weights[0], p=2) ** 2

def configure_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

def train_model(model, adj_norm, features_sparse, labels, train_idx, val_idx, epochs, patience):
    optimizer = configure_optimizer(model, CONFIG["lr"])
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(adj_norm, features_sparse)
        loss_train = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss = loss_train + model.l2_loss()
        loss.backward()
        optimizer.step()
        # validation
        model.eval()
        with torch.no_grad():
            logits_val = model(adj_norm, features_sparse)
            loss_val = F.cross_entropy(logits_val[val_idx], labels[val_idx])
        model.train()
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate_and_predict(model, adj_norm, features_sparse):
    model.eval()
    with torch.no_grad():
        logits_all = model(adj_norm, features_sparse)
        preds_all = logits_all.argmax(dim=1).cpu().numpy()
    return logits_all, preds_all

def main():
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    d = load_data()
    features_np = d["features"]
    edges_np = d["edges"]
    labels_np = d["labels"]
    train_idx_np = d["train_idx"]
    val_idx_np = d["val_idx"]
    test_idx_np = d["test_idx"]

    N = features_np.shape[0]
    labels = torch.from_numpy(labels_np).long()
    train_idx = torch.from_numpy(train_idx_np).long()
    val_idx = torch.from_numpy(val_idx_np).long()
    test_idx = torch.from_numpy(test_idx_np).long()

    # Determine number of classes from training labels (ignore -1 in test)
    train_labels = labels[train_idx]
    num_classes = int(train_labels.max().item()) + 1

    adj_norm, features_sparse = preprocess_adj_and_features(features_np, edges_np, N)

    model = GCN(
        in_features=features_np.shape[1],
        hidden_features=CONFIG["hidden"],
        out_features=num_classes,
        dropout=CONFIG["dropout"],
        weight_decay=CONFIG["weight_decay"],
        num_layers=CONFIG["layers"]
    )

    epochs = 1 if os.environ.get("SMOKE") == "1" else CONFIG["epochs"]
    patience = CONFIG["patience"]

    model = train_model(model, adj_norm, features_sparse, labels, train_idx, val_idx, epochs, patience)

    _, preds_all = evaluate_and_predict(model, adj_norm, features_sparse)

    # validation accuracy from best model
    with torch.no_grad():
        logits_val = model(adj_norm, features_sparse)
        acc_val = (logits_val[val_idx].argmax(dim=1) == labels[val_idx]).float().mean().item()
    print(f"Validation accuracy: {acc_val:.4f}")

    nio.write_predictions(preds_all)

if __name__ == "__main__":
    main()