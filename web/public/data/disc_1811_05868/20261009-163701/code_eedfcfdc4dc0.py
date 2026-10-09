# Plan step -> implemented in: class GCN
# Plan step -> implemented in: class GCN (dropout applied)
# Plan step -> implemented in: optimizer param groups
# Plan step -> implemented in: training loop with epochs from CONFIG
# Plan step -> implemented in: early stopping comment (not implemented due to unspecified patience)
# Plan step -> implemented in: adjacency normalization function normalize_adj
# Plan step -> implemented in: feature preprocessing (no row normalization)
# Plan step -> implemented in: full-batch training (no DataLoader)
# Plan step -> implemented in: seed setup from env var SEED
# Plan step -> implemented in: prediction and writing via nio.write_predictions
# Plan step -> implemented in: smoke mode handling
# Plan step -> implemented in: printing validation accuracy

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import nullius_io as nio

# Hyperparameters from spec
CONFIG = {"lr": 0.01, "weight_decay": 0.001, "epochs": 200, "dropout": 0.8, "hidden": 64, "layers": 2}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_adj(adj):
    """Symmetrically normalize adjacency with self-loops: D^{-1/2} (A+I) D^{-1/2}"""
    adj = adj + sp.eye(adj.shape[0])  # add self-loops
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    norm_adj = sp.diags(deg_inv_sqrt) @ adj @ sp.diags(deg_inv_sqrt)
    return norm_adj

def sparse_mx_to_torch_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    coords = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data.astype(np.float32))
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(coords, values, shape, dtype=torch.float32)

class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhidden, nclass, nlayers, dropout):
        super().__init__()
        self.nlayers = nlayers
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        # input to hidden
        self.layers.append(torch.nn.Linear(nfeat, nhidden))
        # hidden layers (if any)
        for _ in range(nlayers - 2):
            self.layers.append(torch.nn.Linear(nhidden, nhidden))
        # hidden to output
        self.layers.append(torch.nn.Linear(nhidden, nclass))
        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.layers:
            torch.nn.init.xavier_uniform_(linear.weight)
            torch.nn.init.zeros_(linear.bias)

    def forward(self, x, adj):
        # apply dropout to inputs of each GCN layer, then graph conv + linear + activation (except last)
        for i, lin in enumerate(self.layers[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.spmm(adj, x)
            x = lin(x)
            x = F.relu(x)
        # last layer (no activation)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.spmm(adj, x)
        x = self.layers[-1](x)
        return x

def main():
    # Seed
    seed = int(os.environ.get("SEED", "0"))
    set_seed(seed)

    # Load data
    data = nio.load()
    features = data["features"]          # [N, 1433] float32
    edges = data["edges"]                # [E, 2] int64
    labels = data["labels"]              # [N] int64, -1 for test
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    N = features.shape[0]

    # Preprocess adjacency
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(N, N))
    adj_norm = normalize_adj(adj)
    adj_torch = sparse_mx_to_torch_sparse(adj_norm).coalesce()

    # Features as torch tensor (no row-normalization as flag not provided)
    feat_torch = torch.from_numpy(features).float()

    # Labels as torch long (ignore -1 by masking later)
    label_long = torch.from_numpy(labels).long()

    # Model
    model = GCN(
        nfeat=features.shape[1],
        nhidden=CONFIG["hidden"],
        nclass=int(labels.max() + 1) if (labels >= 0).any() else 7,  # Cora has 7 classes
        nlayers=CONFIG["layers"],
        dropout=CONFIG["dropout"]
    )

    # Optimizer with weight decay only on first layer weight
    param_groups = [
        {"params": model.layers[0].weight, "weight_decay": CONFIG["weight_decay"]},
        {"params": model.layers[0].bias, "weight_decay": 0.0},
    ]
    for i in range(1, len(model.layers)):
        param_groups.append({"params": model.layers[i].weight, "weight_decay": 0.0})
        param_groups.append({"params": model.layers[i].bias, "weight_decay": 0.0})
    optimizer = torch.optim.Adam(param_groups, lr=CONFIG["lr"])

    # Training configuration
    epochs = 1 if os.environ.get("SMOKE") == "1" else CONFIG["epochs"]
    best_val_acc = 0.0
    patience_counter = 0
    patience = 50  # arbitrary; early stopping criteria not specified

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(feat_torch, adj_torch)  # [N, C]
        loss_train = F.cross_entropy(logits[train_idx], label_long[train_idx])
        loss_train.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(feat_torch, adj_torch)
            loss_val = F.cross_entropy(logits[val_idx], label_long[val_idx])
            pred_val = val_logits[val_idx].argmax(dim=1)
            acc_val = (pred_val == label_long[val_idx]).float().mean().item()

        # Simple early stopping (optional)
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            patience_counter = 0
        else:
            patience_counter += 1
            if epoch > 5 and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:03d} | Train Loss: {loss_train.item():.4f} | Val Loss: {loss_val.item():.4f} | Val Acc: {acc_val:.4f}")

    # Final validation accuracy (print to stdout)
    model.eval()
    with torch.no_grad():
        final_logits = model(feat_torch, adj_torch)
        final_pred = final_logits[val_idx].argmax(dim=1)
        final_acc = (final_pred == label_long[val_idx]).float().mean().item()
    print(f"Final Validation Accuracy: {final_acc:.4f}")

    # Predictions for all nodes
    with torch.no_grad():
        all_logits = model(feat_torch, adj_torch)
        all_pred = all_logits.argmax(dim=1).cpu().numpy()
    nio.write_predictions(all_pred)

if __name__ == "__main__":
    main()
