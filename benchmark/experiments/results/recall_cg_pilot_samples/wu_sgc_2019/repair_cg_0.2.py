import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

# Hyperparameters as required
CONFIG = {"lr": 0.2, "weight_decay": 5e-06, "epochs": 100, "dropout": 0, "hidden": 0}

# Plan step -> implemented in: SGC class (via get_model)
# Plan step -> implemented in: row_normalize function
# Plan step -> implemented in: aug_norm_adj function
# Plan step -> implemented in: sgc_precompute function
# Plan step -> implemented in: optimizer inside train_model
# Plan step -> implemented in: train_model function
# Plan step -> implemented in: validation tracking inside train_model
# Plan step -> implemented in: predict function
# Plan step -> implemented in: set_seed function
# Plan step -> implemented in: CPU only (no cuda)
# Plan step -> implemented in: CONFIG hidden=0 etc. (unused for SGC but passed)

def set_seed(seed, cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def row_normalize(mx):
    """Row-normalize sparse or dense matrix"""
    if sp.issparse(mx):
        row = np.array(mx.sum(axis=1)).flatten()
    else:
        row = mx.sum(axis=1)
    # avoid division by zero
    row_inv = np.power(row, -1.0)
    row_inv[np.isinf(row_inv)] = 0.0
    if sp.issparse(mx):
        inv = sp.diags(row_inv)
        mx = inv.dot(mx)
    else:
        mx = mx * row_inv[:, np.newaxis]
    return mx

def aug_norm_adj(edges, num_nodes):
    """Adjacency matrix with self-loops and symmetric normalization (AugNormAdj)"""
    data = np.ones(edges.shape[0], dtype=np.float32)
    row = edges[:, 0]
    col = edges[:, 1]
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
    # add self-loops
    adj = adj + sp.eye(num_nodes, dtype=np.float32)
    # degree matrix
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    # normalized adjacency
    adj_norm = D_inv_sqrt.dot(adj).dot(D_inv_sqrt)
    # convert to torch sparse tensor
    adj_norm = adj_norm.tocoo()
    indices = torch.from_numpy(
        np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64)
    )
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    shape = adj_norm.shape
    adj_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
    adj_tensor = adj_tensor.coalesce()
    return adj_tensor

def sgc_precompute(features, adj, degree):
    """Propagate features `degree` times using the normalized adjacency."""
    for _ in range(degree):
        features = torch.spmm(adj, features)
    return features

class SGC(nn.Module):
    """Simple Graph Convolution (SGC) model: a single linear layer."""
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()
        self.linear = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.linear(x)

def get_model(model_opt, nfeat, nclass, nhid=0, dropout=0, cuda=False):
    if model_opt == "SGC":
        model = SGC(nfeat=nfeat, nclass=nclass)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))
    if cuda:
        model.cuda()
    return model

def accuracy(output, labels):
    preds = output.argmax(dim=1)
    correct = (preds == labels).sum().float()
    return correct / len(labels)

def train_model(model, train_features, train_labels, val_features, val_labels, epochs, lr, weight_decay):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss = F.cross_entropy(output, train_labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(val_features)
            val_acc = accuracy(val_output, val_labels)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc

def predict(model, features):
    model.eval()
    with torch.no_grad():
        output = model(features)
        return output.argmax(dim=1)

def main():
    seed = int(os.environ.get("SEED", 0))
    set_seed(seed, cuda=False)

    smoke = os.environ.get("SMOKE") == "1"

    data = np.load("/data/cora_public.npz")
    features = data["features"].astype(np.float32)          # [2708, 1433]
    edges = data["edges"].astype(np.int64)                  # [E, 2]
    labels = data["labels"].astype(np.int64)                # [2708]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    num_nodes = features.shape[0]

    # Step 2: Feature preprocessing (row-normalize)
    features_norm = row_normalize(features)

    # Step 3: Adjacency preprocessing (AugNormAdj)
    adj = aug_norm_adj(edges, num_nodes)

    # Convert features to torch tensor
    features_tensor = torch.from_numpy(features_norm).float()

    # Step 4: Graph propagation (degree = 2 from argparse default)
    degree = 2
    propagated_features = sgc_precompute(features_tensor, adj, degree)

    # Prepare label tensors (ignore -1 for test)
    label_tensor = torch.from_numpy(labels).long()

    # Index features/labels for splits
    train_features = propagated_features[train_idx]
    train_labels = label_tensor[train_idx]
    val_features = propagated_features[val_idx]
    val_labels = label_tensor[val_idx]
    test_features = propagated_features[test_idx]
    test_labels = label_tensor[test_idx]  # contains -1, but we won't use for loss

    # Model
    nfeat = propagated_features.shape[1]
    nclass = int(len(np.unique(labels[labels >= 0])))  # should be 7
    model = get_model("SGC", nfeat=nfeat, nclass=nclass,
                      nhid=CONFIG["hidden"], dropout=CONFIG["dropout"], cuda=False)

    # Training epochs
    epochs = 1 if smoke else CONFIG["epochs"]

    # Step 5-7: Train with validation tracking
    model, best_val_acc = train_model(
        model,
        train_features,
        train_labels,
        val_features,
        val_labels,
        epochs=epochs,
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Step 8: Predict on test (full or subset)
    if smoke:
        test_subset = np.sort(test_idx)[:20]
        test_pred = predict(model, propagated_features[test_subset])
        # Write predictions.csv with only these ids
        out_path = "/output/predictions.csv"
        os.makedirs("/output", exist_ok=True)
        with open(out_path, "w") as f:
            f.write("id,prediction\n")
            for idx, pred in zip(test_subset, test_pred.tolist()):
                f.write(f"{idx},{pred}\n")
    else:
        test_pred = predict(model, test_features)
        out_path = "/output/predictions.csv"
        os.makedirs("/output", exist_ok=True)
        with open(out_path, "w") as f:
            f.write("id,prediction\n")
            for idx, pred in zip(test_idx, test_pred.tolist()):
                f.write(f"{idx},{pred}\n")

    # Validation predictions file (always required)
    if smoke:
        val_subset = np.sort(val_idx)[:20]
        val_pred = predict(model, propagated_features[val_subset])
        val_out_path = "/output/val_predictions.csv"
        with open(val_out_path, "w") as f:
            f.write("id,prediction\n")
            for idx, pred in zip(val_subset, val_pred.tolist()):
                f.write(f"{idx},{pred}\n")
    else:
        val_pred = predict(model, val_features)
        val_out_path = "/output/val_predictions.csv"
        with open(val_out_path, "w") as f:
            f.write("id,prediction\n")
            for idx, pred in zip(val_idx, val_pred.tolist()):
                f.write(f"{idx},{pred}\n")

    # Print final validation accuracy
    print(best_val_acc.item())

if __name__ == "__main__":
    main()
