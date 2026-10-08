# Plan step -> implemented in: function or class name
# 1. Model – SGC: a single linear layer W ∈ ℝ^{F×C} applied to k‑step propagated features; bias term included -> class SGC
# 2. Feature preprocessing – row‑normalize the raw bag‑of‑words matrix (float32) -> function row_normalize
# 3. Adjacency preprocessing – compute symmetrically normalized adjacency with self‑loops (AugNormAdj) -> function aug_norm_adj
# 4. Graph propagation – obtain X̂ = (Ã)^k X with k = degree = 2; implemented via sgc_precompute -> function sgc_precompute
# 5. Optimizer – Adam, learning rate lr = 0.2, weight decay = 5e‑6 -> optimizer uses CONFIG["lr"] and CONFIG["weight_decay"]
# 6. Training – run for epochs = 100 on the full batch (all training nodes); no dropout -> train_loop function
# 7. Validation – after each epoch compute accuracy on val_idx; keep the model with highest validation accuracy -> train_loop tracks best model
# 8. Testing – evaluate the selected model on test_idx; output class predictions -> predict function
# 9. Randomness – read integer seed from environment variable SEED; set numpy and torch seeds accordingly -> seed setup after imports
# 10. Device – CPU only -> no CUDA code, tensors stay on CPU
# 11. Additional notes – No hidden layers, no learning‑rate schedule, no patience or heads -> CONFIG["hidden"]=0, no schedule used

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONFIG = {"lr": 0.2, "weight_decay": 5e-06, "epochs": 100, "dropout": 0, "hidden": 0, "degree": 2}

def row_normalize(mx):
    """Row-normalize sparse or dense matrix"""
    if isinstance(mx, torch.Tensor):
        rowsum = mx.sum(dim=1, keepdim=True)
        rowsum[rowsum == 0] = 1.0
        return mx / rowsum
    else:
        raise TypeError("Expected torch.Tensor")

def aug_norm_adj(adj):
    """Adjacency with self-loops and symmetric normalization (Ã = D^{-1/2} (A+I) D^{-1/2})"""
    N = adj.shape[0]
    # add self-loops
    idx = torch.arange(N, dtype=torch.long)
    self_loop_idx = torch.stack([idx, idx], dim=0)
    self_loop_val = torch.ones(N, dtype=torch.float32)
    if adj.is_sparse:
        adj_idx = adj._indices()
        adj_val = adj._values()
        idx_all = torch.cat([adj_idx, self_loop_idx], dim=1)
        val_all = torch.cat([adj_val, self_loop_val])
    else:
        idx_all = self_loop_idx
        val_all = self_loop_val
    adj_new = torch.sparse_coo_tensor(idx_all, val_all, (N, N)).coalesce()
    # compute degree
    rowsum = torch.sparse.sum(adj_new, dim=1).to_dense()
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    # create D^{-1/2} * A * D^{-1/2}
    vals = adj_new._values() * d_inv_sqrt[adj_new._indices()[0]] * d_inv_sqrt[adj_new._indices()[1]]
    adj_norm = torch.sparse_coo_tensor(adj_new._indices(), vals, (N, N)).coalesce()
    return adj_norm

def sgc_precompute(features, adj, degree):
    """Compute (adj)^degree * features"""
    out = features
    for _ in range(degree):
        out = torch.sparse.mm(adj, out)
    return out

def accuracy(output, labels):
    preds = output.max(1)[1]
    correct = preds.eq(labels).sum().float()
    return correct / labels.numel()

class SGC(nn.Module):
    def __init__(self, nfeat, nclass):
        super().__init__()
        self.W = nn.Linear(nfeat, nclass, bias=True)

    def forward(self, x):
        return self.W(x)

def predict(model, features, idx):
    model.eval()
    with torch.no_grad():
        logits = model(features[idx])
        return logits.argmax(dim=1)

def main():
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    smoke = os.environ.get("SMOKE", "0") == "1"

    data = np.load("/data/cora_public.npz", allow_pickle=False)
    features_np = data["features"].astype(np.float32)
    edges = data["edges"]  # shape [E,2]
    labels_np = data["labels"].astype(np.int64)
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    N = features_np.shape[0]
    # features as torch tensor
    features = torch.from_numpy(features_np)
    features = row_normalize(features)

    # labels as torch tensor
    labels = torch.from_numpy(labels_np)

    # adjacency from edges
    edge_idx = torch.from_numpy(edges.T).long()  # [2, E]
    edge_val = torch.ones(edge_idx.shape[1], dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_idx, edge_val, (N, N)).coalesce()

    # adjacency normalization (AugNormAdj)
    adj_norm = aug_norm_adj(adj)

    # graph propagation
    degree = CONFIG["degree"]
    propagated = sgc_precompute(features, adj_norm, degree)

    # model
    train_labels = labels[train_idx]
    num_classes = int(train_labels.max().item() + 1)
    model = SGC(propagated.shape[1], num_classes)

    optimizer = optim.Adam(model.parameters(),
                           lr=CONFIG["lr"],
                           weight_decay=CONFIG["weight_decay"])

    epochs = 1 if smoke else CONFIG["epochs"]
    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(propagated[train_idx])
        loss = F.cross_entropy(output, train_labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(propagated[val_idx])
            val_acc = accuracy(val_output, labels[val_idx])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # test predictions
    test_pred = predict(model, propagated, test_idx)
    # validation predictions
    val_pred = predict(model, propagated, val_idx)

    # output directories
    os.makedirs("/output", exist_ok=True)

    # write test predictions
    if smoke:
        test_ids = sorted(test_idx)[:20]
    else:
        test_ids = sorted(test_idx)
    with open("/output/predictions.csv", "w") as f:
        f.write("id,prediction\n")
        for idx in test_ids:
            f.write(f"{idx},{int(test_pred[idx])}\n")

    # write validation predictions
    if smoke:
        val_ids = sorted(val_idx)[:20]
    else:
        val_ids = sorted(val_idx)
    with open("/output/val_predictions.csv", "w") as f:
        f.write("id,prediction\n")
        for idx in val_ids:
            f.write(f"{idx},{int(val_pred[idx])}\n")

    print(f"Best validation accuracy: {best_val_acc.item():.4f}")

if __name__ == "__main__":
    main()
