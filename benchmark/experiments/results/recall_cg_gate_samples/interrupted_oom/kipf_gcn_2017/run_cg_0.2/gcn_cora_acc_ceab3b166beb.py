

# Plan step -> implemented in: GCN class
# Plan step -> implemented in: GraphConvolution class
# Plan step -> implemented in: training loop with dropout
# Plan step -> implemented in: weight decay applied via optimizer
# Plan step -> implemented in: Adam optimizer with lr from CONFIG
# Plan step -> implemented in: feature row-normalization and adjacency normalization
# Plan step -> implemented in: full-batch training
# Plan step -> implemented in: epoch loop with early stopping patience
# Plan step -> implemented in: validation set used for early stopping
# Plan step -> implemented in: test accuracy from best model
# Plan step -> implemented in: seeding from ENV SEED
# Plan step -> implemented in: no learning rate schedule
# Plan step -> implemented in: follows official TF GCN design
# Plan step -> implemented in: batch size full, layers 2, heads 1 (unspecified)
# Plan step -> implemented in: writing predictions via nullius_io

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import nullius_io as nio

CONFIG = {"lr": 0.01, "weight_decay": 0.0005, "epochs": 200,
          "dropout": 0.5, "hidden": 16, "patience": 10}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.dropout = dropout
        self.act = act

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)
        x = torch.mm(adj, x)
        if self.act is not None:
            x = self.act(x)
        return x

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, dropout=dropout, act=F.relu)
        self.gc2 = GraphConvolution(nhid, nclass, dropout=dropout, act=None)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        return x

def main():
    seed = int(os.environ.get("SEED", "123"))
    set_seed(seed)

    smoke = os.environ.get("SMOKE") == "1"

    data = nio.load()
    features = data["features"].astype(np.float32)          # [N, 1433]
    edges = data["edges"]                                   # [E, 2]
    labels = data["labels"].astype(np.int64)                # [N]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    N = features.shape[0]

    # ---- Preprocess features: row-normalize ----
    rowsum = features.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1
    features_norm = features / rowsum
    features_tensor = torch.from_numpy(features_norm)

    # ---- Build adjacency matrix ----
    adj = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(N, N))
    adj = adj + sp.eye(N)                     # add self-loops
    # normalize: D^{-1/2} (A+I) D^{-1/2}
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    deg_inv_sqrt_mat = sp.diags(deg_inv_sqrt)
    adj_norm = deg_inv_sqrt_mat @ adj @ deg_inv_sqrt_mat
    adj_tensor = torch.from_numpy(adj_norm.toarray()).float()

    # ---- Labels tensor (ignore -1 for test) ----
    labels_tensor = torch.from_numpy(labels)

    model = GCN(nfeat=features.shape[1],
                nhid=CONFIG["hidden"],
                nclass=int(labels_tensor.max().item() + 1),
                dropout=CONFIG["dropout"])
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CONFIG["lr"],
                                 weight_decay=CONFIG["weight_decay"])

    best_val_loss = float('inf')
    best_state = None
    epochs_since_improve = 0

    max_epochs = 1 if smoke else CONFIG["epochs"]

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features_tensor, adj_tensor)  # [N, C]
        loss_train = F.cross_entropy(output[train_idx], labels_tensor[train_idx])
        loss = loss_train
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_output = model(features_tensor, adj_tensor)
            val_loss = F.cross_entropy(val_output[val_idx], labels_tensor[val_idx])
            val_pred = val_output[val_idx].argmax(dim=1)
            val_acc = (val_pred == labels_tensor[val_idx]).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= CONFIG["patience"]:
            break

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        final_output = model(features_tensor, adj_tensor)
        final_pred = final_output.argmax(dim=1).cpu().numpy()

    # print final validation accuracy
    with torch.no_grad():
        val_output = model(features_tensor, adj_tensor)
        val_pred = val_output[val_idx].argmax(dim=1)
        val_acc = (val_pred == labels_tensor[val_idx]).float().mean().item()
    print(f"Final validation accuracy: {val_acc:.4f}")

    # write predictions for all nodes (helper will select test/val indices)
    nio.write_predictions(final_pred)

if __name__ == "__main__":
    main()