# Plan step 1 -> implemented in: class GCN
# Plan step 2 -> implemented in: class GCN (layer definitions)
# Plan step 3 -> implemented in: training loop (loss computation)
# Plan step 4 -> implemented in: optimizer.Adam
# Plan step 5 -> implemented in: weight initialization (Glorot uniform) and bias zeros
# Plan step 6 -> implemented in: row-normalize features (L1 norm)
# Plan step 7 -> implemented in: build adjacency with self-loops and symmetric normalization
# Plan step 8 -> implemented in: for-loop over epochs with max 200
# Plan step 9 -> implemented in: validation set used for early stopping only
# Plan step 10-> implemented in: final prediction on test_idx
# Plan step 11-> implemented in: dropout applied to inputs of each layer (rate 0.5)
# Plan step 12-> implemented in: masked accuracy printed (validation)
# Plan step 13-> implemented in: hyperparameters as per paper
# Plan step 14-> implemented in: seeding from env SEED
# Plan step 15-> implemented in: writing predictions.csv and val_predictions.csv

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def main():
    # ----- SEED -----
    seed = int(os.environ.get("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----- LOAD DATA -----
    data = np.load("/data/cora_public.npz", allow_pickle=True)
    features = data["features"].astype(np.float32)          # [2708, 1433]
    edges = data["edges"].astype(np.int64)                  # [E, 2]
    labels = data["labels"].astype(np.int64)                # [2708]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    N = features.shape[0]
    num_classes = int(labels.max()) + 1  # labels are 0..6, -1 for test

    # ----- FEATURE NORMALIZATION (L1 per row) -----
    row_sums = features.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    features = features / row_sums
    features_t = torch.from_numpy(features)  # dense

    # ----- LABELS TENSOR -----
    labels_t = torch.from_numpy(labels)      # int64, -1 for test nodes

    # ----- ADJACENCY MATRIX -----
    # edges already contain both directions; add self-loops
    self_loop_edges = np.arange(N)[:, None].repeat(2, axis=1)  # Not correct, fix below
    # Actually we need pairs (i,i)
    self_loop_edges = np.column_stack((np.arange(N), np.arange(N)))
    all_edges = np.concatenate([edges, self_loop_edges], axis=0)  # [E+ N, 2]
    # Build sparse adjacency (COO)
    idx = torch.from_numpy(all_edges.T).long()  # [2, E+N]
    vals = torch.ones(idx.size(1), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, vals, torch.Size([N, N]), dtype=torch.float32)

    # ----- SYMMETRIC NORMALIZATION Â = D^{-1/2} (A+I) D^{-1/2} -----
    # Compute degree vector
    deg = torch.sparse.sum(adj, dim=1).to_dense()  # [N]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Create normalized adjacency values
    row_inv = deg_inv_sqrt[idx[0]]
    col_inv = deg_inv_sqrt[idx[1]]
    norm_vals = vals * row_inv * col_inv
    adj_norm = torch.sparse_coo_tensor(idx, norm_vals, torch.Size([N, N]), dtype=torch.float32)
    adj_norm = adj_norm.coalesce()

    # ----- MODEL DEFINITION -----
    class GCNLayer(Module):
        def __init__(self, in_features, out_features, dropout=0., act=F.relu):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.dropout = dropout
            self.act = act
            self.weight = Parameter(torch.Tensor(in_features, out_features))
            self.bias = Parameter(torch.Tensor(out_features))
            self.reset_parameters()

        def reset_parameters(self):
            # Glorot uniform
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)

        def forward(self, x, adj):
            x = F.dropout(x, self.dropout, training=self.training)
            support = torch.matmul(x, self.weight)  # [N, out]
            # sparse adj * dense support
            output = torch.sparse.mm(adj, support)  # [N, out]
            output = output + self.bias
            if self.act is not None:
                output = self.act(output)
            return output

    class GCN(nn.Module):
        def __init__(self, nfeat, nhidden, nclass, dropout=0.5):
            super().__init__()
            self.gc1 = GCNLayer(nfeat, nhidden, dropout=dropout, act=F.relu)
            self.gc2 = GCNLayer(nhidden, nclass, dropout=dropout, act=lambda x: x)

        def forward(self, x, adj):
            x = self.gc1(x, adj)
            x = self.gc2(x, adj)
            return x

    model = GCN(nfeat=features_t.shape[1], nhidden=16, nclass=num_classes, dropout=0.5)

    # ----- OPTIMIZER -----
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)  # we add L2 manually

    # ----- TRAINING -----
    smoke_mode = os.environ.get("SMOKE", "0") == "1"
    max_epochs = 1 if smoke_mode else 200
    patience = 10
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    lambda_l2 = 5e-4

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features_t, adj_norm)  # [N, C]
        loss_train = F.cross_entropy(logits[train_idx], labels_t[train_idx])
        # L2 regularization on first layer weights only
        l2_reg = lambda_l2 * torch.norm(model.gc1.weight) ** 2
        loss = loss_train + l2_reg
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_val = model(features_t, adj_norm)
            loss_val = F.cross_entropy(logits_val[val_idx], labels_t[val_idx])
            pred_val = logits_val[val_idx].argmax(dim=1)
            acc_val = (pred_val == labels_t[val_idx]).float().mean().item()

        # Early stopping check
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Print final validation accuracy
    model.eval()
    with torch.no_grad():
        logits_final = model(features_t, adj_norm)
        pred_final = logits_final[val_idx].argmax(dim=1)
        acc_final = (pred_final == labels_t[val_idx]).float().mean().item()
    print(f"Validation accuracy: {acc_final:.4f}")

    # ----- PREDICTIONS -----
    with torch.no_grad():
        logits_test = model(features_t, adj_norm)
        pred_test = logits_test.argmax(dim=1)

    # Helper to write CSV
    def write_csv(id_array, pred_array, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("id,prediction\n")
            for idx, p in zip(id_array, pred_array):
                f.write(f"{int(idx)},{int(p)}\n")

    if smoke_mode:
        # Take first 20 sorted ids
        test_sub = np.sort(test_idx)[:20]
        val_sub = np.sort(val_idx)[:20]
        write_csv(test_sub, pred_test[test_sub].cpu().numpy(), "/output/predictions.csv")
        write_csv(val_sub,  pred_test[val_sub].cpu().numpy(),   "/output/val_predictions.csv")
    else:
        write_csv(test_idx, pred_test[test_idx].cpu().numpy(), "/output/predictions.csv")
        write_csv(val_idx,  pred_test[val_idx].cpu().numpy(),   "/output/val_predictions.csv")

if __name__ == "__main__":
    main()
