

# Plan step -> implemented in: load_data function
# Plan step -> implemented in: preprocess_adjacency function
# Plan step -> implemented in: GCN class
# Plan step -> implemented in: train_model function (inner training with early stopping)
# Plan step -> implemented in: self_training_loop function (outer iterative pseudo-labeling with confidence-based selection)
# Plan step -> implemented in: main script orchestrating seed, data loading, training, prediction, and output

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

def load_data():
    d = nio.load()
    features = torch.from_numpy(d['features']).float()          # [2708, 1433]
    edges = torch.from_numpy(d['edges']).long()                # [E, 2]
    labels = torch.from_numpy(d['labels']).long()              # [2708]
    train_idx = torch.from_numpy(d['train_idx']).long()
    val_idx = torch.from_numpy(d['val_idx']).long()
    test_idx = torch.from_numpy(d['test_idx']).long()
    N = features.shape[0]
    return features, edges, labels, train_idx, val_idx, test_idx, N

def preprocess_adjacency(edges, N):
    # Add self-loop
    self_loop = torch.arange(N, dtype=torch.long)
    self_loop_idx = torch.stack([self_loop, self_loop], dim=0)   # [2, N]
    idx_all = torch.cat([edges.t(), self_loop_idx], dim=1)       # [2, E+N]
    vals = torch.ones(idx_all.shape[1], dtype=torch.float32)
    # Compute degree for normalization
    deg = torch.zeros(N, dtype=torch.float32)
    deg.index_add_(0, idx_all[0], vals)
    deg.index_add_(0, idx_all[1], vals)   # because edges are undirected (both directions present)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    # Normalize values: val / sqrt(deg[u]*deg[v])
    norm_vals = vals * deg_inv_sqrt[idx_all[0]] * deg_inv_sqrt[idx_all[1]]
    adj_norm = torch.sparse_coo_tensor(idx_all, norm_vals, (N, N), dtype=torch.float32)
    return adj_norm.coalesce()

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.linear1 = nn.Linear(nfeat, nhid)
        self.linear2 = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        # x: [N, nfeat], adj: [N, N] sparse
        support = self.linear1(x)          # [N, nhid]
        support = F.relu(support)
        support = self.dropout(support)
        support = self.linear2(support)    # [N, nclass]
        output = torch.spmm(adj, support)  # [N, nclass]
        return output

def train_model(model, features, adj, pseudo_labels, labeled_idx, val_idx,
                lr=0.001, weight_decay=5e-4, max_epochs=2000, patience=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = F.cross_entropy(logits[labeled_idx], pseudo_labels[labeled_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj)
            val_pred = val_logits[val_idx].argmax(dim=1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc

def self_training_loop(features, adj_norm, labels, train_idx, val_idx, test_idx, N,
                       seed, smoke_mode=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    valid_labels = labels[labels != -1]
    nclass = int(valid_labels.max().item()) + 1

    labeled_idx = train_idx.clone()
    pseudo_labels = torch.full_like(labels, -1)
    pseudo_labels[labeled_idx] = labels[labeled_idx]

    hidden_dim = 128
    dropout = 0.3
    lr = 0.001
    weight_decay = 5e-4
    max_epochs_inner = 1 if smoke_mode else 2000
    patience_inner = 500
    max_outer_iters = 1 if smoke_mode else 40
    topk = 100
    threshold = 0.0

    best_overall_val_acc = 0.0
    best_model_state = None

    for itr in range(max_outer_iters):
        model = GCN(nfeat=features.shape[1], nhid=hidden_dim,
                    nclass=nclass, dropout=dropout)
        model, val_acc = train_model(model, features, adj_norm, pseudo_labels,
                                     labeled_idx, val_idx,
                                     lr=lr, weight_decay=weight_decay,
                                     max_epochs=max_epochs_inner,
                                     patience=patience_inner)

        if val_acc > best_overall_val_acc:
            best_overall_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.eval()
        with torch.no_grad():
            logits = model(features, adj_norm)
            probs = F.softmax(logits, dim=1)
            confidence, pred_labels = torch.max(probs, dim=1)

        confidence[labeled_idx] = -1.0
        cand_mask = confidence > threshold
        cand_indices = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)
        if cand_indices.numel() == 0:
            break

        if cand_indices.numel() <= topk:
            selected = cand_indices
        else:
            _, topk_idx = torch.topk(confidence[cand_indices], topk)
            selected = cand_indices[topk_idx]

        if selected.numel() == 0:
            break

        labeled_idx = torch.unique(torch.cat([labeled_idx, selected]))
        pseudo_labels[selected] = pred_labels[selected]

    if best_model_state is not None:
        best_model = GCN(nfeat=features.shape[1], nhid=hidden_dim,
                         nclass=nclass, dropout=dropout)
        best_model.load_state_dict(best_model_state)
    else:
        best_model = model

    best_model.eval()
    with torch.no_grad():
        final_logits = best_model(features, adj_norm)
        final_pred = final_logits.argmax(dim=1)
    return final_pred

if __name__ == '__main__':
    seed = int(os.environ.get('SEED', 1024))
    smoke = os.environ.get('SMOKE', '0') == '1'

    features, edges, labels, train_idx, val_idx, test_idx, N = load_data()
    adj_norm = preprocess_adjacency(edges, N)

    preds = self_training_loop(features, adj_norm, labels, train_idx, val_idx, test_idx,
                               N, seed, smoke_mode=smoke)

    nio.write_predictions(preds.numpy())

    val_acc = (preds[val_idx] == labels[val_idx]).float().mean().item()
    print(f'Validation accuracy: {val_acc:.4f}')