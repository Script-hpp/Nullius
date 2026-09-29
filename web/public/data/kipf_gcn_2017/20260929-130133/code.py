import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

def main():
    # ----- Seed -----
    seed_str = os.environ.get('SEED', '0')
    try:
        seed = int(seed_str)
    except ValueError:
        seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ----- Load data -----
    data = np.load('/data/cora_public.npz', allow_pickle=False)
    features_np = data['features'].astype(np.float32)          # [2708, 1433]
    edges = data['edges'].astype(np.int64)                     # [E, 2]
    labels_np = data['labels'].astype(np.int64)                # [2708]
    train_idx = data['train_idx'].astype(np.int64)
    val_idx = data['val_idx'].astype(np.int64)
    test_idx = data['test_idx'].astype(np.int64)

    N = features_np.shape[0]
    num_classes = int(labels_np.max()) + 1  # assuming labels 0..6

    # ----- Feature preprocessing: row-normalize (L1) -----
    row_sums = features_np.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    features_np = features_np / row_sums
    features = torch.from_numpy(features_np)  # [N, F]

    # ----- Labels -----
    labels = torch.from_numpy(labels_np)      # [N], -1 for test nodes

    # ----- Adjacency matrix (binary, symmetric) + self-loops -----
    src = torch.from_numpy(edges[:, 0])       # [E]
    dst = torch.from_numpy(edges[:, 1])       # [E]

    # degree from A (each undirected edge appears twice, once per direction)
    deg = torch.zeros(N, dtype=torch.float32)
    deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    # add self-loops
    deg += 1.0

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # edge values for normalized adjacency (excluding self-loops)
    edge_vals = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

    # self-loop values
    loop_idx = torch.arange(N, dtype=torch.long)
    loop_vals = deg_inv_sqrt * deg_inv_sqrt

    # combine
    all_src = torch.cat([src, loop_idx])
    all_dst = torch.cat([dst, loop_idx])
    all_vals = torch.cat([edge_vals, loop_vals])

    adj_norm = torch.sparse_coo_tensor(
        torch.stack([all_src, all_dst]), all_vals, size=(N, N)
    ).coalesce()

    # ----- Model definition -----
    class GCNLayer(nn.Module):
        def __init__(self, in_features, out_features, dropout=0.5, bias=True, activation=None):
            super().__init__()
            self.dropout = dropout
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            nn.init.xavier_uniform_(self.linear.weight)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)
            self.act = (lambda x: x) if activation is None else activation

        def forward(self, x, adj):
            x = F.dropout(x, p=self.dropout, training=self.training)
            support = self.linear(x)  # XW + b
            out = torch.sparse.mm(adj, support)  # Â * (XW + b)
            return self.act(out)

    class GCN(nn.Module):
        def __init__(self, nfeat, nhid, nclass, dropout=0.5):
            super().__init__()
            self.gcn1 = GCNLayer(nfeat, nhid, dropout=dropout, activation=F.relu)
            self.gcn2 = GCNLayer(nhid, nclass, dropout=dropout, activation=lambda x: x)

        def forward(self, x, adj):
            x = self.gcn1(x, adj)
            x = self.gcn2(x, adj)
            return x

    model = GCN(nfeat=features.shape[1], nhid=16, nclass=num_classes, dropout=0.5)

    # ----- Optimizer -----
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0)  # L2 handled manually
    l2_lambda = 5e-4

    # ----- Training settings -----
    max_epochs = 200
    patience = 10
    best_val_loss = float('inf')
    wait = 0
    best_state = None

    smoke = os.environ.get('SMOKE', '0') == '1'

    if smoke:
        max_epochs = 1
        test_subset = test_idx[:20]
    else:
        test_subset = test_idx

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj_norm)  # [N, C]

        # loss on training set
        loss_train = F.cross_entropy(logits[train_idx], labels[train_idx])

        # L2 regularization on first layer weight
        l2_loss = l2_lambda * torch.sum(model.gcn1.linear.weight ** 2)
        loss = loss_train + l2_loss

        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            logits_val = model(features, adj_norm)
            loss_val = F.cross_entropy(logits_val[val_idx], labels[val_idx])
            pred_val = logits_val[val_idx].argmax(dim=1)
            acc_val = (pred_val == labels[val_idx]).float().mean().item()

        # early stopping check
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # final validation accuracy (print to stdout)
    model.eval()
    with torch.no_grad():
        logits_val = model(features, adj_norm)
        pred_val = logits_val[val_idx].argmax(dim=1)
        acc_val = (pred_val == labels[val_idx]).float().mean().item()
    print(f'Validation Accuracy: {acc_val:.4f}')

    # ----- Test predictions -----
    model.eval()
    with torch.no_grad():
        logits_test = model(features, adj_norm)
        preds_test = logits_test[test_subset].argmax(dim=1)

    # ----- Output CSV -----
    os.makedirs('/output', exist_ok=True)
    out_path = '/output/predictions.csv'
    with open(out_path, 'w') as f:
        f.write('id,prediction\n')
        for idx, pred in zip(test_subset, preds_test.cpu().numpy()):
            f.write(f'{int(idx)},{int(pred)}\n')

if __name__ == '__main__':
    main()
