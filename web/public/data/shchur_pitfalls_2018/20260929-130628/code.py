import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def main():
    # ----- environment -----
    seed = int(os.getenv('SEED', '0'))
    smoke = os.getenv('SMOKE') == '1'
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----- load data -----
    data = np.load('/data/cora_lcc_public.npz')
    features = torch.from_numpy(data['features']).float()          # [N, F]
    edges = torch.from_numpy(data['edges']).long()                # [E, 2] directed
    labels = torch.from_numpy(data['labels']).long()              # [N]
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx = torch.from_numpy(data['val_idx']).long()
    test_idx = torch.from_numpy(data['test_idx']).long()

    N = features.shape[0]

    # ----- build normalized adjacency (A_hat) -----
    # edges already contain both directions; ensure shape [2, E]
    edge_index = edges.t().contiguous()                           # [2, E]
    # add self-loops
    self_loop = torch.arange(N, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, self_loop], dim=1)       # [2, E+ N]

    # compute degree
    deg = torch.bincount(edge_index[0], minlength=N).float()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

    # values for symmetric normalization
    norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
    adj = torch.sparse_coo_tensor(edge_index, norm, size=(N, N)).coalesce()

    # ----- model -----
    class GCN(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
            super().__init__()
            self.lin1 = nn.Linear(in_dim, hidden_dim)
            self.lin2 = nn.Linear(hidden_dim, out_dim)
            self.dropout = dropout

        def forward(self, x, adj):
            # first graph conv
            x = torch.sparse.mm(adj, x)
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # second graph conv
            x = torch.sparse.mm(adj, x)
            x = self.lin2(x)
            return x

    model = GCN(in_dim=features.shape[1], hidden_dim=64, out_dim=int(labels.max().item()) + 1, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # ----- training -----
    max_epochs = 1 if smoke else 10000
    patience = 100
    best_val_acc = 0.0
    patience_cnt = 0
    best_pred = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(features, adj)                     # [N, C]
        loss = F.cross_entropy(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            out = model(features, adj)
            pred = out.argmax(dim=-1)
            val_acc = (pred[val_idx] == labels[val_idx]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_cnt = 0
                best_pred = pred.detach().cpu().clone()
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

    print(f"Validation accuracy: {best_val_acc:.4f}")

    # ----- predictions -----
    if smoke:
        # predict only first 20 test nodes
        test_subset = test_idx[:20]
        preds = best_pred[test_subset].numpy()
    else:
        preds = best_pred[test_idx].numpy()

    # ----- write output -----
    os.makedirs('/output', exist_ok=True)
    with open('/output/predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for idx, p in zip(test_subset if smoke else test_idx, preds):
            f.write(f'{int(idx)},{int(p)}\n')

if __name__ == '__main__':
    main()
