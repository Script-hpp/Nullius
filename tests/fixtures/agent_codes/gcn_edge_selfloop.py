import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def main():
    # Seed
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    data = np.load('/data/cora_public.npz')
    features = torch.from_numpy(data['features']).float()          # [N, 1433]
    edges = torch.from_numpy(data['edges']).long()                # [E, 2]
    labels = torch.from_numpy(data['labels']).long()              # [N]
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx = torch.from_numpy(data['val_idx']).long()
    test_idx = torch.from_numpy(data['test_idx']).long()

    N = features.shape[0]

    # Add self-loops
    self_loop = torch.stack([torch.arange(N, dtype=torch.long),
                             torch.arange(N, dtype=torch.long)], dim=1)
    all_edges = torch.cat([edges, self_loop], dim=0)               # [E+N, 2]

    # Compute degree
    deg = torch.zeros(N)
    deg.index_add_(0, all_edges[:, 0], torch.ones(all_edges.size(0), dtype=torch.float))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Compute normalized adjacency values
    src, dst = all_edges[:, 0], all_edges[:, 1]
    norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

    # Build sparse adjacency matrix
    indices = all_edges.t()                                        # [2, E+N]
    adj_norm = torch.sparse_coo_tensor(indices, norm, size=(N, N)).coalesce()

    # GCN layer definition
    class GCNLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)

        def forward(self, x: Tensor, adj: Tensor) -> Tensor:
            support = x @ self.weight                               # [N, out]
            out = torch.sparse.mm(adj, support)                     # [N, out]
            return out

    # Model
    class GCN(nn.Module):
        def __init__(self, nfeat, nhidden, nclass, dropout):
            super().__init__()
            self.gcn1 = GCNLayer(nfeat, nhidden)
            self.gcn2 = GCNLayer(nhidden, nclass)
            self.dropout = dropout

        def forward(self, x, adj):
            x = self.gcn1(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gcn2(x, adj)
            return x

    model = GCN(nfeat=features.shape[1],
                nhidden=16,
                nclass=int(labels.max().item() + 1),
                dropout=0.5)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None
    patience = 100
    wait = 0
    max_epochs = 200

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj_norm)
        loss_train = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss_train.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(features, adj_norm)
            loss_val = F.cross_entropy(logits[val_idx], labels[val_idx])
            pred_val = logits[val_idx].argmax(dim=1)
            acc_val = (pred_val == labels[val_idx]).float().mean().item()

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_val_acc = acc_val
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test inference
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_norm)
        test_logits = logits[test_idx]
        predictions = test_logits.argmax(dim=1).cpu().numpy()

    # Write predictions
    os.makedirs('/output', exist_ok=True)
    with open('/output/predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for idx, pred in zip(test_idx.cpu().numpy(), predictions):
            f.write(f'{int(idx)},{int(pred)}\n')

    print(f'Validation accuracy: {best_val_acc:.4f}')

if __name__ == '__main__':
    main()
