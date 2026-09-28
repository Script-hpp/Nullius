import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

def main():
    seed = int(os.getenv('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = np.load('/data/cora_public.npz', allow_pickle=False)
    features = data['features'].astype(np.float32)          # (2708, 1433)
    edges = data['edges'].astype(np.int64)                  # (E, 2)
    labels = data['labels'].astype(np.int64)                # (2708,)
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    N = features.shape[0]
    # Build adjacency matrix (including both directions already)
    adj = sparse.coo_matrix((np.ones(edges.shape[0], dtype=np.float32),
                             (edges[:, 0], edges[:, 1])), shape=(N, N))
    adj = adj + adj.T  # ensure symmetry (though already both directions)
    # Add self-loops
    adj = adj + sparse.eye(N, dtype=np.float32)
    # Convert to dense for simplicity
    adj = adj.toarray()
    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = np.sum(axis=1, a=adj)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
    adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
    adj_norm = torch.from_numpy(adj_norm).float()

    # Torch tensors
    X = torch.from_numpy(features).float()
    y = torch.from_numpy(labels).long()

    # Model definition
    class GCN(torch.nn.Module):
        def __init__(self, in_dim, hidden, out_dim, dropout):
            super().__init__()
            self.lin1 = torch.nn.Linear(in_dim, hidden, bias=False)
            self.lin2 = torch.nn.Linear(hidden, out_dim, bias=False)
            self.dropout = dropout

        def forward(self, x, adj):
            x = torch.mm(adj, x)
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.mm(adj, x)
            x = self.lin2(x)
            return x

    model = GCN(in_dim=features.shape[1], hidden=16, out_dim=7, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_loss = float('inf')
    best_state = None
    patience = 100
    wait = 0
    max_epochs = 200

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X, adj_norm)
        loss = F.cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X, adj_norm)
            val_loss = F.cross_entropy(val_out[val_idx], y[val_idx])
            pred = val_out[val_idx].argmax(dim=1)
            val_acc = (pred == y[val_idx]).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_out = model(X, adj_norm)
        test_pred = test_out.argmax(dim=1).cpu().numpy()

    # Compute final validation accuracy for printing
    with torch.no_grad():
        val_out = model(X, adj_norm)
        val_pred = val_out[val_idx].argmax(dim=1)
        final_val_acc = (val_pred == y[val_idx]).float().mean().item()
    print(f'Validation Accuracy: {final_val_acc:.4f}')

    os.makedirs('/output', exist_ok=True)
    with open('/output/predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for idx in test_idx:
            f.write(f'{idx},{test_pred[idx]}\n')

if __name__ == '__main__':
    main()
