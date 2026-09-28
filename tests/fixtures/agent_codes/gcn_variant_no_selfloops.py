import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

def main():
    # Seed
    seed = int(os.getenv('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    data = np.load('/data/cora_public.npz')
    features = torch.from_numpy(data['features']).float()          # [N, F]
    edges = data['edges']                                          # [E, 2]
    labels = torch.from_numpy(data['labels']).long()               # [N]
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx = torch.from_numpy(data['val_idx']).long()
    test_idx = torch.from_numpy(data['test_idx']).long()

    N = features.shape[0]

    # Build adjacency matrix (already symmetric)
    A = torch.zeros((N, N), dtype=torch.float32)
    A[edges[:, 0], edges[:, 1]] = 1.0

    # Add self-loops
    A_hat = A.clone()

    # Symmetric normalization: D^{-1/2} A_hat D^{-1/2}
    deg = A_hat.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    A_norm = deg_inv_sqrt[:, None] * A_hat * deg_inv_sqrt[None, :]

    # Row‑wise L2 normalization of features
    features = F.normalize(features, p=2, dim=1)

    # GCN model (2 layers)
    class GCN(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, adj):
            super().__init__()
            self.adj = adj  # [N, N] normalized adjacency
            self.lin1 = nn.Linear(in_features, hidden_features)
            self.lin2 = nn.Linear(hidden_features, out_features)
            self.dropout = nn.Dropout(p=0.5)
            self.act = nn.ReLU()
            self.reset_parameters()

        def reset_parameters(self):
            xavier_uniform_(self.lin1.weight)
            xavier_uniform_(self.lin2.weight)
            nn.init.zeros_(self.lin1.bias)
            nn.init.zeros_(self.lin2.bias)

        def forward(self, x):
            # x: [N, F]
            x = torch.mm(self.adj, x)          # Â̂ X
            x = self.lin1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = torch.mm(self.adj, x)          # Â̂ H
            x = self.lin2(x)
            return x

    model = GCN(in_features=features.shape[1],
                hidden_features=16,
                out_features=7,
                adj=A_norm)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Training with early stopping
    max_epochs = 200
    patience = 20
    best_val_loss = float('inf')
    best_state = None
    epochs_without_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(features)
        loss = criterion(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(features)
            val_loss = criterion(out[val_idx], labels[val_idx])
            val_pred = val_out[val_idx].argmax(dim=1)
            val_acc = (val_pred == labels[val_idx]).float().mean()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= patience:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation accuracy (to stdout)
    model.eval()
    with torch.no_grad():
        final_out = model(features)
        final_pred = final_out[val_idx].argmax(dim=1)
        final_acc = (final_pred == labels[val_idx]).float().mean()
        print(f'Validation accuracy: {final_acc.item():.4f}')

    # Test predictions
    with torch.no_grad():
        test_out = model(features)
        test_pred = test_out[test_idx].argmax(dim=1).cpu().numpy()

    # Write predictions
    os.makedirs('/output', exist_ok=True)
    with open('/output/predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for idx, pred in zip(test_idx.cpu().numpy(), test_pred):
            f.write(f'{idx},{pred}\n')

if __name__ == '__main__':
    main()
