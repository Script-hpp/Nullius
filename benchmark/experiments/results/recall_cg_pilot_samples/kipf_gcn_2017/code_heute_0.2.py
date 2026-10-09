# Plan step -> implemented in:
# 1. Set random seeds -> set_seed()
# 2. Load data -> load_data()
# 3. Build adjacency matrix and normalize -> build_normalized_adjacency()
# 4. Define GCN model -> GCN class
# 5. Loss -> compute_loss()
# 6. Optimizer -> create_optimizer()
# 7. Training loop -> train_model()
# 8. Validation accuracy for model selection -> evaluate()
# 9. Inference -> predict()
#10. Write predictions -> write_predictions()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

def set_seed():
    seed = int(os.getenv('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    data = np.load('/data/cora_public.npz')
    features = torch.from_numpy(data['features']).float()          # [N, F]
    edges = torch.from_numpy(data['edges']).long()                # [E, 2]
    labels = torch.from_numpy(data['labels']).long()              # [N]
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx = torch.from_numpy(data['val_idx']).long()
    test_idx = torch.from_numpy(data['test_idx']).long()
    return features, edges, labels, train_idx, val_idx, test_idx

def build_normalized_adjacency(edges, num_nodes):
    # Ensure symmetry (edges already contain both directions, but we make sure)
    # Add self-loops
    loop_idx = torch.arange(num_nodes, dtype=torch.long)
    loop_edges = torch.stack([loop_idx, loop_idx], dim=1)          # [N, 2]
    all_edges = torch.cat([edges, loop_edges], dim=0)              # [E+2N, 2]
    # Make symmetric: for each (u,v) add (v,u) if not present
    # Simpler: create sparse adjacency and then symmetrize by adding its transpose
    idx = all_edges.t()                                            # [2, E+2N]
    values = torch.ones(idx.size(1), dtype=torch.float32)
    adj = torch.sparse.FloatTensor(idx, values, torch.Size([num_nodes, num_nodes]))
    adj = adj + adj.t()                                            # ensure symmetry
    # Remove duplicate entries that may have arisen (values become 2)
    adj = adj.coalesce()
    # Set values to 1 (since we only need binary adjacency with self-loops)
    adj_values = adj.values().clone()
    adj_values[:] = 1.0
    adj = torch.sparse.FloatTensor(adj.indices(), adj_values, adj.size())
    # Compute degree vector
    deg = torch.sparse.sum(adj, dim=1).to_dense()                  # [N]
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    # Create D^{-1/2} as sparse diagonal
    diag_idx = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)
    D_inv_sqrt = torch.sparse.FloatTensor(diag_idx, deg_inv_sqrt, torch.Size([num_nodes, num_nodes]))
    # Normalized adjacency: D^{-1/2} A D^{-1/2}
    adj_norm = torch.sparse.mm(D_inv_sqrt, adj)
    adj_norm = torch.sparse.mm(adj_norm, D_inv_sqrt)
    return adj_norm.coalesce()

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = nn.Linear(nfeat, nhid, bias=False)
        self.gc2 = nn.Linear(nhid, nclass, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        # GCN layer: support = A_hat * X
        support = torch.sparse.mm(adj, x)          # [N, nhid] after first linear
        out = self.gc1(support)
        out = F.relu(out)
        out = self.dropout(out)
        support2 = torch.sparse.mm(adj, out)
        out = self.gc2(support2)
        return out

def compute_loss(logits, labels, idx):
    return F.cross_entropy(logits[idx], labels[idx])

def create_optimizer(model, lr=0.01, weight_decay=5e-4):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def evaluate(model, features, adj, labels, idx):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        loss = F.cross_entropy(logits[idx], labels[idx])
        pred = logits[idx].max(1)[1]
        acc = pred.eq(labels[idx]).float().mean().item()
    return loss, acc

def train_model(model, features, adj, labels, train_idx, val_idx, epochs=200, patience=20):
    optimizer = create_optimizer(model)
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = compute_loss(logits, labels, train_idx)
        loss.backward()
        optimizer.step()
        val_loss, val_acc = evaluate(model, features, adj, labels, val_idx)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict(model, features, adj, idx):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        pred = logits[idx].max(1)[1]
    return pred.cpu().numpy()

def write_predictions(ids, preds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('id,prediction\n')
        for i, p in zip(ids, preds):
            f.write(f'{i},{p}\n')

def main():
    set_seed()
    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features.size(0)
    adj = build_normalized_adjacency(edges, num_nodes)

    smoke = os.getenv('SMOKE', '0') == '1'
    if smoke:
        epochs = 1
        test_subset = torch.sort(test_idx)[0][:20]
        val_subset = torch.sort(val_idx)[0][:20]
    else:
        epochs = 200
        test_subset = test_idx
        val_subset = val_idx

    model = GCN(nfeat=features.size(1), nhid=16, nclass=7, dropout=0.5)
    model = train_model(model, features, adj, labels, train_idx, val_idx, epochs=epochs, patience=20)

    # Validation predictions (for selection file)
    val_pred = predict(model, features, adj, val_subset)
    write_predictions(val_subset.numpy(), val_pred, '/output/val_predictions.csv')

    # Test predictions
    test_pred = predict(model, features, adj, test_subset)
    write_predictions(test_subset.numpy(), test_pred, '/output/predictions.csv')

    # Print final validation accuracy (on full val_idx)
    _, final_val_acc = evaluate(model, features, adj, labels, val_idx)
    print(f'Validation accuracy: {final_val_acc:.4f}')

if __name__ == '__main__':
    main()
