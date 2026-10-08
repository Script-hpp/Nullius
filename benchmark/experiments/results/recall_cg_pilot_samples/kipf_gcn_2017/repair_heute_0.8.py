

# Plan step 1 -> implemented in: set_seeds
# Plan step 2 -> implemented in: load_data
# Plan step 3 -> implemented in: normalize_adjacency
# Plan step 4 -> implemented in: GCNConv class and GCN class
# Plan step 5 -> implemented in: train_step (loss computation)
# Plan step 6 -> implemented in: configure_optimizer
# Plan step 7 -> implemented in: training_loop with early stopping
# Plan step 8 -> implemented in: evaluate_validation (model selection)
# Plan step 9 -> implemented in: predict_test
# Plan step 10 -> implemented in: write_predictions

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seeds():
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

def normalize_adjacency(edge_index, num_nodes):
    """
    edge_index: [E, 2] (directed edges, already includes both directions)
    Returns normalized adjacency matrix with self-loops as a sparse COO tensor:
        \hat{A} = D^{-1/2} (A + I) D^{-1/2}
    """
    # Add self-loops
    loop_index = torch.arange(num_nodes, dtype=torch.long).unsqueeze(1).repeat(1, 2)
    loop_index = loop_index[:, [0, 0]]  # (i,i)
    edge_index_with_loop = torch.cat([edge_index, loop_index], dim=0)  # [E + N, 2]

    # Compute degree
    deg = torch.zeros(num_nodes, dtype=torch.float)
    deg.index_add_(0, edge_index_with_loop[:, 0], torch.ones(edge_index_with_loop.size(0)))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

    # Compute normalized edge weights
    src, dst = edge_index_with_loop.t()
    norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

    # Build sparse COO tensor
    indices = torch.stack([src, dst], dim=0)          # [2, E+N]
    size = (num_nodes, num_nodes)
    adj_t = torch.sparse_coo_tensor(indices, norm, size, dtype=torch.float).coalesce()
    return adj_t

class GCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x, adj_t):
        # x: [N, in_features]
        # adj_t: SparseTensor [N, N] (COO)
        support = x @ self.weight                     # [N, out_features]
        out = torch.sparse.mm(adj_t, support)         # [N, out_features]
        return out

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, adj_t)
        return x

def configure_optimizer(model, lr=0.01, weight_decay=5e-4):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def train_step(model, optimizer, features, adj_t, labels, idx):
    model.train()
    optimizer.zero_grad()
    logits = model(features, adj_t)
    loss = F.cross_entropy(logits[idx], labels[idx])
    loss.backward()
    optimizer.step()
    return loss.item(), logits.detach()

def evaluate(model, features, adj_t, labels, idx):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_t)
        loss = F.cross_entropy(logits[idx], labels[idx]).item()
        acc = accuracy(logits[idx], labels[idx])
    return loss, acc, logits

def write_predictions(path, idx, preds):
    """
    idx: tensor of node ids (sorted or otherwise)
    preds: tensor of predictions same length as idx
    Writes CSV with header `id,prediction`.
    """
    with open(path, 'w') as f:
        f.write('id,prediction\n')
        for node_id, pred in zip(idx.tolist(), preds.tolist()):
            f.write(f'{node_id},{pred}\n')

def main():
    set_seeds()
    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features.size(0)
    adj_t = normalize_adjacency(edges, num_nodes)

    # Number of classes: labels are 0..6, -1 for test nodes (ignore)
    nclass = int(labels.max().item()) + 1  # 7
    model = GCN(nfeat=features.size(1), nhid=16, nclass=nclass, dropout=0.5)
    optimizer = configure_optimizer(model)

    smoke = os.getenv('SMOKE') == '1'
    max_epochs = 1 if smoke else 200
    patience = 20 if not smoke else 1  # unused in smoke

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, _ = train_step(model, optimizer, features, adj_t, labels, train_idx)
        val_loss, val_acc, _ = evaluate(model, features, adj_t, labels, val_idx)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stop at epoch {epoch}')
            break
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    # Load best model
    model.load_state_dict(best_state)
    _, val_acc, val_logits = evaluate(model, features, adj_t, labels, val_idx)
    print(f'Final validation accuracy: {val_acc:.4f}')

    # Test predictions (full test set)
    model.eval()
    with torch.no_grad():
        test_logits = model(features, adj_t)
        test_pred = test_logits[test_idx].argmax(dim=-1)

    # Validation predictions (subset in smoke mode)
    if smoke:
        sorted_val = torch.sort(val_idx)[0]
        val_idx_use = sorted_val[:20]
        val_pred_use = val_logits[val_idx_use].argmax(dim=-1)
    else:
        val_idx_use = val_idx
        val_pred_use = val_logits[val_idx_use].argmax(dim=-1)

    # Write outputs
    write_predictions('/output/predictions.csv', test_idx, test_pred)
    write_predictions('/output/val_predictions.csv', val_idx_use, val_pred_use)

if __name__ == '__main__':
    main()