

# Plan step -> implemented in: function or class name
# 1. Model -> class GCN
# 2. Hyperparameters -> variables lr, weight_decay, dropout_prob, hidden_size, num_layers
# 3. Input -> function preprocess_adjacency
# 4. Labels -> conversion to torch.LongTensor inside training loop
# 5. Training -> function train_model
# 6. Early Stopping -> logic inside train_model using validation loss history
# 7. Model Selection -> restoring best state based on lowest validation loss
# 8. Seeding -> set_seeds function called at start
# 9. Output -> write_predictions function

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.sparse import mm as spmm

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_adjacency(edge_list, num_nodes):
    """
    edge_list: np.ndarray [E,2] directed pairs (u,v) possibly containing both directions.
    Returns normalized adjacency matrix (A_hat) as a torch sparse COO tensor:
        A_hat = D^{-1/2} (A+I) D^{-1/2}
    """
    # Remove self-loops
    mask = edge_list[:,0] != edge_list[:,1]
    edges = edge_list[mask]
    # Ensure undirected: we already have both directions, but keep unique
    edges = np.unique(edges, axis=0)
    # Add self-loops
    self_loops = np.arange(num_nodes)[:,None].repeat(2, axis=1)
    self_loops = np.concatenate([self_loops, self_loops[:,::-1]], axis=0)  # not needed; we'll add I later via eye
    # Build adjacency with self-loops (A+I)
    idx = np.concatenate([edges, np.arange(num_nodes)[:,None].repeat(2, axis=1)], axis=0)
    # Actually easier: create adjacency without self-loops then add I in degree computation
    # We'll compute A tilde = A + I directly in degree
    # Build sparse tensor for A (without self-loops)
    i = torch.LongTensor(edges.T)  # [2, E]
    v = torch.FloatTensor(np.ones(len(edges)))
    A = torch.sparse.FloatTensor(i, v, torch.Size([num_nodes, num_nodes]))
    # Compute degrees of A+I
    # degree = sum(A, dim=1) + 1 (for self-loop)
    deg = torch.sparse.sum(A, dim=1).to_dense() + 1.0
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    # Create D^{-1/2} * (A+I) * D^{-1/2}
    # First left multiply: D^{-1/2} * (A+I)
    i_left = i
    v_left = v * deg_inv_sqrt[i[0]]  # scale source node
    A_left = torch.sparse.FloatTensor(i_left, v_left, torch.Size([num_nodes, num_nodes]))
    # Then right multiply: (D^{-1/2} * (A+I)) * D^{-1/2}
    i_right = i_left
    v_right = v_left * deg_inv_sqrt[i[1]]  # scale target node
    A_hat = torch.sparse.FloatTensor(i_right, v_right, torch.Size([num_nodes, num_nodes]))
    return A_hat.coalesce()

class GCN(nn.Module):
    def __init__(self, nfeat, nhidden, nclass, dropout, num_layers=2):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(nfeat, nclass))
        else:
            self.layers.append(nn.Linear(nfeat, nhidden))
            for _ in range(num_layers-2):
                self.layers.append(nn.Linear(nhidden, nhidden))
            self.layers.append(nn.Linear(nhidden, nclass))

    def forward(self, x, adj):
        """
        x: [N, nfeat] dense feature matrix
        adj: normalized adjacency matrix [N,N] sparse
        """
        l = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = spmm(adj, x)   # Ax
            x = layer(x)       # AxW
            if i != l-1:       # activation and dropout for all but last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

def train_model(model, features, adj, labels, train_idx, val_idx, lr, weight_decay, max_epochs=200, smoke=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')
    best_state = None
    val_loss_history = []
    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.cross_entropy(output[train_idx], labels[train_idx])
        loss_train.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            output_val = model(features, adj)
            loss_val = F.cross_entropy(output_val[val_idx], labels[val_idx])
            pred_val = output_val[val_idx].argmax(dim=1)
            acc_val = (pred_val == labels[val_idx]).float().mean().item()
        model.train()

        val_loss_history.append(loss_val.item())
        # Early stopping condition: stop if current val loss > mean of last 10
        if len(val_loss_history) >= 10:
            recent_mean = np.mean(val_loss_history[-10:])
            if loss_val.item() > recent_mean:
                break

        # Track best model
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if smoke and epoch >= 0:  # smoke mode: exactly 1 epoch
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    # Return final validation accuracy (from best state)
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        pred = output[val_idx].argmax(dim=1)
        acc = (pred == labels[val_idx]).float().mean().item()
    return acc

def write_predictions(ids, preds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('id,prediction\n')
        for idx, p in zip(ids, preds):
            f.write(f'{int(idx)},{int(p)}\n')

def main():
    seed = int(os.environ.get('SEED', '0'))
    smoke = os.environ.get('SMOKE', '0') == '1'
    set_seeds(seed)

    data = np.load('/data/cora_lcc_public.npz', allow_pickle=False)
    features = torch.FloatTensor(data['features'])          # [N, 1433]
    edges = data['edges']                                   # [E,2]
    labels = torch.LongTensor(data['labels'])               # [N], -1 for test
    train_idx = torch.LongTensor(data['train_idx'])
    val_idx = torch.LongTensor(data['val_idx'])
    test_idx = torch.LongTensor(data['test_idx'])

    N = features.shape[0]
    adj = preprocess_adjacency(edges, N)                    # normalized adjacency (A+I) hat

    # Hyperparameters from optimized config (as per plan)
    lr = 0.01
    weight_decay = 0.001
    dropout_prob = 0.8
    hidden_size = 64
    num_layers = 2
    nfeat = features.shape[1]
    nclass = int(labels[labels >= 0].max().item()) + 1      # should be 7

    model = GCN(nfeat=nfeat, nhidden=hidden_size, nclass=nclass,
                dropout=dropout_prob, num_layers=num_layers)

    val_acc = train_model(model, features, adj, labels, train_idx, val_idx,
                          lr, weight_decay, max_epochs=200, smoke=smoke)
    print(f'Validation accuracy: {val_acc:.4f}')

    # Predict on validation set (for auxiliary output)
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        val_pred = logits[val_idx].argmax(dim=1).cpu().numpy()
        test_pred = logits[test_idx].argmax(dim=1).cpu().numpy()

    # Order of rows: as given in idx arrays
    write_predictions(val_idx.numpy(), val_pred, '/output/val_predictions.csv')
    write_predictions(test_idx.numpy(), test_pred, '/output/predictions.csv')

    # In smoke mode, restrict to first 20 sorted ids as required
    if smoke:
        # rewrite files with only first 20 sorted ids
        def truncate(path, idx_arr):
            sorted_idx = np.sort(idx_arr)[:20]
            # need corresponding predictions: we already have full predictions array aligned with original order;
            # we will re-index by picking those positions from original idx order.
            # Simpler: recompute predictions for those ids (but we can slice from full preds using mapping)
            # We'll create mapping from original idx to position.
            pos = {int(i): p for i, p in zip(idx_arr.numpy(),
                                             val_pred if 'val' in path else test_pred)}
            with open(path, 'w') as f:
                f.write('id,prediction\n')
                for i in sorted_idx:
                    f.write(f'{i},{pos[i]}\n')
        truncate('/output/val_predictions.csv', val_idx)
        truncate('/output/predictions.csv', test_idx)

if __name__ == '__main__':
    main()