# Plan step 1 -> load_data: function load_data
# Plan step 2 -> preprocess_features: function preprocess_features
# Plan step 3 -> build_adjacency: function build_adjacency
# Plan step 4 -> build_model: class GCN
# Plan step 5 -> train_model: function train
# Plan step 6 -> predict: function predict
# Plan step 7 -> write_output: function write_predictions

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_data():
    data = np.load('/data/cora_public.npz')
    features = data['features'].astype(np.float32)
    edges = data['edges']
    labels = data['labels'].astype(np.int64)
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']
    return features, edges, labels, train_idx, val_idx, test_idx

def preprocess_features(features):
    """Row-normalize feature matrix (L1 norm)."""
    row_sum = features.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sum[row_sum == 0] = 1.0
    return features / row_sum

def build_adjacency(num_nodes, edges):
    """Build symmetric binary adjacency with self-loops and symmetric normalization."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    # Make symmetric (ensure undirected)
    adj = np.maximum(adj, adj.T)
    # Add self-loops
    adj += np.eye(num_nodes, dtype=np.float32)
    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = adj.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
    return adj_norm

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0., act=F.relu, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    def forward(self, x, adj):
        x = self.dropout(x)
        support = torch.matmul(adj, x)
        out = torch.matmul(support, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return self.act(out)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.gc1 = GCNLayer(nfeat, nhid, dropout=dropout, act=F.relu)
        self.gc2 = GCNLayer(nhid, nclass, dropout=dropout, act=lambda x: x)
    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        return x

def train(model, features, adj, labels, train_mask, val_mask, optimizer, l2_lambda, max_epochs, patience):
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.cross_entropy(output[train_mask], labels[train_mask])
        # L2 regularization on first-layer weights only
        l2_reg = l2_lambda * torch.norm(model.gc1.weight) ** 2
        loss = loss_train + l2_reg
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.cross_entropy(output[val_mask], labels[val_mask])
        if loss_val < best_val_loss - 1e-4:
            best_val_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    # Return validation accuracy
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        preds = output.argmax(dim=1)
        correct = (preds[val_mask] == labels[val_mask]).float().mean()
    return float(correct)

def predict(model, features, adj, idx):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        return output[idx].argmax(dim=1).cpu().numpy()

def write_predictions(path, indices, preds):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('id,prediction\n')
        for i, p in zip(indices, preds):
            f.write(f'{int(i)},{int(p)}\n')

def main():
    seed = int(os.environ.get('SEED', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    smoke = os.environ.get('SMOKE') == '1'

    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features.shape[0]

    # Preprocess
    features = preprocess_features(features)
    adj_np = build_adjacency(num_nodes, edges)

    # Convert to torch tensors
    device = torch.device('cpu')
    features_t = torch.from_numpy(features).to(device)
    adj_t = torch.from_numpy(adj_np).to(device)
    labels_t = torch.from_numpy(labels).to(device)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask[val_idx] = True

    # Hyperparameters from paper
    nhid = 16
    dropout = 0.5
    lr = 0.01
    l2_lambda = 5e-4
    max_epochs = 1 if smoke else 200
    patience = 10

    model = GCN(nfeat=features.shape[1], nhid=nhid, nclass=int(labels.max()+1), dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)  # we handle L2 manually

    val_acc = train(model, features_t, adj_t, labels_t, train_mask, val_mask,
                    optimizer, l2_lambda, max_epochs, patience)
    print(f'Validation accuracy: {val_acc:.4f}')

    # Predictions
    test_pred = predict(model, features_t, adj_t, test_idx)
    val_pred = predict(model, features_t, adj_t, val_idx)

    if smoke:
        # Use only first 20 sorted indices
        test_idx_use = np.sort(test_idx)[:20]
        val_idx_use = np.sort(val_idx)[:20]
        test_pred_use = test_pred[np.argsort(test_idx)][:20]
        val_pred_use = val_pred[np.argsort(val_idx)][:20]
    else:
        test_idx_use = test_idx
        val_idx_use = val_idx
        test_pred_use = test_pred
        val_pred_use = val_pred

    write_predictions('/output/predictions.csv', test_idx_use, test_pred_use)
    write_predictions('/output/val_predictions.csv', val_idx_use, val_pred_use)

if __name__ == '__main__':
    main()
