# Plan step -> implemented in: function or class name
# 1. Load npz; row-normalize features (L1) -> load_data()
# 2. Build symmetric binary adjacency from edges (add self-loops); compute symmetrically normalized adjacency -> build_adjacency()
# 3. Model: 2-layer GCN – Layer1: GraphConvolution(1433→16, ReLU, dropout=0.5, sparse_inputs=True); Layer2: GraphConvolution(16→7, linear, dropout=0.5) -> GraphConvolution, GCN
# 4. Weight init: Xavier (Glorot) for weights, zeros for bias -> GraphConvolution.reset_parameters()
# 5. Loss: masked softmax cross-entropy on train_idx -> masked_cross_entropy()
# 6. L2 regularization: 5e-4 on first-layer weights only -> training loop adds l2_reg
# 7. Optimizer: Adam, lr 0.01 -> configure_optimizer()
# 8. Train: max 200 epochs, early stopping on val loss (patience=10) using val_idx -> train_model()
# 9. Seed NumPy and PyTorch from SEED env var -> set_seeds()
# 10. Predict on test_idx; write predictions.csv -> predict_and_save()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

def set_seeds():
    seed = int(os.environ.get('SEED', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(npz_path='/data/cora_public.npz'):
    data = np.load(npz_path, allow_pickle=True)
    features = data['features'].astype(np.float32)          # [N, F]
    edges = data['edges'].astype(np.int64)                  # [E, 2]
    labels = data['labels'].astype(np.int64)                # [N]
    train_idx = data['train_idx'].astype(np.int64)
    val_idx = data['val_idx'].astype(np.int64)
    test_idx = data['test_idx'].astype(np.int64)
    # Row-normalize features (L1)
    row_sums = features.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0   # avoid div by zero
    features = features / row_sums
    return features, edges, labels, train_idx, val_idx, test_idx

def build_adjacency(edges, num_nodes):
    # edges already contain both directions; ensure binary
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    # make symmetric (in case)
    adj = adj + adj.T
    adj.data = np.clip(adj.data, 0, 1)
    # add self-loops
    adj = adj + sp.eye(num_nodes, dtype=np.float32, format='coo')
    adj.data = np.clip(adj.data, 0, 1)
    # symmetric normalization: D^{-1/2} A D^{-1/2}
    row_sum = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = sp.diags(d_inv_sqrt)
    adj_norm = d_mat @ adj @ d_mat
    # convert to torch sparse tensor
    adj_norm = adj_norm.tocoo()
    indices = torch.from_numpy(np.vstack((adj_norm.row, adj_norm.col)).astype(np.int64))
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    shape = torch.Size(adj_norm.shape)
    adj_torch = torch.sparse_coo_tensor(indices, values, shape)
    return adj_torch.coalesce()

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to arXiv:1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0., act=F.relu, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.use_bias = bias
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        # input: [N, in_features] dense
        # adj: torch sparse [N, N] normalized adjacency
        x = F.dropout(input, self.dropout, training=self.training)
        support = torch.sparse.mm(adj, x)          # [N, in_features]
        output = support @ self.weight             # [N, out_features]
        if self.use_bias:
            output = output + self.bias
        if self.act is not None:
            output = self.act(output)
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, dropout=dropout, act=F.relu)
        self.gc2 = GraphConvolution(nhid, nclass, dropout=dropout, act=lambda x: x)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        return x

def masked_cross_entropy(logits, labels, mask):
    loss = F.cross_entropy(logits[mask], labels[mask])
    return loss

def configure_optimizer(model, lr=0.01, weight_decay=5e-4):
    # L2 only on first layer weights
    optimizer = torch.optim.Adam([
        {'params': model.gc1.parameters(), 'weight_decay': weight_decay},
        {'params': model.gc2.parameters(), 'weight_decay': 0.0}
    ], lr=lr)
    return optimizer

def train_model(model, features, adj, labels, train_idx, val_idx, epochs=200, patience=10, lr=0.01):
    optimizer = configure_optimizer(model, lr=lr)
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = masked_cross_entropy(output, labels, train_idx)
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = masked_cross_entropy(output, labels, val_idx)
            preds_val = output[val_idx].argmax(dim=1)
            acc_val = (preds_val == labels[val_idx]).float().mean().item()

        if loss_val < best_val_loss - 1e-4:
            best_val_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # restore best
    model.load_state_dict(best_state)
    # final validation accuracy
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        preds_val = output[val_idx].argmax(dim=1)
        acc_val = (preds_val == labels[val_idx]).float().mean().item()
    print(f'Validation accuracy: {acc_val:.4f}')
    return model

def predict_and_save(model, features, adj, test_idx, val_idx, smoke_mode=False):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        # test predictions
        test_logits = output[test_idx]
        test_pred = test_logits.argmax(dim=1).cpu().numpy()
        # validation predictions
        val_logits = output[val_idx]
        val_pred = val_logits.argmax(dim=1).cpu().numpy()
    if smoke_mode:
        # only first 20 sorted indices
        test_idx_use = np.sort(test_idx)[:20]
        val_idx_use = np.sort(val_idx)[:20]
        test_pred = test_pred[:20]
        val_pred = val_pred[:20]
    else:
        test_idx_use = test_idx
        val_idx_use = val_idx
    # write test predictions
    os.makedirs('/output', exist_ok=True)
    with open('/output/predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for idx, pred in zip(test_idx_use, test_pred):
            f.write(f'{int(idx)},{int(pred)}\n')
    # write validation predictions
    with open('/output/val_predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for idx, pred in zip(val_idx_use, val_pred):
            f.write(f'{int(idx)},{int(pred)}\n')

def main():
    set_seeds()
    features_np, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features_np.shape[0]
    adj = build_adjacency(edges, num_nodes)

    features = torch.FloatTensor(features_np)
    labels = torch.LongTensor(labels)

    model = GCN(nfeat=features.shape[1],
                nhid=16,
                nclass=int(labels.max().item() + 1),
                dropout=0.5)

    smoke = os.environ.get('SMOKE', '0') == '1'
    if smoke:
        epochs = 1
    else:
        epochs = 200

    model = train_model(model, features, adj, labels, train_idx, val_idx,
                        epochs=epochs, patience=10, lr=0.01)

    predict_and_save(model, features, adj, test_idx.numpy(),
                     val_idx.numpy(), smoke_mode=smoke)

if __name__ == '__main__':
    main()
