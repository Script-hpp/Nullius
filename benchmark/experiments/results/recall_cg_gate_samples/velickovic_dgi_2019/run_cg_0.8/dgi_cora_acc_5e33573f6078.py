

# Plan step -> implemented in: function or class name
# 1. Load data -> load_data_via_nio (via nullius_io)
# 2. Preprocess features -> preprocess_features
# 3. Build and normalize adjacency -> build_adjacency, normalize_adj
# 4. Initialize DGI model -> DGI class
# 5. Pretrain DGI with BCE loss -> train_dgi function
# 6. Extract node embeddings -> model.embed method
# 7. Train logistic regression classifier -> LogReg class and train_logreg function
# 8. Repeat logistic regression 50 times for validation accuracy -> evaluation loop
# 9. Final predictions for all nodes -> final logistic regression
# 10. Write predictions via nio.write_predictions

import os
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp

# Hyperparameters from spec
CONFIG = {
    "lr": 0.001,
    "weight_decay": 0.0,
    "epochs": 10000,
    "batch_size": 1,
    "dropout": 0.0,
    "hidden": 512,
    "patience": 20,
    "logreg_lr": 0.01,
    "logreg_weight_decay": 0.0,
    "logreg_epochs": 100
}

# ------------------ Utility functions ------------------
def preprocess_features(features):
    """Row-normalize feature matrix and return dense numpy array."""
    rowsum = np.array(features.sum(axis=1)).flatten()
    r_inv = np.power(rowsum, -1.0)
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features_norm = r_mat_inv.dot(features)
    return features_norm.todense()  # numpy array


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col))
    values = sparse_mx.data
    shape = sparse_mx.shape
    tensor = torch.sparse_coo_tensor(
        torch.LongTensor(coords),
        torch.FloatTensor(values),
        torch.Size(shape)
    )
    return tensor


def build_adjacency(edges, num_nodes):
    """Construct adjacency matrix from edge list (already contains both directions)."""
    data = np.ones(edges.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
    return adj


# ------------------ Model components ------------------
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True, dropout=0.0):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        if act == 'prelu':
            self.act = nn.PReLU()
        else:
            self.act = act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        out = self.act(out)
        out = self.dropout(out)
        return out


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, dim=1)
        else:
            # mask assumed same shape as seq
            return torch.sum(seq * msk, dim=1) / torch.sum(msk, dim=1)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
        return logits


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation, dropout):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation, dropout=dropout)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)
        c = self.read(h_1, msk)
        c = self.sigm(c)
        h_2 = self.gcn(seq2, adj, sparse)
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret

    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)
        return h_1.detach(), c.detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        return self.fc(seq)


# ------------------ Training helpers ------------------
def train_dgi(model, features, adj, sparse, optimizer, b_xent_loss, epochs, patience, device):
    cnt_wait = 0
    best = 1e9
    best_epoch = 0
    nb_nodes = features.shape[1]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]

        lbl_1 = torch.ones(CONFIG["batch_size"], nb_nodes, device=device)
        lbl_2 = torch.zeros(CONFIG["batch_size"], nb_nodes, device=device)
        lbl = torch.cat((lbl_1, lbl_2), dim=1)  # shape [1, 2*nb_nodes]

        logits = model(features, shuf_fts, adj, sparse, None, None, None)
        loss = b_xent_loss(logits, lbl)

        if loss < best:
            best = loss
            best_epoch = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            break

        loss.backward()
        optimizer.step()

    model.load_state_dict(torch.load('best_dgi.pkl'))
    return model


def train_logreg(model, train_embs, train_labels, optimizer, ce_loss, epochs):
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(train_embs)
        loss = ce_loss(logits, train_labels)
        loss.backward()
        optimizer.step()
    return model


def eval_logreg(model, embs, labels):
    model.eval()
    with torch.no_grad():
        logits = model(embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels).float() / labels.shape[0]
    return acc.item()


# ------------------ Main script ------------------
def main():
    # Seed
    seed = int(os.environ.get('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Smoke mode
    smoke = os.environ.get('SMOKE') == '1'
    epochs = 1 if smoke else CONFIG["epochs"]

    # Load data via nullius_io
    import nullius_io as nio
    data = nio.load()
    features_np = data['features']          # [2708, 1433] float32
    edges_np = data['edges']                # [E, 2] int64
    labels_np = data['labels']              # [2708] int64, -1 for test
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    num_nodes = features_np.shape[0]
    ft_size = features_np.shape[1]

    # Build adjacency
    adj_sp = build_adjacency(edges_np, num_nodes)   # scipy sparse (COO)
    # Add self-loop and normalize
    adj_norm = normalize_adj(adj_sp + sp.eye(num_nodes))

    # Preprocess features
    features_norm = preprocess_features(features_np)  # numpy [N, F] float64 -> cast later
    features_norm = features_norm.astype(np.float32)

    # Prepare torch tensors
    device = torch.device('cpu')
    features_tensor = torch.FloatTensor(features_norm).unsqueeze(0).to(device)  # [1, N, F]

    if True:  # sparse=True as in original code
        adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)
        sparse_flag = True
    else:
        adj_dense = torch.FloatTensor(adj_norm.todense()).unsqueeze(0).to(device)
        adj_tensor = adj_dense
        sparse_flag = False

    # Number of classes (from training labels)
    nb_classes = len(np.unique(labels_np[train_idx]))

    # Initialize DGI model
    model = DGI(ft_size, CONFIG["hidden"], activation='prelu', dropout=CONFIG["dropout"])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CONFIG["lr"],
                                 weight_decay=CONFIG["weight_decay"])
    b_xent_loss = nn.BCEWithLogitsLoss()

    # Pretrain DGI
    model = train_dgi(model, features_tensor, adj_tensor, sparse_flag,
                      optimizer, b_xent_loss,
                      epochs, CONFIG["patience"], device)

    # Extract embeddings
    embeds, _ = model.embed(features_tensor, adj_tensor, sparse_flag, None)
    embeds = embeds.squeeze(0)  # [N, hidden]

    # Prepare tensors for downstream classification
    train_embs = embeds[train_idx]
    val_embs = embeds[val_idx]
    test_embs = embeds[test_idx]

    train_labels = torch.LongTensor(labels_np[train_idx]).to(device)
    val_labels = torch.LongTensor(labels_np[val_idx]).to(device)

    # Repeated logistic regression for validation accuracy estimation
    val_accs = []
    for _ in range(50):
        logreg = LogReg(CONFIG["hidden"], nb_classes).to(device)
        opt = torch.optim.Adam(logreg.parameters(), lr=CONFIG["logreg_lr"], weight_decay=CONFIG["logreg_weight_decay"])
        ce_loss = nn.CrossEntropyLoss()
        logreg = train_logreg(logreg, train_embs, train_labels, opt, ce_loss, CONFIG["logreg_epochs"])
        acc = eval_logreg(logreg, val_embs, val_labels)
        val_accs.append(acc)

    avg_val_acc = np.mean(val_accs)
    print(f'Validation accuracy: {avg_val_acc:.5f}')

    # Final logistic regression on full training data for final predictions
    final_logreg = LogReg(CONFIG["hidden"], nb_classes).to(device)
    final_opt = torch.optim.Adam(final_logreg.parameters(), lr=CONFIG["logreg_lr"], weight_decay=CONFIG["logreg_weight_decay"])
    final_ce = nn.CrossEntropyLoss()
    final_logreg = train_logreg(final_logreg, train_embs, train_labels, final_opt, final_ce, CONFIG["logreg_epochs"])

    # Predict all nodes
    with torch.no_grad():
        all_logits = final_logreg(embeds)
        all_preds = torch.argmax(all_logits, dim=1).cpu().numpy().astype(np.int64)

    # Write predictions (helper expects predictions for every node)
    nio.write_predictions(all_preds)


if __name__ == '__main__':
    main()