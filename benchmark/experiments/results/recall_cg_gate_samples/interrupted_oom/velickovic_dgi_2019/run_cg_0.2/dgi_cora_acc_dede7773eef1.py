

# Plan step -> implemented in: DGI class (includes GCN, AvgReadout, Discriminator)
# Plan step -> implemented in: preprocess_features and normalize_adj functions
# Plan step -> implemented in: CONFIG dict and training loop
# Plan step -> implemented in: model.embed method
# Plan step -> implemented in: logistic regression evaluation loop
# Plan step -> implemented in: final logistic regression and nio.write_predictions call

import os
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp

import nullius_io as nio

CONFIG = {
    "lr": 0.001,
    "weight_decay": 0.0,
    "epochs": 10000,
    "batch_size": 1,
    "dropout": 0.0,
    "hidden": 512,
    "patience": 20,
}

def preprocess_features(features):
    """Row-normalize feature matrix and return dense ndarray"""
    assert sp.isspmatrix(features)
    rowsum = np.array(features.sum(axis=1)).flatten()
    r_inv = np.power(rowsum, -1.0)
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense().astype(np.float32)

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
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    coords = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(coords, values, shape)

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True, dropout=0.0):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
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
            msk = msk.unsqueeze(-1)
            sum_ = torch.sum(seq * msk, dim=1)
            count = torch.sum(msk, dim=1)
            return sum_ / count

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
    def __init__(self, n_in, n_h, activation, dropout=0.0):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation, bias=True, dropout=dropout)
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
    def forward(self, seq):
        return self.fc(seq)

def main():
    seed = int(os.environ.get('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = nio.load()
    features = data['features']          # [N, F] numpy float32
    edges = data['edges']                # [E, 2] int64
    labels_int = data['labels']          # [N] int64, -1 for test
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    N = features.shape[0]
    F = features.shape[1]

    # Preprocess features
    features_sp = sp.csr_matrix(features)
    features_norm = preprocess_features(features_sp)  # [N, F] numpy float32

    # Build adjacency matrix
    adj = sp.csr_matrix((np.ones(edges.shape[0], dtype=np.float32),
                         (edges[:, 0], edges[:, 1])), shape=(N, N))
    adj = adj + sp.eye(N)                     # add self-loop
    adj_norm = normalize_adj(adj)             # symmetrically normalized
    sparse = True
    if sparse:
        sp_adj = sparse_mx_to_torch_sparse_tensor(adj_norm)
    else:
        adj_dense = torch.FloatTensor(adj_norm.toarray())[np.newaxis, :, :]

    # Features tensor with batch dimension
    features_tensor = torch.FloatTensor(features_norm)[np.newaxis, :, :]  # [1, N, F]

    # Labels one-hot (for logistic regression)
    nb_classes = 7
    labels_onehot = np.zeros((N, nb_classes), dtype=np.float32)
    for i, lbl in enumerate(labels_int):
        if lbl >= 0:
            labels_onehot[i, int(lbl)] = 1.0
    labels_tensor = torch.FloatTensor(labels_onehot)[np.newaxis, :, :]  # [1, N, C]

    # Index tensors
    train_idx_t = torch.LongTensor(train_idx)
    val_idx_t = torch.LongTensor(val_idx)
    test_idx_t = torch.LongTensor(test_idx)

    # Model, optimizer, losses
    model = DGI(F, CONFIG["hidden"], activation='prelu', dropout=CONFIG["dropout"])
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CONFIG["lr"],
                                 weight_decay=CONFIG["weight_decay"])
    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()

    # Training loop
    nb_epochs = 1 if os.environ.get('SMOKE') == '1' else CONFIG["epochs"]
    patience = CONFIG["patience"]
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):
        model.train()
        optimizer.zero_grad()

        idx = np.random.permutation(N)
        shuf_fts = features_tensor[:, idx, :]

        lbl_1 = torch.ones(CONFIG["batch_size"], N)
        lbl_2 = torch.zeros(CONFIG["batch_size"], N)
        lbl = torch.cat((lbl_1, lbl_2), 1)  # [1, 2N]

        logits = model(features_tensor, shuf_fts,
                       sp_adj if sparse else adj_dense,
                       sparse, None, None, None)
        loss = b_xent(logits, lbl)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            break

        loss.backward()
        optimizer.step()

    # Load best model
    model.load_state_dict(torch.load('best_dgi.pkl'))
    model.eval()

    # Extract embeddings
    embeds, _ = model.embed(features_tensor,
                            sp_adj if sparse else adj_dense,
                            sparse, None)  # embeds: [1, N, hidden]
    train_embs = embeds[0, train_idx_t]
    val_embs = embeds[0, val_idx_t]
    test_embs = embeds[0, test_idx_t]
    train_lbls = torch.argmax(labels_tensor[0, train_idx_t], dim=1)
    val_lbls = torch.argmax(labels_tensor[0, val_idx_t], dim=1)
    test_lbls = torch.argmax(labels_tensor[0, test_idx_t], dim=1)

    # Logistic regression evaluation (50 repeats)
    accs = []
    for _ in range(50):
        log = LogReg(CONFIG["hidden"], nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=CONFIG["lr"], weight_decay=0.0)
        for __ in range(100):
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward()
            opt.step()
        log.eval()
        with torch.no_grad():
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            accs.append(acc.item() * 100)  # percentage
    avg_acc = np.mean(accs)
    print(f'Average accuracy: {avg_acc:.2f}%')

    # Final logistic regression for predictions on all nodes
    final_log = LogReg(CONFIG["hidden"], nb_classes)
    opt_f = torch.optim.Adam(final_log.parameters(), lr=CONFIG["lr"], weight_decay=0.0)
    for __ in range(100):
        final_log.train()
        opt_f.zero_grad()
        logits = final_log(train_embs)
        loss = xent(logits, train_lbls)
        loss.backward()
        opt_f.step()
    final_log.eval()
    with torch.no_grad():
        all_logits = final_log(embeds[0])          # [N, nb_classes]
        preds_all = torch.argmax(all_logits, dim=1)  # [N]
    nio.write_predictions(preds_all.cpu().numpy())

if __name__ == '__main__':
    main()