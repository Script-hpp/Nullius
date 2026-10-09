

# Plan step -> implemented in: preprocess_features
# Plan step -> implemented in: normalize_adj
# Plan step -> implemented in: sparse_mx_to_torch_sparse_tensor
# Plan step -> implemented in: GCN class
# Plan step -> implemented in: AvgReadout class
# Plan step -> implemented in: Discriminator class
# Plan step -> implemented in: DGI class
# Plan step -> implemented in: LogReg class
# Plan step -> implemented in: training loop inside main()
# Plan step -> implemented in: early stopping inside training loop
# Plan step -> implemented in: model.embed() call
# Plan step -> implemented in: logistic regression training & validation accuracy loop
# Plan step -> implemented in: final logistic regression and nio.write_predictions()

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

# Seed handling
seed = int(os.getenv("SEED", "0"))
np.random.seed(seed)
torch.manual_seed(seed)

def preprocess_features(features):
    """Row-normalize feature matrix and return dense ndarray."""
    # features: numpy array [N, F]
    rowsum = np.array(features.sum(1))  # shape (N,)
    r_inv = np.power(rowsum, -1).astype(np.float32)
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = np.diag(r_inv)
    features_norm = r_mat_inv.dot(features)
    return features_norm, None

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    coords = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data.astype(np.float32))
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(coords, values, shape)

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        if act == "prelu":
            self.act = nn.PReLU()
        else:
            self.act = act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter("bias", None)
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
        return self.act(out)

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

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
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.dropout = nn.Dropout(CONFIG["dropout"])

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)
        h_1 = self.dropout(h_1)
        c = self.read(h_1, msk)
        c = self.sigm(c)
        h_2 = self.gcn(seq2, adj, sparse)
        h_2 = self.dropout(h_2)
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        h_1 = self.dropout(h_1)
        c = self.read(h_1, msk)
        return h_1.detach(), c.detach()

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.weights_init(self.fc)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        return self.fc(seq)

def main():
    # Load data via nullius_io
    d = nio.load()
    features = d["features"]          # [N, F] float32
    edges = d["edges"]                # [E, 2] int64
    labels = d["labels"]              # [N] int64, -1 for test
    train_idx = d["train_idx"]
    val_idx = d["val_idx"]
    test_idx = d["test_idx"]

    N = features.shape[0]

    # Build adjacency matrix (add self-loop)
    adj = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(N, N))
    adj = adj + sp.eye(N)
    adj_norm = normalize_adj(adj)

    # Preprocess features
    features_norm, _ = preprocess_features(features)   # dense np.ndarray [N, F]

    # Prepare torch tensors
    features_tensor = torch.FloatTensor(features_norm[np.newaxis, :, :])  # [1, N, F]
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_norm)
    sparse_flag = True

    # Model, optimizer, loss
    ft_size = features.shape[1]
    hid_units = CONFIG["hidden"]
    activation = nn.PReLU()  # nonlinearity='prelu'
    model = DGI(ft_size, hid_units, activation)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    b_xent = nn.BCEWithLogitsLoss()

    # Training hyperparameters
    nb_nodes = N
    batch_size = CONFIG["batch_size"]
    patience = CONFIG["patience"]
    nb_epochs = CONFIG["epochs"]
    smoke = os.getenv("SMOKE") == "1"
    if smoke:
        nb_epochs = 1  # smoke mode: exactly 1 epoch

    best = 1e9
    best_t = 0
    cnt_wait = 0

    for epoch in range(nb_epochs):
        model.train()
        optimizer.zero_grad()
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features_tensor[:, idx, :]  # [1, N, F]

        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)  # [1, 2*N]

        logits = model(
            features_tensor,
            shuf_fts,
            adj_tensor if sparse_flag else adj_tensor,
            sparse_flag,
            None,
            None,
            None,
        )
        loss = b_xent(logits, lbl)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), "best_dgi.pt")
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            # Early stopping
            break

        loss.backward()
        optimizer.step()

    # Load best model
    model.load_state_dict(torch.load("best_dgi.pt"))
    model.eval()
    with torch.no_grad():
        embeds, _ = model.embed(
            features_tensor,
            adj_tensor if sparse_flag else adj_tensor,
            sparse_flag,
            None,
        )
        embeds = embeds.squeeze(0)  # [N, hid_units]

    train_embs = embeds[train_idx]
    val_embs = embeds[val_idx]

    train_labels = torch.from_numpy(labels[train_idx]).long()
    val_labels = torch.from_numpy(labels[val_idx]).long()

    # Repeated logistic regression to estimate validation accuracy
    accs = []
    for _ in range(50):
        log = LogReg(hid_units, 7)  # 7 classes in Cora
        opt = torch.optim.Adam(log.parameters(), lr=CONFIG["lr"], weight_decay=0.0)
        log.train()
        for __ in range(100):
            opt.zero_grad()
            logits = log(train_embs)
            loss = nn.CrossEntropyLoss()(logits, train_labels)
            loss.backward()
            opt.step()
        log.eval()
        with torch.no_grad():
            val_logits = log(val_embs)
            preds = torch.argmax(val_logits, dim=1)
            acc = (preds == val_labels).float().mean()
            accs.append(acc.item())

    avg_val_acc = np.mean(accs)
    print(f"{avg_val_acc:.4f}")

    # Final logistic regression model for predictions (trained on train set)
    final_log = LogReg(hid_units, 7)
    opt_final = torch.optim.Adam(final_log.parameters(), lr=CONFIG["lr"], weight_decay=0.0)
    final_log.train()
    for _ in range(100):
        opt_final.zero_grad()
        logits = final_log(train_embs)
        loss = nn.CrossEntropyLoss()(logits, train_labels)
        loss.backward()
        opt_final.step()
    final_log.eval()
    with torch.no_grad():
        all_logits = final_log(embeds)          # [N, 7]
        preds_all = torch.argmax(all_logits, dim=1)  # [N]
    preds_np = preds_all.cpu().numpy()

    # Write predictions via nullius_io (handles test/val selection)
    nio.write_predictions(preds_np)

if __name__ == "__main__":
    main()