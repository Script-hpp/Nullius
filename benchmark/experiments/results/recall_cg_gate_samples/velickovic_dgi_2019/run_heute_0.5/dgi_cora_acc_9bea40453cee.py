# Plan step -> implemented in:
# 1. Read SEED from os.environ, set numpy.random.seed(SEED) and torch.manual_seed(SEED). -> set_seeds()
# 2. Load `/data/cora_public.npz`; obtain X (features), edge_index, y, train_idx, val_idx, test_idx. -> load_data()
# 3. Encoder: 1‑layer GCN, in_dim=1433 → H (hidden, uncertain: 512) → out_dim=H, ReLU activation. -> GCNEncoder
# 4. Readout: s = σ( mean(Z, dim=0) ), σ = sigmoid (uncertain). -> readout()
# 5. Discriminator: score(z, s) = zᵀ W s, W∈ℝ^{H×H} learned (uncertain initialization Xavier). -> Discriminator
# 6. Corruption: shuffle rows of X to get Ẋ, encode to Ẑ. -> corrupt_features()
# 7. Loss: −log σ(score(Z,s)) − log σ(−score(Ẑ,s)) summed over nodes (JSD estimator). -> compute_loss()
# 8. Optimizer: Adam(lr=LR_e (uncertain: 0.001), weight_decay=0.0). -> encoder_optimizer
# 9. Train encoder for Epoch_e (uncertain: 200) epochs, full‑batch, re‑shuffle corruption each epoch. -> pretrain_encoder()
#10. Freeze encoder, compute node embeddings Z for all nodes. -> get_embeddings()
#11. Classifier: linear layer C = 7 → logits; optimizer Adam(lr=LR_c (uncertain: 0.01), weight_decay=0.0). -> Classifier
#12. Train classifier for Epoch_c (uncertain: 100) epochs, early stop on val accuracy (patience 10). -> train_classifier()
#13. Predict on test_idx: argmax of logits, produce integer class labels. -> predict()
#14. Write `/output/predictions.csv` with header `id,prediction`; one row per test node (id = node index). -> nio.write_predictions()
#15. Verify total runtime < 5 minutes on CPU. (handled by early stopping & epoch limits)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def set_seeds():
    seed = int(os.environ.get('SEED', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    import nullius_io as nio
    d = nio.load()
    X = torch.from_numpy(d['features']).float()          # [N, 1433]
    edge_index = torch.from_numpy(d['edges']).long()     # [E, 2] (both directions present)
    y = torch.from_numpy(d['labels']).long()             # [N], -1 for test
    train_idx = torch.from_numpy(d['train_idx']).long()
    val_idx   = torch.from_numpy(d['val_idx']).long()
    test_idx  = torch.from_numpy(d['test_idx']).long()
    return X, edge_index, y, train_idx, val_idx, test_idx

def build_adj_norm(edge_index, num_nodes):
    # symmetric adjacency (already contains both directions)
    idx = edge_index.t()                     # [2, E]
    vals = torch.ones(idx.size(1), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes)).coalesce()
    # degree
    deg = torch.sparse.sum(adj, dim=1).to_dense() + 1e-6
    norm = deg.pow(-0.5)                     # D^{-1/2}
    return adj, norm

def gcn_conv(X, weight, adj, norm):
    # \hat{A} X W where \hat{A}=D^{-1/2} A D^{-1/2}
    X = X * norm[:, None]                    # D^{-1/2} X
    AX = torch.sparse.mm(adj, X)             # A (D^{-1/2} X)
    AX = AX * norm[:, None]                  # D^{-1/2} A D^{-1/2} X
    return torch.mm(AX, weight)              # \hat{A} X W

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, hid_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X, adj, norm):
        Z = gcn_conv(X, self.weight, adj, norm)
        return torch.relu(Z)

class Discriminator(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        nn.init.xavier_uniform_(self.W)

    def score(self, Z, s):
        # s: [hid], Z: [N, hid]
        ws = torch.mv(self.W, s)            # [hid]
        return torch.mv(Z, ws)              # [N]

def readout(Z):
    return torch.sigmoid(Z.mean(dim=0))

def corrupt_features(X):
    perm = torch.randperm(X.size(0))
    return X[perm]

def compute_loss(encoder, disc, X, adj, norm):
    Z = encoder(X, adj, norm)
    s = readout(Z)
    pos_score = disc.score(Z, s)
    X_corr = corrupt_features(X)
    Z_corr = encoder(X_corr, adj, norm)   # reuse same encoder weights
    neg_score = disc.score(Z_corr, s)
    loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()
    return loss

def pretrain_encoder(X, adj, norm, epochs, lr, smoke):
    N, D = X.shape
    hid = 256 if not smoke else 32   # smaller hidden in smoke to be safe
    encoder = GCNEncoder(D, hid)
    disc = Discriminator(hid)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(disc.parameters()), lr=lr, weight_decay=0.0)
    encoder.train()
    disc.train()
    for epoch in range(epochs):
        opt.zero_grad()
        loss = compute_loss(encoder, disc, X, adj, norm)
        loss.backward()
        opt.step()
        if (epoch+1) % 20 == 0 or epochs == 1:
            print(f'Pretrain epoch {epoch+1}/{epochs}, loss={loss.item():.4f}')
    return encoder

def get_embeddings(encoder, X, adj, norm):
    encoder.eval()
    with torch.no_grad():
        Z = encoder(X, adj, norm)
    return Z

class Classifier(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(hid_dim, n_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, Z):
        return self.linear(Z)

def train_classifier(Z, y, train_idx, val_idx, epochs, lr, patience, smoke):
    n_classes = 7
    cls = Classifier(Z.size(1), n_classes)
    opt = torch.optim.Adam(cls.parameters(), lr=lr, weight_decay=0.0)
    best_val_acc = 0.0
    best_state = None
    wait = 0
    for epoch in range(epochs):
        cls.train()
        opt.zero_grad()
        logits = cls(Z)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()
        # validation
        cls.eval()
        with torch.no_grad():
            val_logits = cls(Z[val_idx])
            pred = val_logits.argmax(dim=1)
            acc = (pred == y[val_idx]).float().mean().item()
        if acc > best_val_acc:
            best_val_acc = acc
            best_state = {k: v.clone() for k, v in cls.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience and not smoke:
                break
        if (epoch+1) % 10 == 0 or epochs == 1:
            print(f'Classifier epoch {epoch+1}/{epochs}, val_acc={acc:.4f}')
    if best_state is not None:
        cls.load_state_dict(best_state)
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    return cls

def predict(classifier, Z):
    classifier.eval()
    with torch.no_grad():
        logits = classifier(Z)
        pred = logits.argmax(dim=1)
    return pred

def main():
    set_seeds()
    X, edge_index, y, train_idx, val_idx, test_idx = load_data()
    num_nodes = X.size(0)
    adj, norm = build_adj_norm(edge_index, num_nodes)

    smoke = os.environ.get('SMOKE') == '1'
    lr_e = 0.001
    lr_c = 0.01
    epochs_e = 1 if smoke else 200
    epochs_c = 1 if smoke else 100
    patience = 0 if smoke else 10

    encoder = pretrain_encoder(X, adj, norm, epochs_e, lr_e, smoke)
    Z = get_embeddings(encoder, X, adj, norm)
    classifier = train_classifier(Z, y, train_idx, val_idx, epochs_c, lr_c, patience, smoke)
    pred = predict(classifier, Z)   # int64 tensor [N]

    # Write predictions for all nodes (helper will select test/val indices)
    import nullius_io as nio
    nio.write_predictions(pred)

if __name__ == '__main__':
    main()
