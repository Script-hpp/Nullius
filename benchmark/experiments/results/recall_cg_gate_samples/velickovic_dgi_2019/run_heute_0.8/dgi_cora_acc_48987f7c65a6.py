

# Plan step -> implemented in: set_seed
# Plan step -> implemented in: load_data
# Plan step -> implemented in: GCNEncoder
# Plan step -> implemented in: readout
# Plan step -> implemented in: Discriminator
# Plan step -> implemented in: corrupt_features
# Plan step -> implemented in: dgi_loss
# Plan step -> implemented in: train_encoder
# Plan step -> implemented in: get_embeddings
# Plan step -> implemented in: Classifier
# Plan step -> implemented in: train_classifier
# Plan step -> implemented in: predict
# Plan step -> implemented in: write_predictions
# Plan step -> implemented in: main

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed):
    """Step 1: Read SEED from os.environ, set numpy.random.seed(SEED) and torch.manual_seed(SEED)."""
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    """Step 2: Load `/data/cora_public.npz`; obtain X (features), edge_index, y, train_idx, val_idx, test_idx."""
    import nullius_io as nio
    d = nio.load()
    X = torch.from_numpy(d['features']).float()          # [N, F]
    edge_index = torch.from_numpy(d['edges']).t().long() # [2, E] (src, dst)
    y = torch.from_numpy(d['labels']).long()             # [N]
    train_idx = torch.from_numpy(d['train_idx']).long()
    val_idx   = torch.from_numpy(d['val_idx']).long()
    test_idx  = torch.from_numpy(d['test_idx']).long()
    return X, edge_index, y, train_idx, val_idx, test_idx

def readout(Z):
    """Step 4: s = σ( mean(Z, dim=0) ), σ = sigmoid."""
    return torch.sigmoid(torch.mean(Z, dim=0))

class GCNEncoder(nn.Module):
    """Step 3: Encoder: 1‑layer GCN, in_dim=1433 → H (hidden, uncertain: 256) → out_dim=H, ReLU activation.
       Implements random‑walk normalization (D^{-1}A) with sparse tensors."""
    def __init__(self, in_dim, hidden_dim, edge_index, num_nodes):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim, bias=False)
        src, dst = edge_index.unbind(dim=0)  # [E], [E]
        # out-degree normalization constants
        deg_out = torch.bincount(src, minlength=num_nodes).float().clamp(min=1.0)
        norm_src = 1.0 / deg_out                     # [N]
        # values for sparse tensor D^{-1}A: norm_src[src] for each edge src->dst
        values = norm_src[src]                       # [E]
        indices = edge_index.t()                     # [2, E] (row=src, col=dst)
        self.adj_norm = torch.sparse_coo_tensor(
            indices, values, size=(num_nodes, num_nodes), dtype=torch.float32
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, X, edge_index):
        # X: [N, F], edge_index: [2, E] (unused, structure stored in self.adj_norm)
        support = self.lin(X)                          # [N, H]
        aggregated = torch.sparse.mm(self.adj_norm, support)  # [N, H]
        out = support + aggregated                     # (I + D^{-1}A) * support
        return F.relu(out)

class Discriminator(nn.Module):
    """Step 5: Discriminator: score(z, s) = zᵀ W s, W∈ℝ^{H×H} learned (uncertain) initialization Xavier."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, z, s):
        # z: [N, H], s: [H]
        return torch.sum((z @ self.W) * s, dim=1)      # [N]

def corrupt_features(X):
    """Step 6: Corruption: shuffle rows of X to get Ẋ."""
    perm = torch.randperm(X.size(0))
    return X[perm]

def dgi_loss(pos, neg):
    """Step 7: Loss: −log σ(score(Z,s)) − log σ(−score(Ẑ,s)) summed over nodes."""
    return -torch.mean(F.logsigmoid(pos)) - torch.mean(F.logsigmoid(-neg))

def train_encoder(encoder, discriminator, X, edge_index, epochs, lr, smoke):
    """Steps 8‑9: Train encoder with DGI loss (full‑batch, reshuffle corruption each epoch)."""
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(discriminator.parameters()),
        lr=lr, weight_decay=0.0
    )
    encoder.train()
    discriminator.train()
    N = X.size(0)
    for epoch in range(epochs):
        optimizer.zero_grad()
        Z = encoder(X, edge_index)                           # [N, H]
        s = readout(Z)                                       # [H]
        pos = discriminator(Z, s)                            # [N]
        X_corr = corrupt_features(X)                         # [N, F]
        Z_corr = encoder(X_corr, edge_index)                 # [N, H]
        neg = discriminator(Z_corr, s)                       # [N]
        loss = dgi_loss(pos, neg)
        loss.backward()
        optimizer.step()
        if smoke and epoch == 0:
            break
    encoder.eval()
    discriminator.eval()

def get_embeddings(encoder, X, edge_index):
    """Step 10: Freeze encoder, compute node embeddings Z for all nodes."""
    with torch.no_grad():
        return encoder(X, edge_index)

class Classifier(nn.Module):
    """Step 11: Classifier: linear layer C = 7 → logits."""
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, z):
        return self.linear(z)

def train_classifier(classifier, Z, y, train_idx, val_idx, epochs, lr, patience, smoke):
    """Step 12: Train classifier for epochs, early stop on val accuracy (patience 10)."""
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=0.0)
    best_val_acc = 0.0
    wait = 0
    best_state = None
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(Z)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        # validation
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(Z[val_idx])
            pred = val_logits.argmax(dim=1)
            acc = (pred == y[val_idx]).float().mean().item()
        if acc > best_val_acc:
            best_val_acc = acc
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
        if smoke and epoch == 0:
            break
    classifier.load_state_dict(best_state)
    return best_val_acc

def predict(classifier, Z):
    """Step 13: Predict on test_idx: argmax of logits, produce integer class labels."""
    with torch.no_grad():
        logits = classifier(Z)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=1)
    return pred, logits

def write_predictions(logits):
    """Step 14: Write predictions via helper (expects logits or labels for all nodes)."""
    import nullius_io as nio
    nio.write_predictions(logits)

def main():
    seed = int(os.environ.get('SEED', 0))
    set_seed(seed)
    smoke = os.environ.get('SMOKE') == '1'

    X, edge_index, y, train_idx, val_idx, test_idx = load_data()
    N, F = X.shape
    n_classes = 7
    hidden_dim = 256  # reduced from 512 to meet time constraints; same mechanism

    encoder = GCNEncoder(F, hidden_dim, edge_index, N)
    discriminator = Discriminator(hidden_dim)

    # encoder training
    encoder_epochs = 1 if smoke else 200
    train_encoder(encoder, discriminator, X, edge_index,
                  epochs=encoder_epochs, lr=0.001, smoke=smoke)

    # get embeddings
    Z = get_embeddings(encoder, X, edge_index)

    # classifier
    classifier = Classifier(hidden_dim, n_classes)
    classifier_epochs = 1 if smoke else 100
    val_acc = train_classifier(classifier, Z, y, train_idx, val_idx,
                               epochs=classifier_epochs, lr=0.01,
                               patience=10, smoke=smoke)
    print(f"Validation accuracy: {val_acc:.4f}")

    _, logits = predict(classifier, Z)   # logits shape [N, 7]
    write_predictions(logits)

if __name__ == "__main__":
    main()