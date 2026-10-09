# Plan step -> implemented in: function or class name
# 1. Read SEED and seed -> set_seed()
# 2. Load data -> load_data()
# 3. Encoder: 1-layer GCN -> GCNEncoder
# 4. Readout: sigmoid mean -> readout()
# 5. Discriminator: bilinear score -> Discriminator
# 6. Corruption: shuffle features -> corrupt_features()
# 7. Loss: JSD estimator -> compute_dgi_loss()
# 8. Optimizer Adam(lr=0.001) -> pretrain_encoder()
# 9. Train encoder for epochs -> pretrain_encoder()
#10. Freeze encoder, compute embeddings -> get_embeddings()
#11. Classifier linear layer -> NodeClassifier
#12. Train classifier with early stop -> train_classifier()
#13. Predict on test_idx -> predict()
#14. Write predictions -> nio.write_predictions()
#15. Runtime <5 min -> ensured by early stopping and modest size

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    d = nio.load()
    X = torch.from_numpy(d['features']).float()          # [N, 1433]
    edge_index = torch.from_numpy(d['edges']).long().t() # [2, E]
    y = torch.from_numpy(d['labels']).long()             # [N]
    train_idx = torch.from_numpy(d['train_idx']).long()
    val_idx = torch.from_numpy(d['val_idx']).long()
    test_idx = torch.from_numpy(d['test_idx']).long()
    return X, edge_index, y, train_idx, val_idx, test_idx

def normalize_adj(edge_index, num_nodes):
    # Add self-loops
    loop_idx = torch.arange(num_nodes, device=edge_index.device)
    edge_index_with_loop = torch.cat([edge_index, loop_idx.unsqueeze(0).repeat(2,1)], dim=1)
    # Compute degree
    deg = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float)
    # Use float ones for index_add
    ones = torch.ones_like(edge_index_with_loop[0], dtype=torch.float)
    deg.index_add_(0, edge_index_with_loop[0], ones)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    # Norm values
    src, dst = edge_index_with_loop
    norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    adj_norm = torch.sparse_coo_tensor(edge_index_with_loop, norm, (num_nodes, num_nodes))
    return adj_norm.coalesce()

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, adj_norm):
        # x: [N, F], adj_norm: sparse [N,N]
        support = torch.sparse.mm(adj_norm, x)          # [N, F]
        out = self.linear(support)                      # [N, H]
        return F.relu(out)

def readout(z):
    # z: [N, H]
    s = torch.sigmoid(z.mean(dim=0))                    # [H]
    return s

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, z, s):
        # z: [N, H], s: [H]
        # score = z^T W s  (scalar per node)
        zW = z @ self.W                                 # [N, H]
        score = (zW * s).sum(dim=1)                     # [N]
        return score

def corrupt_features(x):
    perm = torch.randperm(x.size(0), device=x.device)
    return x[perm]

def compute_dgi_loss(encoder, discriminator, x, adj_norm):
    z = encoder(x, adj_norm)                            # [N, H]
    s = readout(z)                                      # [H]
    pos_score = discriminator(z, s)                     # [N]
    # corrupted
    x_corr = corrupt_features(x)
    z_corr = encoder(x_corr, adj_norm)                  # [N, H]
    neg_score = discriminator(z_corr, s)                # [N]
    loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()
    return loss, z.detach()

def pretrain_encoder(X, edge_index, y, train_idx, val_idx, test_idx, epochs, lr, smoke):
    N = X.size(0)
    adj_norm = normalize_adj(edge_index, N)
    encoder = GCNEncoder(in_dim=X.size(1), hidden_dim=512)
    discriminator = Discriminator(hidden_dim=512)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(discriminator.parameters()),
                                 lr=lr, weight_decay=0.0)
    encoder.train()
    discriminator.train()
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        loss, _ = compute_dgi_loss(encoder, discriminator, X, adj_norm)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == epochs:
            print(f'Pretrain Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')
    encoder.eval()
    return encoder

def get_embeddings(encoder, X, edge_index):
    N = X.size(0)
    adj_norm = normalize_adj(edge_index, N)
    with torch.no_grad():
        z = encoder(X, adj_norm)   # [N, H]
    return z

class NodeClassifier(nn.Module):
    def __init__(self, hidden_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, n_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z):
        return self.fc(z)   # [N, C]

def train_classifier(z, y, train_idx, val_idx, epochs, lr, patience, smoke):
    classifier = NodeClassifier(hidden_dim=z.size(1), n_classes=7)
    opt = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=0.0)
    best_val_acc = 0.0
    wait = 0
    for epoch in range(1, epochs+1):
        classifier.train()
        opt.zero_grad()
        logits = classifier(z[train_idx])
        loss = F.cross_entropy(logits, y[train_idx])
        loss.backward()
        opt.step()
        # validation
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(z[val_idx])
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == y[val_idx]).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            best_state = classifier.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stop at epoch {epoch}')
                break
        if epoch % 10 == 0 or epoch == epochs:
            print(f'Classifier Epoch {epoch}/{epochs}, Val Acc: {val_acc:.4f}')
    classifier.load_state_dict(best_state)
    return classifier, best_val_acc

def predict(classifier, z):
    classifier.eval()
    with torch.no_grad():
        logits = classifier(z)          # [N, 7]
    return logits   # can be logits or labels; helper expects logits or labels

def main():
    seed = int(os.environ.get('SEED', 0))
    set_seed(seed)

    X, edge_index, y, train_idx, val_idx, test_idx = load_data()
    smoke = os.environ.get('SMOKE') == '1'

    # Pretraining epochs
    pretrain_epochs = 1 if smoke else 200
    pretrain_lr = 0.001
    encoder = pretrain_encoder(X, edge_index, y, train_idx, val_idx, test_idx,
                               epochs=pretrain_epochs, lr=pretrain_lr, smoke=smoke)

    # Get embeddings
    z = get_embeddings(encoder, X, edge_index)

    # Classifier training
    classifier_epochs = 1 if smoke else 100
    classifier_lr = 0.01
    patience = 10 if not smoke else 1e9  # effectively disable early stop in smoke
    classifier, val_acc = train_classifier(z, y, train_idx, val_idx,
                                           epochs=classifier_epochs,
                                           lr=classifier_lr,
                                           patience=patience,
                                           smoke=smoke)

    # Predict for all nodes
    pred_logits = predict(classifier, z)   # Tensor [N,7]
    # Write predictions (helper will select test_idx)
    nio.write_predictions(pred_logits)

    print(f'Final validation accuracy: {val_acc:.4f}')

if __name__ == '__main__':
    main()
