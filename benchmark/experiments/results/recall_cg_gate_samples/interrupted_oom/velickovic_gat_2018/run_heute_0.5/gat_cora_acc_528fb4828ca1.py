

# Plan step -> implemented in: set_seeds
# Plan step -> implemented in: load_data
# Plan step -> implemented in: GATLayer
# Plan step -> implemented in: GATModel
# Plan step -> implemented in: train_loop
# Plan step -> implemented in: predict_and_write

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

def set_seeds():
    """Set random seeds for reproducibility."""
    seed = int(os.environ.get('SEED', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    """Load Cora dataset via nullius_io helper."""
    d = nio.load()  # returns dict with keys: features, edges, labels, train_idx, val_idx, test_idx
    features = torch.from_numpy(d['features']).float()          # [N, F]
    edges = torch.from_numpy(d['edges']).long()                # [E, 2] (bidirectional)
    labels = torch.from_numpy(d['labels']).long()              # [N]
    train_idx = torch.from_numpy(d['train_idx']).long()
    val_idx = torch.from_numpy(d['val_idx']).long()
    test_idx = torch.from_numpy(d['test_idx']).long()
    return features, edges, labels, train_idx, val_idx, test_idx

class GATLayer(nn.Module):
    """
    Dense GAT layer (multi-head support via concatenation).
    Uses adjacency matrix for O(N^2) attention, fully vectorized.
    """
    def __init__(self, in_features, out_features, heads=1, concat=True,
                 dropout=0.6, attn_dropout=0.6, negative_slope=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.negative_slope = negative_slope

        # Linear transformation for each head
        self.W = nn.Parameter(torch.Tensor(in_features, out_features * heads))
        # Attention vectors: split into left and right parts later
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_features))

        self.reset_parameters()
        self.leaky_relu = nn.LeakyReLU(self.negative_slope)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W, gain=gain)
        nn.init.xavier_uniform_(self.att, gain=gain)

    def forward(self, x, adj):
        """
        x: [N, in_features]
        adj: [N, N] bool tensor (True where edge exists, includes self-loops)
        Returns: [N, out_features * heads] if concat else [N, out_features]
        """
        N = x.size(0)
        # Input dropout
        x_drop = F.dropout(x, p=self.dropout, training=self.training)
        # Linear projection
        h = torch.matmul(x_drop, self.W)          # [N, heads * out_features]
        h = h.view(N, self.heads, self.out_features)  # [N, heads, out_features]

        # Split attention vector
        att_l = self.att[:, :, :self.out_features]   # [1, heads, out_features]
        att_r = self.att[:, :, self.out_features:]   # [1, heads, out_features]

        # Compute a_l and a_r: [N, heads]
        a_l = (h * att_l).sum(dim=-1)   # [N, heads]
        a_r = (h * att_r).sum(dim=-1)   # [N, heads]

        # Attention scores: e_ij = LeakyReLU(a_l_i + a_r_j)
        e = self.leaky_relu(a_l.unsqueeze(1) + a_r.unsqueeze(0))  # [N, N, heads]

        # Mask non-edges
        e_masked = e.masked_fill(~adj.unsqueeze(-1), float('-inf'))

        # Softmax over destination dimension (j)
        max_e = e_masked.max(dim=1, keepdim=True)[0]          # [N, 1, heads]
        exp_e = torch.exp(e_masked - max_e)                   # [N, N, heads]
        sum_exp = exp_e.sum(dim=1, keepdim=True) + 1e-16      # [N, 1, heads]
        alpha = exp_e / sum_exp                               # [N, N, heads]

        # Attention dropout
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

        # Weighted aggregation: output_i = sum_j alpha_ij * h_j
        # einsum: 'nij,njf->nif' -> [N, heads, out_features]
        out = torch.einsum('nij,njf->nif', alpha, h)

        if self.concat:
            out = out.contiguous().view(N, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)  # average over heads -> [N, out_features]
        return out

class GATModel(nn.Module):
    """
    2-layer GAT model as specified in the plan.
    """
    def __init__(self, nfeat, nhidden, nclass, heads=8, dropout=0.6):
        super(GATModel, self).__init__()
        self.dropout = dropout
        # First multi-head GAT layer
        self.gat1 = GATLayer(
            in_features=nfeat,
            out_features=nhidden // heads,  # per-head output dimension
            heads=heads,
            concat=True,
            dropout=dropout,
            attn_dropout=dropout
        )
        # Second single-head GAT layer (output = nclass)
        self.gat2 = GATLayer(
            in_features=nhidden,
            out_features=nclass,
            heads=1,
            concat=False,
            dropout=dropout,
            attn_dropout=dropout
        )
        self.elu = nn.ELU()

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, adj)
        x = self.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, adj)  # [N, nclass]
        return x  # raw logits

def train_loop(model, features, adj, labels, train_idx, val_idx, max_epochs, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj)
            val_loss = criterion(val_logits[val_idx], labels[val_idx])
            pred = val_logits[val_idx].argmax(dim=1)
            acc = (pred == labels[val_idx]).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Return validation accuracy of best model
    model.eval()
    with torch.no_grad():
        val_logits = model(features, adj)
        pred = val_logits[val_idx].argmax(dim=1)
        acc = (pred == labels[val_idx]).float().mean().item()
    print(f"Validation Accuracy: {acc:.4f}")
    return acc

def predict_and_write(model, features, adj):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)  # [N, nclass]
    # Convert to numpy for nullius_io
    preds_np = logits.argmax(dim=1).cpu().numpy()
    nio.write_predictions(preds_np)

def main():
    set_seeds()
    features, edges, labels, train_idx, val_idx, test_idx = load_data()

    # Determine number of classes (ignore -1 labels)
    valid_labels = labels[labels >= 0]
    nclass = int(valid_labels.max().item() + 1) if valid_labels.numel() > 0 else 7
    nfeat = features.size(1)

    # Smoke mode handling
    smoke = os.environ.get('SMOKE') == '1'
    if smoke:
        max_epochs = 1
        patience = 100  # irrelevant, but keep large
    else:
        max_epochs = 10000
        patience = 100

    hidden_dim = 64  # as per plan (8 heads * 8 per head)
    heads = 8

    # Build adjacency matrix (include self-loops)
    N = features.size(0)
    adj = torch.zeros(N, N, dtype=torch.bool)
    adj[edges[:, 0], edges[:, 1]] = True
    adj.fill_diagonal_(True)  # add self-loops

    model = GATModel(nfeat=nfeat, nhidden=hidden_dim, nclass=nclass,
                     heads=heads, dropout=0.6)

    train_loop(model, features, adj, labels, train_idx, val_idx,
               max_epochs=max_epochs, patience=patience)

    predict_and_write(model, features, adj)

if __name__ == "__main__":
    main()