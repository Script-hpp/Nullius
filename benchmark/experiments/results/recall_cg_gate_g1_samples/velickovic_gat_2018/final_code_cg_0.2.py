

# Plan step -> implemented in: set_seeds
# Plan step -> implemented in: load_data
# Plan step -> implemented in: preprocess_features
# Plan step -> implemented in: build_adj_bias
# Plan step -> implemented in: GATAttnHead
# Plan step -> implemented in: GATLayer
# Plan step -> implemented in: GAT
# Plan step -> implemented in: train_loop
# Plan step -> implemented in: main

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {"lr": 0.005, "weight_decay": 0.0005, "epochs": 100000, "batch_size": 1, "dropout": 0.6, "patience": 100, "heads": 8}

def set_seeds():
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    d = nio.load()
    features = d["features"].astype(np.float32)          # (N, F)
    edges = d["edges"].astype(np.int64)                  # (E, 2)
    labels = d["labels"].astype(np.int64)                # (N,)
    train_idx = d["train_idx"]
    val_idx = d["val_idx"]
    test_idx = d["test_idx"]
    return features, edges, labels, train_idx, val_idx, test_idx

def preprocess_features(feats):
    """Row-normalize feature matrix."""
    rowsum = np.sum(feats, axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    return feats / rowsum

def build_adj_bias(num_nodes, edges):
    """Adjacency matrix with self-loops, then bias mask where 0 for edge, -1e9 otherwise."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    if edges.size > 0:
        adj[edges[:, 0], edges[:, 1]] = 1.0
    # add self-loops
    np.fill_diagonal(adj, 1.0)
    # bias: 0 where adj>0, else a large negative
    bias = np.where(adj > 0, 0.0, -1e9).astype(np.float32)
    return bias

class GATAttnHead(nn.Module):
    """Single GAT attention head matching the TensorFlow implementation."""
    def __init__(self, in_sz, out_sz, dropout, activation, residual=False):
        super().__init__()
        self.dropout = dropout
        self.activation = activation
        self.residual = residual
        self.fc_W = nn.Linear(in_sz, out_sz, bias=False)      # W for seq_fts
        self.fc_f1 = nn.Linear(out_sz, 1, bias=False)         # for f_1
        self.fc_f2 = nn.Linear(out_sz, 1, bias=False)         # for f_2
        self.bias = nn.Parameter(torch.zeros(out_sz))         # bias_add
        if residual:
            self.proj_W = nn.Linear(in_sz, out_sz, bias=False) if in_sz != out_sz else None
        else:
            self.proj_W = None

    def forward(self, x, bias_mat, training):
        """
        x: (N, in_sz)
        bias_mat: (N, N) tensor (0 for edge, -1e9 for non-edge)
        Returns: (N, out_sz)
        """
        if self.dropout > 0.0 and training:
            x = F.dropout(x, p=self.dropout, training=True)

        seq_fts = self.fc_W(x)                     # (N, out_sz)

        f1 = self.fc_f1(seq_fts)                   # (N, 1)
        f2 = self.fc_f2(seq_fts)                   # (N, 1)
        logits = f1 + f2.transpose(0, 1)           # (N, N)

        coefs = F.leaky_relu(logits) + bias_mat    # (N, N)
        coefs = F.softmax(coefs, dim=-1)           # normalize over destinations

        if self.dropout > 0.0 and training:
            coefs = F.dropout(coefs, p=self.dropout, training=True)
        if self.dropout > 0.0 and training:
            seq_fts_dropped = F.dropout(seq_fts, p=self.dropout, training=True)
        else:
            seq_fts_dropped = seq_fts

        vals = torch.matmul(coefs, seq_fts_dropped)   # (N, out_sz)
        ret = vals + self.bias                         # bias_add

        if self.residual:
            if self.proj_W is not None:
                res = self.proj_W(x)
            else:
                res = x
            ret = ret + res

        out = self.activation(ret)
        return out

class GATLayer(nn.Module):
    """Multi-head GAT layer (used for hidden layers)."""
    def __init__(self, in_sz, out_sz, n_heads, dropout, activation, residual=False):
        super().__init__()
        self.heads = nn.ModuleList([
            GATAttnHead(in_sz, out_sz, dropout, activation, residual=False)
            for _ in range(n_heads)
        ])
        self.residual = residual

    def forward(self, x, bias_mat, training):
        head_outs = [head(x, bias_mat, training) for head in self.heads]
        out = torch.cat(head_outs, dim=-1)   # (N, out_sz * n_heads)
        if self.residual:
            # not needed as residual=False in our usage
            pass
        return out

class GAT(nn.Module):
    """GAT model with two layers as per the specification."""
    def __init__(self, ft_size, nb_classes, hid_units, n_heads, dropout, activation=F.elu, residual=False):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        in_sz = ft_size
        for l in range(len(hid_units)):
            out_sz = hid_units[l]
            self.hidden_layers.append(GATLayer(in_sz, out_sz, n_heads[l], dropout, activation, residual=False))
            in_sz = out_sz * n_heads[l]
        # Output heads: multiple heads, each predicting logits, then averaged
        self.out_heads = nn.ModuleList([
            GATAttnHead(in_sz, nb_classes, dropout, lambda x: x, residual=False)
            for _ in range(n_heads[-1])
        ])

    def forward(self, x, bias_mat, training):
        for layer in self.hidden_layers:
            x = layer(x, bias_mat, training)
        # Collect logits from each output head
        logits_list = [head(x, bias_mat, training) for head in self.out_heads]
        # Average over heads
        logits = torch.stack(logits_list, dim=0).mean(dim=0)
        return logits

def masked_loss(logits, label_indices, mask):
    """Masked softmax cross-entropy loss."""
    # Replace invalid labels (-1) with 0 to avoid gather error; masked positions contribute zero loss.
    safe_labels = label_indices.clone()
    safe_labels[mask == 0] = 0
    loss_per_node = F.cross_entropy(logits, safe_labels, reduction='none')
    loss = (loss_per_node * mask).sum() / mask.sum()
    return loss

def masked_accuracy(logits, label_indices, mask):
    """Masked accuracy."""
    preds = logits.argmax(dim=-1)
    correct = (preds == label_indices).float()
    correct = correct * mask
    return correct.sum() / mask.sum()

def train_loop():
    set_seeds()
    features_raw, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features_raw.shape[0]
    feat_dim = features_raw.shape[1]
    num_classes = int(labels.max() + 1)  # labels are 0..6 for labeled nodes

    # Preprocess
    features = preprocess_features(features_raw)               # (N, F)
    bias_mat = build_adj_bias(num_nodes, edges)                # (N, N)

    # Torch tensors
    features_t = torch.from_numpy(features).float()            # (N, F)
    label_indices_t = torch.from_numpy(labels).long()          # (N,)
    bias_t = torch.from_numpy(bias_mat)                        # (N, N)

    # Masks (float)
    train_mask = torch.zeros(num_nodes, dtype=torch.float32)
    train_mask[train_idx] = 1.0
    val_mask = torch.zeros(num_nodes, dtype=torch.float32)
    val_mask[val_idx] = 1.0

    # Model
    hid_units = [8]                     # units per head in first layer
    n_heads = [CONFIG["heads"], 1]      # [8, 1] as per spec
    model = GAT(
        ft_size=feat_dim,
        nb_classes=num_classes,
        hid_units=hid_units,
        n_heads=n_heads,
        dropout=CONFIG["dropout"],
        activation=F.elu,
        residual=False,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    epochs = 1 if os.environ.get("SMOKE") == "1" else CONFIG["epochs"]
    patience = CONFIG["patience"]
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features_t, bias_t, training=True)
        loss = masked_loss(logits, label_indices_t, train_mask)
        acc = masked_accuracy(logits, label_indices_t, train_mask)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features_t, bias_t, training=False)
            val_loss = masked_loss(val_logits, label_indices_t, val_mask)
            val_acc = masked_accuracy(val_logits, label_indices_t, val_mask)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | loss {loss.item():.4f} acc {acc.item():.4f} | "
                f"val_loss {val_loss.item():.4f} val_acc {val_acc.item():.4f}"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final predictions (inference mode, dropout off)
    model.eval()
    with torch.no_grad():
        final_logits = model(features_t, bias_t, training=False)
    preds = final_logits.argmax(dim=-1).cpu().numpy()  # (N,)

    nio.write_predictions(preds)
    print(f"Best validation accuracy: {best_val_acc:.4f}")

def main():
    train_loop()

if __name__ == "__main__":
    main()