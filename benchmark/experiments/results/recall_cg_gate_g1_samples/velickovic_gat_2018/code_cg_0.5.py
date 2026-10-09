# Plan step -> implemented in: class GAT
# Plan step -> implemented in: __init__ of GAT (hid_units)
# Plan step -> implemented in: __init__ of GAT (n_heads)
# Plan step -> implemented in: __init__ of GAT (dropout rates)
# Plan step -> implemented in: AttentionHead (activation)
# Plan step -> implemented in: __init__ of GAT (residual flag)
# Plan step -> implemented in: optimizer setup (lr, weight_decay)
# Plan step -> implemented in: optimizer choice (Adam)
# Plan step -> implemented in: training loop (batch_size, epochs)
# Plan step -> implemented in: training loop (early stopping with patience)
# Plan step -> implemented in: preprocessing (feature row-norm, adj bias)
# Plan step -> implemented in: seed setup
# Plan step -> implemented in: training loop (batch size 1, dropout applied)
# Plan step -> implemented in: restore best state, predict argmax
# Plan step -> implemented in: nio.write_predictions call

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

# Hyperparameters from spec (read only from this dict)
CONFIG = {
    "lr": 0.005,
    "weight_decay": 0.0005,
    "epochs": 100000,
    "batch_size": 1,
    "dropout": 0.6,
    "patience": 100,
    "heads": 8,
}

def set_seed(seed=None):
    if seed is None:
        seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

def row_normalize_features(feats):
    """Row-normalize feature matrix (each row sums to 1)."""
    row_sum = feats.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return feats / row_sum

def build_adj_bias(num_nodes, edges):
    """Adjacency matrix with self-loops, then bias mask for attention."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    u = edges[:, 0]
    v = edges[:, 1]
    adj[u, v] = 1.0
    # add self-loops
    np.fill_diagonal(adj, 1.0)
    # bias: 0 where edge exists, large negative otherwise
    NEG_INF = -1e9
    bias = NEG_INF * (1.0 - adj)
    return bias  # shape [N, N]

class AttentionHead(nn.Module):
    """
    Single GAT attention head.
    Mirrors TensorFlow implementation from utils/layers.py attn_head.
    """
    def __init__(self, in_features, out_features, dropout, attn_drop, ffd_drop, activation=F.elu):
        super().__init__()
        self.dropout = dropout
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.activation = activation

        # Linear projection W (no bias)
        self.W = nn.Linear(in_features, out_features, bias=False)
        # For attention coefficients: a single-layer MLP producing scalars
        self.W_f1 = nn.Linear(out_features, 1, bias=False)
        self.W_f2 = nn.Linear(out_features, 1, bias=False)
        # Bias added after neighbor aggregation (tf.contrib.layers.bias_add)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.W_f1.weight)
        nn.init.xavier_uniform_(self.W_f2.weight)

    def forward(self, x, bias_mat, training):
        """
        x:      (B, N, in_features)
        bias_mat: (B, N, N) or broadcastable
        Returns: (B, N, out_features) after activation.
        """
        if self.ffd_drop != 0.0:
            x = F.dropout(x, p=self.ffd_drop, training=training)

        # Feature projection
        h = self.W(x)                     # (B, N, out_features)

        # Compute unnormalized attention scores
        f_1 = self.W_f1(h)                # (B, N, 1)
        f_2 = self.W_f2(h)                # (B, N, 1)
        logits = f_1 + f_2.transpose(1, 2)  # (B, N, N)

        # LeakyReLU (alpha=0.2 as in TF default)
        logits = F.leaky_relu(logits, negative_slope=0.2)

        # Add bias mask (masked softmax)
        logits = logits + bias_mat

        # Normalize attention coefficients
        coefs = F.softmax(logits, dim=-1)  # (B, N, N)

        if self.attn_drop != 0.0:
            coefs = F.dropout(coefs, p=self.attn_drop, training=training)

        # Second feature dropout (as in original code)
        if self.ffd_drop != 0.0:
            h = F.dropout(h, p=self.ffd_drop, training=training)

        # Weighted sum of neighbors
        z = torch.matmul(coefs, h)         # (B, N, out_features)

        # Add bias (equivalent to tf.contrib.layers.bias_add)
        z = z + self.bias

        return self.activation(z)

class GAT(nn.Module):
    """
    GAT model matching the paper's architecture:
    - 1 hidden layer with 8 attention heads each outputting 8 features -> concat => 64
    - ELU activation on hidden layer
    - Output layer: 1 attention head producing logits for each class, averaged over heads
    """
    def __init__(self, feat_size, hidden_units, n_heads, nb_classes, dropout, attn_drop, ffd_drop, residual=False):
        super().__init__()
        self.hidden_units = hidden_units   # e.g., [8]
        self.n_heads = n_heads             # e.g., [8, 1]
        self.dropout = dropout
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.residual = residual

        # Build hidden layers
        self.layers = nn.ModuleList()
        in_ch = feat_size
        for l in range(len(hidden_units)):
            out_ch = hidden_units[l]
            heads = n_heads[l]
            heads_list = nn.ModuleList()
            for _ in range(heads):
                heads_list.append(
                    AttentionHead(in_ch, out_ch, dropout, attn_drop, ffd_drop, activation=F.elu)
                )
            self.layers.append(heads_list)
            in_ch = out_ch * heads  # after concatenation

        # Output layer (logits)
        self.output_heads = nn.ModuleList()
        for _ in range(n_heads[-1]):
            self.output_heads.append(
                AttentionHead(in_ch, nb_classes, dropout, attn_drop, ffd_drop, activation=lambda x: x)
            )

    def forward(self, x, bias_mat, training):
        h = x
        for heads_list in self.layers:
            head_outs = [head(h, bias_mat, training) for head in heads_list]
            h = torch.cat(head_outs, dim=-1)  # concat along feature dim
        # Output logits, average over heads
        logits = [head(h, bias_mat, training) for head in self.output_heads]
        logits = torch.stack(logits, dim=0).mean(dim=0)
        return logits

def masked_softmax_cross_entropy(logits, labels, mask):
    """
    labels: one-hot tensor (B, N, C)
    mask:   bool tensor (B, N) indicating which nodes to consider
    """
    log_logits = F.log_softmax(logits, dim=-1)
    loss = -(labels * log_logits).sum(dim=-1)  # (B, N)
    loss = loss * mask.float()
    return loss.sum() / mask.float().sum()

def masked_accuracy(logits, labels, mask):
    """
    Returns accuracy averaged over masked nodes.
    """
    preds = logits.argmax(dim=-1)
    true_labels = labels.argmax(dim=-1)
    correct = (preds == true_labels).float()
    correct = correct * mask.float()
    return correct.sum() / mask.float().sum()

def main():
    # Load data via helper
    data = nio.load()
    features = data['features'].astype(np.float32)          # (N, F)
    edges = data['edges'].astype(np.int64)                  # (E, 2)
    labels = data['labels'].astype(np.int64)                # (N,)
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    N = features.shape[0]
    F = features.shape[1]
    C = int(labels.max() + 1)  # number of classes (labels are 0..6, -1 for test)

    # Preprocess features: row-normalize
    features_norm = row_normalize_features(features)

    # Build adjacency bias matrix
    bias_np = build_adj_bias(N, edges)  # (N, N)

    # Convert to torch tensors
    features_tensor = torch.from_numpy(features_norm).unsqueeze(0)          # (1, N, F)
    bias_tensor = torch.from_numpy(bias_np).unsqueeze(0)                   # (1, N, N)

    # Prepare one-hot labels for train and val
    def to_one_hot(idx):
        oh = np.zeros((N, C), dtype=np.float32)
        oh[idx, labels[idx]] = 1.0
        return oh

    y_train_np = to_one_hot(train_idx)
    y_val_np   = to_one_hot(val_idx)
    # test labels are hidden (-1), we don't need them for training

    y_train_tensor = torch.from_numpy(y_train_np).unsqueeze(0)  # (1, N, C)
    y_val_tensor   = torch.from_numpy(y_val_np).unsqueeze(0)

    # Masks
    train_mask_np = np.zeros(N, dtype=bool)
    train_mask_np[train_idx] = True
    val_mask_np   = np.zeros(N, dtype=bool)
    val_mask_np[val_idx]   = True

    train_mask_tensor = torch.from_numpy(train_mask_np).unsqueeze(0)  # (1, N)
    val_mask_tensor   = torch.from_numpy(val_mask_np).unsqueeze(0)

    # Model instantiation
    model = GAT(
        feat_size=F,
        hidden_units=[8],
        n_heads=[8, 1],
        nb_classes=C,
        dropout=CONFIG["dropout"],
        attn_drop=CONFIG["dropout"],
        ffd_drop=CONFIG["dropout"],
        residual=False
    )

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CONFIG["lr"],
                                 weight_decay=CONFIG["weight_decay"])

    # Smoke mode override
    if os.environ.get("SMOKE") == "1":
        epochs = 1
    else:
        epochs = CONFIG["epochs"]

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(features_tensor, bias_tensor, training=True)
        loss = masked_softmax_cross_entropy(logits, y_train_tensor, train_mask_tensor)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features_tensor, bias_tensor, training=False)
            val_loss = masked_softmax_cross_entropy(val_logits, y_val_tensor, val_mask_tensor)
            val_acc = masked_accuracy(val_logits, y_val_tensor, val_mask_tensor)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:04d} | loss {loss.item():.4f} | val_loss {val_loss.item():.4f} | val_acc {val_acc.item():.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final inference (no dropout)
    model.eval()
    with torch.no_grad():
        final_logits = model(features_tensor, bias_tensor, training=False)
        final_logits = final_logits.squeeze(0)  # (N, C)
        preds = final_logits.argmax(dim=-1).cpu().numpy()  # (N,)

    # Write predictions via helper (will select test_idx internally)
    nio.write_predictions(preds)

    # Print final validation accuracy (from last validation eval)
    print(f"Final validation accuracy: {val_acc.item():.4f}")

if __name__ == "__main__":
    main()
