
# Plan step -> implemented in: set_seed
# Plan step -> implemented in: row_normalize_features
# Plan step -> implemented in: MultiHeadGATLayer
# Plan step -> implemented in: GAT
# Plan step -> implemented in: masked_softmax_cross_entropy
# Plan step -> implemented in: masked_accuracy
# Plan step -> implemented in: main
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {"lr": 0.005, "weight_decay": 0.0005, "epochs": 100000, "batch_size": 1, "dropout": 0.6, "patience": 100, "heads": 8}

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

class MultiHeadGATLayer(nn.Module):
    """
    Multi-head GAT layer using dense adjacency mask.
    If concat=True, outputs are concatenated; else averaged (used for output layer).
    """
    def __init__(self, in_features, out_features, n_heads, dropout, attn_drop, ffd_drop,
                 concat=True, adj=None, activation=F.elu):
        super().__init__()
        self.concat = concat
        self.adj = adj  # (N, N) float32 with 1 for edges inc self-loops
        self.activation = activation
        self.dropout = dropout
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.n_heads = n_heads

        # Parameters per head
        self.Ws = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_features, in_features)) for _ in range(n_heads)
        ])
        self.a1s = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_features, 1)) for _ in range(n_heads)
        ])
        self.a2s = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_features, 1)) for _ in range(n_heads)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_features)) for _ in range(n_heads)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.Ws:
            nn.init.xavier_uniform_(w)
        for a in self.a1s:
            nn.init.xavier_uniform_(a)
        for a in self.a2s:
            nn.init.xavier_uniform_(a)
        for b in self.biases:
            nn.init.zeros_(b)

    def forward(self, x, training):
        """
        x: (N, in_features)
        returns: (N, out_features * n_heads) if concat else (N, out_features)
        """
        N = x.size(0)
        head_outputs = []
        for i in range(self.n_heads):
            W = self.Ws[i]          # (out, in)
            a1 = self.a1s[i].squeeze(-1)  # (out,)
            a2 = self.a2s[i].squeeze(-1)  # (out,)
            bias = self.biases[i]         # (out,)

            # Feature dropout on input
            if self.ffd_drop != 0.0:
                x_drop = F.dropout(x, p=self.ffd_drop, training=training)
            else:
                x_drop = x

            # Linear projection
            h = x_drop @ W.t()          # (N, out)

            # Feature dropout after projection
            if self.ffd_drop != 0.0:
                h = F.dropout(h, p=self.ffd_drop, training=training)

            # Attention scores
            f1 = h @ a1.unsqueeze(-1)   # (N, 1)
            f2 = h @ a2.unsqueeze(-1)   # (N, 1)
            e = f1 + f2.t()             # (N, N)
            e = F.leaky_relu(e, negative_slope=0.2)

            # Mask non-edges
            mask = (self.adj == 0)
            e = e.masked_fill(mask, float('-inf'))

            # Softmax
            alpha = F.softmax(e, dim=1)         # (N, N)

            # Attention dropout
            if self.attn_drop != 0.0:
                alpha = F.dropout(alpha, p=self.attn_drop, training=training)

            # Weighted sum
            h_prime = alpha @ h                 # (N, out)

            # Add bias
            h_prime = h_prime + bias

            # Activation (skip for output layer if concat=False and activation is identity later)
            if self.activation is not None:
                h_prime = self.activation(h_prime)

            head_outputs.append(h_prime)

        if self.concat:
            return torch.cat(head_outputs, dim=-1)
        else:
            # average over heads
            return torch.stack(head_outputs, dim=0).mean(dim=0)

class GAT(nn.Module):
    def __init__(self, feat_size, hidden_units, n_heads, nb_classes, dropout, attn_drop, ffd_drop, adj):
        super().__init__()
        self.layers = nn.ModuleList()
        in_ch = feat_size
        for li in range(len(hidden_units)):
            out_ch = hidden_units[li]
            heads = n_heads[li]
            self.layers.append(
                MultiHeadGATLayer(
                    in_features=in_ch,
                    out_features=out_ch,
                    n_heads=heads,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    ffd_drop=ffd_drop,
                    concat=True,
                    adj=adj,
                    activation=F.elu
                )
            )
            in_ch = out_ch * heads
        # Output layer
        self.output_layer = MultiHeadGATLayer(
            in_features=in_ch,
            out_features=nb_classes,
            n_heads=n_heads[-1],
            dropout=dropout,
            attn_drop=attn_drop,
            ffd_drop=ffd_drop,
            concat=False,
            adj=adj,
            activation=None  # linear
        )

    def forward(self, x, training):
        h = x
        for layer in self.layers:
            h = layer(h, training)
        logits = self.output_layer(h, training)
        return logits

def masked_softmax_cross_entropy(logits, labels, mask):
    """
    logits: (N, C)
    labels: (N, C) one-hot
    mask:   (N,) bool
    """
    log_logits = F.log_softmax(logits, dim=-1)
    loss = -(labels * log_logits).sum(dim=-1)   # (N,)
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
    C = int(labels.max() + 1)  # labels are 0..6, -1 for test

    # Preprocess features: row-normalize
    features_norm = row_normalize_features(features)

    # Build adjacency matrix with self-loops
    adj_np = np.zeros((N, N), dtype=np.float32)
    adj_np[edges[:, 0], edges[:, 1]] = 1.0
    # add self-loops
    adj_np[np.arange(N), np.arange(N)] = 1.0
    adj_tensor = torch.from_numpy(adj_np)  # (N, N)

    # Convert to torch tensors
    features_tensor = torch.from_numpy(features_norm).unsqueeze(0)  # (1, N, F)
    if CONFIG["batch_size"] > 1:
        features_tensor = features_tensor.repeat(CONFIG["batch_size"], 1, 1)
    else:
        # keep batch dim 1
        pass

    # One-hot labels for train and val
    def to_one_hot(idx):
        oh = np.zeros((N, C), dtype=np.float32)
        oh[idx, labels[idx]] = 1.0
        return oh

    y_train_np = to_one_hot(train_idx)
    y_val_np   = to_one_hot(val_idx)

    y_train_tensor = torch.from_numpy(y_train_np).unsqueeze(0)  # (1, N, C)
    if CONFIG["batch_size"] > 1:
        y_train_tensor = y_train_tensor.repeat(CONFIG["batch_size"], 1, 1)
    y_val_tensor   = torch.from_numpy(y_val_np).unsqueeze(0)
    if CONFIG["batch_size"] > 1:
        y_val_tensor = y_val_tensor.repeat(CONFIG["batch_size"], 1, 1)

    # Masks
    train_mask_np = np.zeros(N, dtype=bool)
    train_mask_np[train_idx] = True
    val_mask_np   = np.zeros(N, dtype=bool)
    val_mask_np[val_idx]   = True

    train_mask_tensor = torch.from_numpy(train_mask_np).unsqueeze(0)  # (1, N)
    if CONFIG["batch_size"] > 1:
        train_mask_tensor = train_mask_tensor.repeat(CONFIG["batch_size"], 1)
    val_mask_tensor   = torch.from_numpy(val_mask_np).unsqueeze(0)
    if CONFIG["batch_size"] > 1:
        val_mask_tensor = val_mask_tensor.repeat(CONFIG["batch_size"], 1)

    # Model
    model = GAT(
        feat_size=F,
        hidden_units=[8],
        n_heads=[CONFIG["heads"], 1],
        nb_classes=C,
        dropout=CONFIG["dropout"],
        attn_drop=CONFIG["dropout"],
        ffd_drop=CONFIG["dropout"],
        adj=adj_tensor
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

        logits = model(features_tensor, training=True)
        loss = masked_softmax_cross_entropy(logits, y_train_tensor, train_mask_tensor)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features_tensor, training=False)
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
        final_logits = model(features_tensor, training=False)
        final_logits = final_logits.squeeze(0)  # (N, C)
        preds = final_logits.argmax(dim=-1).cpu().numpy()  # (N,)

    # Write predictions via helper (will select test_idx internally)
    nio.write_predictions(preds)

    # Print final validation accuracy (from last validation eval)
    print(f"Final validation accuracy: {val_acc.item():.4f}")

if __name__ == "__main__":
    main()