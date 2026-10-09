# Plan step -> implemented in: function or class name
# 1. Load data -> load_data()
# 2. Preprocess features (row-normalize) -> preprocess_features()
# 3. Build adjacency with self-loops and bias matrix -> build_bias()
# 4. Create PyTorch GAT model matching TensorFlow specification -> GAT class
# 5. Training loop with early stopping on validation loss -> train()
# 6. Use hyperparameters from CONFIG dict only -> referenced throughout
# 7. Handle SEED for reproducibility -> set_seeds()
# 8. Handle SMOKE mode (1 epoch) -> adjust epochs
# 9. Output predictions via nullius_io -> nio.write_predictions()
# 10. Print final validation accuracy -> print()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

# Hyperparameters (read only from this dict)
CONFIG = {
    "lr": 0.005,
    "weight_decay": 0.0005,
    "epochs": 100000,
    "batch_size": 1,
    "dropout": 0.6,
    "patience": 100,
    "heads": 8,
    # Architecture specifics (not in CONFIG but fixed per spec)
    "hid_units": [8],
    "n_heads": [8, 1],
    "activation": F.elu,
    "residual": False,
}


def set_seeds():
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_bias(num_nodes, edges):
    """Adjacency with self-loops -> bias matrix (0 for edge, -1e9 otherwise)."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    u, v = edges[:, 0], edges[:, 1]
    adj[u, v] = 1.0
    # add self-loops
    np.fill_diagonal(adj, 1.0)
    bias = np.where(adj > 0, 0.0, -1e9).astype(np.float32)
    return bias  # (N, N)


class AttnHead(nn.Module):
    """Single GAT attention head as in the TensorFlow implementation."""

    def __init__(self, in_sz, out_sz, dropout, activation, residual=False):
        super().__init__()
        self.dropout = dropout
        self.activation = activation
        self.residual = residual
        self.fc = nn.Linear(in_sz, out_sz, bias=False)          # W for seq_fts
        self.fc_f1 = nn.Linear(out_sz, 1, bias=False)           # for f_1
        self.fc_f2 = nn.Linear(out_sz, 1, bias=False)           # for f_2
        self.bias = nn.Parameter(torch.zeros(out_sz))           # bias_add
        if residual:
            self.proj = (
                nn.Linear(in_sz, out_sz, bias=False)
                if in_sz != out_sz
                else None
            )
        else:
            self.proj = None

    def forward(self, seq, bias_mat, attn_drop, ffd_drop):
        # seq: (B, N, in_sz)
        if self.dropout != 0.0:
            seq = F.dropout(seq, p=self.dropout, training=True)
        seq_fts = self.fc(seq)                     # (B, N, out_sz)

        f_1 = self.fc_f1(seq_fts)                  # (B, N, 1)
        f_2 = self.fc_f2(seq_fts)                  # (B, N, 1)
        logits = f_1 + f_2.transpose(1, 2)         # (B, N, N)
        logits = F.leaky_relu(logits)
        coefs = logits + bias_mat                  # (B, N, N)
        coefs = F.softmax(coefs, dim=-1)           # softmax over neighbours
        if self.dropout != 0.0:
            coefs = F.dropout(coefs, p=self.dropout, training=True)
        if self.dropout != 0.0:
            seq_fts = F.dropout(seq_fts, p=self.dropout, training=True)
        vals = torch.matmul(coefs, seq_fts)        # (B, N, out_sz)
        ret = vals + self.bias                     # bias_add
        if self.residual:
            if self.proj is not None:
                seq = self.proj(seq)
            ret = ret + seq
        ret = self.activation(ret)
        return ret


class GAT(nn.Module):
    """GAT model matching the specifications."""

    def __init__(
        self,
        ft_size,
        nb_classes,
        hid_units,
        n_heads,
        dropout,
        activation=F.elu,
        residual=False,
    ):
        super().__init__()
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.dropout = dropout
        self.activation = activation
        self.residual = residual

        # Build hidden layers
        self.layers = nn.ModuleList()
        in_sz = ft_size
        for l in range(len(hid_units)):
            heads = nn.ModuleList()
            out_sz = hid_units[l]
            for _ in range(n_heads[l]):
                heads.append(
                    AttnHead(
                        in_sz,
                        out_sz,
                        dropout,
                        activation,
                        residual=False,
                    )
                )
            self.layers.append(heads)
            in_sz = out_sz * n_heads[l]

        # Output layer
        self.out_heads = nn.ModuleList()
        for _ in range(n_heads[-1]):
            self.out_heads.append(
                AttnHead(
                    in_sz,
                    nb_classes,
                    dropout,
                    activation=lambda x: x,
                    residual=False,
                )
            )

    def forward(self, features, bias, attn_drop, ffd_drop):
        """
        features: (B, N, F)
        bias:     (B, N, N)
        Returns logits: (B, N, nb_classes)
        """
        h = features
        for heads in self.layers:
            head_outs = [head(h, bias, attn_drop, ffd_drop) for head in heads]
            h = torch.cat(head_outs, dim=-1)  # concat multi-head outputs
        out_heads = [head(h, bias, attn_drop, ffd_drop) for head in self.out_heads]
        logits = torch.stack(out_heads, dim=0).mean(dim=0)  # average over heads
        return logits


def masked_loss(logits, label_indices, mask):
    """Masked softmax cross-entropy loss."""
    log_softmax = F.log_softmax(logits.squeeze(0), dim=-1)   # (N, C)
    loss_per_node = F.nll_loss(
        log_softmax, label_indices, ignore_index=-1, reduction="none"
    )  # (N,)
    loss = (loss_per_node * mask.squeeze(0)).sum() / mask.squeeze(0).sum()
    return loss


def masked_accuracy(logits, label_indices, mask):
    """Masked accuracy."""
    probs = logits.squeeze(0)
    preds = probs.argmax(dim=-1)
    correct = (preds == label_indices).float()
    acc = (correct * mask.squeeze(0)).sum() / mask.squeeze(0).sum()
    return acc


def train():
    set_seeds()
    features_raw, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features_raw.shape[0]
    feat_dim = features_raw.shape[1]
    num_classes = int(labels.max() + 1)  # labels are 0..6 for labeled nodes

    # Preprocess
    features = preprocess_features(features_raw)               # (N, F)
    bias_np = build_bias(num_nodes, edges)                     # (N, N)

    # Torch tensors (add batch dimension)
    features_t = torch.from_numpy(features).float().unsqueeze(0)      # (1, N, F)
    bias_t = torch.from_numpy(bias_np).float().unsqueeze(0)           # (1, N, N)
    label_indices_t = torch.from_numpy(labels).long()                 # (N,)

    # Masks
    train_mask_np = np.zeros(num_nodes, dtype=np.float32)
    train_mask_np[train_idx] = 1.0
    val_mask_np = np.zeros(num_nodes, dtype=np.float32)
    val_mask_np[val_idx] = 1.0
    test_mask_np = np.zeros(num_nodes, dtype=np.float32)
    test_mask_np[test_idx] = 1.0

    train_mask_t = torch.from_numpy(train_mask_np).float().unsqueeze(0)  # (1, N)
    val_mask_t = torch.from_numpy(val_mask_np).float().unsqueeze(0)
    test_mask_t = torch.from_numpy(test_mask_np).float().unsqueeze(0)

    # Model
    model = GAT(
        ft_size=feat_dim,
        nb_classes=num_classes,
        hid_units=CONFIG["hid_units"],
        n_heads=CONFIG["n_heads"],
        dropout=CONFIG["dropout"],
        activation=CONFIG["activation"],
        residual=CONFIG["residual"],
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
        logits = model(features_t, bias_t, CONFIG["dropout"], CONFIG["dropout"])
        loss = masked_loss(logits, label_indices_t, train_mask_t)
        acc = masked_accuracy(logits, label_indices_t, train_mask_t)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features_t, bias_t, CONFIG["dropout"], CONFIG["dropout"])
            val_loss = masked_loss(val_logits, label_indices_t, val_mask_t)
            val_acc = masked_accuracy(val_logits, label_indices_t, val_mask_t)

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

    # Final predictions (using dropout as in training/val)
    model.eval()
    with torch.no_grad():
        final_logits = model(features_t, bias_t, CONFIG["dropout"], CONFIG["dropout"])
    preds = final_logits.squeeze(0).argmax(dim=-1).cpu().numpy()  # (N,)

    nio.write_predictions(preds)
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
