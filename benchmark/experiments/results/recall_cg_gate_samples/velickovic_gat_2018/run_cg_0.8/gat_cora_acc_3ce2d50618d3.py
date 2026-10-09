import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

# Plan step -> implemented in: load_and_preprocess_data
# Plan step -> implemented in: GATLayer
# Plan step -> implemented in: MultiHeadGATLayer
# Plan step -> implemented in: GATNet
# Plan step -> implemented in: masked_softmax_cross_entropy
# Plan step -> implemented in: masked_accuracy
# Plan step -> implemented in: train_loop
# Plan step -> implemented in: evaluate_and_output
# Plan step -> implemented in: main

CONFIG = {
    "lr": 0.005,
    "weight_decay": 0.0005,
    "epochs": 100000,
    "batch_size": 2,
    "dropout": 0.6,
    "patience": 100,
    "heads": 8,
    "hidden": 8  # features per attention head (fixed from paper)
}


def load_and_preprocess_data():
    """Load data via nullius_io, row‑normalize features, build adjacency bias."""
    data = nio.load()
    feats = data['features'].astype(np.float32)          # [N, F]
    edges = data['edges'].astype(np.int64)               # [E, 2]
    labels = data['labels'].astype(np.int64)             # [N]
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    N = feats.shape[0]

    # Row‑normalize features (as in preprocess_features)
    row_sum = feats.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    feats_norm = feats / row_sum
    features = torch.from_numpy(feats_norm)              # [N, F]

    # Adjacency with self‑loops
    adj = torch.zeros((N, N), dtype=torch.float32)
    u, v = edges[:, 0], edges[:, 1]
    adj[u, v] = 1.0
    idx = torch.arange(N)
    adj[idx, idx] = 1.0

    # Bias matrix: 0 for existing edge, -1e9 otherwise
    bias = torch.where(adj > 0,
                       torch.tensor(0.0, dtype=torch.float32),
                       torch.tensor(-1e9, dtype=torch.float32))  # [N, N]

    labels_tensor = torch.from_numpy(labels)             # [N]
    train_mask = torch.zeros(N, dtype=torch.float32)
    train_mask[train_idx] = 1.0
    val_mask = torch.zeros(N, dtype=torch.float32)
    val_mask[val_idx] = 1.0
    test_mask = torch.zeros(N, dtype=torch.float32)
    test_mask[test_idx] = 1.0

    return features, bias, labels_tensor, train_mask, val_mask, test_mask


class GATLayer(nn.Module):
    """
    Single attention head as in the TensorFlow reference.
    Supports batched input (batch dimension optional).
    Applies dropout to features and attention coefficients,
    uses leaky_relu (negative slope 0.2) for attention,
    and optionally an activation after the aggregation.
    """
    def __init__(self, in_features, out_features, dropout, alpha=0.2,
                 activation=F.elu, residual=False):
        super().__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.activation = activation
        self.residual = residual

        # Linear projection without bias (use_bias=False in TF)
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention weights a_L and a_R
        self.a_L = nn.Parameter(torch.Tensor(out_features))
        self.a_R = nn.Parameter(torch.Tensor(out_features))
        # Bias added after aggregation (as in tf.contrib.layers.bias_add)
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.xavier_uniform_(self.a_L)
        nn.init.xavier_uniform_(self.a_R)
        nn.init.zeros_(self.bias)

    def forward(self, x, bias_mat):
        # x: [N, in_f] or [B, N, in_f]
        # bias_mat: [N, N] or [B, N, N]
        if x.dim() == 2:
            # add batch dim
            x = x.unsqueeze(0)          # [1, N, in_f]
            squeeze_back = True
            if bias_mat.dim() == 2:
                bias_mat = bias_mat.unsqueeze(0)  # [1, N, N]
        else:
            squeeze_back = False
            if bias_mat.dim() == 2:
                bias_mat = bias_mat.unsqueeze(0).expand(x.size(0), -1, -1)

        if self.training:
            x = F.dropout(x, p=self.dropout, training=True)

        proj = self.W(x)  # [B, N, out_f]

        f1 = torch.matmul(proj, self.a_L.unsqueeze(-1))  # [B, N, 1]
        f2 = torch.matmul(proj, self.a_R.unsqueeze(-1))  # [B, N, 1]
        e = f1 + f2.transpose(-2, -1)                    # [B, N, N]
        e = F.leaky_relu(e, negative_slope=self.alpha)
        e = e + bias_mat

        alpha = F.softmax(e, dim=1)                      # [B, N, N]
        if self.training:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
            proj = F.dropout(proj, p=self.dropout, training=True)

        out = torch.matmul(alpha, proj)                  # [B, N, out_f]
        out = out + self.bias                            # broadcast bias

        if self.residual:
            if x.shape[-1] != out.shape[-1]:
                res = self.W(x)
                out = out + res
            else:
                out = out + x

        if self.activation is not None:
            out = self.activation(out)

        if squeeze_back:
            out = out.squeeze(0)
        return out


class MultiHeadGATLayer(nn.Module):
    """
    Multi‑head wrapper: concatenates heads (concat=True) or averages them (concat=False).
    Works with batched input.
    """
    def __init__(self, in_features, out_features, n_heads, dropout,
                 alpha=0.2, concat=True, activation=F.elu, residual=False):
        super().__init__()
        self.concat = concat
        self.heads = nn.ModuleList([
            GATLayer(in_features, out_features, dropout, alpha,
                     activation=activation, residual=residual)
            for _ in range(n_heads)
        ])
        self.out_dim = n_heads * out_features if concat else out_features

    def forward(self, x, bias):
        head_outs = [head(x, bias) for head in self.heads]
        if self.concat:
            return torch.cat(head_outs, dim=1)
        else:
            return torch.stack(head_outs, dim=0).mean(dim=0)


class GATNet(nn.Module):
    """
    Two‑layer GAT as used in the Cora experiment:
    - Layer 1: 8 heads, each 8‑dim output → concatenated 64, ELU
    - Layer 2: 1 head, 7‑dim logits (no activation)
    """
    def __init__(self, nfeat, nhidden, nheads, nclass, dropout):
        super().__init__()
        self.att1 = MultiHeadGATLayer(
            nfeat, nhidden, nheads=nheads,
            dropout=dropout, concat=True,
            activation=F.elu, residual=False
        )
        self.att2 = MultiHeadGATLayer(
            self.att1.out_dim, nclass, nheads=1,
            dropout=dropout, concat=False,
            activation=None, residual=False
        )

    def forward(self, x, bias):
        x = self.att1(x, bias)
        x = self.att2(x, bias)
        return x


def masked_softmax_cross_entropy(logits, labels, mask):
    """Masked cross‑entropy loss, mean over masked nodes.
    logits: [B, N, C], labels: [B, N], mask: [B, N]"""
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                           labels.view(-1),
                           reduction='none')
    loss = loss * mask.view(-1)
    return loss.mean() if mask.sum() > 0 else torch.tensor(0.0, device=logits.device)


def masked_accuracy(logits, labels, mask):
    """Masked accuracy.
    logits: [B, N, C], labels: [B, N], mask: [B, N]"""
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float()
    acc = (correct * mask).sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0, device=logits.device)
    return acc


def train_loop(features, bias, labels, train_mask, val_mask, config):
    """Train GAT with early stopping on validation loss."""
    # Create batched tensors according to batch_size
    B = config["batch_size"]
    features_b = features.unsqueeze(0).repeat(B, 1, 1)          # [B, N, F]
    bias_b = bias.unsqueeze(0).repeat(B, 1, 1)                # [B, N, N]
    labels_b = labels.unsqueeze(0).repeat(B, 1)               # [B, N]
    train_mask_b = train_mask.unsqueeze(0).repeat(B, 1)       # [B, N]
    val_mask_b = val_mask.unsqueeze(0).repeat(B, 1)           # [B, N]

    model = GATNet(
        nfeat=features.shape[1],
        nhidden=config["hidden"],
        nheads=config["heads"],
        nclass=int(labels.max().item() + 1),
        dropout=config["dropout"]
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    max_epochs = 1 if os.getenv("SMOKE") == "1" else config["epochs"]
    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features_b, bias_b)
        loss = masked_softmax_cross_entropy(logits, labels_b, train_mask_b)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features_b, bias_b)
            val_loss = masked_softmax_cross_entropy(val_logits, labels_b, val_mask_b)
            val_acc = masked_accuracy(val_logits, labels_b, val_mask_b)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= config["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            print(f"Epoch {epoch:04d} | loss {loss.item():.4f} | val_loss {val_loss.item():.4f} | val_acc {val_acc.item():.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        final_logits = model(features_b, bias_b)
        final_val_acc = masked_accuracy(final_logits, labels_b, val_mask_b)
    print(f"Final validation accuracy: {final_val_acc.item():.4f}")
    # Return logits for the first graph in the batch (they are identical)
    return final_logits[0]


def evaluate_and_output(logits):
    """Convert logits to class predictions and write via nullius_io."""
    pred = logits.argmax(dim=1).cpu().numpy()
    nio.write_predictions(pred)


def main():
    # Seed
    seed = int(os.getenv("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Data
    features, bias, labels, train_mask, val_mask, test_mask = load_and_preprocess_data()

    # Train
    logits = train_loop(features, bias, labels, train_mask, val_mask, CONFIG)

    # Output predictions for all nodes
    evaluate_and_output(logits)


if __name__ == "__main__":
    main()
