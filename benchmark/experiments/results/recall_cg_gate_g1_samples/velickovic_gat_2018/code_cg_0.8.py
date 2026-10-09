# Plan step -> implemented in: function or class name
# 1. Load data and preprocessing -> load_and_preprocess
# 2. Build adjacency mask for attention -> build_adjacency_mask
# 3. Define GraphAttentionLayer (core GAT mechanism) -> GraphAttentionLayer
# 4. Define multi-head wrapper -> MultiHeadGATLayer
# 5. Define full GAT model -> GAT
# 6. Training loop with early stopping -> train_model
# 7. Prediction and output -> predict_and_write

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

# Hyperparameters must be read only from this dict
CONFIG = {
    "lr": 0.005,
    "weight_decay": 0.0005,
    "epochs": 100000,
    "batch_size": 1,
    "dropout": 0.6,
    "patience": 100,
    "heads": 8,
    "hidden_units": [8],   # per-head output size in hidden layer
    "nhead_output": 1,     # heads in output layer
}


def load_and_preprocess():
    """Load data via nullius_io, row‑normalize features, prepare labels and masks."""
    d = nio.load()
    features = d["features"].astype(np.float32)          # [N, F_in]
    edges = d["edges"].astype(np.int64)                  # [E, 2]
    labels = d["labels"].astype(np.int64)                # [N]
    train_idx = d["train_idx"]
    val_idx = d["val_idx"]
    test_idx = d["test_idx"]

    # Row‑normalize features (as in process.preprocess_features)
    row_sum = features.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    features = features / row_sum

    N = features.shape[0]
    # Build adjacency (including self-loops)
    adj = np.zeros((N, N), dtype=np.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    np.arange(N)  # just to silence unused warning
    adj[np.arange(N), np.arange(N)] = 1.0

    # Masks for train/val/test
    train_mask = np.zeros(N, dtype=np.float32)
    val_mask = np.zeros(N, dtype=np.float32)
    test_mask = np.zeros(N, dtype=np.float32)
    train_mask[train_idx] = 1.0
    val_mask[val_idx] = 1.0
    test_mask[test_idx] = 1.0

    # One‑hot label tensor (only where label >=0)
    nclass = int(labels.max() + 1) if labels.any() >= 0 else 7
    label_onehot = np.zeros((N, nclass), dtype=np.float32)
    valid = labels >= 0
    label_onehot[np.arange(N)[valid], labels[valid]] = 1.0

    return (
        torch.from_numpy(features),
        torch.from_numpy(label_onehot),
        torch.from_numpy(train_mask),
        torch.from_numpy(val_mask),
        torch.from_numpy(test_mask),
        torch.from_numpy(adj),
        nclass,
    )


def build_adjacency_mask(adj: torch.Tensor) -> torch.Tensor:
    """Return bias mask: 0 for allowed edges (including self), -1e9 otherwise."""
    NEG_INF = -1e9
    mask = torch.where(adj > 0, torch.tensor(0.0, device=adj.device),
                       torch.tensor(NEG_INF, device=adj.device))
    return mask  # shape [N, N]


class GraphAttentionLayer(nn.Module):
    """
    Single GAT attention head as described in the TensorFlow reference.
    Applies dropout to input and attention coefficients, uses leaky ReLU,
    masks non‑neighbors with a large negative bias, and optionally adds
    a residual connection.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
        alpha: float = 0.2,
        concat: bool = True,
        residual: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.residual = residual

        # Weight matrix for feature transformation (no bias)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        # Attention weight vectors
        self.a_l = nn.Parameter(torch.empty(size=(out_features, 1)))
        self.a_r = nn.Parameter(torch.empty(size=(out_features, 1)))

        if residual and in_features != out_features:
            self.residual_W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        else:
            self.register_parameter("residual_W", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        if self.residual_W is not None:
            nn.init.xavier_uniform_(self.residual_W.data, gain=1.414)

    def forward(
        self,
        x: torch.Tensor,
        adj_mask: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        """
        x:          [N, in_features]
        adj_mask:   [N, N] bias mask (0 for allowed, -1e9 for disallowed)
        training:   bool flag for dropout behavior
        Returns:    [N, out_features] if concat=False else [N, n_heads*out_features]
        """
        # Dropout on input features
        if training:
            x_drop = F.dropout(x, p=self.dropout, training=True)
        else:
            x_drop = x

        # Linear transformation
        z = torch.mm(x_drop, self.W)  # [N, out_features]

        # Second dropout (after linear transform) as in TF code
        if training:
            z_drop = F.dropout(z, p=self.dropout, training=True)
        else:
            z_drop = z

        # Compute attention scores
        f1 = torch.mm(z_drop, self.a_l)  # [N, 1]
        f2 = torch.mm(z_drop, self.a_r)  # [N, 1]
        e = f1 + f2.t()                  # [N, N]
        e = F.leaky_relu(e, negative_slope=self.alpha)

        # Mask before softmax
        e_masked = e + adj_mask
        alpha = F.softmax(e_masked, dim=1)  # [N, N]

        # Dropout on attention coefficients
        if training:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # Linear combination of neighbors' features
        z_prime = torch.mm(alpha, z_drop)  # [N, out_features]

        # Residual connection
        if self.residual:
            if self.residual_W is not None:
                res = torch.mm(x, self.residual_W)
            else:
                res = x
            z_prime = z_prime + res

        # Activation (ELU for hidden, linear for output handled externally)
        if self.concat:
            return F.elu(z_prime)
        else:
            return z_prime  # linear


class MultiHeadGATLayer(nn.Module):
    """Wrapper for multiple attention heads with concatenation or averaging."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        dropout: float,
        concat: bool = True,
        residual: bool = False,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                GraphAttentionLayer(
                    in_features,
                    out_features,
                    dropout,
                    concat=False,
                    residual=residual,
                )
                for _ in range(n_heads)
            ]
        )
        self.concat = concat

    def forward(self, x, adj_mask, training):
        head_outs = [head(x, adj_mask, training) for head in self.heads]
        if self.concat:
            return torch.cat(head_outs, dim=-1)  # [N, n_heads * out_features]
        else:
            return torch.mean(torch.stack(head_outs, dim=0), dim=0)  # [N, out_features]


class GAT(nn.Module):
    """Full GAT model matching the TensorFlow reference architecture."""

    def __init__(
        self,
        nfeat: int,
        nhidden: int,
        nclass: int,
        nheads_hidden: int,
        nheads_output: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_layer = MultiHeadGATLayer(
            nfeat,
            nhidden,
            nheads_hidden,
            dropout,
            concat=True,
            residual=False,
        )
        self.output_layer = MultiHeadGATLayer(
            nhidden * nheads_hidden,
            nclass,
            nheads_output,
            dropout,
            concat=False,
            residual=False,
        )

    def forward(self, x, adj_mask, training):
        x = self.hidden_layer(x, adj_mask, training)
        logits = self.output_layer(x, adj_mask, training)
        return logits  # [N, nclass]


def train_model(
    model,
    features,
    labels_onehot,
    train_mask,
    val_mask,
    adj_mask,
    epochs,
    lr,
    weight_decay,
    patience,
    smoke_mode=False,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(features, adj_mask, training=True)
        loss_per_node = -(labels_onehot * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        loss = (loss_per_node * train_mask).sum() / train_mask.sum()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj_mask, training=False)
            val_loss_per_node = -(labels_onehot * F.log_softmax(val_logits, dim=-1)).sum(dim=-1)
            val_loss = (val_loss_per_node * val_mask).sum() / val_mask.sum()
            val_pred = torch.argmax(val_logits, dim=-1)
            val_label = torch.argmax(labels_onehot, dim=-1)
            val_correct = (val_pred == val_label).float() * val_mask
            val_acc = val_correct.sum() / val_mask.sum()
        model.train()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if smoke_mode:
                # In smoke mode we still break early to respect patience logic
                pass
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Return final validation accuracy (from last evaluation)
    return val_acc.item()


def predict_and_write(model, features, adj_mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_mask, training=False)
        pred = torch.argmax(logits, dim=-1).cpu().numpy()
    nio.write_predictions(pred)


def main():
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    smoke = os.environ.get("SMOKE") == "1"

    features, labels_onehot, train_mask, val_mask, test_mask, adj, nclass = load_and_preprocess()
    adj_mask = build_adjacency_mask(adj)

    # Model dimensions from CONFIG
    nhidden = CONFIG["hidden_units"][0]          # per-head hidden size
    nheads_hidden = CONFIG["heads"]
    nheads_output = CONFIG["nhead_output"]
    dropout = CONFIG["dropout"]

    model = GAT(
        nfeat=features.shape[1],
        nhidden=nhidden,
        nclass=nclass,
        nheads_hidden=nheads_hidden,
        nheads_output=nheads_output,
        dropout=dropout,
    )

    epochs = CONFIG["epochs"]
    if smoke:
        epochs = 1  # force exactly one epoch in smoke mode

    val_acc = train_model(
        model=model,
        features=features,
        labels_onehot=labels_onehot,
        train_mask=train_mask,
        val_mask=val_mask,
        adj_mask=adj_mask,
        epochs=epochs,
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        patience=CONFIG["patience"],
        smoke_mode=smoke,
    )

    print(f"Validation accuracy: {val_acc:.4f}")

    predict_and_write(model, features, adj_mask)


if __name__ == "__main__":
    main()