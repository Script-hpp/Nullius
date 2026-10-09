# Plan step -> implemented in: function or class name
# 1. Load data and preprocessing -> load_and_preprocess
# 2. Build edge index (including self-loops) -> load_and_preprocess
# 3. Define sparse attention head operation -> SparseMultiHeadGATLayer._head_forward
# 4. Define multi-head wrapper -> SparseMultiHeadGATLayer
# 5. Define full GAT model -> GAT
# 6. Training loop with early stopping -> train_model
# 7. Prediction and output -> predict_and_write

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {
    "lr": 0.005,
    "weight_decay": 0.0005,
    "epochs": 100000,
    "batch_size": 1,
    "dropout": 0.6,
    "patience": 100,
    "heads": 8,
    "nhead_output": 1,
    "negative_slope": 0.2,
    "eps": 1e-16,
}


def load_and_preprocess():
    d = nio.load()
    features = d["features"].astype(np.float32)          # [N, F_in]
    edges = d["edges"].astype(np.int64)                  # [E, 2] (directed, both directions)
    labels = d["labels"].astype(np.int64)                # [N]
    train_idx = d["train_idx"]
    val_idx = d["val_idx"]
    test_idx = d["test_idx"]

    # Row‑normalize features (as in process.preprocess_features)
    row_sum = features.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    features = features / row_sum

    N = features.shape[0]
    nclass = 7  # Cora has 7 classes

    # One‑hot label tensor (only where label >=0)
    label_onehot = np.zeros((N, nclass), dtype=np.float32)
    valid = labels >= 0
    label_onehot[np.arange(N)[valid], labels[valid]] = 1.0

    # Masks
    train_mask = np.zeros(N, dtype=np.float32)
    val_mask = np.zeros(N, dtype=np.float32)
    test_mask = np.zeros(N, dtype=np.float32)
    train_mask[train_idx] = 1.0
    val_mask[val_idx] = 1.0
    test_mask[test_idx] = 1.0

    # Build edge index (src, dst) including self-loops
    edges_tensor = torch.from_numpy(edges)  # [E, 2]
    self_loop = torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(1, 2)  # [N, 2]
    all_edges = torch.cat([edges_tensor, self_loop], dim=0)  # [E+N, 2]
    edge_index = all_edges.t().contiguous()  # [2, E_total]

    return (
        torch.from_numpy(features),
        torch.from_numpy(label_onehot),
        torch.from_numpy(train_mask),
        torch.from_numpy(val_mask),
        torch.from_numpy(test_mask),
        edge_index,
        nclass,
    )


class SparseMultiHeadGATLayer(nn.Module):
    """
    Multi-head GAT layer using sparse adjacency (edge_index) for efficient attention.
    If concat=True, outputs are concatenated; else averaged.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        dropout: float,
        concat: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout

        # Parameters for each head
        self.W_list = nn.ParameterList()
        self.a_l_list = nn.ParameterList()
        self.a_r_list = nn.ParameterList()
        self.bias_list = nn.ParameterList()

        for _ in range(n_heads):
            self.W_list.append(nn.Parameter(torch.Tensor(in_features, out_features)))
            self.a_l_list.append(nn.Parameter(torch.Tensor(out_features, 1)))
            self.a_r_list.append(nn.Parameter(torch.Tensor(out_features, 1)))
            self.bias_list.append(nn.Parameter(torch.Tensor(out_features)))

        self.reset_parameters()

    def reset_parameters(self):
        for W in self.W_list:
            nn.init.xavier_uniform_(W.data, gain=1.414)
        for a_l in self.a_l_list:
            nn.init.xavier_uniform_(a_l.data, gain=1.414)
        for a_r in self.a_r_list:
            nn.init.xavier_uniform_(a_r.data, gain=1.414)
        for b in self.bias_list:
            nn.init.zeros_(b.data)

    def _head_forward(self, x, edge_index, W, a_l, a_r, bias, training):
        """
        x: [N, in_features]
        edge_index: [2, E]
        Returns: [N, out_features]
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]  # [E]

        # Linear transformation
        Wh = x @ W  # [N, out_features]

        # Attention scores
        el = (Wh @ a_l).squeeze(-1)  # [N]
        er = (Wh @ a_r).squeeze(-1)  # [N]
        e = F.leaky_relu(el[src] + er[dst], negative_slope=CONFIG["negative_slope"])  # [E]

        # Softmax per source node
        e_max = torch.scatter_reduce(
            e, src, reduce="amax", dim_size=N
        )  # [N]
        e_exp = torch.exp(e - e_max[src])
        e_sum = torch.scatter_reduce(
            e_exp, src, reduce="sum", dim_size=N
        )  # [N]
        alpha = e_exp / (e_sum[src] + CONFIG["eps"])  # [E]

        # Dropout on attention coefficients
        if training:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # Message passing
        Wh_j = Wh[dst]  # [E, out_features]
        weighted = alpha.unsqueeze(-1) * Wh_j  # [E, out_features]
        agg = torch.scatter_reduce(
            weighted, src, reduce="sum", dim_size=N
        )  # [N, out_features]

        out = agg + bias  # [N, out_features]
        return out

    def forward(self, x, edge_index, training):
        head_outputs = []
        for i in range(self.n_heads):
            out = self._head_forward(
                x,
                edge_index,
                self.W_list[i],
                self.a_l_list[i],
                self.a_r_list[i],
                self.bias_list[i],
                training,
            )
            head_outputs.append(out)

        if self.concat:
            return torch.cat(head_outputs, dim=-1)  # [N, n_heads * out_features]
        else:
            return torch.stack(head_outputs, dim=0).mean(dim=0)  # [N, out_features]


class GAT(nn.Module):
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
        self.hidden_layer = SparseMultiHeadGATLayer(
            nfeat,
            nhidden,
            nheads_hidden,
            dropout,
            concat=True,
        )
        self.output_layer = SparseMultiHeadGATLayer(
            nhidden * nheads_hidden,
            nclass,
            nheads_output,
            dropout,
            concat=False,
        )

    def forward(self, x, edge_index, training):
        x = self.hidden_layer(x, edge_index, training)
        logits = self.output_layer(x, edge_index, training)
        return logits  # [N, nclass]


def masked_cross_entropy(logits, labels_onehot, mask):
    log_prob = F.log_softmax(logits, dim=-1)
    loss = -(labels_onehot * log_prob).sum(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss


def masked_accuracy(logits, labels_onehot, mask):
    pred = logits.argmax(dim=-1)
    true = labels_onehot.argmax(dim=-1)
    correct = (pred == true).float() * mask
    acc = correct.sum() / mask.sum()
    return acc


def train_model(
    model,
    features,
    labels_onehot,
    train_mask,
    val_mask,
    edge_index,
    epochs,
    lr,
    weight_decay,
    patience,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, edge_index, training=True)
        loss = masked_cross_entropy(logits, labels_onehot, train_mask)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features, edge_index, training=False)
            val_loss = masked_cross_entropy(val_logits, labels_onehot, val_mask)
            val_acc = masked_accuracy(val_logits, labels_onehot, val_mask)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Return last validation accuracy (could also return best, but we follow original)
    model.eval()
    with torch.no_grad():
        val_logits = model(features, edge_index, training=False)
        val_acc = masked_accuracy(val_logits, labels_onehot, val_mask)
    return val_acc.item()


def predict_and_write(model, features, edge_index):
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index, training=False)
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).squeeze(0).cpu().numpy()
    nio.write_predictions(pred)


def main():
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    features, labels_onehot, train_mask, val_mask, test_mask, edge_index, nclass = load_and_preprocess()

    nhidden = CONFIG["heads"]  # per-head hidden size = 8 (as per spec: each head outputs 8 features)
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
    val_acc = train_model(
        model=model,
        features=features,
        labels_onehot=labels_onehot,
        train_mask=train_mask,
        val_mask=val_mask,
        edge_index=edge_index,
        epochs=epochs,
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        patience=CONFIG["patience"],
    )

    print(f"Validation accuracy: {val_acc:.4f}")

    predict_and_write(model, features, edge_index)


if __name__ == "__main__":
    main()
