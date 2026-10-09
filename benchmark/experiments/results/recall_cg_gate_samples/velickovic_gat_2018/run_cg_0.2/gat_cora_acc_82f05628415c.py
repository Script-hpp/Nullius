

# Plan step -> implemented in: load_and_preprocess
# Plan step -> implemented in: GATHead
# Plan step -> implemented in: GAT
# Plan step -> implemented in: masked_loss_and_acc
# Plan step -> implemented in: train_model
# Plan step -> implemented in: set_seed

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
    "batch_size": 2,
    "dropout": 0.6,
    "patience": 100,
    "heads": 8,
    "hidden_units": 8,  # output features per head in the first layer
    "num_classes": 7,   # Cora has 7 classes
}

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_and_preprocess():
    d = nio.load()
    features = d['features'].astype(np.float32)          # [N, F]
    edges = d['edges']                                   # [E, 2]
    labels = d['labels']                                 # [N]
    train_idx = d['train_idx']
    val_idx = d['val_idx']
    test_idx = d['test_idx']

    N = features.shape[0]

    # row‑normalize features
    row_sum = features.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    features = features / row_sum

    # adjacency with self‑loops (using edge list)
    u, v = edges[:, 0], edges[:, 1]
    edge_index = np.vstack([u, v])                       # [2, E]
    # add self-loops
    loop = np.arange(N)
    edge_index = np.concatenate([edge_index, np.vstack([loop, loop])], axis=1)

    # torch tensors
    features_t = torch.from_numpy(features)              # [N, F]
    edge_index_t = torch.from_numpy(edge_index).long()   # [2, E]

    # masks and labels (long)
    valid_labels = labels[labels >= 0]
    num_classes = int(valid_labels.max() + 1)            # 7 for Cora
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    labels_long = torch.from_numpy(labels).long()        # [N], -1 for test nodes

    return features_t, edge_index_t, labels_long, train_mask, val_mask, test_idx

class GATHead(nn.Module):
    def __init__(self, in_features, out_features, dropout, negative_slope=0.2):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=False)
        # attention vector a, split into left and right halves later
        self.a = nn.Parameter(torch.empty(size=(2 * out_features,)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.xavier_uniform_(self.a, gain=gain)

    def forward(self, x, edge_index, training=False):
        # x: [N, in_features]
        N = x.size(0)
        h = self.W(x)                         # [N, out_features]
        if self.dropout > 0 and training:
            h = F.dropout(h, p=self.dropout, training=training)

        src, dst = edge_index                 # each [E]
        h_src = h[src]                        # [E, out]
        h_dst = h[dst]                        # [E, out]

        # compute e = LeakyReLU(a^T [h_src || h_dst])
        a_l = self.a[:self.out_features]
        a_r = self.a[self.out_features:]
        e = (h_src * a_l).sum(dim=-1) + (h_dst * a_r).sum(dim=-1)
        e = self.leaky_relu(e)

        # softmax per source node
        # e_max per source
        e_max = torch.scatter_reduce(e, 0, src, reduce='amax', dim_size=N)
        e_max_exp = e_max[src]
        e_exp = torch.exp(e - e_max_exp)
        e_sum = torch.scatter_reduce(e_exp, 0, src, reduce='sum', dim_size=N)
        e_sum_exp = e_sum[src]
        alpha = e_exp / (e_sum_exp + 1e-16)   # [E]

        if self.dropout > 0 and training:
            alpha = F.dropout(alpha, p=self.dropout, training=training)

        # message: weight destination node features
        msg = h_dst * alpha.unsqueeze(-1)     # [E, out]
        out = torch.zeros_like(h)             # [N, out]
        out.index_add_(0, src, msg)           # aggregate by source index
        out = F.elu(out)                      # activation for hidden layers
        return out

class GAT(nn.Module):
    def __init__(self, in_features, hidden_units, n_heads, num_classes, dropout):
        super().__init__()
        self.n_layers = len(n_heads)
        assert len(hidden_units) == self.n_layers - 1
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for l in range(self.n_layers):
            in_ch = in_features if l == 0 else hidden_units[l-1] * n_heads[l-1]
            out_ch = hidden_units[l] if l < self.n_layers-1 else num_classes
            heads = nn.ModuleList()
            for _ in range(n_heads[l]):
                heads.append(GATHead(in_ch, out_ch, dropout))
            self.layers.append(heads)

    def forward(self, x, edge_index, training=False):
        h = x
        for l, heads in enumerate(self.layers):
            head_outs = [head(h, edge_index, training=training) for head in heads]
            if l == len(self.layers) - 1:
                # output layer: average over heads
                h = torch.mean(torch.stack(head_outs, dim=0), dim=0)
            else:
                h = torch.cat(head_outs, dim=-1)
        return h

def masked_loss_and_acc(logits, labels, mask):
    """
    logits: [N, C]
    labels: [N] long (ignore index -1 for unlabeled)
    mask:   [N] bool
    """
    # filter only masked nodes for numeric stability
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device), 0.0
    logits_masked = logits[mask]
    labels_masked = labels[mask]
    loss = F.cross_entropy(logits_masked, labels_masked)
    preds = logits_masked.argmax(dim=-1)
    acc = (preds == labels_masked).float().mean().item()
    return loss, acc

def train_model():
    features, edge_index, labels, train_mask, val_mask, test_idx = load_and_preprocess()
    N = features.size(0)
    C = CONFIG["num_classes"]

    model = GAT(
        in_features=features.size(1),
        hidden_units=[CONFIG["hidden_units"]],
        n_heads=[CONFIG["heads"], 1],
        num_classes=C,
        dropout=CONFIG["dropout"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    best_val_loss = float('inf')
    patience_cnt = 0
    best_state = None

    epochs = CONFIG["epochs"]
    if os.environ.get("SMOKE") == "1":
        epochs = 2

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, edge_index, training=True)   # [N, C]
        loss, acc = masked_loss_and_acc(logits, labels, train_mask)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features, edge_index, training=False)
            val_loss, val_acc = masked_loss_and_acc(val_logits, labels, val_mask)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            print(f"Epoch {epoch:04d} | loss {loss:.4f} acc {acc:.4f} | val loss {val_loss:.4f} val acc {val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # final validation accuracy (optional)
    model.eval()
    with torch.no_grad():
        final_logits = model(features, edge_index, training=False)
        _, final_val_acc = masked_loss_and_acc(final_logits, labels, val_mask)
    print(f"Final validation accuracy: {final_val_acc:.4f}")

    # inference for all nodes
    model.eval()
    with torch.no_grad():
        all_logits = model(features, edge_index, training=False)   # [N, C]
        pred = torch.argmax(all_logits, dim=-1).cpu().numpy()
    nio.write_predictions(pred)

if __name__ == "__main__":
    seed = int(os.environ.get("SEED", 42))
    set_seed(seed)
    train_model()