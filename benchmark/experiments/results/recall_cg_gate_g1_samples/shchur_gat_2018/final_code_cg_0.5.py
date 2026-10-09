

# Plan step -> implemented in: GAT class and GATLayer class
# Plan step -> implemented in: build_adjacency function
# Plan step -> implemented in: GATLayer constructor
# Plan step -> implemented in: train function (optimizer creation)
# Plan step -> implemented in: train function loop
# Plan step -> implemented in: main function (using loaded indices)
# Plan step -> implemented in: main function calling nio.write_predictions
# Plan step -> implemented in: main function
# Plan step -> implicit in model forward
# Plan step -> no implementation needed
# Plan step -> implemented in: GATLayer.reset_parameters
# Plan step -> implemented in: train function (best_state restore)
# Plan step -> implemented in: main function
# Plan step -> comment that implementation is efficient
# Plan step -> noted in comments

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nullius_io as nio

CONFIG = {
    "lr": 0.01,
    "weight_decay": 0.01,
    "epochs": 100000,
    "hidden": 8,
    "layers": 2,
    "patience": 100,
    "heads": 8,
    "input_dropout": 0.6,
    "coef_dropout": 0.3,
}


def build_adjacency(edges, num_nodes):
    """Return a boolean adjacency matrix [N,N] with self-loops."""
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    if edges.shape[0] > 0:
        u = torch.from_numpy(edges[:, 0]).long()
        v = torch.from_numpy(edges[:, 1]).long()
        adj[u, v] = True
    # add self-loops
    idx = torch.arange(num_nodes)
    adj[idx, idx] = True
    return adj


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, concat=True,
                 activation=F.elu, input_dropout=0.6, coef_dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.activation = activation
        self.input_dropout = input_dropout
        self.coef_dropout = coef_dropout

        self.W = nn.Parameter(torch.empty(num_heads, in_dim, out_dim))
        self.a_i = nn.Parameter(torch.empty(num_heads, out_dim))
        self.a_j = nn.Parameter(torch.empty(num_heads, out_dim))
        self.bias = nn.Parameter(torch.empty(num_heads, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_i)
        nn.init.xavier_uniform_(self.a_j)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x: [N, in_dim]
        adj: [N, N] bool tensor indicating edges (including self-loops)
        """
        N = x.size(0)
        # input dropout
        if self.training:
            x = F.dropout(x, p=self.input_dropout, training=True)

        # linear transform per head: [H, N, out_dim]
        xW = torch.einsum('ni,hio->hno', x, self.W)  # [H, N, out_dim]

        # attention coefficients
        att_i = torch.einsum('hno,ho->hn', xW, self.a_i)  # [H, N]
        att_j = torch.einsum('hno,ho->hn', xW, self.a_j)  # [H, N]
        e = att_i.unsqueeze(2) + att_j.unsqueeze(1)      # [H, N, N]
        e = F.leaky_relu(e, negative_slope=0.2)

        # mask non-edges
        e_masked = e.masked_fill(~adj, float('-inf'))

        # softmax over destination dimension
        alpha = F.softmax(e_masked, dim=2)               # [H, N, N]

        # coefficient dropout
        if self.training:
            alpha = F.dropout(alpha, p=self.coef_dropout, training=True)

        # aggregate: [H, N, out_dim]
        out = torch.einsum('hin,hnj->hij', alpha, xW)    # [H, N, out_dim]
        out = out + self.bias.unsqueeze(1)               # [H, N, out_dim]

        if self.concat:
            out = out.transpose(0, 1).contiguous().view(N, -1)  # [N, H*out_dim]
        else:
            out = out.mean(dim=0)                           # [N, out_dim]

        if self.activation is not None:
            out = self.activation(out)
        return out


class GAT(nn.Module):
    def __init__(self, num_features, num_classes, cfg):
        super().__init__()
        self.layers = nn.ModuleList()
        heads_per_layer = [cfg["heads"], 1]
        for li in range(cfg["layers"]):
            in_dim = num_features if li == 0 else cfg["hidden"] * heads_per_layer[li - 1]
            out_dim = cfg["hidden"] if li < cfg["layers"] - 1 else num_classes
            concat = li < cfg["layers"] - 1  # concat for all but last layer
            self.layers.append(
                GATLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_heads=heads_per_layer[li],
                    concat=concat,
                    activation=F.elu if li < cfg["layers"] - 1 else None,
                    input_dropout=cfg["input_dropout"],
                    coef_dropout=cfg["coef_dropout"],
                )
            )

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x


def train(model, features, adj, labels, idx_train, idx_val, cfg):
    """Train the model with early stopping on validation loss/accuracy."""
    device = features.device
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj)
            val_loss = F.cross_entropy(val_logits[idx_val], labels[idx_val])
            val_pred = val_logits[idx_val].argmax(dim=1)
            val_acc = (val_pred == labels[idx_val]).float().mean()
        model.train()

        improved = False
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            improved = True
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            improved = True

        if improved:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg["patience"]:
            break

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_acc.item()


def main():
    # seed
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load data
    data = nio.load()
    features_np = data["features"]          # [N, F] float32
    edges_np = data["edges"]                # [E, 2] int64
    labels_np = data["labels"]              # [N] int64, -1 for test
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    N = features_np.shape[0]
    num_classes = len(np.unique(labels_np[labels_np >= 0]))

    # torch tensors
    features = torch.from_numpy(features_np).float()
    labels = torch.from_numpy(labels_np).long()
    adj = build_adjacency(edges_np, N).to(features.device)
    features = features.to(features.device)
    labels = labels.to(features.device)

    # smoke mode
    if os.environ.get("SMOKE") == "1":
        cfg = CONFIG.copy()
        cfg["epochs"] = 1
    else:
        cfg = CONFIG

    model = GAT(features.shape[1], num_classes, cfg)
    val_acc = train(model, features, adj, labels, train_idx, val_idx, cfg)

    # final inference
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)  # [N, C]

    # output logits for all nodes (helper will select test indices)
    nio.write_predictions(logits.cpu().numpy())

    # print final validation accuracy as requested
    print(f"{val_acc:.4f}")


if __name__ == "__main__":
    main()