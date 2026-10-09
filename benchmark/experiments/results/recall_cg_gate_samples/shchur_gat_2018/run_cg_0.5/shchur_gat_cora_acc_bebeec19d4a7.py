

# Plan step -> implemented in: set_seed
# Plan step -> implemented in: load_data
# Plan step -> implemented in: build_adjacency
# Plan step -> implemented in: GATLayer
# Plan step -> implemented in: GAT
# Plan step -> implemented in: train_model
# Plan step -> implemented in: main

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {"epochs": 100000, "patience": 100, "heads": 8}
# Additional hyperparameters (read only from CONFIG)
CONFIG["lr"] = 0.01
CONFIG["weight_decay"] = 5e-4
CONFIG["input_dropout"] = 0.6
CONFIG["attn_dropout"] = 0.6
CONFIG["hidden_size"] = 8
CONFIG["num_layers"] = 2
CONFIG["num_classes"] = 7  # Cora has 7 classes


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    return nio.load()


def build_adjacency(edges_np, N):
    """Return edge_index [2, E] with self-loops added."""
    edge_index = torch.from_numpy(edges_np).t().long()  # [2, E]
    # add self-loops
    loop = torch.arange(N, device=edge_index.device).unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop], dim=1)
    return edge_index


class GATLayer(nn.Module):
    """
    Single GAT layer with multi-head attention.
    Operates on a sparse graph given by edge_index.
    """

    def __init__(self, in_dim, out_dim, heads, concat, dropout_input, dropout_attn):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout_input = dropout_input
        self.dropout_attn = dropout_attn

        self.W = nn.Parameter(torch.Tensor(heads, in_dim, out_dim))
        self.att = nn.Parameter(torch.Tensor(heads, 2 * out_dim, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        """
        x: [N, in_dim]
        edge_index: [2, E] (source nodes in dim 0, target nodes in dim 1)
        """
        N = x.size(0)
        # Linear transformation: [N, heads, out_dim]
        h = torch.einsum('ni,hoi->nho', x, self.W)  # [N, heads, out_dim]
        if self.training:
            h = F.dropout(h, p=self.dropout_input, training=True)

        out_heads = []
        for h_idx in range(self.heads):
            Wh = h[:, h_idx, :]  # [N, out_dim]

            # Source and target node features for each edge
            Wh_src = Wh[edge_index[0]]  # [E, out_dim]
            Wh_dst = Wh[edge_index[1]]  # [E, out_dim]

            # Attention coefficients
            a_input = torch.cat([Wh_src, Wh_dst], dim=-1)  # [E, 2*out_dim]
            e = (a_input * self.att[h_idx].squeeze(-1)).sum(dim=-1)  # [E]
            e = F.leaky_relu(e, negative_slope=0.2)

            # Softmax per source node
            exp_e = torch.exp(e)
            sum_exp = torch.zeros(N, device=x.device).index_add_(0, edge_index[0], exp_e)
            alpha = exp_e / (sum_exp[edge_index[0]] + 1e-16)  # [E]
            if self.training:
                alpha = F.dropout(alpha, p=self.dropout_attn, training=True)

            # Message passing
            msg = alpha.unsqueeze(-1) * Wh_src  # [E, out_dim]
            agg = torch.zeros_like(Wh)  # [N, out_dim]
            agg.index_add_(0, edge_index[1], msg)  # aggregate to targets

            out_heads.append(agg)

        # Combine heads
        if self.concat:
            out = torch.cat(out_heads, dim=-1)  # [N, heads * out_dim]
            out = F.elu(out)  # activation except for final layer
        else:
            out = torch.stack(out_heads, dim=0).mean(dim=0)  # [N, out_dim]
            # no activation here (final layer)
        return out


class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, heads_per_layer,
                 input_drop, attn_drop):
        super().__init__()
        assert len(heads_per_layer) == num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                l_in = in_dim
            else:
                l_in = hidden_dim * heads_per_layer[i - 1]
            if i == num_layers - 1:
                l_out = out_dim
            else:
                l_out = hidden_dim
            concat = (i != num_layers - 1)
            layer = GATLayer(l_in, l_out, heads_per_layer[i],
                             concat, input_drop, attn_drop)
            self.layers.append(layer)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


def train_model(model, features, labels, train_idx, val_idx, edge_index, config):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_since_improve = 0
    best_state = None
    max_epochs = 1 if os.environ.get("SMOKE") == "1" else config["epochs"]

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, edge_index)
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features, edge_index)
            val_loss = criterion(val_logits[val_idx], labels[val_idx])
            val_pred = val_logits[val_idx].argmax(dim=-1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()

        improved = False
        if val_loss < best_val_loss - 1e-6:
            improved = True
            best_val_loss = val_loss
        if val_acc > best_val_acc + 1e-6:
            improved = True
            best_val_acc = val_acc

        if improved:
            epochs_since_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= config["patience"]:
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:4d} | loss {loss.item():.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    seed = int(os.environ.get("SEED", "0"))
    set_seed(seed)
    data = load_data()
    features_np = data["features"]
    edges_np = data["edges"]
    labels_np = data["labels"]
    train_idx_np = data["train_idx"]
    val_idx_np = data["val_idx"]
    test_idx_np = data["test_idx"]

    N = features_np.shape[0]
    num_classes = CONFIG["num_classes"]

    features = torch.from_numpy(features_np).float()
    labels = torch.from_numpy(labels_np).long()
    train_idx = torch.from_numpy(train_idx_np).long()
    val_idx = torch.from_numpy(val_idx_np).long()
    test_idx = torch.from_numpy(test_idx_np).long()

    edge_index = build_adjacency(edges_np, N)

    heads_per_layer = [CONFIG["heads"], 1]  # first layer 8 heads, second layer 1 head
    model = GAT(
        num_layers=CONFIG["num_layers"],
        in_dim=features.shape[1],
        hidden_dim=CONFIG["hidden_size"],
        out_dim=num_classes,
        heads_per_layer=heads_per_layer,
        input_drop=CONFIG["input_dropout"],
        attn_drop=CONFIG["attn_dropout"]
    )

    model = train_model(model, features, labels, train_idx, val_idx, edge_index, CONFIG)

    model.eval()
    with torch.no_grad():
        all_logits = model(features, edge_index)

    # Write predictions for all nodes (logits or class predictions accepted)
    nio.write_predictions(all_logits.cpu().numpy())


if __name__ == "__main__":
    main()