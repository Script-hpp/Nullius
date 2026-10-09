

# Plan step -> implemented in: set_seed
# Plan step -> implemented in: load_data
# Plan step -> implemented in: build_edge_index
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

# Literal config as required
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

def build_edge_index(edges_np, N):
    """Return edge_index [2, E] including self-loops."""
    edge_idx = torch.from_numpy(edges_np).t().contiguous()  # [2, E]
    # add self-loops
    loop = torch.arange(N, dtype=torch.long)
    self_loop = torch.stack([loop, loop], dim=0)  # [2, N]
    edge_index = torch.cat([edge_idx, self_loop], dim=1)
    return edge_index

class GATLayer(nn.Module):
    """
    Single GAT layer with multi-head attention.
    Operates on a precomputed edge_index (including self-loops).
    """
    def __init__(self, in_dim, out_dim, heads, input_drop, attn_drop, edge_index, N, concat=True, activation=F.elu):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.activation = activation
        self.edge_index = edge_index  # [2, E]
        self.N = N
        self.input_drop = input_drop
        self.attn_drop = attn_drop

        # Parameters per head: W [heads, in_dim, out_dim], a [heads, 2*out_dim], bias [heads, out_dim]
        self.W = nn.Parameter(torch.Tensor(heads, in_dim, out_dim))
        self.a = nn.Parameter(torch.Tensor(heads, 2 * out_dim))
        self.bias = nn.Parameter(torch.Tensor(heads, out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        x: [N, in_dim]
        Returns: [N, out_dim * heads] if concat else [N, out_dim]
        """
        # Apply input dropout (same dropout mask for all heads)
        x_drop = F.dropout(x, p=self.input_drop, training=self.training)

        # Linear projection for all heads: [N, heads, out_dim]
        # x_drop: [N, in_dim] -> [N, 1, in_dim] * [heads, in_dim, out_dim] -> [N, heads, out_dim]
        x_proj = torch.einsum('ni,hio->nho', x_drop, self.W)  # [N, heads, out_dim]

        src, dst = self.edge_index  # each [E]
        head_outputs = []

        for h in range(self.heads):
            # Projected features for this head: [N, out_dim]
            Wh = x_proj[:, h, :]  # [N, out_dim]

            # Compute attention scores for edges
            Wh_src = Wh[src]  # [E, out_dim]
            Wh_dst = Wh[dst]  # [E, out_dim]
            a_vec = self.a[h]  # [2*out_dim]
            # Concatenate Wh_src and Wh_dst
            cat = torch.cat([Wh_src, Wh_dst], dim=-1)  # [E, 2*out_dim]
            e = F.leaky_relu((cat * a_vec).sum(dim=-1), negative_slope=0.2)  # [E]

            # Apply attention dropout
            e = F.dropout(e, p=self.attn_drop, training=self.training)

            # Softmax per source node (normalize over incoming edges to src)
            # We'll compute alpha per edge
            alpha = torch.zeros_like(e)
            # Loop over nodes (N <= 2485) – acceptable
            for node in range(self.N):
                mask = (src == node)
                if mask.any():
                    e_node = e[mask]
                    alpha_node = F.softmax(e_node, dim=0)
                    alpha[mask] = alpha_node

            # Message passing: alpha_ij * Wh_j
            msg = alpha.unsqueeze(-1) * Wh_dst  # [E, out_dim]
            # Aggregate to source nodes
            agg = torch.zeros(self.N, self.out_dim, device=Wh.device)
            agg.index_add_(0, src, msg)  # sum over messages for each src

            # Add bias
            agg = agg + self.bias[h]

            # Activation (except possibly last layer handled outside)
            if self.activation is not None and self.concat:
                agg = self.activation(agg)

            head_outputs.append(agg)

        if self.concat:
            # [N, heads, out_dim] -> [N, heads * out_dim]
            return torch.cat(head_outputs, dim=-1)
        else:
            # average over heads
            return torch.mean(torch.stack(head_outputs, dim=0), dim=0)


class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, heads_per_layer,
                 input_drop, attn_drop, edge_index, N):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                l_in = in_dim
            else:
                l_in = hidden_dim * heads_per_layer[i-1]
            if i == num_layers-1:
                l_out = out_dim
            else:
                l_out = hidden_dim
            concat = (i != num_layers-1)  # last layer averages heads (concat=False)
            layer = GATLayer(l_in, l_out, heads_per_layer[i],
                             input_drop, attn_drop, edge_index, N,
                             concat=concat,
                             activation=F.elu if i != num_layers-1 else None)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train_model(model, features, labels, train_idx, val_idx, config):
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
        logits = model(features)
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features)
            val_loss = criterion(logits[val_idx], labels[val_idx])
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

    edge_index = build_edge_index(edges_np, N)

    heads_per_layer = [CONFIG["heads"], 1]  # first layer 8 heads, second layer 1 head
    model = GAT(
        num_layers=CONFIG["num_layers"],
        in_dim=features.shape[1],
        hidden_dim=CONFIG["hidden_size"],
        out_dim=num_classes,
        heads_per_layer=heads_per_layer,
        input_drop=CONFIG["input_dropout"],
        attn_drop=CONFIG["attn_dropout"],
        edge_index=edge_index,
        N=N
    )

    model = train_model(model, features, labels, train_idx, val_idx, CONFIG)

    model.eval()
    with torch.no_grad():
        all_logits = model(features)

    # Write predictions for all nodes (logits or class predictions accepted)
    nio.write_predictions(all_logits.cpu().numpy())

if __name__ == "__main__":
    main()