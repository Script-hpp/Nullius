

# Plan step -> implemented in: GAT class
# Plan step -> implemented in: GATLayer
# Plan step -> implemented in: train_model
# Plan step -> implemented in: build_adjacency
# Plan step -> implemented in: set_seed
# Plan step -> implemented in: main (loading data, training, prediction)

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

def build_adjacency(edges_np, N):
    """Adjacency matrix with self-loops, dense format."""
    adj = torch.zeros(N, N, dtype=torch.float32)
    adj[edges_np[:, 0], edges_np[:, 1]] = 1.0
    idx = torch.arange(N, dtype=torch.long)
    adj[idx, idx] = 1.0  # add self-loops
    return adj

class GATLayer(nn.Module):
    """
    Single GAT layer (multi-head attention).
    Implements the mechanism described in the GAT paper:
    - Linear projection of features
    - Attention coefficients computed via a single-layer feedforward network
    - LeakyReLU, softmax over neighbors, dropout
    - Aggregation and optional activation
    """
    def __init__(self, in_dim, out_dim, heads, input_drop, attn_drop, adj_matrix, concat=True, activation=F.elu):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat  # if False, average over heads (used for last layer)
        self.adj = adj_matrix  # [N, N] binary with self-loops
        self.activation = activation

        # Dropout rates
        self.input_drop = input_drop
        self.attn_drop = attn_drop

        # Learnable parameters per head
        self.W = nn.Parameter(torch.Tensor(heads, in_dim, out_dim))  # [H, in_dim, out_dim]
        self.a = nn.Parameter(torch.Tensor(heads, 2 * out_dim, 1))  # [H, 2*out_dim, 1]

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier init for W and a
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h):
        """
        h: [N, in_dim]
        Returns: [N, out_dim * heads] if concat else [N, out_dim]
        """
        N = h.size(0)
        h_drop = F.dropout(h, p=self.input_drop, training=self.training)  # [N, in_dim]

        # Linear projection: [N, heads, out_dim]
        # h_drop: [N, in_dim] -> [N, 1, in_dim] * [1, heads, in_dim, out_dim] -> [N, heads, out_dim]
        h_proj = torch.einsum("ni,hio->nho", h_drop, self.W)  # [N, heads, out_dim]

        # Prepare for attention: compute a^T [Wh_i || Wh_j]
        # Wh_i: [N, heads, out_dim] -> repeat for j dimension
        # We'll compute scores for all pairs efficiently using broadcasting
        # First, create [N, heads, out_dim, 1] and [1, heads, out_dim, N] then combine
        # Actually easier: compute for each head separately
        head_outputs = []
        for h_idx in range(self.heads):
            Wh = h_proj[:, h_idx, :]  # [N, out_dim]
            # Compute attention scores: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
            # a: [2*out_dim, 1]
            a = self.a[h_idx].squeeze(-1)  # [2*out_dim]
            # Wh_i repeated N times: [N, out_dim] -> [N, 1, out_dim]
            # Wh_j repeated N times: [N, out_dim] -> [1, N, out_dim]
            Wh_i = Wh.unsqueeze(1)  # [N, 1, out_dim]
            Wh_j = Wh.unsqueeze(0)  # [1, N, out_dim]
            # Concatenate: [N, N, 2*out_dim]
            a_input = torch.cat([Wh_i.repeat(1, N, 1), Wh_j.repeat(N, 1, 1)], dim=-1)
            # e = LeakyReLU(a^T a_input) -> [N, N]
            e = F.leaky_relu(torch.matmul(a_input, a.unsqueeze(-1)).squeeze(-1), negative_slope=0.2)

            # Mask with adjacency (including self-loops)
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(self.adj > 0, e, zero_vec)  # [N, N]

            # Softmax over neighbors
            attention = F.softmax(attention, dim=1)  # [N, N]
            attention = F.dropout(attention, p=self.attn_drop, training=self.training)

            # Aggregate: h'_i = sum_j alpha_ij * Wh_j
            h_prime = torch.matmul(attention, Wh)  # [N, out_dim]

            if self.activation is not None and self.concat:
                h_prime = self.activation(h_prime)

            head_outputs.append(h_prime)

        if self.concat:
            # [N, heads, out_dim] -> [N, heads * out_dim]
            output = torch.cat(head_outputs, dim=-1)
        else:
            # average over heads
            output = torch.mean(torch.stack(head_outputs, dim=0), dim=0)

        return output

class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, heads_per_layer,
                 input_drop, attn_drop, adj_matrix):
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
                             input_drop, attn_drop, adj_matrix,
                             concat=concat, activation=F.elu if i != num_layers-1 else None)
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

    adj = build_adjacency(edges_np, N)

    heads_per_layer = [CONFIG["heads"], 1]  # first layer 8 heads, second layer 1 head
    model = GAT(
        num_layers=CONFIG["num_layers"],
        in_dim=features.shape[1],
        hidden_dim=CONFIG["hidden_size"],
        out_dim=num_classes,
        heads_per_layer=heads_per_layer,
        input_drop=CONFIG["input_dropout"],
        attn_drop=CONFIG["attn_dropout"],
        adj_matrix=adj
    )

    model = train_model(model, features, labels, train_idx, val_idx, CONFIG)

    model.eval()
    with torch.no_grad():
        all_logits = model(features)

    # Write predictions for all nodes (logits or class predictions accepted)
    nio.write_predictions(all_logits.cpu().numpy())

if __name__ == "__main__":
    main()