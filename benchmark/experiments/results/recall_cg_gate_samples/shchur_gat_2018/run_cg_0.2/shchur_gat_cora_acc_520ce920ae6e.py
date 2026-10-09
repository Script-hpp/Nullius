# Plan step -> implemented in: load_and_seed
# Plan step -> implemented in: preprocess_adj
# Plan step -> implemented in: GATModel
# Plan step -> implemented in: train
# Plan step -> implemented in: evaluate
# Plan step -> implemented in: main

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

# Hyperparameters read only from this dict
CONFIG = {
    "epochs": 100000,
    "patience": 100,
    "heads": 8,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "hidden_size": 8,
    "num_layers": 2,
    "input_dropout_prob": 0.6,
    "coeff_dropout_prob": 0.6,
}

def load_and_seed():
    """Load data, set random seeds from env SEED."""
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    d = nio.load()
    features = torch.from_numpy(d["features"].astype(np.float32))  # [N, 1433]
    edges = torch.from_numpy(d["edges"].astype(np.int64))          # [E, 2]
    labels = torch.from_numpy(d["labels"].astype(np.int64))       # [N]
    train_idx = torch.from_numpy(d["train_idx"].astype(np.int64))
    val_idx = torch.from_numpy(d["val_idx"].astype(np.int64))
    test_idx = torch.from_numpy(d["test_idx"].astype(np.int64))
    return features, edges, labels, train_idx, val_idx, test_idx

def preprocess_adj(edges, num_nodes):
    """Add self-loops and return edge_index as torch.LongTensor [2, E']."""
    # edges already contain both directions; make them unique? we keep as is.
    edge_index = edges.t()  # [2, E]
    # self-loops
    loop_index = torch.arange(num_nodes, dtype=torch.long)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)  # [2, N]
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index

class GATLayer(nn.Module):
    """
    Single GAT layer (multi-head) matching the TensorFlow description.
    Supports concatenation of heads except for the final layer.
    """
    def __init__(self, in_dim, out_dim, num_heads, concat=True,
                 dropout=0.6, attn_dropout=0.6, negative_slope=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.negative_slope = negative_slope

        # Parameters per head
        self.W = nn.Parameter(torch.Tensor(num_heads, in_dim, out_dim))
        self.a_l = nn.Parameter(torch.Tensor(num_heads, out_dim, 1))
        self.a_r = nn.Parameter(torch.Tensor(num_heads, out_dim, 1))

        self.leakyrelu = nn.LeakyReLU(self.negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a_l.data)
        nn.init.xavier_uniform_(self.a_r.data)

    def forward(self, x, edge_index):
        """
        x: [N, in_dim]
        edge_index: [2, E] (includes self-loops)
        Returns: [N, out_dim * num_heads] if concat else [N, out_dim]
        """
        H, C = self.num_heads, self.out_dim
        N = x.size(0)

        # Dropout on input features
        x_drop = F.dropout(x, p=self.dropout, training=self.training)

        # Linear transform per head: [N, H, C]
        x_w = torch.einsum('ij,hjk->ihk', x_drop, self.W)  # [N, H, C]

        # Compute attention scores
        # a_l * x_w -> [N, H, 1], a_r * x_w -> [N, H, 1]
        a_l = self.a_l.permute(0, 2, 1)  # [H, 1, C]
        a_r = self.a_r.permute(0, 2, 1)
        el = (x_w * a_l).sum(dim=-1)  # [N, H]
        er = (x_w * a_r).sum(dim=-1)  # [N, H]

        # Prepare edge-wise attention
        src, dst = edge_index  # each [E]
        # el_src: [E, H], er_dst: [E, H]
        el_src = el.index_select(0, src)  # [E, H]
        er_dst = er.index_select(0, dst)  # [E, H]
        e = el_src + er_dst               # [E, H]
        e = self.leakyrelu(e)             # [E, H]

        # Softmax per source node
        # Compute exp
        e_exp = torch.exp(e)              # [E, H]
        # Sum over destination nodes per source and head
        # We need to sum for each (src, head)
        # Use scatter_add over source dimension
        # Create flattened indices for scattering: src * H + head
        # But easier: loop over heads (H small <=8)
        alpha = torch.zeros_like(e_exp)
        for h in range(H):
            e_exp_h = e_exp[:, h]         # [E]
            src_h = src
            # sum per source
            sum_h = torch.zeros(N, device=x.device)
            sum_h.scatter_add_(0, src_h, e_exp_h)
            # avoid division by zero
            sum_h = sum_h.clamp(min=1e-16)
            alpha_h = e_exp_h / sum_h[src_h]
            alpha[:, h] = alpha_h

        # Apply dropout to attention coefficients
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

        # Weighted aggregation: [E, H] * [E, H, C]? Actually we need to multiply alpha[e, h] * x_w[dst, h, :]
        # Compute output per head: for each head, aggregate messages from neighbors
        # We'll compute using index_add
        out = torch.zeros(N, H, C, device=x.device)
        for h in range(H):
            alpha_h = alpha[:, h]          # [E]
            # message = alpha_h[:, None] * x_w[dst, h, :]  # [E, C]
            msg = alpha_h.unsqueeze(-1) * x_w.index_select(0, dst)[:, h, :]  # [E, C]
            # aggregate to src nodes
            out[:, h, :].index_add_(0, src, msg)

        if self.concat:
            # [N, H*C]
            out = out.view(N, H * C)
        else:
            # average over heads
            out = out.mean(dim=1)

        return out

class GATModel(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_classes, num_layers,
                 heads_per_layer, dropout, attn_dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            in_dim = feat_dim if i == 0 else hidden_dim * heads_per_layer[i-1]
            out_dim = hidden_dim if i < num_layers-1 else num_classes
            concat = i < num_layers-1  # concat for all but last
            self.layers.append(
                GATLayer(in_dim, out_dim, heads_per_layer[i],
                         concat=concat, dropout=dropout,
                         attn_dropout=attn_dropout)
            )

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
        return x  # logits [N, num_classes]

def train(model, features, edge_index, labels, train_idx, val_idx,
          lr, weight_decay, epochs, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(features, edge_index)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, edge_index)
            val_loss = F.cross_entropy(val_logits[val_idx], labels[val_idx]).item()
            val_pred = val_logits[val_idx].argmax(dim=-1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()
        model.train()

        # Check improvement
        improved = False
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            improved = True
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            improved = True

        if improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # Load best model? Not stored; we just return the last model (early stopping ensures best recent)
    return model

def evaluate(model, features, edge_index, labels, idx):
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index)
        loss = F.cross_entropy(logits[idx], labels[idx]).item()
        pred = logits[idx].argmax(dim=-1)
        acc = (pred == labels[idx]).float().mean().item()
    return loss, acc, logits

def main():
    features, edges, labels, train_idx, val_idx, test_idx = load_and_seed()
    num_nodes = features.size(0)
    edge_index = preprocess_adj(edges, num_nodes)

    model = GATModel(
        feat_dim=features.size(1),
        hidden_dim=CONFIG["hidden_size"],
        num_classes=7,
        num_layers=CONFIG["num_layers"],
        heads_per_layer=[CONFIG["heads"], 1] if CONFIG["num_layers"] == 2 else [CONFIG["heads"]] * CONFIG["num_layers"],
        dropout=CONFIG["input_dropout_prob"],
        attn_dropout=CONFIG["coeff_dropout_prob"]
    )

    smoke = os.environ.get("SMOKE") == "1"
    epochs_to_run = 1 if smoke else CONFIG["epochs"]
    patience_to_use = 1 if smoke else CONFIG["patience"]

    model = train(model, features, edge_index, labels, train_idx, val_idx,
                  lr=CONFIG["lr"],
                  weight_decay=CONFIG["weight_decay"],
                  epochs=epochs_to_run,
                  patience=patience_to_use)

    val_loss, val_acc, _ = evaluate(model, features, edge_index, labels, val_idx)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Get predictions (logits) for all nodes
    model.eval()
    with torch.no_grad():
        all_logits = model(features, edge_index)  # [N, 7]
    # nio.write_predictions accepts either class predictions or logits
    nio.write_predictions(all_logits)

if __name__ == "__main__":
    main()
