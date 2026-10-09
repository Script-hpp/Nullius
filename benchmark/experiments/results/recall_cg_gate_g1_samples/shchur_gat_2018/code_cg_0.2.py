import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

# Plan step 1 -> implemented in: class GAT
# Plan step 2 -> implemented in: function preprocess_adjacency
# Plan step 3 -> implemented in: hyperparameters CONFIG["input_dropout"] and CONFIG["coef_dropout"] used in GATLayer
# Plan step 4 -> implemented in: optimizer creation (Adam) using CONFIG["lr"] and CONFIG["weight_decay"]
# Plan step 5 -> implemented in: training loop with early stopping based on CONFIG["patience"]
# Plan step 6 -> implemented in: using train_idx and val_idx from loaded data
# Plan step 7 -> implemented in: using test_idx for final prediction
# Plan step 8 -> implemented in: setting seeds from environment variable SEED
# Plan step 9 -> implemented in: full‑batch training (no mini‑batching logic)
# Plan step 10 -> implemented in: learning‑rate decay disabled (no decay applied)
# Plan step 11 -> implemented in: weight initialization with Glorot uniform (xavier_uniform_)
# Plan step 12 -> implemented in: early stopping restores best model state
# Plan step 13 -> implemented in: calling nio.write_predictions with final logits
# Plan step 14 -> implemented in: ensuring runtime <5 minutes (comment)
# Plan step 15 -> implemented in: all other settings follow CONFIG and defaults

CONFIG = {
    "lr": 0.01,
    "weight_decay": 0.01,
    "epochs": 100000,
    "hidden": 8,
    "layers": 2,
    "patience": 100,
    "heads": 8,
    "input_dropout": 0.6,
    "coef_dropout": 0.3
}


def preprocess_adjacency(edges, num_nodes):
    """
    Build undirected adjacency with self-loops and return edge_index [2, E].
    """
    edge_index = edges.t()  # [2, E]
    # add self-loops
    self_loop = torch.arange(num_nodes, dtype=torch.long)
    self_loop = self_loop.repeat(2, 1)  # [2, N]
    edge_index = torch.cat([edge_index, self_loop], dim=1)
    # make sure adjacency is undirected (already both directions in edges, but keep unique)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, concat=True, dropout=0.6, attn_dropout=0.3, negative_slope=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.negative_slope = negative_slope

        self.W = nn.ParameterList()
        self.a_l = nn.ParameterList()
        self.a_r = nn.ParameterList()
        self.bias_out = nn.ParameterList()
        self.bias_att_l = nn.ParameterList()
        self.bias_att_r = nn.ParameterList()

        for _ in range(num_heads):
            w = nn.Parameter(torch.Tensor(in_dim, out_dim))
            al = nn.Parameter(torch.Tensor(out_dim, 1))
            ar = nn.Parameter(torch.Tensor(out_dim, 1))
            b_out = nn.Parameter(torch.Tensor(out_dim))
            b_al = nn.Parameter(torch.Tensor(1))
            b_ar = nn.Parameter(torch.Tensor(1))
            self.W.append(w)
            self.a_l.append(al)
            self.a_r.append(ar)
            self.bias_out.append(b_out)
            self.bias_att_l.append(b_al)
            self.bias_att_r.append(b_ar)

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.W:
            nn.init.xavier_uniform_(w)
        for al in self.a_l:
            nn.init.xavier_uniform_(al)
        for ar in self.a_r:
            nn.init.xavier_uniform_(ar)
        for b_out in self.bias_out:
            nn.init.zeros_(b_out)
        for b_al in self.bias_att_l:
            nn.init.zeros_(b_al)
        for b_ar in self.bias_att_r:
            nn.init.zeros_(b_ar)

    def forward(self, x, edge_index):
        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)

        N = x.size(0)
        head_outputs = []

        for i in range(self.num_heads):
            W = self.W[i]
            a_l = self.a_l[i]
            a_r = self.a_r[i]
            b_al = self.bias_att_l[i]
            b_ar = self.bias_att_r[i]
            b_out = self.bias_out[i]

            z = torch.matmul(x, W)  # [N, out_dim]

            if self.training and self.dropout > 0:
                z = F.dropout(z, p=self.dropout, training=True)

            att_l = torch.matmul(z, a_l) + b_al  # [N,1]
            att_r = torch.matmul(z, a_r) + b_ar  # [N,1]

            src = edge_index[0]
            dst = edge_index[1]

            e = att_l[src].squeeze(-1) + att_r[dst].squeeze(-1)  # [E]
            e = F.leaky_relu(e, negative_slope=self.negative_slope)

            exp_e = torch.exp(e)
            sum_exp = torch.zeros(N, device=x.device)
            sum_exp.index_add_(0, src, exp_e)
            alpha = exp_e / (sum_exp[src] + 1e-16)  # [E]

            if self.training and self.attn_dropout > 0:
                alpha = F.dropout(alpha, p=self.attn_dropout, training=True)

            z_dst = z[dst]  # [E, out_dim]
            weighted = z_dst * alpha.unsqueeze(-1)  # [E, out_dim]
            out = torch.zeros(N, z.size(1), device=x.device)
            out.index_add_(0, src, weighted)
            out = out + b_out
            head_outputs.append(out)

        if self.concat:
            output = torch.cat(head_outputs, dim=-1)
        else:
            output = torch.stack(head_outputs, dim=0).mean(dim=0)
        return output


class GAT(nn.Module):
    def __init__(self, num_features, num_classes, hidden, heads, dropout, attn_dropout):
        super().__init__()
        self.layer1 = GATLayer(num_features, hidden, heads, concat=True,
                               dropout=dropout, attn_dropout=attn_dropout)
        self.layer2 = GATLayer(hidden * heads, num_classes, 1, concat=False,
                               dropout=dropout, attn_dropout=attn_dropout)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = F.elu(x)
        x = self.layer2(x, edge_index)
        return x


def main():
    seed = int(os.environ.get('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)

    smoke = os.environ.get('SMOKE') == '1'

    d = nio.load()
    features = torch.from_numpy(d['features']).float()
    labels = torch.from_numpy(d['labels']).long()
    train_idx = torch.from_numpy(d['train_idx']).long()
    val_idx = torch.from_numpy(d['val_idx']).long()
    test_idx = torch.from_numpy(d['test_idx']).long()
    edges = torch.from_numpy(d['edges']).long()

    num_nodes = features.shape[0]
    edge_index = preprocess_adjacency(edges, num_nodes)

    num_features = features.shape[1]
    num_labels = int(labels.max().item()) + 1  # labels are 0..6, -1 for test ignored

    model = GAT(num_features=num_features,
                num_classes=num_labels,
                hidden=CONFIG["hidden"],
                heads=CONFIG["heads"],
                dropout=CONFIG["input_dropout"],
                attn_dropout=CONFIG["coef_dropout"])

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CONFIG["lr"],
                                 weight_decay=CONFIG["weight_decay"])

    max_epochs = 1 if smoke else CONFIG["epochs"]
    patience = CONFIG["patience"]
    wait = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, edge_index)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features, edge_index)
            val_loss = F.cross_entropy(val_logits[val_idx], labels[val_idx]).item()
            pred = val_logits[val_idx].argmax(dim=1)
            val_acc = (pred == labels[val_idx]).float().mean().item()

        improved = False
        if val_loss < best_val_loss - 1e-6:
            improved = True
            best_val_loss = val_loss
        if val_acc > best_val_acc + 1e-6:
            improved = True
            best_val_acc = val_acc
        if improved:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        all_logits = model(features, edge_index)

    nio.write_predictions(all_logits.detach().cpu().numpy())
    # Print final validation accuracy (best observed)
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
