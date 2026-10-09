

# Plan step -> implemented in: CONFIG dict
# Plan step -> implemented in: build_adjacency function
# Plan step -> implemented in: GATLayer class
# Plan step -> implemented in: GAT class
# Plan step -> implemented in: optimizer initialization
# Plan step -> implemented in: training loop with early stopping
# Plan step -> implemented in: use of val_idx for validation
# Plan step -> implemented in: prediction on test_idx via helper
# Plan step -> implemented in: seed setting from env SEED
# Plan step -> implemented in: full-batch training (no loader)
# Plan step -> implemented in: no learning rate decay
# Plan step -> implemented in: Xavier/Glorot uniform init in GATLayer
# Plan step -> implemented in: saving/restoring best model state
# Plan step -> implemented in: nio.write_predictions call
# Plan step -> implemented in: ensuring runtime <5 min (comment)

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
    """Build dense adjacency matrix with self-loops."""
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    if edges.size > 0:
        u = torch.from_numpy(edges[:, 0]).long()
        v = torch.from_numpy(edges[:, 1]).long()
        adj[u, v] = 1.0
        adj[v, u] = 1.0  # ensure symmetry (though edges already contain both directions)
    # add self-loops
    adj.fill_diagonal_(1.0)
    return adj

class GATLayer(nn.Module):
    """
    Single GAT layer (multi-head attention) using dense operations.
    """
    def __init__(self, in_features, out_features, num_heads, concat=True,
                 dropout=0.6, attn_dropout=0.3, alpha=0.2):
        super().__init__()
        self.concat = concat
        self.num_heads = num_heads
        self.out_features = out_features
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.alpha = alpha

        # Parameters per head
        self.Ws = nn.ParameterList()
        self.att_vecs = nn.ParameterList()
        self.biases = nn.ParameterList()

        for _ in range(num_heads):
            w = nn.Parameter(torch.empty(in_features, out_features))
            att = nn.Parameter(torch.empty(2 * out_features))
            b = nn.Parameter(torch.empty(out_features))
            self.Ws.append(w)
            self.att_vecs.append(att)
            self.biases.append(b)

        self.reset_parameters()
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def reset_parameters(self):
        for w in self.Ws:
            nn.init.xavier_uniform_(w, gain=1.414)
        for att in self.att_vecs:
            nn.init.xavier_uniform_(att, gain=1.414)
        for b in self.biases:
            nn.init.zeros_(b)

    def forward(self, h, adj):
        """
        h: [N, in_features]
        adj: [N, N] dense adjacency (with self-loops)
        """
        if self.training:
            h = F.dropout(h, self.dropout, training=True)

        heads_out = []
        for i in range(self.num_heads):
            W = self.Ws[i]
            att = self.att_vecs[i]
            bias = self.biases[i]

            # Linear transformation
            z = torch.mm(h, W)  # [N, out_features]

            # Attention scores: a^T [z_i || z_j] = a1^T z_i + a2^T z_j
            a1 = att[:self.out_features]   # [out_features]
            a2 = att[self.out_features:]   # [out_features]
            # [N, 1] + [1, N] -> [N, N]
            e = torch.mm(z, a1.unsqueeze(1)) + torch.mm(z, a2.unsqueeze(1)).t()
            e = self.leakyrelu(e)

            # Mask non-neighbors
            mask = (adj == 0)
            e = e.masked_fill(mask, float('-inf'))

            # Attention coefficients
            alpha = F.softmax(e, dim=1)          # [N, N]
            alpha = F.dropout(alpha, self.attn_dropout, training=self.training)

            # Dropout on transformed features (as in official implementation)
            z_drop = F.dropout(z, self.dropout, training=self.training)

            # Weighted aggregation
            head_out = torch.mm(alpha, z_drop)   # [N, out_features]
            head_out = head_out + bias           # add bias
            heads_out.append(head_out)

        if self.concat:
            out = torch.cat(heads_out, dim=1)    # [N, num_heads * out_features]
        else:
            out = torch.stack(heads_out, dim=0).mean(dim=0)  # [N, out_features]

        # Apply activation (ELU) after each hidden layer (not after final layer)
        if self.concat and not getattr(self, 'is_last_layer', False):
            out = F.elu(out)
        return out

class GAT(nn.Module):
    def __init__(self, nfeat, nhidden, nclass, nlayers, heads, dropout, attn_dropout):
        super().__init__()
        assert len(heads) == nlayers
        self.layers = nn.ModuleList()
        in_ch = nfeat
        for l in range(nlayers):
            out_ch = nhidden if l < nlayers - 1 else nclass
            concat = True if l < nlayers - 1 else False
            layer = GATLayer(
                in_features=in_ch,
                out_features=out_ch,
                num_heads=heads[l],
                concat=concat,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            # Mark last layer so activation is skipped
            if l == nlayers - 1:
                layer.is_last_layer = True
            self.layers.append(layer)
            # Update input channels for next layer
            if concat:
                in_ch = out_ch * heads[l]
            else:
                in_ch = out_ch  # heads[l] is 1 for last layer anyway

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
        return x

def main():
    # Seed
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    data = nio.load()
    features = torch.from_numpy(data["features"]).float()          # [N, F]
    edges = data["edges"].astype(np.int64)                         # [E, 2]
    labels = torch.from_numpy(data["labels"]).long()               # [N]
    train_idx = torch.from_numpy(data["train_idx"]).long()
    val_idx = torch.from_numpy(data["val_idx"]).long()
    test_idx = torch.from_numpy(data["test_idx"]).long()

    N = features.shape[0]
    adj = build_adjacency(edges, N)                                # [N, N]

    # Model
    model = GAT(
        nfeat=features.shape[1],
        nhidden=CONFIG["hidden"],
        nclass=int(labels.max().item() + 1),  # 7 classes
        nlayers=CONFIG["layers"],
        heads=[CONFIG["heads"], 1],           # first layer 8 heads, second layer 1 head
        dropout=CONFIG["input_dropout"],
        attn_dropout=CONFIG["coef_dropout"],
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Masks
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Training loop
    epochs = CONFIG["epochs"]
    patience = CONFIG["patience"]
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_state = None

    # Smoke mode: exactly 1 epoch, ignore early stopping
    smoke = os.environ.get("SMOKE") == "1"
    if smoke:
        epochs = 1
        patience = epochs  # effectively disable early stopping

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj)
            val_loss = F.cross_entropy(val_logits[val_mask], labels[val_mask])
            val_pred = val_logits.argmax(dim=-1)
            val_acc = (val_pred[val_mask] == labels[val_mask]).float().mean().item()
        model.train()

        # Early stopping logic (GAT criterion: stop if neither loss improves nor acc improves)
        improved = False
        if val_loss < best_val_loss or val_acc > best_val_acc:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            improved = True
        else:
            epochs_no_improve += 1

        if not smoke and epochs_no_improve >= patience:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation (optional, for logging)
    model.eval()
    with torch.no_grad():
        final_logits = model(features, adj)
        final_val_acc = (final_logits[val_mask].argmax(dim=-1) == labels[val_mask]).float().mean().item()
    print(f"Validation Accuracy: {final_val_acc:.4f}")

    # Prepare predictions for all nodes (logits)
    preds_np = final_logits.detach().cpu().numpy()
    # nio.write_predictions accepts logits or class indices
    nio.write_predictions(preds_np)

if __name__ == "__main__":
    main()