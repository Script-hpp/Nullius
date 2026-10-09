# Plan step 1 -> implemented in: GAT class
# Plan step 2 -> implemented in: data loading and adjacency building in main()
# Plan step 3 -> implemented in: GATLayer dropout arguments
# Plan step 4 -> implemented in: optimizer initialization
# Plan step 5 -> implemented in: training loop with early stopping
# Plan step 6 -> implemented in: use of val_idx for validation
# Plan step 7 -> implemented in: prediction on test_idx via helper
# Plan step 8 -> implemented in: seed setting from env SEED
# Plan step 9 -> implemented in: full-batch training (no loader)
# Plan step 10 -> implemented in: no learning rate decay
# Plan step 11 -> implemented in: Xavier/Glorot uniform init in GATLayer
# Plan step 12 -> implemented in: saving/restoring best model state
# Plan step 13 -> implemented in: nio.write_predictions call
# Plan step 14 -> implemented in: ensuring runtime <5 min (comment)
# Plan step 15 -> implemented in: following configs/defaults elsewhere

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

# Hyperparameters from spec
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
    # add self-loops
    adj.fill_diagonal_(1.0)
    return adj

class GATLayer(nn.Module):
    """
    Single GAT layer (multi-head attention).
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

        # Weight matrix for linear transformation
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features * num_heads)))
        # Attention vector a (split into a1, a2 per head)
        self.a = nn.Parameter(torch.empty(size=(num_heads, 2 * out_features, 1)))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        """
        h: [N, in_features]
        adj: [N, N] dense adjacency (with self-loops)
        """
        N = h.size(0)
        # Dropout on input features
        h_drop = F.dropout(h, self.dropout, training=self.training)

        # Linear transformation
        h_prime = torch.matmul(h_drop, self.W)          # [N, out_features * num_heads]
        h_prime = h_prime.view(N, self.num_heads, self.out_features)  # [N, num_heads, out_features]

        # Prepare attention weights
        # a1: [num_heads, out_features], a2: [num_heads, out_features]
        a1 = self.a[:, :self.out_features, 0]   # [num_heads, out_features]
        a2 = self.a[:, self.out_features:, 0]   # [num_heads, out_features]

        # Compute attention scores per head: e_ij = a1^T Wh_i + a2^T Wh_j
        # Wh_i dot a1 -> [N, num_heads]
        attn_for_self = torch.sum(h_prime * a1.unsqueeze(0), dim=-1)  # [N, num_heads]
        attn_for_neigh = torch.sum(h_prime * a2.unsqueeze(0), dim=-1)  # [N, num_heads]
        # Combine to get [num_heads, N, N]
        e = attn_for_self.unsqueeze(2) + attn_for_neigh.unsqueeze(1)   # [N, num_heads, N]
        e = e.permute(1, 0, 2)                                          # [num_heads, N, N]
        e = self.leakyrelu(e)

        # Mask non-neighbors
        mask = adj.unsqueeze(0) == 0   # [1, N, N]
        e = e.masked_fill(mask, -1e9)

        # Softmax -> attention coefficients
        attn = F.softmax(e, dim=-1)                     # [num_heads, N, N]
        attn = F.dropout(attn, self.attn_dropout, training=self.training)

        # Aggregate features
        # h_prime_heads: [num_heads, N, out_features]
        h_prime_heads = h_prime.permute(1, 0, 2)
        # [num_heads, N, out_features] = attn @ h_prime_heads
        h_double_prime = torch.einsum('hnm,hmd->hnd', attn, h_prime_heads)  # [num_heads, N, out_features]
        h_double_prime = h_double_prime.permute(1, 0, 2)                    # [N, num_heads, out_features]

        if self.concat:
            # Concatenate heads -> [N, num_heads * out_features]
            output = h_double_prime.reshape(N, -1)
        else:
            # Average across heads -> [N, out_features]
            output = h_double_prime.mean(dim=1)
        return output

class GAT(nn.Module):
    def __init__(self, nfeat, nhidden, nclass, nlayers, heads, dropout, attn_dropout):
        super().__init__()
        assert len(heads) == nlayers
        self.layers = nn.ModuleList()
        in_ch = nfeat
        for l in range(nlayers):
            # Determine output features for this layer
            out_ch = nhidden if l < nlayers - 1 else nclass
            concat = True if l < nlayers - 1 else False
            self.layers.append(
                GATLayer(
                    in_features=in_ch,
                    out_features=out_ch,
                    num_heads=heads[l],
                    concat=concat,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
            )
            # Update input channels for next layer
            if concat:
                in_ch = out_ch * heads[l]
            else:
                in_ch = out_ch  # heads[l] is 1 for last layer anyway

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < len(self.layers) - 1:
                x = F.elu(x)
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
