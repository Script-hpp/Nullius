

# Plan step -> implemented in: main function loading data via nullius_io
# Plan step -> implemented in: GAT class definition
# Plan step -> implemented in: optimizer and loss setup in main
# Plan step -> implemented in: training loop with early stopping in main
# Plan step -> implemented in: seed initialization from SEED environment variable in main
# Plan step -> implemented in: validation accuracy tracking and best model saving in main
# Plan step -> implemented in: final prediction and writing via nullius_io in main

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

def main():
    # ----- Seed -----
    seed = int(os.environ.get('SEED', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----- Load data -----
    data = nio.load()  # returns dict with keys: features, edges, labels, train_idx, val_idx, test_idx
    features = torch.from_numpy(data['features']).float()          # [N, F]
    edges = torch.from_numpy(data['edges']).long()                 # [E, 2] directed pairs
    labels = torch.from_numpy(data['labels']).long()               # [N]
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx = torch.from_numpy(data['val_idx']).long()
    test_idx = torch.from_numpy(data['test_idx']).long()

    N = features.size(0)
    num_classes = int(labels.max().item() + 1)  # labels are 0..6 for labeled nodes

    # ----- Build adjacency list (neighbors) including self-loops -----
    neighbors = [[] for _ in range(N)]
    src = edges[:, 0]
    dst = edges[:, 1]
    for s, t in zip(src.tolist(), dst.tolist()):
        neighbors[s].append(t)
    # add self-loops
    for i in range(N):
        neighbors[i].append(i)

    # ----- Smoke mode -----
    if os.environ.get('SMOKE') == '1':
        max_epochs = 1
        patience = 0  # irrelevant
    else:
        max_epochs = 10000
        patience = 100

    # ----- Model hyperparameters -----
    in_features = features.size(1)
    hidden_dim_per_head = 8      # first layer output per head
    num_heads = 8                # first layer heads
    dropout = 0.6
    lr = 0.005
    weight_decay = 5e-4

    # ----- GAT definition -----
    class GAT(nn.Module):
        def __init__(self, in_feat, hid_dim, n_heads, n_classes, drop=0.6, alpha=0.2):
            super().__init__()
            self.n_heads = n_heads
            self.hid_dim = hid_dim
            self.drop = drop
            self.leakyrelu = nn.LeakyReLU(alpha)

            # First layer: projection to multi-head space
            self.W = nn.Parameter(torch.empty(in_feat, hid_dim * n_heads))
            # Attention vectors per head (split into left/right)
            self.a_l = nn.Parameter(torch.empty(n_heads, hid_dim))
            self.a_r = nn.Parameter(torch.empty(n_heads, hid_dim))

            # Second layer: single head projecting to logits
            self.W_out = torch.nn.Parameter(torch.empty(hid_dim * n_heads, n_classes))
            self.a_out_l = nn.Parameter(torch.empty(n_classes))
            self.a_out_r = nn.Parameter(torch.empty(n_classes))

            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.W.data, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.a_l.data, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.a_r.data, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.W_out.data, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.a_out_l.data, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.a_out_r.data, gain=nn.init.calculate_gain('relu'))

        def forward(self, x, neigh_list):
            """
            x: [N, F] node features
            neigh_list: list of lists, neigh_list[i] = list of neighbor indices of i (includes self)
            Returns: logits [N, n_classes]
            """
            N = x.size(0)
            # -------- First multi-head GAT layer ----------
            # Linear projection
            x_drop = F.dropout(x, p=self.drop, training=self.training)
            Wh = torch.matmul(x_drop, self.W)          # [N, n_heads * hid_dim]
            Wh = Wh.view(N, self.n_heads, self.hid_dim)  # [N, n_heads, hid_dim]

            head_outputs = []
            for h in range(self.n_heads):
                Wh_h = Wh[:, h, :]                     # [N, hid_dim]
                # Attention scores: a_l^T Wh_i + a_r^T Wh_j
                attn_for_i = (Wh_h * self.a_l[h]).sum(dim=-1)  # [N]
                attn_for_j = (Wh_h * self.a_r[h]).sum(dim=-1)  # [N]

                out_h = torch.zeros_like(Wh_h)         # [N, hid_dim]
                for i in range(N):
                    node_neigh = neigh_list[i]
                    if not node_neigh:
                        continue
                    # e_ij = LeakyReLU(attn_i[i] + attn_j[j])
                    e = attn_for_i[i] + attn_for_j[node_neigh]   # [len(neigh)]
                    e = self.leakyrelu(e)
                    alpha = F.softmax(e, dim=0)
                    alpha_drop = F.dropout(alpha, p=self.drop, training=self.training)
                    # Weighted sum of neighbor features
                    weighted = (alpha_drop[:, None] * Wh_h[node_neigh]).sum(dim=0)
                    out_h[i] = weighted
                head_outputs.append(out_h)

            # Concatenate heads -> [N, n_heads * hid_dim]
            x_cat = torch.cat(head_outputs, dim=-1)
            x_elu = F.elu(x_cat)

            # -------- Second GAT layer (single head) ----------
            x_drop2 = F.dropout(x_elu, p=self.drop, training=self.training)
            Wh_out = torch.matmul(x_drop2, self.W_out)   # [N, n_classes]
            attn_i_out = (Wh_out * self.a_out_l).sum(dim=-1)  # [N]
            attn_j_out = (Wh_out * self.a_out_r).sum(dim=-1)  # [N]

            out = torch.zeros_like(Wh_out)               # [N, n_classes]
            for i in range(N):
                node_neigh = neigh_list[i]
                if not node_neigh:
                    continue
                e = attn_i_out[i] + attn_j_out[node_neigh]   # [len(neigh)]
                e = self.leakyrelu(e)
                alpha = F.softmax(e, dim=0)
                alpha_drop = F.dropout(alpha, p=self.drop, training=self.training)
                weighted = (alpha_drop[:, None] * Wh_out[node_neigh]).sum(dim=0)
                out[i] = weighted
            return out  # logits

    model = GAT(in_features, hidden_dim_per_head, num_heads, num_classes, dropout).to(torch.device('cpu'))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, neighbors)
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, neighbors)
            val_loss = criterion(val_logits[val_idx], labels[val_idx])
            _, val_pred = torch.max(val_logits[val_idx], dim=1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        final_logits = model(features, neighbors)
        # Predictions for all nodes (as required by nullius_io)
        preds = torch.argmax(final_logits, dim=1)  # [N] long

    # Output predictions via helper
    nio.write_predictions(preds.cpu().numpy())

    # Print final validation accuracy
    print(f'Best validation accuracy: {best_val_acc:.4f}')

if __name__ == '__main__':
    main()