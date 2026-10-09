# Plan step -> Load data: load data via nullius_io
# Plan step -> Define GAT model: GAT class with two layers, multi-head attention
# Plan step -> Set up optimizer (Adam) and loss (CrossEntropy)
# Plan step -> Training loop with early stopping on validation loss
# Plan step -> Seed initialization from SEED environment variable
# Plan step -> Validation accuracy tracking and best model saving
# Plan step -> Final prediction and writing via nullius_io

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

    # ----- Build adjacency list (neighbors) -----
    neighbors = [[] for _ in range(N)]
    src = edges[:, 0]
    dst = edges[:, 1]
    for s, t in zip(src.tolist(), dst.tolist()):
        neighbors[s].append(t)

    # ----- Model hyperparameters -----
    in_features = features.size(1)
    hidden_dim_per_head = 8      # first layer output per head
    num_heads = 8                # first layer heads
    dropout = 0.6
    lr = 0.005
    weight_decay = 5e-4
    max_epochs = 10000
    patience = 100

    # ----- Smoke mode -----
    if os.environ.get('SMOKE') == '1':
        max_epochs = 1  # exactly one epoch, ignore patience

    # ----- GAT definition -----
    class GAT(nn.Module):
        def __init__(self, in_feat, hid_dim, n_heads, n_classes, drop=0.6, alpha=0.2):
            super().__init__()
            self.n_heads = n_heads
            self.hid_dim = hid_dim
            self.drop = drop
            self.leakyrelu = nn.LeakyReLU(alpha)

            # First layer: per-head weight and attention vector
            self.W = nn.ParameterList([
                nn.Parameter(torch.empty(in_feat, hid_dim)) for _ in range(n_heads)
            ])
            self.a = nn.ParameterList([
                nn.Parameter(torch.empty(2 * hid_dim)) for _ in range(n_heads)
            ])

            # Second layer: single head, input = concatenated heads
            self.W_out = nn.Parameter(torch.empty(hid_dim * n_heads, n_classes))
            self.a_out = nn.Parameter(torch.empty(2 * n_classes))

            self.reset_parameters()

        def reset_parameters(self):
            for w in self.W:
                nn.init.xavier_uniform_(w.data, gain=nn.init.calculate_gain('relu'))
            for a_vec in self.a:
                nn.init.xavier_uniform_(a_vec.data, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.W_out.data, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.a_out.data, gain=nn.init.calculate_gain('relu'))

            # Note: we rely on dropout for regularization; bias not used.

        def forward(self, x, neigh_list):
            """
            x: [N, F] node features
            neigh_list: list of lists, neigh_list[i] = list of neighbor indices of i (directed)
            Returns: logits [N, n_classes]
            """
            N = x.size(0)
            # Dropout on input features
            x_drop = F.dropout(x, p=self.drop, training=self.training)

            head_outputs = []
            for h in range(self.n_heads):
                W_h = self.W[h]
                a_h = self.a[h]
                # Linear transform
                z = torch.matmul(x_drop, W_h)          # [N, hid_dim]
                # Prepare output accumulator
                out_h = torch.zeros_like(z)

                for i in range(N):
                    # Include self-loop for attention
                    node_neigh = neigh_list[i] + [i]
                    z_i = z[i]
                    # Compute unnormalized attention scores
                    e_vals = []
                    for j in node_neigh:
                        z_j = z[j]
                        concat = torch.cat([z_i, z_j], dim=-1)   # [2*hid_dim]
                        e = self.leakyrelu((a_h * concat).sum(dim=-1))
                        e_vals.append(e)
                    e_tensor = torch.stack(e_vals) if e_vals else torch.empty(0)
                    if e_tensor.numel() > 0:
                        alpha = F.softmax(e_tensor, dim=0)
                        alpha_drop = F.dropout(alpha, p=self.drop, training=self.training)
                        # Weighted sum of neighbor transformed features
                        weighted = torch.sum(alpha_drop[:, None] * z[node_neigh], dim=0)
                        out_h[i] = weighted
                    else:
                        out_h[i] = torch.zeros_like(z_i)
                head_outputs.append(out_h)

            # Concatenate heads -> [N, hid_dim * n_heads]
            x_cat = torch.cat(head_outputs, dim=-1)
            x_elu = F.elu(x_cat)

            # Second layer (single head)
            x_drop2 = F.dropout(x_elu, p=self.drop, training=self.training)
            z_out = torch.matmul(x_drop2, self.W_out)     # [N, n_classes]

            out = torch.zeros_like(z_out)
            for i in range(N):
                node_neigh = neigh_list[i] + [i]
                z_i_out = z_out[i]
                e_vals = []
                for j in node_neigh:
                    z_j_out = z_out[j]
                    concat = torch.cat([z_i_out, z_j_out], dim=-1)   # [2*n_classes]
                    e = self.leakyrelu((self.a_out * concat).sum(dim=-1))
                    e_vals.append(e)
                e_tensor = torch.stack(e_vals) if e_vals else torch.empty(0)
                if e_tensor.numel() > 0:
                    alpha = F.softmax(e_tensor, dim=0)
                    alpha_drop = F.dropout(alpha, p=self.drop, training=self.training)
                    weighted = torch.sum(alpha_drop[:, None] * z_out[node_neigh], dim=0)
                    out[i] = weighted
                else:
                    out[i] = torch.zeros_like(z_i_out)
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
            val_loss = criterion(logits[val_idx], labels[val_idx])  # using same logits? recompute below
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
