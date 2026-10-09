

# Plan step -> Load data via nullius_io: main()
# Plan step -> Define GAT model: GAT class
# Plan step -> Set up optimizer (Adam) and loss (CrossEntropy): in training setup
# Plan step -> Training loop with early stopping on validation loss: training loop
# Plan step -> Seed initialization from SEED environment variable: start of main()
# Plan step -> Validation accuracy tracking and best model saving: training loop (tracks loss, also computes acc)
# Plan step -> Final prediction and writing via nullius_io: end of main()

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
    data = nio.load()  # dict with keys: features, edges, labels, train_idx, val_idx, test_idx
    features = torch.from_numpy(data['features']).float()          # [N, F]
    edges = torch.from_numpy(data['edges']).long()                 # [E, 2]
    labels = torch.from_numpy(data['labels']).long()               # [N]
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx = torch.from_numpy(data['val_idx']).long()
    test_idx = torch.from_numpy(data['test_idx']).long()

    N = features.size(0)
    num_classes = int(labels.max().item() + 1)  # labels are 0..6 for labeled nodes

    # ----- Build adjacency matrix (including self-loops) -----
    adj = torch.zeros(N, N, dtype=torch.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    adj += torch.eye(N)  # add self-loops
    # Ensure binary (0/1) – some entries could become 2 if edge already existed as self-loop, clamp
    adj = torch.clamp(adj, min=0.0, max=1.0)

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
        def __init__(self, in_feat, hid_dim, n_heads, n_classes, drop=0.6, alpha=0.2, adj_matrix=None):
            super().__init__()
            self.n_heads = n_heads
            self.hid_dim = hid_dim
            self.drop = drop
            self.n_classes = n_classes
            self.leakyrelu = nn.LeakyReLU(alpha)

            # Store adjacency matrix (dense, N x N) as buffer
            if adj_matrix is None:
                raise ValueError("Adjacency matrix must be provided")
            self.register_buffer('adj', adj_matrix)  # [N, N]

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
                # Normal init to avoid fan_in/fan_out issues
                nn.init.normal_(a_vec.data, mean=0.0, std=0.1)
            nn.init.xavier_uniform_(self.W_out.data, gain=nn.init.calculate_gain('relu'))
            nn.init.normal_(self.a_out.data, mean=0.0, std=0.1)

        def forward(self, x):
            """
            x: [N, F] node features
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

                # Split attention vector
                a_l = a_h[:self.hid_dim]
                a_r = a_h[self.hid_dim:]

                # Compute attention scores: e_ij = LeakyReLU(a_l^T z_i + a_r^T z_j)
                el = (z * a_l).sum(dim=1, keepdim=True)   # [N,1]
                er = (z * a_r).sum(dim=1, keepdim=True)   # [N,1]
                e = el + er.t()                           # [N,N]
                e = self.leakyrelu(e)

                # Mask non-neighbors (including self-loops already in adj)
                # Set e where adj==0 to a large negative value before softmax
                e_masked = e * self.adj + -1e9 * (1.0 - self.adj)

                # Softmax over j (neighbor dimension)
                alpha = torch.softmax(e_masked, dim=1)    # [N,N]
                alpha_drop = F.dropout(alpha, p=self.drop, training=self.training)

                # Aggregate neighbors' features
                z_agg = torch.matmul(alpha_drop, z)       # [N, hid_dim]
                head_outputs.append(z_agg)

            # Concatenate heads -> [N, hid_dim * n_heads]
            x_cat = torch.cat(head_outputs, dim=-1)
            x_elu = F.elu(x_cat)

            # Second layer (single head)
            x_drop2 = F.dropout(x_elu, p=self.drop, training=self.training)
            z_out = torch.matmul(x_drop2, self.W_out)     # [N, n_classes]

            # Attention for second layer
            a_out = self.a_out
            a_l_out = a_out[:self.n_classes]
            a_r_out = a_out[self.n_classes:]
            el_out = (z_out * a_l_out).sum(dim=1, keepdim=True)   # [N,1]
            er_out = (z_out * a_r_out).sum(dim=1, keepdim=True)   # [N,1]
            e_out = el_out + er_out.t()                           # [N,N]
            e_out = self.leakyrelu(e_out)
            e_masked_out = e_out * self.adj + -1e9 * (1.0 - self.adj)
            alpha_out = torch.softmax(e_masked_out, dim=1)        # [N,N]
            alpha_out_drop = F.dropout(alpha_out, p=self.drop, training=self.training)
            out = torch.matmul(alpha_out_drop, z_out)             # [N, n_classes]
            return out  # logits

    model = GAT(in_features, hidden_dim_per_head, num_heads, num_classes,
                dropout, adj_matrix=adj).to(torch.device('cpu'))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features)
            val_loss = criterion(val_logits[val_idx], labels[val_idx])
            _, val_pred = torch.max(val_logits[val_idx], dim=1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
        final_logits = model(features)
        # Predictions for all nodes (as required by nullius_io)
        preds = torch.argmax(final_logits, dim=1)  # [N] long

    # Output predictions via helper
    nio.write_predictions(preds.cpu().numpy())

    # Print final validation accuracy (from last epoch)
    print(f'Best validation loss: {best_val_loss:.4f}')

if __name__ == '__main__':
    main()