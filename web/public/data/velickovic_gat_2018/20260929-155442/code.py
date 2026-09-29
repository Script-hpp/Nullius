import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def main():
    # ----- Seed -----
    seed = int(os.environ.get('SEED', 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ----- Load data -----
    data = np.load('/data/cora_public.npz')
    features = torch.FloatTensor(data['features'])          # [N, F]
    edges = torch.LongTensor(data['edges'])                 # [E, 2] (directed, symmetric)
    labels = torch.LongTensor(data['labels'])               # [N]
    train_idx = torch.LongTensor(data['train_idx'])
    val_idx = torch.LongTensor(data['val_idx'])
    test_idx = torch.LongTensor(data['test_idx'])

    N = features.shape[0]
    num_classes = int(labels.max().item()) + 1  # labels have -1 for test, but max is 6

    # ----- Build adjacency (with self-loops) -----
    adj = torch.zeros(N, N, dtype=torch.bool)
    adj[edges[:, 0], edges[:, 1]] = True
    adj.fill_diagonal_(True)   # add self-loops

    smoke = os.environ.get('SMOKE') == '1'

    # ----- GAT Layer -----
    class GATLayer(nn.Module):
        def __init__(self, in_features, out_features, n_heads, concat=True,
                     dropout=0.6, alpha=0.2, add_skip=False):
            super().__init__()
            self.n_heads = n_heads
            self.concat = concat
            self.dropout = dropout
            self.alpha = alpha
            self.out_features = out_features // n_heads if concat else out_features
            self.add_skip = add_skip

            # Weight matrices for each head
            self.W = nn.Parameter(torch.empty(n_heads, in_features, self.out_features))
            # Attention vectors (a) for each head
            self.a = nn.Parameter(torch.empty(n_heads, 2 * self.out_features, 1))

            if add_skip:
                self.skip_weight = nn.Parameter(torch.empty(in_features,
                                                          n_heads * self.out_features if concat else self.out_features))
            else:
                self.register_parameter('skip_weight', None)

            self.leakyrelu = nn.LeakyReLU(self.alpha)
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.W.data, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.a.data, gain=nn.init.calculate_gain('relu'))
            if self.skip_weight is not None:
                nn.init.xavier_uniform_(self.skip_weight.data, gain=nn.init.calculate_gain('relu'))

        def forward(self, h, adj):
            if self.training:
                h = F.dropout(h, p=self.dropout, training=True)

            N = h.size(0)
            # Linear transformation: h_prime shape [N, n_heads, out_features]
            h_prime = torch.einsum('ij,hjk->hik', h, self.W)  # [n_heads, N, out]
            h_prime = h_prime.permute(1, 0, 2)                # [N, n_heads, out]

            # Prepare attention
            a1 = self.a[:, :self.out_features, :]   # [n_heads, out, 1]
            a2 = self.a[:, self.out_features:, :]   # [n_heads, out, 1]
            a1_s = a1.squeeze(-1)                   # [n_heads, out]
            a2_s = a2.squeeze(-1)                   # [n_heads, out]

            # Compute e_ij = a1^T h_i + a2^T h_j per head
            e1 = torch.einsum('nhi,ho->nh', h_prime, a1_s)  # [N, n_heads]
            e2 = torch.einsum('nhi,ho->nh', h_prime, a2_s)  # [N, n_heads]
            e1 = e1.transpose(0, 1)                          # [n_heads, N]
            e2 = e2.transpose(0, 1)                          # [n_heads, N]
            e = e1[:, :, None] + e2[:, None, :]              # [n_heads, N, N]
            e = self.leakyrelu(e)

            # Mask with adjacency
            mask = adj.unsqueeze(0)          # [1, N, N]
            e = e.masked_fill(~mask, float('-inf'))

            # Attention coefficients
            alpha = F.softmax(e, dim=-1)     # [n_heads, N, N]
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            # Weighted aggregation
            h_prime_trans = h_prime.transpose(0, 1)      # [n_heads, N, out]
            weighted = torch.bmm(alpha, h_prime_trans)   # [n_heads, N, out]
            weighted = weighted.transpose(0, 1)          # [N, n_heads, out]

            if self.concat:
                out = weighted.reshape(N, -1)            # [N, n_heads*out]
                if self.skip_weight is not None:
                    skip = h @ self.skip_weight
                    out = out + skip
                out = F.elu(out)
            else:
                out = weighted.mean(dim=1)               # [N, out]
                if self.skip_weight is not None:
                    skip = h @ self.skip_weight
                    out = out + skip
            return out

    # ----- Model -----
    class GAT(nn.Module):
        def __init__(self, nfeat, nhidden, nclass, nheads, dropout=0.6):
            super().__init__()
            self.att1 = GATLayer(nfeat, nhidden * nheads, nheads,
                                 concat=True, dropout=dropout, add_skip=True)
            self.att2 = GATLayer(nhidden * nheads, nclass, 1,
                                 concat=False, dropout=dropout, add_skip=False)

        def forward(self, x, adj):
            x = self.att1(x, adj)
            x = self.att2(x, adj)
            return x

    model = GAT(nfeat=features.shape[1],
                nhidden=8,          # 8 dims per head -> 8*8=64
                nclass=num_classes,
                nheads=8,
                dropout=0.6)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.005,
                                 weight_decay=0.0005)

    # ----- Training loop -----
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience = 100
    wait = 0
    max_epochs = 10000

    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        logits = model(features, adj)
        loss_train = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss_train.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_v = model(features, adj)
            loss_val = F.cross_entropy(logits_v[val_idx], labels[val_idx])
            pred_v = logits_v[val_idx].argmax(dim=-1)
            acc_val = (pred_v == labels[val_idx]).float().mean().item()
        model.train()

        # Early stopping logic (improve both)
        if acc_val > best_val_acc and loss_val.item() < best_val_loss:
            best_val_acc = acc_val
            best_val_loss = loss_val.item()
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if wait >= patience:
            break

        if smoke and epoch >= 0:  # smoke mode runs exactly 1 epoch
            break

    # Restore best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_v = model(features, adj)
        acc_val = (logits_v[val_idx].argmax(dim=-1) == labels[val_idx]).float().mean().item()
    print(f'{acc_val:.4f}')

    # ----- Test predictions -----
    with torch.no_grad():
        logits_t = model(features, adj)
        if smoke:
            subset = test_idx[:20]
            preds = logits_t[subset].argmax(dim=-1).cpu().numpy()
            ids = subset.cpu().numpy()
        else:
            preds = logits_t[test_idx].argmax(dim=-1).cpu().numpy()
            ids = test_idx.cpu().numpy()

    out_path = '/output/predictions.csv'
    with open(out_path, 'w') as f:
        f.write('id,prediction\n')
        for idx, p in zip(ids, preds):
            f.write(f'{int(idx)},{int(p)}\n')


if __name__ == '__main__':
    main()
