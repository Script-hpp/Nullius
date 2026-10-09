

# Plan step 1 -> implemented in: set_seeds()
# Plan step 2 -> implemented in: load_data()
# Plan step 3 -> implemented in: GATLayer and GATModel classes
# Plan step 4 -> implemented in: configure_optimizer()
# Plan step 5 -> implemented in: compute_loss()
# Plan step 6 -> implemented in: train_loop()
# Plan step 7 -> implemented in: early_stopping()
# Plan step 8 -> implemented in: predict_test()
# Plan step 9 -> implemented in: write_predictions_via_helper()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio


def set_seeds():
    seed = int(os.environ.get('SEED', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    d = nio.load()
    features = torch.from_numpy(d['features']).float()          # [N, F]
    edges = torch.from_numpy(d['edges']).long()                # [E, 2] (directed, both ways)
    labels = torch.from_numpy(d['labels']).long()              # [N]
    train_idx = torch.from_numpy(d['train_idx']).long()
    val_idx = torch.from_numpy(d['val_idx']).long()
    test_idx = torch.from_numpy(d['test_idx']).long()
    edge_index = edges.t().contiguous()                        # [2, E]
    return features, edge_index, labels, train_idx, val_idx, test_idx


class GATLayer(nn.Module):
    """
    Single GAT layer (multi-head, optional concatenation).
    Implements the standard GAT mechanism with softmax over incoming edges.
    """
    def __init__(self, in_features, out_features, heads=1, concat=True,
                 dropout=0.6, attn_dropout=0.6, negative_slope=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.negative_slope = negative_slope

        # Linear transformation shared across heads
        self.W = nn.Linear(in_features, out_features * heads, bias=False)
        # Attention weights: one vector per head
        self.attn = nn.Parameter(torch.empty(size=(heads, 2 * out_features)))

        self.leakyrelu = nn.LeakyReLU(self.negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.xavier_uniform_(self.attn, gain=gain)

    def forward(self, x, edge_index):
        """
        x: [N, in_features]
        edge_index: [2, E] where edge_index[0] = source nodes, edge_index[1] = target nodes
        """
        # Apply dropout to input features
        x = F.dropout(x, p=self.dropout, training=self.training)

        N = x.size(0)
        # Linear projection
        h = self.W(x)                              # [N, heads * out_features]
        h = h.view(N, self.heads, self.out_features)  # [N, heads, out_features]

        src = edge_index[0]
        dst = edge_index[1]

        # Features of source and target nodes per head
        h_src = h[src]      # [E, heads, out_features]
        h_dst = h[dst]      # [E, heads, out_features]

        # Concatenate for attention computation
        h_cat = torch.cat([h_src, h_dst], dim=-1)  # [E, heads, 2*out_features]

        # Attention scores (before softmax)
        e = (h_cat * self.attn).sum(dim=-1)        # [E, heads]
        e = self.leakyrelu(e)

        # Compute attention coefficients per head (softmax over incoming edges to each node)
        alpha = torch.zeros_like(e)
        for h_i in range(self.heads):
            e_h = e[:, h_i]                         # [E]
            exp_e_h = torch.exp(e_h)
            # Sum over incoming edges to each destination node
            sum_exp = torch.zeros(N, device=x.device)
            sum_exp.index_add_(0, dst, exp_e_h)     # [N]
            # Normalize
            alpha_h = exp_e_h / (sum_exp[dst] + 1e-16)
            alpha[:, h_i] = alpha_h

        # Dropout on attention coefficients
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

        # Aggregate neighbor features per head
        out_heads = []
        for h_i in range(self.heads):
            alpha_h = alpha[:, h_i].unsqueeze(-1)   # [E, 1]
            h_src_h = h_src[:, h_i, :]              # [E, out_features]
            # Message from source to target
            msg = alpha_h * h_src_h                  # [E, out_features]
            # Accumulate messages per target node
            out_h = torch.zeros(N, self.out_features, device=x.device)
            out_h.index_add_(0, dst, msg)            # [N, out_features]
            out_heads.append(out_h)

        if self.concat:
            out = torch.cat(out_heads, dim=-1)       # [N, heads * out_features]
        else:
            out = torch.stack(out_heads, dim=0).mean(dim=0)  # [N, out_features]

        return out


class GATModel(nn.Module):
    def __init__(self, nfeat, nhidden, nclass, heads=8, dropout=0.6):
        super().__init__()
        self.gat1 = GATLayer(
            in_features=nfeat,
            out_features=nhidden // heads,  # per-head output so that concat gives nhidden
            heads=heads,
            concat=True,
            dropout=dropout,
            attn_dropout=dropout
        )
        self.gat2 = GATLayer(
            in_features=nhidden,
            out_features=nclass,
            heads=1,
            concat=False,
            dropout=dropout,
            attn_dropout=dropout
        )
        self.elu = nn.ELU()

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.elu(x)
        x = self.gat2(x, edge_index)
        return x  # logits


def configure_optimizer(model, lr=0.005, weight_decay=5e-4):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def compute_loss(logits, labels, idx):
    return F.cross_entropy(logits[idx], labels[idx])


def train_loop(model, optimizer, features, edge_index, labels,
               train_idx, val_idx, max_epochs, patience):
    best_val_loss = float('inf')
    patience_cnt = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, edge_index)
        loss_train = compute_loss(logits, labels, train_idx)
        loss_train.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits_val = model(features, edge_index)
            loss_val = compute_loss(logits_val, labels, val_idx)

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_test(model, features, edge_index, test_idx):
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index)
        pred = logits[test_idx].argmax(dim=1)
    return pred


def write_predictions_via_helper(logits_all):
    # Helper expects logits [N, num_classes] or class predictions [N]
    nio.write_predictions(logits_all.detach())


def main():
    set_seeds()
    features, edge_index, labels, train_idx, val_idx, test_idx = load_data()

    nfeat = features.shape[1]
    nclass = int(labels.max().item() + 1)  # labels are 0..6 for labeled nodes
    nhidden = 64
    heads = 8

    model = GATModel(nfeat=nfeat, nhidden=nhidden, nclass=nclass, heads=heads, dropout=0.6)
    optimizer = configure_optimizer(model, lr=0.005, weight_decay=5e-4)

    smoke = os.environ.get('SMOKE') == '1'
    max_epochs = 1 if smoke else 10000
    patience = 100

    model = train_loop(
        model, optimizer, features, edge_index, labels,
        train_idx, val_idx, max_epochs, patience
    )

    # Final validation accuracy
    model.eval()
    with torch.no_grad():
        logits_all = model(features, edge_index)
        pred_val = logits_all[val_idx].argmax(dim=1)
        acc = (pred_val == labels[val_idx]).float().mean().item()
        print(f'Validation accuracy: {acc:.4f}')

    # Write predictions for all nodes (logits)
    write_predictions_via_helper(logits_all)


if __name__ == '__main__':
    main()