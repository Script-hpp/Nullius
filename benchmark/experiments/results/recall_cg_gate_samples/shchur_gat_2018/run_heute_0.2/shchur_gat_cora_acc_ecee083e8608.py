# Plan step -> implemented in: function or class name
# Load data -> implemented in: load_data
# Model: 2-layer GAT -> implemented in: class GAT
# Optimizer -> implemented in: configure_optimizer
# Loss -> implemented in: training loop (cross entropy)
# Training -> implemented in: train function
# Seed -> implemented in: set_seed
# Validation -> implemented in: validate function
# Testing -> implemented in: predict function
# Runtime limit -> implemented via max_epochs and early stopping
# Smoke mode -> handled in main conditional

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio


def load_data():
    """Load the Cora LCC dataset and return torch tensors."""
    data = nio.load()
    features = torch.from_numpy(data['features']).float()
    edges = torch.from_numpy(data['edges']).long()          # [E, 2]
    labels = torch.from_numpy(data['labels']).long()
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx = torch.from_numpy(data['val_idx']).long()
    test_idx = torch.from_numpy(data['test_idx']).long()
    return features, edges, labels, train_idx, val_idx, test_idx


def set_seed(seed: int):
    """Seed numpy and torch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


class GATLayer(nn.Module):
    """
    Single GAT layer supporting multi-head attention.
    If concat=True, outputs are concatenated; otherwise averaged.
    """
    def __init__(self, in_features, out_features, n_heads, concat=True,
                 dropout=0.6, negative_slope=0.2):
        super().__init__()
        self.concat = concat
        self.n_heads = n_heads
        self.out_features = out_features
        self.dropout = dropout
        self.negative_slope = negative_slope

        # Weight matrices for each head: [n_heads, in_features, out_features]
        self.W = nn.Parameter(torch.empty(n_heads, in_features, out_features))
        # Attention parameters: [n_heads, 2 * out_features]
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, edge_index):
        """
        h: [N, in_features]
        edge_index: [2, E] (directed edges, includes both directions)
        Returns: [N, n_heads * out_features] if concat else [N, out_features]
        """
        N = h.size(0)
        # Dropout on input features
        h_drop = F.dropout(h, self.dropout, training=self.training)

        # Linear transformation: [N, n_heads, out_features]
        Wh = torch.einsum('ni,hio->nh o', h_drop, self.W)  # (N, heads, out)

        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]

        # Wh for source and target nodes: [E, heads, out]
        Wh_src = Wh[src]
        Wh_dst = Wh[dst]

        # Prepare attention input: [E, heads, 2*out]
        a_input = torch.cat([Wh_src, Wh_dst], dim=-1)

        # Compute unnormalized attention scores: [E, heads]
        e = torch.einsum('ehi,hi->eh', a_input, self.a)
        e = F.leaky_relu(e, self.negative_slope)

        # Softmax per source node
        exp_e = torch.exp(e)  # [E, heads]
        sum_exp = torch.zeros(N, self.n_heads, device=h.device)
        for head in range(self.n_heads):
            sum_exp[:, head].index_add_(0, src, exp_e[:, head])
        alpha = exp_e / (sum_exp[src] + 1e-16)  # [E, heads]

        # Dropout on attention coefficients
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # Aggregate messages: weighted sum of Wh_src per head (message from src to dst)
        if self.concat:
            # Output per head, then concatenate
            head_outputs = []
            for head in range(self.n_heads):
                alpha_head = alpha[:, head].unsqueeze(-1)          # [E,1]
                Wh_src_head = Wh_src[:, head, :]                  # [E, out]
                weighted = Wh_src_head * alpha_head               # [E, out]
                out_head = torch.zeros(N, self.out_features, device=h.device)
                out_head.index_add_(0, dst, weighted)             # accumulate to destination
                head_outputs.append(out_head)
            out = torch.cat(head_outputs, dim=-1)                 # [N, heads*out]
        else:
            # Average over heads
            out_accum = torch.zeros(N, self.out_features, device=h.device)
            for head in range(self.n_heads):
                alpha_head = alpha[:, head].unsqueeze(-1)
                Wh_src_head = Wh_src[:, head, :]
                weighted = Wh_src_head * alpha_head
                out_head = torch.zeros(N, self.out_features, device=h.device)
                out_head.index_add_(0, dst, weighted)
                out_accum += out_head
            out = out_accum / self.n_heads                       # [N, out]
        return out


class GAT(nn.Module):
    """2-layer GAT as described in the plan."""
    def __init__(self, nfeat, nhidden, nclass, nheads, dropout):
        super().__init__()
        self.dropout = dropout
        # First layer: multi-head, concat=True
        self.attention = GATLayer(
            nfeat, nhidden, n_heads=nheads,
            concat=True, dropout=dropout
        )
        # Second layer: single head, concat=False (output logits)
        self.out_att = GATLayer(
            nhidden * nheads, nclass, n_heads=1,
            concat=False, dropout=dropout
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.attention(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, edge_index)
        return x  # logits


def configure_optimizer(model, lr=0.005, weight_decay=5e-4):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def validate(model, features, edge_index, labels, idx):
    """Return loss and accuracy on the given index set."""
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index)
        loss = F.cross_entropy(logits[idx], labels[idx])
        pred = logits[idx].argmax(dim=-1)
        acc = (pred == labels[idx]).float().mean().item()
    return loss.item(), acc


def train(model, optimizer, features, edge_index, labels,
          train_idx, val_idx, max_epochs, patience):
    """Training loop with early stopping on validation loss."""
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, edge_index)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        val_loss, val_acc = validate(model, features, edge_index, labels, val_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

        if epoch % 10 == 0:
            print(f'Epoch {epoch:04d} | Loss {loss.item():.4f} | '
                  f'Val loss {val_loss:.4f} | Val acc {val_acc:.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, features, edge_index):
    """Return predicted class for every node (as numpy array)."""
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index)
        preds = logits.argmax(dim=-1)
    return preds.cpu().numpy()


def main():
    # ----- Seed -----
    seed = int(os.environ.get('SEED', 0))
    set_seed(seed)

    # ----- Load data -----
    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    edge_index = edges.t().contiguous()  # [2, E]

    # ----- Model -----
    model = GAT(
        nfeat=features.shape[1],
        nhidden=8,
        nclass=7,
        nheads=8,
        dropout=0.6
    )

    # ----- Optimizer -----
    optimizer = configure_optimizer(model)

    # ----- Training configuration -----
    max_epochs = 10000
    patience = 100
    if os.environ.get('SMOKE') == '1':
        max_epochs = 1
        patience = 1  # effectively disables early stopping for smoke run

    # ----- Train -----
    model = train(
        model, optimizer, features, edge_index, labels,
        train_idx, val_idx, max_epochs, patience
    )

    # ----- Final validation accuracy -----
    _, final_val_acc = validate(model, features, edge_index, labels, val_idx)
    print(f'Final validation accuracy: {final_val_acc:.4f}')

    # ----- Predictions for all nodes -----
    preds_all = predict(model, features, edge_index)
    nio.write_predictions(preds_all)


if __name__ == '__main__':
    main()
