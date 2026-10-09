# Plan step -> implemented in: load_data
# Plan step -> implemented in: set_seeds
# Plan step -> implemented in: GATConv
# Plan step -> implemented in: GAT
# Plan step -> implemented in: configure_optimizer
# Plan step -> implemented in: compute_loss
# Plan step -> implemented in: validate
# Plan step -> implemented in: train
# Plan step -> implemented in: test_and_write

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

def load_data():
    data = np.load('/data/cora_lcc_public.npz')
    features = torch.from_numpy(data['features']).float()          # [N, F_in]
    edges = torch.from_numpy(data['edges']).long()                # [E, 2] (directed, both directions)
    labels = torch.from_numpy(data['labels']).long()              # [N]
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx = torch.from_numpy(data['val_idx']).long()
    test_idx = torch.from_numpy(data['test_idx']).long()
    return features, edges, labels, train_idx, val_idx, test_idx

def set_seeds():
    seed = int(os.environ.get('SEED', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

class GATConv(nn.Module):
    """
    Single GAT layer supporting multiple heads.
    If concat=True, output shape is [N, heads * out_dim];
    else output shape is [N, out_dim] (averaged over heads).
    """
    def __init__(self, in_dim, out_dim, heads, concat, dropout, activation):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.activation = activation

        # Parameters per head: weight matrix W and attention vector a
        self.W = nn.Parameter(torch.Tensor(heads, out_dim, in_dim))
        self.a = nn.Parameter(torch.Tensor(heads, 2 * out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_src, edge_dst):
        """
        x: [N, in_dim]
        edge_src, edge_dst: [E] (source -> target)
        Returns: [N, out_dim * heads] if concat else [N, out_dim]
        """
        N = x.size(0)
        # Dropout on input features
        if self.training:
            x = F.dropout(x, p=self.dropout, training=True)

        # Linear transformation per head: [heads, N, out_dim]
        # x @ W[h]^T for each head h
        h_prime = torch.einsum('ni,hio->hno', x, self.W)  # [heads, N, out_dim]

        # Compute attention scores for each edge and each head
        src_h = h_prime[:, edge_src, :]          # [heads, E, out_dim]
        dst_h = h_prime[:, edge_dst, :]          # [heads, E, out_dim]
        concat = torch.cat([src_h, dst_h], dim=-1)  # [heads, E, 2*out_dim]
        # e = LeakyReLU( concat * a ) summed over last dim
        e = F.leaky_relu((concat * self.a.unsqueeze(1)).sum(dim=-1))  # [heads, E]

        # Softmax over edges sharing same source node (per head)
        exp_e = torch.exp(e)                     # [heads, E]
        # sum over source nodes per head
        edge_src_exp = edge_src.unsqueeze(0).expand(self.heads, -1)  # [heads, E]
        sum_exp = torch.zeros(self.heads, N, device=x.device, dtype=x.dtype)
        sum_exp.scatter_add_(1, edge_src_exp, exp_e)               # [heads, N]
        # normalize
        alpha = exp_e / (sum_exp.gather(1, edge_src_exp) + 1e-16)  # [heads, E]

        # Dropout on attention coefficients
        if self.training:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # Message: alpha * src_h -> [heads, E, out_dim]
        msg = alpha.unsqueeze(-1) * src_h      # [heads, E, out_dim]

        # Aggregate messages to destination nodes per head
        out = torch.zeros(self.heads, N, self.out_dim, device=x.device, dtype=x.dtype)
        edge_dst_exp = edge_dst.unsqueeze(0).expand(self.heads, -1)  # [heads, E]
        # Expand index to match msg shape for scatter_add
        idx = edge_dst_exp.unsqueeze(-1).expand(-1, -1, self.out_dim)  # [heads, E, out_dim]
        out.scatter_add_(1, idx, msg)                                   # [heads, N, out_dim]

        # Merge heads
        if self.concat:
            out = out.transpose(0, 1).contiguous().view(N, -1)   # [N, heads*out_dim]
        else:
            out = out.mean(dim=0)                               # [N, out_dim]

        if self.activation is not None:
            out = self.activation(out)
        return out

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads, dropout):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads, concat=True,
                             dropout=dropout, activation=nn.ELU())
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1,
                             concat=False, dropout=dropout, activation=None)

    def forward(self, x, edge_src, edge_dst):
        x = self.conv1(x, edge_src, edge_dst)
        x = self.conv2(x, edge_src, edge_dst)
        return x

def configure_optimizer(model, lr=0.005, weight_decay=5e-4):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def compute_loss(logits, labels, idx):
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(logits[idx], labels[idx])

def validate(model, features, edge_src, edge_dst, labels, val_idx):
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_src, edge_dst)
        loss = compute_loss(logits, labels, val_idx)
        preds = logits[val_idx].argmax(dim=1)
        acc = (preds == labels[val_idx]).float().mean().item()
    return loss, acc

def train(model, optimizer, features, edge_src, edge_dst, labels,
          train_idx, val_idx, max_epochs=10000, patience=100, smoke=False):
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, edge_src, edge_dst)
        loss = compute_loss(logits, labels, train_idx)
        loss.backward()
        optimizer.step()

        # Validation
        val_loss, val_acc = validate(model, features, edge_src, edge_dst, labels, val_idx)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if smoke:
            # Smoke mode: only 1 epoch
            break

        if epochs_no_improve >= patience:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    # Return best validation accuracy (compute again)
    _, best_val_acc = validate(model, features, edge_src, edge_dst, labels, val_idx)
    return best_val_acc

def test_and_write(model, features, edge_src, edge_dst):
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_src, edge_dst)
        preds = logits.argmax(dim=1)  # [N] integer predictions
    nio.write_predictions(preds)

def main():
    set_seeds()
    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    edge_src = edges[:, 0]
    edge_dst = edges[:, 1]

    num_features = features.shape[1]
    num_classes = len(torch.unique(labels[labels != -1]))  # should be 7
    model = GAT(in_dim=num_features, hidden_dim=8, out_dim=num_classes,
                heads=8, dropout=0.6)

    optimizer = configure_optimizer(model)

    smoke = os.environ.get('SMOKE') == '1'
    val_acc = train(model, optimizer, features, edge_src, edge_dst, labels,
                    train_idx, val_idx, max_epochs=10000, patience=100,
                    smoke=smoke)

    print(f'Validation accuracy: {val_acc:.4f}')
    test_and_write(model, features, edge_src, edge_dst)

if __name__ == '__main__':
    main()
