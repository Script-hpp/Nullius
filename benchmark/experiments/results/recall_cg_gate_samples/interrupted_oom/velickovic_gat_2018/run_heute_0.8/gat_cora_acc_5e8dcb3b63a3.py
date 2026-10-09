

# Plan step -> implemented in: function or class name
# 1. Set SEED from env and seed numpy & torch -> set_seeds()
# 2. Load data via nullius_io -> load_data()
# 3. Model – 2-layer GAT -> GATLayer class, GATModel class
# 4. Optimizer: Adam -> defined in train()
# 5. Loss: CrossEntropyLoss on train_idx -> train()
# 6. Training: full-batch, early stopping on val loss -> train()
# 7. Validation each epoch -> train()
# 8. Test inference -> predict()
# 9. Write predictions via nullius_io -> main()

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

def set_seeds():
    seed = int(os.environ.get('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    d = nio.load()
    features = torch.from_numpy(d['features']).float()          # [N, F]
    edges = torch.from_numpy(d['edges']).long()                # [E, 2] (bidirectional)
    labels = torch.from_numpy(d['labels']).long()              # [N]
    train_idx = torch.from_numpy(d['train_idx']).long()
    val_idx = torch.from_numpy(d['val_idx']).long()
    test_idx = torch.from_numpy(d['test_idx']).long()
    return features, edges, labels, train_idx, val_idx, test_idx

class GATLayer(nn.Module):
    """
    Multi-head GAT layer.
    - in_features: input dimension per node
    - out_features: output dimension per head
    - heads: number of attention heads
    - concat: if True, concatenate head outputs; else average
    - dropout: dropout rate applied to features and attention coefficients
    """
    def __init__(self, in_features, out_features, heads=1, concat=True,
                 dropout=0.6, alpha=0.2, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.alpha = alpha

        # Weight for each head: (heads, in_features, out_features)
        self.W = nn.Parameter(torch.Tensor(heads, in_features, out_features))
        # Attention weight vector: (heads, 2*out_features)
        self.a = nn.Parameter(torch.Tensor(heads, 2 * out_features))
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x, edge_list):
        """
        x: (N, in_features)
        edge_list: tuple (src, dst) each LongTensor of length E
        Returns: node embeddings of shape (N, out_dim)
        """
        N = x.size(0)
        src, dst = edge_list
        E = src.size(0)

        # Apply dropout to input features
        x_drop = F.dropout(x, p=self.dropout, training=self.training)

        # Linear transformation per head: list of length heads, each (N, out_features)
        head_outputs = []
        for h in range(self.heads):
            # W[h] shape (in_features, out_features)
            h_out = x_drop @ self.W[h]          # (N, out_features)
            head_outputs.append(h_out)
        # head_outputs: list of (N, out_features)
        # Stack to (heads, N, out_features) for easier indexing
        head_outputs = torch.stack(head_outputs, dim=0)   # (H, N, F')

        # Compute attention scores for each edge and each head
        # Wh_src: (H, E, F'), Wh_dst: (H, E, F')
        Wh_src = head_outputs[:, src, :]   # (H, E, F')
        Wh_dst = head_outputs[:, dst, :]   # (H, E, F')
        # Concatenate: (H, E, 2*F')
        concat = torch.cat([Wh_src, Wh_dst], dim=-1)
        # Attention: (H, E) = LeakyReLU(concat * a_h)
        e = (concat * self.a.unsqueeze(1)).sum(dim=-1)   # (H, E)
        e = self.leakyrelu(e)                            # (H, E)

        # Softmax over source nodes (normalize per source)
        alpha = torch.zeros_like(e)
        # We'll process each head separately
        for h in range(self.heads):
            e_h = e[h]                     # (E,)
            # For each unique source node, compute softmax over its outgoing edges
            unique_src = torch.unique(src)
            for s in unique_src:
                mask = (src == s)
                e_src = e_h[mask]          # (num_edges_from_s,)
                alpha_src = F.softmax(e_src, dim=0)   # (num_edges_from_s,)
                alpha[h, mask] = alpha_src

        # Dropout on attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Message passing: aggregate weighted Wh_dst per head
        head_outs = []   # each will be (N, F')
        for h in range(self.heads):
            alpha_h = alpha[h].unsqueeze(-1)   # (E, 1)
            Wh_dst_h = head_outputs[h, dst, :] # (E, F')
            weighted = alpha_h * Wh_dst_h      # (E, F')
            # Sum inbound to each node
            out_h = torch.zeros(N, self.out_features, device=x.device)
            out_h.index_add_(0, dst, weighted) # (N, F')
            head_outs.append(out_h)

        if self.concat:
            # (N, heads * F')
            out = torch.cat(head_outs, dim=-1)
        else:
            # Average heads
            out = torch.stack(head_outs, dim=0).mean(dim=0)

        if self.bias is not None:
            out = out + self.bias
        return out

class GATModel(nn.Module):
    def __init__(self, nfeat, nhidden, nclass, heads=8, dropout=0.6):
        super().__init__()
        self.gat1 = GATLayer(
            nfeat,
            nhidden // heads,   # out_features per head so concat gives nhidden
            heads=heads,
            concat=True,
            dropout=dropout
        )
        self.gat2 = GATLayer(
            nhidden,
            nclass,
            heads=1,
            concat=False,
            dropout=dropout
        )
        self.dropout = dropout

    def forward(self, x, edge_list):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_list)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_list)
        return x

def train(model, features, labels, train_idx, val_idx, edge_list,
          lr=0.005, weight_decay=5e-4, max_epochs=10000, patience=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        logits = model(features, edge_list)
        loss = loss_fn(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, edge_list)
            val_loss = loss_fn(val_logits[val_idx], labels[val_idx])
            _, preds = torch.max(val_logits[val_idx], dim=1)
            acc = (preds == labels[val_idx]).float().mean().item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    # Return final validation accuracy from best state
    model.eval()
    with torch.no_grad():
        val_logits = model(features, edge_list)
        _, preds = torch.max(val_logits[val_idx], dim=1)
        acc = (preds == labels[val_idx]).float().mean().item()
    print(f"Validation Accuracy: {acc:.4f}")
    return acc

def predict(model, features, edge_list):
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_list)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
    return preds.cpu().numpy()

def main():
    set_seeds()
    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    src = edges[:, 0]
    dst = edges[:, 1]
    edge_list = (src, dst)

    N = features.size(0)
    nclass = int(labels.max().item() + 1)   # labels contain -1 for test, max is 6 -> 7 classes
    hidden = 64          # as per plan: heads=8, out_dim_per_head=8
    heads = 8
    dropout = 0.6

    model = GATModel(nfeat=features.size(1),
                     nhidden=hidden,
                     nclass=nclass,
                     heads=heads,
                     dropout=dropout)

    smoke = os.environ.get('SMOKE', '0') == '1'
    if smoke:
        # Run exactly 1 epoch (no early stopping needed)
        train(model, features, labels, train_idx, val_idx, edge_list,
              max_epochs=1, patience=100)
    else:
        train(model, features, labels, train_idx, val_idx, edge_list)

    pred = predict(model, features, edge_list)
    # nio.write_predictions expects predictions for ALL nodes (shape [N])
    nio.write_predictions(pred)

if __name__ == "__main__":
    main()