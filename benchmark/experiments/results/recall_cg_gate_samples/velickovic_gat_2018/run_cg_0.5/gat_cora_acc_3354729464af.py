

# Plan step -> implemented in: set_seed
# Plan step -> implemented in: load_and_preprocess
# Plan step -> implemented in: build_adjacency_bias
# Plan step -> implemented in: GATModel
# Plan step -> implemented in: configure_optimizer
# Plan step -> implemented in: train_model
# Plan step -> implemented in: predict
# Plan step -> implemented in: write_predictions

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {"lr": 0.005, "weight_decay": 0.0005, "epochs": 100000, "batch_size": 2, "dropout": 0.6, "patience": 100, "heads": 8}
# Additional hyperparameters derived from the plan
CONFIG["hidden_per_head"] = 8          # F' per attention head in the first layer
CONFIG["output_heads"] = 1             # single head for the output layer
CONFIG["num_classes"] = 7              # Cora has 7 classes

def set_seed():
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_and_preprocess():
    d = nio.load()
    features = d['features'].astype(np.float32)          # [N, F]
    labels = d['labels'].astype(np.int64)               # [N]
    train_idx = d['train_idx']
    val_idx = d['val_idx']
    test_idx = d['test_idx']
    edges = d['edges'].astype(np.int64)                 # [E, 2] (directed, both directions)

    # Row‑normalize features (as in preprocess_features)
    rowsum = features.sum(axis=1, keepdims=True)
    r_inv = np.power(rowsum, -1.0)
    r_inv[np.isinf(r_inv)] = 0.0
    features_norm = features * r_inv

    features_tensor = torch.from_numpy(features_norm)
    labels_tensor = torch.from_numpy(labels)

    num_nodes = features.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True

    # For training/validation we need labels where they are known; unknown labels stay -1
    label_long = torch.full((num_nodes,), -1, dtype=torch.long)
    label_long[train_idx] = labels_tensor[train_idx]
    label_long[val_idx] = labels_tensor[val_idx]

    return (features_tensor, labels_tensor, train_idx, val_idx, test_idx,
            edges, train_mask, val_mask, test_mask, label_long, num_nodes)

def build_adjacency_bias(edges, num_nodes, batch_size):
    """
    Returns:
        edge_index_batch: LongTensor of shape [2, B*E] containing edge indices
                          with offsets for each graph in the batch.
        Also adds self‑loops.
    """
    # Convert edges to torch and transpose to shape [2, E]
    edge_index = torch.from_numpy(edges).T   # [2, E]
    # Add self‑loops
    self_loop = torch.arange(num_nodes, dtype=torch.long)
    self_loop = torch.stack([self_loop, self_loop], dim=0)   # [2, N]
    edge_index = torch.cat([edge_index, self_loop], dim=1)   # [2, E+num_nodes]

    if batch_size == 1:
        return edge_index
    # Create batch dimension by offsetting node indices
    offsets = torch.arange(batch_size, dtype=torch.long) * num_nodes
    edge_index_list = [edge_index + offset.view(1, -1) for offset in offsets]
    edge_index_batch = torch.cat(edge_index_list, dim=1)
    return edge_index_batch

class AttnHead(nn.Module):
    """
    Single attention head used inside a multi‑head layer.
    Mirrors the TensorFlow implementation: dropout on input and on the
    linearly transformed features, leakyReLU attention, softmax with
    optional dropout on attention coefficients.
    """
    def __init__(self, in_features, out_features, dropout, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, edge_index):
        """
        x:          [BN, F]  (flattened batch*nodes)
        edge_index: [2, E_total]  (batched edge indices)
        Returns:    [BN, out_features]
        """
        if self.dropout:
            x = F.dropout(x, self.dropout, training=self.training)

        # Linear transformation
        Wh = self.W(x)                     # [BN, out_features]
        if self.dropout:
            Wh = F.dropout(Wh, self.dropout, training=self.training)

        # Compute attention scores for each edge
        src, dst = edge_index[0], edge_index[1]
        Wh_src = Wh[src]                   # [E_total, out_features]
        Wh_dst = Wh[dst]                   # [E_total, out_features]
        # Concatenate [Wh_src || Wh_dst]
        a_input = torch.cat([Wh_src, Wh_dst], dim=-1)   # [E_total, 2*out_features]
        e = self.leakyrelu(self.a(a_input)).squeeze(-1) # [E_total]

        # Dropout on attention coefficients (as in the TF code)
        if self.dropout:
            e = F.dropout(e, self.dropout, training=self.training)

        # Sparse softmax per source node (row‑wise)
        attn_adj = torch.sparse_coo_tensor(edge_index, e, size=(x.size(0), x.size(0)))
        attn_adj = torch.sparse.softmax(attn_adj, dim=1)

        # Weighted aggregation
        out = torch.sparse.mm(attn_adj, Wh)   # [BN, out_features]
        out = out + self.bias
        return out

class MultiHeadGATLayer(nn.Module):
    """
    Multi‑head GAT layer.
    If concat=True, outputs are concatenated; otherwise averaged.
    """
    def __init__(self, in_features, out_features, n_heads, dropout, alpha=0.2, concat=True):
        super().__init__()
        self.concat = concat
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            AttnHead(in_features, out_features, dropout, alpha) for _ in range(n_heads)
        ])

    def forward(self, x, edge_index):
        """
        x:          [B, N, F]
        edge_index: [2, B*E_per_graph] (includes self‑loops and batch offsets)
        Returns:    [B, N, out_dim] where out_dim depends on concat.
        """
        B, N, _ = x.shape
        # Flatten batch and node dimensions for shared processing
        x_flat = x.view(B * N, -1)          # [BN, F]

        head_outputs = []
        for head in self.heads:
            out_flat = head(x_flat, edge_index)   # [BN, out_features]
            if self.concat:
                head_outputs.append(out_flat)
            else:
                head_outputs.append(out_flat.unsqueeze(-1))  # [BN, out, 1]

        if self.concat:
            # Concatenate along feature dimension
            out = torch.cat(head_outputs, dim=-1)   # [BN, n_heads*out_features]
        else:
            # Average over heads
            out = torch.mean(torch.cat(head_outputs, dim=-1), dim=-1)  # [BN, out_features]

        # Restore batch dimension
        out = out.view(B, N, -1)
        return out

class GATModel(nn.Module):
    def __init__(self, nfeat, nhidden, nclass, nheads, dropout):
        super().__init__()
        self.dropout = dropout
        # First layer: 8 heads, each output 8 features -> concat => 64
        self.attn1 = MultiHeadGATLayer(
            nfeat, nhidden, nheads[0], dropout, concat=True
        )
        # Second layer: 1 head, output = nclass, concat=False (average)
        self.attn2 = MultiHeadGATLayer(
            nhidden * nheads[0], nclass, nheads[1], dropout, concat=False
        )

    def forward(self, x, edge_index):
        """
        x:          [B, N, F]
        edge_index: [2, B*E_per_graph]
        Returns:    logits [B, N, nclass]
        """
        x = self.attn1(x, edge_index)
        x = F.elu(x)
        x = self.attn2(x, edge_index)
        return x

def configure_optimizer(model):
    return torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

def train_model(model, optimizer, features_batch, edge_index_batch,
                label_long, train_mask, val_mask, epochs, patience):
    best_val_loss = float('inf')
    best_state = None
    epochs_without_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features_batch, edge_index_batch)          # [B, N, C]
        # Flatten for loss/masking
        logits_flat = logits.view(-1, logits.size(-1))
        label_flat = label_long.view(-1)
        train_mask_flat = train_mask.view(-1)

        loss = F.cross_entropy(logits_flat[train_mask_flat], label_flat[train_mask_flat])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features_batch, edge_index_batch)
            val_logits_flat = val_logits.view(-1, val_logits.size(-1))
            val_loss = F.cross_entropy(val_logits_flat[val_mask.view(-1)],
                                       label_flat[val_mask.view(-1)])
            val_pred = val_logits_flat[val_mask.view(-1)].argmax(dim=-1)
            val_acc = (val_pred == label_long[val_mask]).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: loss {loss.item():.4f}, '
                  f'val loss {val_loss.item():.4f}, val acc {val_acc:.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict(model, features_batch, edge_index_batch):
    model.eval()
    with torch.no_grad():
        logits = model(features_batch, edge_index_batch)          # [B, N, C]
        # Take first batch (all batches are identical)
        probs = logits[0]
        preds = probs.argmax(dim=-1)
    return preds.cpu().numpy()

def write_predictions(preds):
    nio.write_predictions(preds)

def main():
    set_seed()
    (features, labels, train_idx, val_idx, test_idx, edges,
     train_mask, val_mask, test_mask, label_long, num_nodes) = load_and_preprocess()

    # Normalisation already done in load_and_preprocess (features tensor)
    # Prepare batch
    batch_size = CONFIG["batch_size"]
    # Repeat features and masks for batch dimension
    features_batch = features.unsqueeze(0).repeat(batch_size, 1, 1)   # [B, N, F]
    label_long_batch = label_long.unsqueeze(0).repeat(batch_size, 1) # [B, N]
    train_mask_batch = train_mask.unsqueeze(0).repeat(batch_size, 1)
    val_mask_batch = val_mask.unsqueeze(0).repeat(batch_size, 1)

    # Build batched edge index with self‑loops
    edge_index_batch = build_adjacency_bias(edges, num_nodes, batch_size)  # [2, B*E_per_graph]

    # Model
    num_classes = CONFIG["num_classes"]
    model = GATModel(
        nfeat=features.shape[1],
        nhidden=CONFIG["hidden_per_head"],
        nclass=num_classes,
        nheads=[CONFIG["heads"], CONFIG["output_heads"]],
        dropout=CONFIG["dropout"]
    )
    optimizer = configure_optimizer(model)

    epochs = CONFIG["epochs"]
    if os.environ.get("SMOKE") == "1":
        epochs = 1

    model = train_model(model, optimizer, features_batch, edge_index_batch,
                        label_long_batch, train_mask_batch, val_mask_batch,
                        epochs, CONFIG["patience"])

    # Final validation accuracy (optional print)
    model.eval()
    with torch.no_grad():
        val_logits = model(features_batch, edge_index_batch)
        val_logits_flat = val_logits.view(-1, val_logits.size(-1))
        val_acc = (val_logits_flat[val_mask_batch.view(-1)].argmax(dim=-1)
                   == label_long_batch[val_mask_batch.view(-1)]).float().mean().item()
    print(f'Final validation accuracy: {val_acc:.4f}')

    # Prediction for the original graph (use batch_size=1 to avoid duplication)
    single_features = features.unsqueeze(0)          # [1, N, F]
    single_edge_index = build_adjacency_bias(edges, num_nodes, 1)
    preds = predict(model, single_features, single_edge_index)
    write_predictions(preds)

if __name__ == "__main__":
    main()