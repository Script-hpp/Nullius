

# Plan step -> implemented in: load_data
# Plan step -> implemented in: GAT class
# Plan step -> implemented in: GAT.forward (activation)
# Plan step -> implemented in: GATHead.forward (dropout)
# Plan step -> implemented in: CONFIG dict and batch size
# Plan step -> implemented in: optimizer creation
# Plan step -> implemented in: training loop with early stopping
# Plan step -> implemented in: validation step inside loop
# Plan step -> implemented in: preprocess_features, build_adj_bias functions
# Plan step -> implemented in: prepare_batch function
# Plan step -> implemented in: set_seed function
# Plan step -> implemented in: loading best state
# Plan step -> implemented in: write_predictions function
# Plan step -> implemented in: target accuracy comment
# Plan step -> implemented in: uncertain values comment
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "lr": 0.005,
    "weight_decay": 0.0005,
    "batch_size": 2,
    "dropout": 0.6,
    "heads": 8  # kept for reference; actual heads used are [8,1] in first and second layer
}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    features = data['features'].astype(np.float32)  # [N, F]
    edges = data['edges'].astype(np.int64)          # [E, 2]
    labels = data['labels'].astype(np.int64)        # [N]
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']
    return features, edges, labels, train_idx, val_idx, test_idx

def preprocess_features(features):
    """Row-normalize feature matrix."""
    rowsum = features.sum(axis=1, keepdims=True)
    r_inv = 1.0 / rowsum
    r_inv[np.isinf(r_inv)] = 0.0
    features_norm = features * r_inv
    return features_norm

def build_adj_bias(num_nodes, edges):
    """Adjacency bias matrix: 0 for existing edges (including self), -1e9 otherwise."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[edges[:,0], edges[:,1]] = 1.0
    np.fill_diagonal(adj, 1.0)  # self-loops
    bias = np.zeros_like(adj)
    bias[adj == 0.0] = -1e9
    return bias  # shape [N, N]

class GATHead(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj_bias, training):
        # x: (B, N, F_in)
        if training:
            x = F.dropout(x, self.dropout, training=True)
        # Linear transformation
        h = torch.matmul(x, self.W)  # (B, N, F_out)
        # Compute attention scores
        a1 = self.a[:self.W.shape[1], :]      # (F_out, 1)
        a2 = self.a[self.W.shape[1]:, :]      # (F_out, 1)
        f1 = torch.matmul(h, a1)              # (B, N, 1)
        f2 = torch.matmul(h, a2)              # (B, N, 1)
        e = f1 + f2.transpose(1, 2)           # (B, N, N)
        e = self.leakyrelu(e)
        # Mask attention
        attention = e + adj_bias              # (B, N, N)
        attention = F.softmax(attention, dim=-1)
        if training:
            attention = F.dropout(attention, self.dropout, training=True)
        if training:
            h = F.dropout(h, self.dropout, training=True)
        # Linear combination
        h_prime = torch.matmul(attention, h)  # (B, N, F_out)
        h_prime = h_prime + self.bias         # bias
        return h_prime

class MultiHeadGAT(nn.Module):
    def __init__(self, in_features, out_features, n_heads, dropout, alpha=0.2, concat=True):
        super().__init__()
        self.concat = concat
        self.heads = nn.ModuleList([
            GATHead(in_features, out_features, dropout, alpha) for _ in range(n_heads)
        ])

    def forward(self, x, adj_bias, training):
        head_outs = [head(x, adj_bias, training) for head in self.heads]
        if self.concat:
            return torch.cat(head_outs, dim=-1)
        else:
            return torch.mean(torch.stack(head_outs, dim=0), dim=0)

class GAT(nn.Module):
    def __init__(self, nfeat, nhidden, nclass, nheads, dropout):
        super().__init__()
        self.attention1 = MultiHeadGAT(nfeat, nhidden, nheads[0], dropout, concat=True)
        self.attention2 = MultiHeadGAT(nhidden * nheads[0], nclass, nheads[1], dropout, concat=False)
        self.elu = nn.ELU()

    def forward(self, x, adj_bias, training):
        x = self.attention1(x, adj_bias, training)
        x = self.elu(x)
        x = self.attention2(x, adj_bias, training)
        return x

def masked_cross_entropy(logits, labels_onehot, mask):
    log_prob = F.log_softmax(logits, dim=-1)
    loss = -torch.sum(labels_onehot * log_prob, dim=-1)  # (B, N)
    loss = loss * mask
    return torch.sum(loss) / torch.sum(mask)

def masked_accuracy(logits, labels_int, mask):
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels_int).float()
    correct = correct * mask
    return torch.sum(correct) / torch.sum(mask)

def prepare_batch(features_tensor, bias_tensor, labels_onehot, mask_tensor, batch_size):
    feat_batch = features_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
    bias_batch = bias_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
    label_batch = labels_onehot.unsqueeze(0).repeat(batch_size, 1, 1)
    mask_batch = mask_tensor.unsqueeze(0).repeat(batch_size, 1).float()
    return feat_batch, bias_batch, label_batch, mask_batch

def main():
    seed = int(os.environ.get("SEED", "0"))
    set_seed(seed)

    features, edges, labels, train_idx, val_idx, test_idx = load_data('/data/cora_public.npz')
    num_nodes = features.shape[0]
    feat_dim = features.shape[1]
    num_classes = len(np.unique(labels[labels >= 0]))  # should be 7

    features_norm = preprocess_features(features)
    features_tensor = torch.from_numpy(features_norm)  # [N, F]

    bias_np = build_adj_bias(num_nodes, edges)        # [N, N]
    bias_tensor = torch.from_numpy(bias_np)           # [N, N]

    # Full labels tensor with -1 for unknown (test) nodes
    labels_tensor = torch.from_numpy(labels)          # [N]

    train_label = labels[train_idx]
    val_label = labels[val_idx]
    train_onehot = np.zeros((num_nodes, num_classes), dtype=np.float32)
    train_onehot[train_idx, train_label] = 1.0
    val_onehot = np.zeros((num_nodes, num_classes), dtype=np.float32)
    val_onehot[val_idx, val_label] = 1.0

    train_onehot_tensor = torch.from_numpy(train_onehot)   # [N, C]
    val_onehot_tensor = torch.from_numpy(val_onehot)
    train_label_tensor = torch.from_numpy(train_label)     # [N]
    val_label_tensor = torch.from_numpy(val_label)

    train_mask = np.zeros(num_nodes, dtype=np.float32)
    train_mask[train_idx] = 1.0
    val_mask = np.zeros(num_nodes, dtype=np.float32)
    val_mask[val_idx] = 1.0

    train_mask_tensor = torch.from_numpy(train_mask)
    val_mask_tensor = torch.from_numpy(val_mask)

    # Use CONFIG["heads"] for first layer, second layer single head as per paper
    nheads = [CONFIG["heads"], 1]

    model = GAT(
        nfeat=feat_dim,
        nhidden=8,
        nclass=num_classes,
        nheads=nheads,
        dropout=CONFIG["dropout"]
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    smoke = os.environ.get("SMOKE") == "1"
    max_epochs = 1 if smoke else 10000
    patience = 100 if not smoke else 1
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    batch_size = CONFIG["batch_size"]

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        feat_batch, bias_batch, label_batch, mask_batch = prepare_batch(
            features_tensor, bias_tensor, train_onehot_tensor, train_mask_tensor, batch_size
        )
        logits = model(feat_batch, bias_batch, training=True)
        loss = masked_cross_entropy(logits, label_batch, mask_batch)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Use full labels tensor for accuracy; mask selects training nodes
            train_acc = masked_accuracy(
                logits,
                labels_tensor.unsqueeze(0).repeat(batch_size, 1),
                mask_batch
            )

        model.eval()
        with torch.no_grad():
            feat_batch_val, bias_batch_val, label_batch_val, mask_batch_val = prepare_batch(
                features_tensor, bias_tensor, val_onehot_tensor, val_mask_tensor, batch_size
            )
            logits_val = model(feat_batch_val, bias_batch_val, training=False)
            val_loss = masked_cross_entropy(logits_val, label_batch_val, mask_batch_val)
            val_acc = masked_accuracy(
                logits_val,
                labels_tensor.unsqueeze(0).repeat(batch_size, 1),
                mask_batch_val
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_val_acc = val_acc.item()
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss {loss.item():.4f} | Train Acc {train_acc.item():.4f} | Val Loss {val_loss.item():.4f} | Val Acc {val_acc.item():.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        feat = features_tensor.unsqueeze(0)      # (1, N, F)
        bias = bias_tensor.unsqueeze(0)          # (1, N, N)
        logits = model(feat, bias, training=False)  # (1, N, C)
        logits = logits.squeeze(0)               # (N, C)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    os.makedirs('/output', exist_ok=True)

    if smoke:
        test_ids_to_write = np.sort(test_idx)[:20]
        val_ids_to_write = np.sort(val_idx)[:20]
    else:
        test_ids_to_write = test_idx
        val_ids_to_write = val_idx

    with open('/output/predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for node_id in test_ids_to_write:
            f.write(f'{node_id},{int(preds[node_id])}\n')

    with open('/output/val_predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for node_id in val_ids_to_write:
            f.write(f'{node_id},{int(preds[node_id])}\n')

    print(f'Best validation accuracy: {best_val_acc:.4f}')

if __name__ == '__main__':
    main()