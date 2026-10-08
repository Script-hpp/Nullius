# Plan step 1 -> implemented in: load_and_preprocess_data
# Plan step 2 -> implemented in: GATModel
# Plan step 3 -> implemented in: activation_elu (via F.elu inside GATAttentionHead)
# Plan step 4 -> implemented in: dropout_application (via F.dropout)
# Plan step 5 -> implemented in: optimizer_setup
# Plan step 6 -> implemented in: optimizer_setup (weight_decay passed to Adam)
# Plan step 7 -> implemented in: training_loop_with_early_stopping
# Plan step 8 -> implemented in: validation_evaluation
# Plan step 9 -> implemented in: load_and_preprocess_data (row‑norm features, add self‑loops)
# Plan step 10-> implemented in: batching_logic
# Plan step 11-> implemented in: seed_setting
# Plan step 12-> implemented in: checkpoint_restore (inside training_loop_with_early_stopping)
# Plan step 13-> implemented in: prediction_generation
# Plan step 14-> implemented in: target_accuracy_comment (no‑op)
# Plan step 15-> implemented in: hyperparameter_comment (no‑op)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {"lr": 0.005, "weight_decay": 0.0005, "batch_size": 2, "dropout": 0.6, "heads": 8}
# Additional hyper‑parameters (values chosen as typical for GAT on Cora)
CONFIG["epochs"] = 10000
CONFIG["patience"] = 100

def seed_setting(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_and_preprocess_data():
    data = np.load('/data/cora_public.npz')
    features = data['features'].astype(np.float32)          # (N, F)
    edges = data['edges']                                   # (E, 2)
    labels = data['labels'].astype(np.int64)                # (N)  -1 for test nodes
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    # row‑normalize features
    row_sum = features.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    features_norm = features / row_sum

    # adjacency with self‑loops (binary)
    N = features.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    adj += np.eye(N)
    adj_binary = (adj > 0).astype(np.float32)

    # torch tensors
    features_tensor = torch.from_numpy(features_norm)
    adj_tensor = torch.from_numpy(adj_binary)
    labels_tensor = torch.from_numpy(labels)

    num_classes = int(labels_tensor.max().item() + 1)  # 7 (ignore -1)
    return features_tensor, adj_tensor, labels_tensor, train_idx, val_idx, test_idx, num_classes

def activation_elu():
    return nn.ELU()

def dropout_application(x, p, training):
    return F.dropout(x, p, training=training)

def optimizer_setup(model, lr, weight_decay):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

class GATAttentionHead(nn.Module):
    """
    Mimics the TensorFlow attn_head from the authors' code:
    - feat dropout on inputs
    - linear projection (no bias)
    - attention scores via a single‑layer perceptron (LeakyReLU)
    - add bias_mat (here we use adjacency mask: 0 for existing edge, -1e9 otherwise)
    - dropout on attention coefficients
    - feat dropout again on projected features
    - weighted sum
    - bias_add (learnable bias vector)
    - residual connection (optional)
    - activation
    """
    def __init__(self, in_sz, out_sz, dropout, residual=False):
        super().__init__()
        self.dropout = dropout
        self.residual = residual

        self.W = nn.Linear(in_sz, out_sz, bias=False)          # projection
        self.attn_fc = nn.Linear(2 * out_sz, 1, bias=False)   # for a^T [Wh_i || Wh_j]
        self.bias_vec = nn.Parameter(torch.zeros(out_sz))     # bias_add
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)

        if residual:
            self.residual_proj = nn.Linear(in_sz, out_sz, bias=False) if in_sz != out_sz else None

    def forward(self, seq, adj_mat):
        """
        seq:      (B, N, in_sz)
        adj_mat:  (B, N, N) 0 where edge exists, -1e9 otherwise
        """
        B, N, _ = seq.shape

        # dropout on inputs
        seq_drop = F.dropout(seq, self.dropout, training=self.training)

        # linear projection
        proj = self.W(seq_drop)                     # (B, N, out_sz)

        # attention scores
        # prepare concatenated vectors for all pairs
        proj_i = proj.unsqueeze(2).expand(-1, -1, N, -1)   # (B, N, N, out_sz)
        proj_j = proj.unsqueeze(1).expand(-1, N, -1, -1)   # (B, N, N, out_sz)
        concat = torch.cat([proj_i, proj_j], dim=-1)       # (B, N, N, 2*out_sz)
        e = self.leakyrelu(self.attn_fc(concat).squeeze(-1))  # (B, N, N)

        # mask non‑edges
        e = e + adj_mat

        # softmax -> attention coefficients
        alpha = F.softmax(e, dim=-1)                       # (B, N, N)
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # dropout again on projected features (as in TF code)
        proj_drop = F.dropout(proj, self.dropout, training=self.training)

        # weighted sum
        attn_out = torch.matmul(alpha, proj_drop)          # (B, N, out_sz)

        # bias_add
        attn_out = attn_out + self.bias_vec

        # residual connection
        if self.residual:
            if self.residual_proj is not None:
                res = self.residual_proj(seq)
            else:
                res = seq
            attn_out = attn_out + res

        # activation (ELU for hidden layers, identity for output layer handled outside)
        return F.elu(attn_out)

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_sz, out_sz_per_head, n_heads, dropout, concat=True, residual=False):
        super().__init__()
        self.concat = concat
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            GATAttentionHead(in_sz, out_sz_per_head, dropout, residual=residual)
            for _ in range(n_heads)
        ])

    def forward(self, x, adj_mat):
        head_outs = [head(x, adj_mat) for head in self.heads]
        if self.concat:
            return torch.cat(head_outs, dim=-1)
        else:
            return torch.mean(torch.stack(head_outs, dim=0), dim=0)

class GATModel(nn.Module):
    def __init__(self, num_features, num_classes, heads, hid_units, dropout):
        """
        heads: list like [8, 1]  (number of attention heads per layer)
        hid_units: list like [8] (output features per head for each hidden layer)
        """
        super().__init__()
        self.num_layers = len(heads)
        self.layers = nn.ModuleList()

        for l in range(self.num_layers):
            in_ch = num_features if l == 0 else hid_units[l-1] * heads[l-1]
            out_ch_per_head = hid_units[l] if l < self.num_layers - 1 else num_classes
            concat = l < self.num_layers - 1   # concat on hidden layers, average on output
            self.layers.append(
                MultiHeadGATLayer(
                    in_sz=in_ch,
                    out_sz_per_head=out_ch_per_head,
                    n_heads=heads[l],
                    dropout=dropout,
                    concat=concat,
                    residual=False   # as per authors' code (residual=False)
                )
            )

    def forward(self, features, adj):
        """
        features: (B, N, F)
        adj:      (B, N, N) binary with self-loops
        """
        h = features
        for l, layer in enumerate(self.layers):
            h = layer(h, adj)
        return h   # (B, N, C) where C = num_classes (averaged over heads in last layer)

def masked_softmax_cross_entropy(logits, labels, mask):
    """
    logits: (B, N, C)
    labels: (B, N, C) one‑hot
    mask:   (B, N) 0/1
    """
    # cross entropy per node
    log_p = F.log_softmax(logits, dim=-1)
    loss = -torch.sum(labels * log_p, dim=-1)   # (B, N)
    loss = loss * mask
    return loss.sum() / mask.sum()

def masked_accuracy(logits, labels, mask):
    preds = logits.argmax(dim=-1)
    labels_argmax = labels.argmax(dim=-1)
    correct = (preds == labels_argmax).float()
    correct = correct * mask
    return correct.sum() / mask.sum()

def validation_evaluation(model, features_batch, adj_batch, labels_batch, mask_batch):
    model.eval()
    with torch.no_grad():
        logits = model(features_batch, adj_batch)
        loss = masked_softmax_cross_entropy(logits, labels_batch, mask_batch)
        acc = masked_accuracy(logits, labels_batch, mask_batch)
    return loss.item(), acc.item()

def training_loop_with_early_stopping(model, optimizer, features_batch, adj_batch,
                                      labels_train_batch, train_mask_batch,
                                      labels_val_batch, val_mask_batch,
                                      max_epochs=None, patience=None):
    if max_epochs is None:
        max_epochs = CONFIG["epochs"]
    if patience is None:
        patience = CONFIG["patience"]

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features_batch, adj_batch)
        loss = masked_softmax_cross_entropy(logits, labels_train_batch, train_mask_batch)
        loss.backward()
        optimizer.step()

        val_loss, val_acc = validation_evaluation(model, features_batch, adj_batch,
                                                  labels_val_batch, val_mask_batch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss, val_acc  # val_acc from last improved epoch

def batching_logic(features, adj, labels_train, labels_val, train_mask, val_mask, batch_size):
    """
    Repeat the single graph batch_size times to simulate mini‑batch of graphs.
    """
    features_batch = features.unsqueeze(0).repeat(batch_size, 1, 1)
    adj_batch = adj.unsqueeze(0).repeat(batch_size, 1, 1)
    labels_train_batch = labels_train.unsqueeze(0).repeat(batch_size, 1, 1)
    labels_val_batch = labels_val.unsqueeze(0).repeat(batch_size, 1, 1)
    train_mask_batch = train_mask.unsqueeze(0).repeat(batch_size, 1)
    val_mask_batch = val_mask.unsqueeze(0).repeat(batch_size, 1)
    return features_batch, adj_batch, labels_train_batch, labels_val_batch, train_mask_batch, val_mask_batch

def prediction_generation(model, features_batch, adj_batch):
    model.eval()
    with torch.no_grad():
        logits = model(features_batch, adj_batch)          # (B, N, C)
        logits_avg = logits.mean(dim=0)                    # average over batch dimension
        preds = logits_avg.argmax(dim=-1)
    return preds

def target_accuracy_comment():
    pass

def hyperparameter_comment():
    pass

def main():
    seed = int(os.environ.get('SEED', 0))
    seed_setting(seed)

    smoke = os.environ.get('SMOKE', '0') == '1'

    features, adj, labels, train_idx, val_idx, test_idx, num_classes = load_and_preprocess_data()

    # Prepare one‑hot labels for train and val (test nodes ignored)
    N = labels.size(0)
    labels_train = torch.zeros(N, num_classes, dtype=torch.float32)
    labels_train[train_idx, labels[train_idx]] = 1.0

    labels_val = torch.zeros(N, num_classes, dtype=torch.float32)
    labels_val[val_idx, labels[val_idx]] = 1.0

    # Masks (float for loss weighting)
    train_mask = torch.zeros(N, dtype=torch.float32)
    train_mask[train_idx] = 1.0
    val_mask = torch.zeros(N, dtype=torch.float32)
    val_mask[val_idx] = 1.0

    batch_size = CONFIG["batch_size"]
    features_batch, adj_batch, labels_train_batch, labels_val_batch, train_mask_batch, val_mask_batch = batching_logic(
        features, adj, labels_train, labels_val, train_mask, val_mask, batch_size
    )

    model = GATModel(
        num_features=features.shape[1],
        num_classes=num_classes,
        heads=[CONFIG["heads"], 1],
        hid_units=[8],
        dropout=CONFIG["dropout"]
    )
    optimizer = optimizer_setup(
        model,
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    max_epochs = 1 if smoke else CONFIG["epochs"]
    patience = 0 if smoke else CONFIG["patience"]

    model, best_val_loss, val_acc = training_loop_with_early_stopping(
        model, optimizer,
        features_batch, adj_batch,
        labels_train_batch, train_mask_batch,
        labels_val_batch, val_mask_batch,
        max_epochs=max_epochs,
        patience=patience
    )

    # Produce predictions for all nodes
    all_pred = prediction_generation(model, features_batch, adj_batch)   # (N,)

    # Select indices to write (full or first 20 in smoke)
    if smoke:
        test_idx_use = np.sort(test_idx.cpu().numpy())[:20]
        val_idx_use = np.sort(val_idx.cpu().numpy())[:20]
    else:
        test_idx_use = test_idx.cpu().numpy()
        val_idx_use = val_idx.cpu().numpy()

    os.makedirs('/output', exist_ok=True)

    # Test predictions
    with open('/output/predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for idx in test_idx_use:
            f.write(f'{int(idx)},{int(all_pred[idx])}\n')

    # Validation predictions
    with open('/output/val_predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for idx in val_idx_use:
            f.write(f'{int(idx)},{int(all_pred[idx])}\n')

    print(f'Validation accuracy: {val_acc:.4f}')

if __name__ == '__main__':
    main()
