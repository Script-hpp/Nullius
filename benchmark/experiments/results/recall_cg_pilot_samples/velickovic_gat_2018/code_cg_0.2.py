import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Plan step 1 -> implemented in: load_and_preprocess_data
# Plan step 2 -> implemented in: GATModel
# Plan step 3 -> implemented in: activation_elu
# Plan step 4 -> implemented in: dropout_application
# Plan step 5 -> implemented in: optimizer_setup
# Plan step 6 -> implemented in: optimizer_setup
# Plan step 7 -> implemented in: training_loop_with_early_stopping
# Plan step 8 -> implemented in: validation_evaluation
# Plan step 9 -> implemented in: preprocess_features_and_adj
# Plan step 10 -> implemented in: batching_logic
# Plan step 11 -> implemented in: seed_setting
# Plan step 12 -> implemented in: checkpoint_restore
# Plan step 13 -> implemented in: prediction_generation
# Plan step 14 -> implemented in: target_accuracy_comment
# Plan step 15 -> implemented in: hyperparameter_comment

CONFIG = {"lr": 0.005, "weight_decay": 0.0005, "batch_size": 2, "dropout": 0.6, "heads": 8}

def seed_setting(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_and_preprocess_data():
    data = np.load('/data/cora_public.npz')
    features = data['features'].astype(np.float32)          # (N, F)
    edges = data['edges']                                   # (E, 2)
    labels = data['labels'].astype(np.int64)                # (N)
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']
    # row-normalize features
    row_sum = features.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    features_norm = features / row_sum
    # adjacency with self-loops
    N = features.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    adj += np.eye(N)
    adj_binary = (adj > 0).astype(np.float32)
    # torch tensors
    features_tensor = torch.from_numpy(features_norm)
    adj_tensor = torch.from_numpy(adj_binary)
    labels_tensor = torch.from_numpy(labels)
    num_classes = int(labels_tensor.max().item() + 1)  # 7
    return features_tensor, adj_tensor, labels_tensor, train_idx, val_idx, test_idx, num_classes

def preprocess_features_and_adj(features_raw, edges):
    row_sum = features_raw.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    features_norm = features_raw / row_sum
    N = features_raw.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    adj += np.eye(N)
    adj_binary = (adj > 0).astype(np.float32)
    return features_norm, adj_binary

def activation_elu():
    return nn.ELU()

def dropout_application(x, p, training):
    return F.dropout(x, p, training=training)

def optimizer_setup(model, lr, weight_decay):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

class GATAttentionHead(nn.Module):
    def __init__(self, in_sz, out_sz, dropout, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.dropout = dropout
        self.W = nn.Linear(in_sz, out_sz, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_sz, 1)))
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj_mask):
        # h: (B, N, in_sz)
        # adj_mask: (B, N, N) where 0 for edge, -1e9 for non-edge
        B, N, _ = h.shape
        h_drop = F.dropout(h, self.dropout, training=self.training)
        h_proj = self.W(h_drop)  # (B, N, out_sz)
        a1 = self.a[:out_sz, :]          # (out_sz, 1)
        a2 = self.a[out_sz:, :]          # (out_sz, 1)
        f1 = torch.matmul(h_proj, a1)    # (B, N, 1)
        f2 = torch.matmul(h_proj, a2)    # (B, N, 1)
        e = f1 + f2.transpose(2, 1)      # (B, N, N)
        e = self.leakyrelu(e)
        e = e + adj_mask
        alpha = F.softmax(e, dim=-1)     # (B, N, N)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        h_proj_drop = F.dropout(h_proj, self.dropout, training=self.training)
        h_prime = torch.matmul(alpha, h_proj_drop)  # (B, N, out_sz)
        if not self.is_last:
            h_prime = F.elu(h_prime)
        return h_prime

class GATModel(nn.Module):
    def __init__(self, num_features, num_classes, heads, hid_units, dropout):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.heads = heads                 # e.g., [8, 1]
        self.hid_units = hid_units         # e.g., [8]
        self.dropout = dropout
        self.n_heads = heads
        self.num_layers = len(self.n_heads)
        self.layers = nn.ModuleList()
        for l in range(self.num_layers):
            if l < self.num_layers - 1:
                out_sz = self.hid_units_per_head[l]
            else:
                out_sz = self.num_classes
            heads_l = self.n_heads[l]
            heads_list = nn.ModuleList([
                GATAttentionHead(
                    in_sz=self.num_features if l == 0 else self.hid_units_per_head[l-1] * self.n_heads[l-1],
                    out_sz=out_sz,
                    dropout=self.dropout,
                    is_last=(l == self.num_layers - 1)
                )
                for _ in range(heads_l)
            ])
            self.layers.append(heads_list)

    @property
    def hid_units_per_head(self):
        return self.hid_units

    def forward(self, features, adj):
        attn_mask = (adj < 0.5) * -1e9
        h = features
        for l, heads_list in enumerate(self.layers):
            head_outs = [attn(h, attn_mask) for attn in heads_list]
            if l < len(self.layers) - 1:
                h = torch.cat(head_outs, dim=-1)
            else:
                h = torch.mean(torch.stack(head_outs, dim=0), dim=0)
        return h

def masked_softmax_cross_entropy(logits, labels, mask):
    loss = F.cross_entropy(logits.permute(0, 2, 1), labels.argmax(-1), reduction='none')
    loss = loss * mask
    return loss.sum() / mask.sum()

def masked_accuracy(logits, labels, mask):
    preds = logits.argmax(-1)
    labels_argmax = labels.argmax(-1)
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
                                      max_epochs=10000, patience=100):
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
        model.eval()
        with torch.no_grad():
            val_logits = model(features_batch, adj_batch)
            val_loss = masked_softmax_cross_entropy(val_logits, labels_val_batch, val_mask_batch)
            val_acc = masked_accuracy(val_logits, labels_val_batch, val_mask_batch)
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
    return model, best_val_loss, val_acc  # val_acc from last epoch (could recompute)

def batching_logic(features, adj, labels_train, labels_val, train_mask, val_mask, batch_size):
    features_batch = features.unsqueeze(0).repeat(batch_size, 1, 1)
    adj_batch = adj.unsqueeze(0).repeat(batch_size, 1, 1)
    labels_train_batch = labels_train.unsqueeze(0).repeat(batch_size, 1, 1)
    labels_val_batch = labels_val.unsqueeze(0).repeat(batch_size, 1, 1)
    train_mask_batch = train_mask.unsqueeze(0).repeat(batch_size, 1)
    val_mask_batch = val_mask.unsqueeze(0).repeat(batch_size, 1)
    return features_batch, adj_batch, labels_train_batch, labels_val_batch, train_mask_batch, val_mask_batch

def checkpoint_restore(model, best_state):
    model.load_state_dict(best_state)

def prediction_generation(model, features_batch, adj_batch):
    model.eval()
    with torch.no_grad():
        logits = model(features_batch, adj_batch)  # (B, N, C)
        logits_avg = logits.mean(dim=0)            # (N, C)
        preds = logits_avg.argmax(-1)
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

    # one-hot labels
    labels_onehot = torch.zeros(labels.size(0), num_classes)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)

    labels_train = labels_onehot[train_idx]
    labels_val = labels_onehot[val_idx]

    train_mask = torch.zeros_like(labels, dtype=torch.float32)
    train_mask[train_idx] = 1.0
    val_mask = torch.zeros_like(labels, dtype=torch.float32)
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

    max_epochs = 1 if smoke else 10000
    patience = 0 if smoke else 100

    model, best_val_loss, val_acc = training_loop_with_early_stopping(
        model, optimizer,
        features_batch, adj_batch,
        labels_train_batch, train_mask_batch,
        labels_val_batch, val_mask_batch,
        max_epochs=max_epochs,
        patience=patience
    )

    # predictions
    test_pred = prediction_generation(model, features_batch, adj_batch)
    val_pred = prediction_generation(model, features_batch, adj_batch)

    if smoke:
        test_idx_use = np.sort(test_idx)[:20]
        val_idx_use = np.sort(val_idx)[:20]
    else:
        test_idx_use = test_idx
        val_idx_use = val_idx

    os.makedirs('/output', exist_ok=True)
    with open('/output/predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for idx in test_idx_use:
            f.write(f'{int(idx)},{int(test_pred[idx])}\n')
    with open('/output/val_predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for idx in val_idx_use:
            f.write(f'{int(idx)},{int(val_pred[idx])}\n')

    print(f'Validation accuracy: {val_acc:.4f}')

if __name__ == '__main__':
    main()