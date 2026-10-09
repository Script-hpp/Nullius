

# Plan step -> implemented in: load_and_preprocess
# Plan step -> implemented in: AttnHead class
# Plan step -> implemented in: GAT class
# Plan step -> implemented in: CONFIG dict
# Plan step -> implemented in: train_model function
# Plan step -> implemented in: set_seed function

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {
    "lr": 0.005,
    "weight_decay": 0.0005,
    "epochs": 100000,
    "batch_size": 2,
    "dropout": 0.6,
    "patience": 100,
    "heads": 8,
}

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_and_preprocess():
    d = nio.load()
    features = d['features'].astype(np.float32)          # [N, F]
    edges = d['edges']                                   # [E, 2]
    labels = d['labels']                                 # [N]
    train_idx = d['train_idx']
    val_idx = d['val_idx']
    test_idx = d['test_idx']

    N = features.shape[0]

    # row‑normalize features
    row_sum = features.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    features = features / row_sum

    # adjacency with self‑loop
    adj = np.zeros((N, N), dtype=np.float32)
    u, v = edges[:, 0], edges[:, 1]
    adj[u, v] = 1.0
    np.fill_diagonal(adj, 1.0)

    # bias matrix: 0 where adj>0 else -1e9
    bias_np = np.where(adj > 0, 0.0, -1e9).astype(np.float32)

    # torch tensors
    features_t = torch.from_numpy(features)               # [N, F]
    bias_t = torch.from_numpy(bias_np)                    # [N, N]

    # masks and labels (one‑hot)
    num_classes = int(labels.max() + 1)                   # ignoring -1 test nodes
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    y_train = torch.zeros(N, num_classes)
    y_val = torch.zeros(N, num_classes)
    y_train[train_idx, labels[train_idx].long()] = 1
    y_val[val_idx, labels[val_idx].long()] = 1

    return features_t, bias_t, y_train, y_val, train_mask, val_mask, test_idx

class AttnHead(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.dropout = dropout
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.fc1 = nn.Linear(out_features, 1, bias=False)
        self.fc2 = nn.Linear(out_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc1.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc2.weight, gain=gain)

    def forward(self, x, bias_mat, training=False):
        # x: [B, N, F_in]
        if self.dropout > 0 and training:
            x = F.dropout(x, p=self.dropout, training=True)
        seq_fts = self.fc(x)                     # [B, N, F_out]

        f_1 = self.fc1(seq_fts)                  # [B, N, 1]
        f_2 = self.fc2(seq_fts)                  # [B, N, 1]
        logits = f_1 + f_2.transpose(1, 2)       # [B, N, N]

        coefs = F.leaky_relu(logits, negative_slope=0.2) + bias_mat
        coefs = F.softmax(coefs, dim=-1)

        if self.dropout > 0 and training:
            coefs = F.dropout(coefs, p=self.dropout, training=True)

        vals = torch.bmm(coefs, seq_fts)         # [B, N, F_out]
        ret = vals + self.bias                   # broadcast bias
        return ret                               # linear output

class GAT(nn.Module):
    def __init__(self, in_features, hidden_units, n_heads, num_classes, dropout):
        super().__init__()
        self.hidden_units = hidden_units          # list, e.g. [8]
        self.n_heads = n_heads                    # list, e.g. [8, 1]
        self.dropout = dropout

        self.layers = nn.ModuleList()
        in_ch = in_features
        for l, (hu, nh) in enumerate(zip(hidden_units, n_heads[:-1])):
            heads = nn.ModuleList()
            for _ in range(nh):
                heads.append(AttnHead(in_ch, hu, dropout))
            self.layers.append(heads)
            in_ch = hu * nh   # concatenated output dim for next layer

        # output layer
        out_heads = nn.ModuleList()
        for _ in range(n_heads[-1]):
            out_heads.append(AttnHead(in_ch, num_classes, dropout))
        self.output_heads = out_heads

    def forward(self, x, bias_mat, training=False):
        h = x
        for heads in self.layers:
            head_outs = [head(h, bias_mat, training=training) for head in heads]
            h = torch.cat(head_outs, dim=-1)   # [B, N, hidden_units * n_heads]
            h = F.elu(h)                       # activation after concatenation
        # output layer
        out_list = [head(h, bias_mat, training=training) for head in self.output_heads]
        logits = torch.stack(out_list, dim=0).mean(dim=0)   # average over heads
        return logits

def masked_loss_and_acc(logits, labels, mask):
    """
    logits: [B, N, C]
    labels: [B, N, C] (one-hot)
    mask:   [B, N] bool
    Returns scalar loss and accuracy.
    """
    B, N, C = logits.shape
    logits_flat = logits.view(-1, C)          # [B*N, C]
    labels_flat = labels.view(-1, C)
    mask_flat = mask.view(-1)                 # [B*N]

    # masked cross entropy
    log_probs = F.log_softmax(logits_flat, dim=-1)
    loss = - (mask_flat.float() * (log_probs * labels_flat).sum(dim=-1)).sum()
    denom = mask_flat.float().sum()
    loss = loss / denom.clamp(min=1.0)

    # accuracy
    preds = logits_flat.argmax(dim=-1)
    correct = (preds == labels_flat.argmax(dim=-1)).float()
    acc = (correct * mask_flat.float()).sum() / denom.clamp(min=1.0)
    return loss.item(), acc.item()

def train_model():
    features, bias, y_train, y_val, train_mask, val_mask, test_idx = load_and_preprocess()
    N = features.shape[0]
    C = y_train.shape[1]

    # prepare batched tensors (repeat for batch_size)
    B = CONFIG["batch_size"]
    batched_features = features.unsqueeze(0).repeat(B, 1, 1)      # [B, N, F]
    batched_bias = bias.unsqueeze(0).repeat(B, 1, 1)            # [B, N, N]
    batched_y_train = y_train.unsqueeze(0).repeat(B, 1, 1)      # [B, N, C]
    batched_y_val = y_val.unsqueeze(0).repeat(B, 1, 1)
    batched_train_mask = train_mask.unsqueeze(0).repeat(B, 1)   # [B, N]
    batched_val_mask = val_mask.unsqueeze(0).repeat(B, 1)

    model = GAT(
        in_features=features.shape[1],
        hidden_units=[CONFIG["heads"]],   # each head outputs 8 features
        n_heads=[CONFIG["heads"], 1],     # 8 heads in first layer, 1 in output
        num_classes=C,
        dropout=CONFIG["dropout"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    best_val_loss = float('inf')
    patience_cnt = 0
    best_state = None

    epochs = CONFIG["epochs"]
    if os.environ.get("SMOKE") == "1":
        epochs = 1

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(batched_features, batched_bias, training=True)   # [B, N, C]
        loss_val, acc_val = masked_loss_and_acc(logits, batched_y_train, batched_train_mask)
        loss_val.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_logits = model(batched_features, batched_bias, training=False)
            val_loss, val_acc = masked_loss_and_acc(val_logits, batched_y_val, batched_val_mask)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:04d} | loss {loss_val:.4f} acc {acc_val:.4f} | val loss {val_loss:.4f} val acc {val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # final validation accuracy (from best state)
    model.eval()
    with torch.no_grad():
        final_logits = model(batched_features, batched_bias, training=False)
        _, final_val_acc = masked_loss_and_acc(final_logits, batched_y_val, batched_val_mask)
    print(f"Final validation accuracy: {final_val_acc:.4f}")

    # prediction for all nodes (single graph)
    model.eval()
    with torch.no_grad():
        single_logits = model(features.unsqueeze(0), bias.unsqueeze(0), training=False)   # [1, N, C]
        pred = torch.argmax(single_logits, dim=-1).squeeze(0).cpu().numpy()
    nio.write_predictions(pred)

if __name__ == "__main__":
    seed = int(os.environ.get("SEED", 42))
    set_seed(seed)
    train_model()