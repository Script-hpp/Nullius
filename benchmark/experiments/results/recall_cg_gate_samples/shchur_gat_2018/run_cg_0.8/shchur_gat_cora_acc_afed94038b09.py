

# Plan step 1 -> implemented in: set_seed
# Plan step 2 -> implemented in: load_data
# Plan step 3 -> implemented in: build_adjacency
# Plan step 4 -> implemented in: GAT class (and GATLayer)
# Plan step 5 -> implemented in: train_loop
# Plan step 6 -> implemented in: train_loop
# Plan step 7 -> implemented in: get_predictions
# Plan step 8 -> implemented in: write_predictions_call
# Plan step 9 -> implemented in: print_validation_accuracy

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {"epochs": 100000, "patience": 100, "heads": 8}
CONFIG["layers"] = 2
CONFIG["hidden"] = 8
CONFIG["lr"] = 0.01
CONFIG["weight_decay"] = 5e-4
CONFIG["input_dropout"] = 0.6
CONFIG["attn_dropout"] = 0.6
CONFIG["num_heads_per_layer"] = [CONFIG["heads"], 1]

def set_seed():
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    import nullius_io as nio
    d = nio.load()
    features = d['features'].astype(np.float32)
    edges = d['edges'].astype(np.int64)
    labels = d['labels'].astype(np.int64)
    train_idx = d['train_idx'].astype(np.int64)
    val_idx = d['val_idx'].astype(np.int64)
    test_idx = d['test_idx'].astype(np.int64)
    return features, edges, labels, train_idx, val_idx, test_idx

def build_adjacency(num_nodes, edges):
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    adj[edges[:, 0], edges[:, 1]] = True
    adj.fill_diagonal_(True)
    return adj

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, concat=True,
                 attn_dropout=0.0, negative_slope=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.attn_dropout = attn_dropout
        self.negative_slope = negative_slope

        self.W = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        self.a = nn.Parameter(torch.empty(num_heads, 2 * out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, adj):
        N = x.size(0)
        Wh = torch.einsum('bij,nj->bni', self.W, x)  # [H, N, F']

        heads_out = []
        for h in range(self.num_heads):
            Wh_h = Wh[h]  # [N, F']
            left = Wh_h @ self.a[h, :self.out_features]      # [N,1]
            right = Wh_h @ self.a[h, self.out_features:]     # [N,1]
            e = left + right.T                               # [N,N]
            e = F.leaky_relu(e, negative_slope=self.negative_slope)
            e_masked = e.masked_fill(~adj, float('-inf'))
            alpha = F.softmax(e_masked, dim=1)               # [N,N]
            if self.training:
                alpha = F.dropout(alpha, p=self.attn_dropout, training=True)
            head_out = alpha @ Wh_h                          # [N, F']
            heads_out.append(head_out)

        heads_out = torch.stack(heads_out, dim=0)  # [H, N, F']
        if self.concat:
            out = heads_out.transpose(0, 1).contiguous().view(N, -1)  # [N, H*F']
        else:
            out = heads_out.mean(dim=0)          # [N, F']
        out = out + self.bias
        return out

class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, num_classes,
                 heads_per_layer, input_dropout, attn_dropout):
        super().__init__()
        self.input_dropout = input_dropout
        layers = []
        for i in range(num_layers):
            in_feat = in_dim if i == 0 else hidden_dim * heads_per_layer[i-1]
            out_feat = hidden_dim if i < num_layers-1 else num_classes
            concat = i < num_layers-1
            layers.append(GATLayer(
                in_features=in_feat,
                out_features=out_feat,
                num_heads=heads_per_layer[i],
                concat=concat,
                attn_dropout=attn_dropout
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, adj):
        if self.training:
            x = F.dropout(x, p=self.input_dropout, training=True)
        for layer in self.layers:
            x = layer(x, adj)
        return x

def train_loop(model, optimizer, x, adj, labels, train_idx, val_idx, epochs, patience):
    best_acc = -float('inf')
    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, adj)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].long())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, adj)
            val_loss = F.cross_entropy(val_logits[val_idx], labels[val_idx].long())
            val_pred = val_logits[val_idx].argmax(dim=-1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()

        improved = (val_loss < best_loss - 1e-4) or (val_acc > best_acc + 1e-4)
        if improved:
            best_loss = val_loss
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if best_state is not None:
                model.load_state_dict(best_state)
            break

    if best_state is not None and epochs_no_improve < patience:
        model.load_state_dict(best_state)
    return best_acc

def get_predictions(model, x, adj):
    model.eval()
    with torch.no_grad():
        logits = model(x, adj)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
    return pred.cpu().numpy()

def write_predictions_call(pred):
    import nullius_io as nio
    nio.write_predictions(pred)

def print_validation_accuracy(acc):
    print(f"{acc:.4f}")

def main():
    set_seed()
    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    x = torch.from_numpy(features)
    labels_tensor = torch.from_numpy(labels)
    adj = build_adjacency(features.shape[0], edges)

    epochs = CONFIG["epochs"]
    patience = CONFIG["patience"]
    layers = CONFIG["layers"]
    hidden = CONFIG["hidden"]
    lr = CONFIG["lr"]
    weight_decay = CONFIG["weight_decay"]
    input_dropout = CONFIG["input_dropout"]
    attn_dropout = CONFIG["attn_dropout"]
    heads_per_layer = CONFIG["num_heads_per_layer"]
    num_classes = len(np.unique(labels_tensor[labels_tensor >= 0]))

    model = GAT(
        num_layers=layers,
        in_dim=features.shape[1],
        hidden_dim=hidden,
        num_classes=num_classes,
        heads_per_layer=heads_per_layer,
        input_dropout=input_dropout,
        attn_dropout=attn_dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = train_loop(
        model, optimizer, x, adj, labels_tensor,
        train_idx, val_idx, epochs, patience
    )
    pred_all = get_predictions(model, x, adj)
    write_predictions_call(pred_all)
    print_validation_accuracy(best_val_acc)

if __name__ == "__main__":
    main()