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
    "hidden_per_head": 8,
    "nb_layers": 1,
    "output_heads": 1
}

# Plan step -> implemented in: set_seed
# Plan step -> implemented in: load_and_preprocess
# Plan step -> implemented in: build_adjacency_bias
# Plan step -> implemented in: GATModel
# Plan step -> implemented in: configure_optimizer
# Plan step -> implemented in: train_model
# Plan step -> implemented in: predict
# Plan step -> implemented in: write_predictions

def set_seed():
    seed = int(os.environ.get("SEED", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_and_preprocess():
    d = nio.load()
    features = d['features']  # numpy float32 [N, F]
    labels = d['labels']      # numpy int64 [N]
    train_idx = d['train_idx']
    val_idx = d['val_idx']
    test_idx = d['test_idx']
    edges = d['edges']        # int64 [E, 2]

    # Row-normalize features
    rowsum = features.sum(axis=1, keepdims=True)
    r_inv = np.power(rowsum, -1.0)
    r_inv[np.isinf(r_inv)] = 0.0
    features_norm = features * r_inv

    features_tensor = torch.from_numpy(features_norm).float()
    labels_tensor = torch.from_numpy(labels).long()

    num_nodes = features.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True

    label_long = torch.full((num_nodes,), -1, dtype=torch.long)
    label_long[train_idx] = labels_tensor[train_idx]
    label_long[val_idx] = labels_tensor[val_idx]

    return (features_tensor, labels_tensor, train_idx, val_idx, test_idx,
            edges, train_mask, val_mask, test_mask, label_long, num_nodes)

def build_adjacency_bias(edges, num_nodes, batch_size):
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    adj[edges[:, 0], edges[:, 1]] = 1.0
    adj[range(num_nodes), range(num_nodes)] = 1.0  # self-loops
    bias = torch.where(adj > 0, torch.tensor(0.0), torch.tensor(-1e9))
    bias_batch = bias.unsqueeze(0).repeat(batch_size, 1, 1)
    return bias_batch

class AttnHead(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, h, adj_bias):
        if self.dropout > 0:
            h = F.dropout(h, self.dropout, training=self.training)
        h_prime = self.W(h)  # [B, N, out]
        B, N, _ = h_prime.shape
        # Prepare pairings for attention
        h_repeat = h_prime.repeat(1, 1, N).view(B, N * N, -1)      # [B, N*N, out]
        h_repeat_inter = h_prime.repeat(1, N, 1)                  # [B, N*N, out]
        a_input = torch.cat([h_repeat, h_repeat_inter], dim=-1)   # [B, N*N, 2*out]
        e = self.leakyrelu(self.a(a_input)).view(B, N, N)         # [B, N, N]
        # Mask via bias (0 for edge, -1e9 for non-edge)
        attention = e + adj_bias
        attention = F.softmax(attention, dim=-1)
        if self.dropout > 0:
            attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime_agg = torch.matmul(attention, h_prime)  # [B, N, out]
        h_prime_agg = h_prime_agg + self.bias
        return h_prime_agg  # pre‑activation

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, dropout, alpha=0.2, concat=True):
        super().__init__()
        self.concat = concat
        self.heads = nn.ModuleList([
            AttnHead(in_features, out_features, dropout, alpha) for _ in range(n_heads)
        ])
        self.output_dim = n_heads * out_features if concat else out_features

    def forward(self, h, adj_bias):
        head_outs = [head(h, adj_bias) for head in self.heads]
        if self.concat:
            out = torch.cat(head_outs, dim=-1)
        else:
            out = torch.stack(head_outs, dim=0).mean(dim=0)
        return out

class GATModel(nn.Module):
    def __init__(self, nfeat, nhidden, nclass, nheads, dropout):
        super().__init__()
        self.dropout = dropout
        self.attn1 = MultiHeadGATLayer(nfeat, nhidden, nheads[0], dropout, concat=True)
        self.attn2 = MultiHeadGATLayer(nhidden * nheads[0], nclass, nheads[1], dropout, concat=False)

    def forward(self, x, adj_bias):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.attn1(x, adj_bias)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        logits = self.attn2(x, adj_bias)
        return logits

def configure_optimizer(model):
    return torch.optim.Adam(model.parameters(),
                            lr=CONFIG["lr"],
                            weight_decay=CONFIG["weight_decay"])

def train_model(model, optimizer, features_batch, bias_batch,
                label_long, train_mask, val_mask, epochs, patience):
    best_val_loss = float('inf')
    best_state = None
    epochs_without_improve = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features_batch, bias_batch)
        loss = F.cross_entropy(logits[0, train_mask], label_long[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features_batch, bias_batch)
            val_loss = F.cross_entropy(val_logits[0, val_mask], label_long[val_mask])
            val_pred = val_logits[0, val_mask].argmax(dim=-1)
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

def predict(model, features_batch, bias_batch):
    model.eval()
    with torch.no_grad():
        logits = model(features_batch, bias_batch)
        preds = logits[0].argmax(dim=-1).cpu().numpy()
    return preds

def write_predictions(preds):
    nio.write_predictions(preds)

def main():
    set_seed()
    (features, labels, train_idx, val_idx, test_idx, edges,
     train_mask, val_mask, test_mask, label_long, num_nodes) = load_and_preprocess()

    num_classes = int(labels.max().item()) + 1
    batch_size = CONFIG["batch_size"]
    bias_batch = build_adjacency_bias(edges, num_nodes, batch_size)
    features_batch = features.unsqueeze(0).repeat(batch_size, 1, 1)

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

    model = train_model(model, optimizer, features_batch, bias_batch,
                        label_long, train_mask, val_mask,
                        epochs, CONFIG["patience"])

    # Final validation accuracy
    model.eval()
    with torch.no_grad():
        val_logits = model(features_batch, bias_batch)
        final_val_acc = (val_logits[0, val_mask].argmax(dim=-1)
                         == label_long[val_mask]).float().mean().item()
    print(f'Final validation accuracy: {final_val_acc:.4f}')

    preds = predict(model, features_batch, bias_batch)
    write_predictions(preds)

if __name__ == "__main__":
    main()
