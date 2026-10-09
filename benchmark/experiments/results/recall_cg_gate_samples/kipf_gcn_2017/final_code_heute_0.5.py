# Plan step -> implemented in: load_data
# Plan step -> implemented in: set_seed
# Plan step -> implemented in: build_normalized_adjacency
# Plan step -> implemented in: class GCN
# Plan step -> implemented in: train
# Plan step -> implemented in: predict
# Plan step -> implemented in: save_predictions
# Plan step -> implemented in: print_validation_accuracy

import os
import numpy as np
import torch
import torch.nn.functional as F
import nullius_io as nio

def load_data():
    d = nio.load()
    features = torch.from_numpy(d['features']).float()          # [N, F]
    edges = torch.from_numpy(d['edges']).long()                # [E, 2]
    labels = torch.from_numpy(d['labels']).long()              # [N]
    train_idx = torch.from_numpy(d['train_idx']).long()
    val_idx = torch.from_numpy(d['val_idx']).long()
    test_idx = torch.from_numpy(d['test_idx']).long()
    return features, edges, labels, train_idx, val_idx, test_idx

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_normalized_adjacency(edges, num_nodes):
    # add self-loops
    self_loop = torch.arange(num_nodes, dtype=torch.long).unsqueeze(1).repeat(1, 2)
    edge_index = torch.cat([edges, self_loop], dim=0).t()      # [2, E']
    # compute degree
    deg = torch.zeros(num_nodes, dtype=torch.float)
    deg.index_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    # compute normalization coefficients
    vals = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
    # create sparse tensor
    A_hat = torch.sparse_coo_tensor(edge_index, vals, size=(num_nodes, num_nodes),
                                    dtype=torch.float).coalesce()
    return A_hat

class GCN(torch.nn.Module):
    def __init__(self, in_features, hidden, out_classes):
        super().__init__()
        self.W0 = torch.nn.Parameter(torch.Tensor(in_features, hidden))
        self.W1 = torch.nn.Parameter(torch.Tensor(hidden, out_classes))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W0)
        torch.nn.init.xavier_uniform_(self.W1)

    def forward(self, A_hat, x):
        # first layer
        h = torch.sparse.mm(A_hat, x @ self.W0)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        # second layer (logits)
        logits = torch.sparse.mm(A_hat, h @ self.W1)
        return logits

def train(model, A_hat, features, labels, train_idx, val_idx, smoke=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val_loss = float('inf')
    patience_counter = 0
    max_epochs = 1 if smoke else 200
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(A_hat, features)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_logits = model(A_hat, features)
            val_loss = F.cross_entropy(val_logits[val_idx], labels[val_idx]).item()
            pred = val_logits[val_idx].argmax(dim=1)
            acc = (pred == labels[val_idx]).float().mean().item()
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 100:
                break
    # restore best
    model.load_state_dict(best_state)
    return best_state, acc

def predict(model, A_hat, features):
    model.eval()
    with torch.no_grad():
        logits = model(A_hat, features)
    return logits

def save_predictions(pred_all):
    # pred_all can be logits or class predictions; helper will handle selection
    nio.write_predictions(pred_all)

def print_validation_accuracy(acc):
    print(f"Validation accuracy: {acc:.4f}")

def main():
    seed = int(os.getenv('SEED', '0'))
    set_seed(seed)

    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features.shape[0]

    A_hat = build_normalized_adjacency(edges, num_nodes)

    model = GCN(in_features=features.shape[1], hidden=16, out_classes=7)

    smoke = os.getenv('SMOKE') == '1'
    _, val_acc = train(model, A_hat, features, labels, train_idx, val_idx, smoke=smoke)

    logits = predict(model, A_hat, features)
    # produce class predictions for all nodes
    pred_all = logits.argmax(dim=1)
    save_predictions(pred_all)

    print_validation_accuracy(val_acc)

if __name__ == "__main__":
    main()
