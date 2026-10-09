

# Plan step -> implemented in: set_seeds
# Plan step -> implemented in: load_data
# Plan step -> implemented in: build_adjacency
# Plan step -> implemented in: add_self_loops
# Plan step -> implemented in: normalize_adjacency
# Plan step -> implemented in: propagate_features
# Plan step -> implemented in: split_data
# Plan step -> implemented in: train_logistic_regression
# Plan step -> implemented in: tune_C
# Plan step -> implemented in: retrain_on_combined
# Plan step -> implemented in: predict_and_save

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time

# Suppress sparse invariant warnings to avoid segfaults on malformed tensors
torch.sparse.check_sparse_tensor_invariants(False)


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    data = np.load('/data/cora_public.npz', allow_pickle=False)
    features = torch.from_numpy(data['features']).float()          # [N, F]
    edges = torch.from_numpy(data['edges']).long()                # [E, 2]
    labels = torch.from_numpy(data['labels']).long()              # [N]
    train_idx = torch.from_numpy(data['train_idx']).long()
    val_idx = torch.from_numpy(data['val_idx']).long()
    test_idx = torch.from_numpy(data['test_idx']).long()
    return features, edges, labels, train_idx, val_idx, test_idx


def build_adjacency(edges, num_nodes):
    # edges: [E, 2] (u, v)
    idx = edges.t()                                               # [2, E]
    values = torch.ones(idx.size(1), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, values, size=(num_nodes, num_nodes))
    return adj.coalesce()


def add_self_loops(adj):
    N = adj.size(0)
    self_loop_idx = torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(1, 2)  # [N, 2]
    self_loop_idx = self_loop_idx.t()                                          # [2, N]
    self_loop_vals = torch.ones(N, dtype=torch.float32)
    idx = torch.cat([adj.indices(), self_loop_idx], dim=1)
    vals = torch.cat([adj.values(), self_loop_vals])
    adj_new = torch.sparse_coo_tensor(idx, vals, size=(N, N))
    return adj_new.coalesce()


def normalize_adjacency(adj):
    # symmetric normalization: D^{-1/2} A D^{-1/2}
    idx = adj.indices()
    vals = adj.values()
    # compute degree
    deg = torch.sparse.sum(adj, dim=1).to_dense()          # [N]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    # normalize values
    norm_vals = vals * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, norm_vals, size=adj.shape).coalesce()


def propagate_features(adj_norm, features, K=2):
    X = features
    for _ in range(K):
        X = torch.sparse.mm(adj_norm, X)
    return X


def split_data(X, y, train_idx, val_idx, test_idx):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test = X[test_idx]  # labels are -1 for test, ignore
    return X_train, y_train, X_val, y_val, X_test


def train_logistic_regression(X_train, y_train, X_val, y_val, C, max_iter=1000):
    """Train a softmax regression model with L2 regularization (inverse strength C).
    Returns the model and validation accuracy (if validation set non-empty)."""
    num_features = X_train.size(1)
    num_classes = int(y_train.max().item() + 1)
    model = nn.Linear(num_features, num_classes, bias=True)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)

    lambda_l2 = 1.0 / C  # inverse of C as in sklearn's LogisticRegression
    optimizer = torch.optim.LBFGS(model.parameters(),
                                  max_iter=max_iter,
                                  line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        logits = model(X_train)
        loss = F.cross_entropy(logits, y_train)
        # L2 regularization on weight and bias
        l2_reg = lambda_l2 * (torch.norm(model.weight, p=2) ** 2 +
                              torch.norm(model.bias, p=2) ** 2)
        loss = loss + l2_reg
        loss.backward()
        return loss

    optimizer.step(closure)

    # Validation accuracy (skip if validation set empty)
    model.eval()
    with torch.no_grad():
        if X_val.shape[0] > 0:
            val_logits = model(X_val)
            val_pred = val_logits.argmax(dim=1)
            val_acc = (val_pred == y_val).float().mean().item()
        else:
            val_acc = 0.0  # placeholder when no validation data
    return model, val_acc


def tune_C(X_train, y_train, X_val, y_val, C_list):
    best_acc = -1.0
    best_C = None
    best_model = None
    for C in C_list:
        model, acc = train_logistic_regression(X_train, y_train, X_val, y_val, C, max_iter=1000)
        if acc > best_acc:
            best_acc = acc
            best_C = C
            best_model = model
    return best_model, best_C, best_acc


def retrain_on_combined(X_train, y_train, X_val, y_val, best_C):
    """Retrain on the union of train and validation sets."""
    X_trval = torch.cat([X_train, X_val], dim=0)
    y_trval = torch.cat([y_train, y_val], dim=0)
    # No validation needed; we can reuse the training function with empty val set.
    model, _ = train_logistic_regression(X_trval, y_trval,
                                         torch.empty(0, dtype=torch.long),  # dummy X_val
                                         torch.empty(0, dtype=torch.long),  # dummy y_val
                                         best_C, max_iter=1000)
    return model


def predict_and_save(model, X, idx, out_path, smoke=False):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        pred = logits.argmax(dim=1).cpu().numpy()
    if smoke:
        # take first 20 sorted indices
        idx_sorted = torch.sort(idx)[0]
        idx_sub = idx_sorted[:20]
        pred_sub = pred[idx_sorted[:20]]
        ids = idx_sub.cpu().numpy()
    else:
        ids = idx.cpu().numpy()
        pred_sub = pred
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('id,prediction\n')
        for node_id, p in zip(ids, pred_sub):
            f.write(f'{int(node_id)},{int(p)}\n')


def main():
    seed = int(os.getenv('SEED', '0'))
    smoke = os.getenv('SMOKE', '0') == '1'
    set_seeds(seed)

    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features.size(0)

    # Step 3: build adjacency
    adj = build_adjacency(edges, num_nodes)
    # Step 3 continued: add self-loops
    adj = add_self_loops(adj)
    # Step 4: symmetric normalization
    adj_norm = normalize_adjacency(adj)
    # Step 5: propagate features with K=2
    X_prop = propagate_features(adj_norm, features, K=2)
    # Step 6: skip row-normalization (as per plan)
    # Step 8: split
    X_train, y_train, X_val, y_val, X_test = split_data(X_prop, labels, train_idx, val_idx, test_idx)

    # Step 9-11: tune C
    C_list = [0.001, 0.01, 0.1, 1, 10, 100]
    best_model, best_C, best_val_acc = tune_C(X_train, y_train, X_val, y_val, C_list)
    print(f'Validation accuracy: {best_val_acc:.4f} (C={best_C})')

    # Step 12: retrain on train+val
    final_model = retrain_on_combined(X_train, y_train, X_val, y_val, best_C)

    # Step 13: predict test
    if smoke:
        # predict only first 20 test nodes
        predict_and_save(final_model, X_test, test_idx, '/output/predictions.csv', smoke=True)
        # val predictions: first 20 val nodes
        predict_and_save(final_model, X_prop[val_idx], val_idx, '/output/val_predictions.csv', smoke=True)
    else:
        predict_and_save(final_model, X_test, test_idx, '/output/predictions.csv', smoke=False)
        predict_and_save(final_model, X_prop[val_idx], val_idx, '/output/val_predictions.csv', smoke=False)


if __name__ == '__main__':
    start = time()
    main()
    print(f'Elapsed time: {time() - start:.2f}s')