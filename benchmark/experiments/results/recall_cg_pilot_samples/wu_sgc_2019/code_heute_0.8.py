# Plan step 1 -> implemented in: set_seeds
# Plan step 2 -> implemented in: load_data
# Plan step 3 -> implemented in: build_normalized_adj
# Plan step 4 -> implemented in: build_normalized_adj (normalization inside)
# Plan step 5 -> implemented in: propagate_features
# Plan step 6 -> implemented in: (row-normalization skipped as per plan)
# Plan step 8 -> implemented in: split_data (using provided indices)
# Plan step 9 -> implemented in: train_model (logistic regression with LBFGS)
# Plan step 10 -> implemented in: tune_hyperparameter (search over C)
# Plan step 11 -> implemented in: tune_hyperparameter (select best C)
# Plan step 12 -> implemented in: retrain_model (train on train+val)
# Plan step 13 -> implemented in: predict_and_write
# Plan step 14 -> implemented in: write_predictions
# Plan step 15 -> implemented in: main (runtime guard)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seeds():
    seed = int(os.environ.get('SEED', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    data = np.load('/data/cora_public.npz', allow_pickle=False)
    features = data['features'].astype(np.float32)
    edges = data['edges'].astype(np.int64)
    labels = data['labels'].astype(np.int64)
    train_idx = data['train_idx'].astype(np.int64)
    val_idx = data['val_idx'].astype(np.int64)
    test_idx = data['test_idx'].astype(np.int64)
    return features, edges, labels, train_idx, val_idx, test_idx

def build_normalized_adj(edges, num_nodes):
    # Add self-loops
    self_loops = np.stack([np.arange(num_nodes), np.arange(num_nodes)], axis=1)
    edge_list = np.concatenate([edges, self_loops], axis=0)  # (E + N, 2)
    # Compute out-degree from original edges (each undirected edge appears twice)
    out_deg = np.bincount(edges[:, 0], minlength=num_nodes).astype(np.float32)
    deg = out_deg + 1.0  # +1 for self-loop
    deg_sqrt_inv = 1.0 / np.sqrt(deg)
    # Normalization values for each directed edge (including self-loops)
    vals = deg_sqrt_inv[edge_list[:, 0]] * deg_sqrt_inv[edge_list[:, 1]]
    indices = torch.from_numpy(edge_list.T).long()
    values = torch.from_numpy(vals).float()
    A_hat = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes)).coalesce()
    return A_hat

def propagate_features(A_hat, X, K=2):
    X_prop = X.clone()
    for _ in range(K):
        X_prop = torch.sparse.mm(A_hat, X_prop)
    return X_prop

def split_data(features, labels, train_idx, val_idx, test_idx):
    X = torch.from_numpy(features).float()
    y = torch.from_numpy(labels).long()
    return X, y, train_idx, val_idx, test_idx

class LogisticRegression(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    def forward(self, x):
        return x @ self.weight + self.bias

def train_model(X, y, idx_train, idx_val, C, max_iter=1000):
    model = LogisticRegression(X.shape[1], int(y.max().item() + 1))
    lambda_l2 = 1.0 / C if C > 0 else 0.0
    optimizer = torch.optim.LBFGS(model.parameters(),
                                  lr=0.1,
                                  max_iter=max_iter,
                                  tolerance_grad=1e-06,
                                  tolerance_change=1e-09,
                                  history_size=100)
    def closure():
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits[idx_train], y[idx_train])
        if lambda_l2 > 0:
            l2_reg = lambda_l2 * (torch.sum(model.weight ** 2) + torch.sum(model.bias ** 2))
            loss = loss + l2_reg
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        val_logits = model(X[idx_val])
        val_pred = val_logits.argmax(dim=1)
        val_acc = (val_pred == y[idx_val]).float().mean().item()
    return model, val_acc

def tune_hyperparameter(X, y, train_idx, val_idx, C_list, smoke=False):
    best_C = None
    best_acc = -1.0
    best_model = None
    max_iter = 1 if smoke else 1000
    for C in C_list:
        model, acc = train_model(X, y, train_idx, val_idx, C, max_iter=max_iter)
        if acc > best_acc:
            best_acc = acc
            best_C = C
            best_model = model
    return best_C, best_acc, best_model

def retrain_model(X, y, train_idx, val_idx, C, smoke=False):
    combined_idx = np.concatenate([train_idx, val_idx])
    max_iter = 1 if smoke else 1000
    model, _ = train_model(X, y, combined_idx, combined_idx, C, max_iter=max_iter)  # dummy val idx not used
    return model

def predict_and_write(model, X, idx_out, out_path):
    model.eval()
    with torch.no_grad():
        logits = model(X[idx_out])
        preds = logits.argmax(dim=1).cpu().numpy()
    write_predictions(idx_out, preds, out_path)

def write_predictions(ids, preds, path):
    with open(path, 'w') as f:
        f.write('id,prediction\n')
        for idx, p in zip(ids, preds):
            f.write(f'{int(idx)},{int(p)}\n')

def main():
    set_seeds()
    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features.shape[0]
    A_hat = build_normalized_adj(edges, num_nodes)
    X_raw = torch.from_numpy(features).float()
    X = propagate_features(A_hat, X_raw, K=2)
    X, y, train_idx, val_idx, test_idx = split_data(features, labels, train_idx, val_idx, test_idx)

    smoke = os.getenv('SMOKE') == '1'
    if smoke:
        val_idx_used = val_idx[:20]
        test_idx_used = test_idx[:20]
    else:
        val_idx_used = val_idx
        test_idx_used = test_idx

    C_candidates = [0.001, 0.01, 0.1, 1, 10, 100]
    best_C, best_val_acc, _ = tune_hyperparameter(
        X, y, train_idx, val_idx_used, C_candidates, smoke=smoke
    )
    print(f'Validation accuracy: {best_val_acc:.4f}')

    # Retrain on train+val (using selected C)
    final_model = retrain_model(X, y, train_idx, val_idx_used, best_C, smoke=smoke)

    # Write test predictions
    predict_and_write(final_model, X, test_idx_used, '/output/predictions.csv')
    # Write validation predictions
    predict_and_write(final_model, X, val_idx_used, '/output/val_predictions.csv')

if __name__ == '__main__':
    main()
main()
