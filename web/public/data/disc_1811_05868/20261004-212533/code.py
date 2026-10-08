

# Plan step -> implemented in: function or class name
# 1. Model -> GCN class
# 2. Hyperparameters -> constants LR, WEIGHT_DECAY, DROPOUT, HIDDEN, NUM_LAYERS
# 3. Input processing -> load_and_preprocess function
# 4. Labels handling -> loss and accuracy computation uses train/val labels
# 5. Training -> train function (full-batch Adam)
# 6. Early stopping -> early stopping condition in training loop
# 7. Model selection -> best state restoration based on validation loss
# 8. Seeding -> set_seeds function called at start
# 9. Output -> write_predictions function for test and val

import os
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import csv

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_and_preprocess(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    features = data['features'].astype(np.float32)          # [N, F]
    edges = data['edges'].astype(np.int64)                  # [E, 2] directed pairs (both directions)
    labels = data['labels'].astype(np.int64)                # [N]
    train_idx = data['train_idx'].astype(np.int64)
    val_idx = data['val_idx'].astype(np.int64)
    test_idx = data['test_idx'].astype(np.int64)
    N = features.shape[0]

    # Remove self-loops from edges (if any)
    mask = edges[:, 0] != edges[:, 1]
    edges_no_self = edges[mask]

    # Add self-loops (will be accounted for in normalization)
    self_loop_edges = np.stack([np.arange(N), np.arange(N)], axis=1)
    all_edges = np.concatenate([edges_no_self, self_loop_edges], axis=0)

    # Build adjacency matrix (COO)
    adj = sp.coo_matrix(
        (np.ones(all_edges.shape[0], dtype=np.float32),
         (all_edges[:, 0], all_edges[:, 1])),
        shape=(N, N),
        dtype=np.float32
    )
    # Normalize: D^{-1/2} A D^{-1/2}
    deg = np.array(adj.sum(axis=1)).flatten() + 1e-12
    deg_inv_sqrt = np.power(deg, -0.5)
    row = adj.row
    col = adj.col
    norm_data = deg_inv_sqrt[row] * adj.data * deg_inv_sqrt[col]
    adj_norm = sp.coo_matrix((norm_data, (row, col)), shape=(N, N))

    # Convert to torch sparse tensor and coalesce to remove duplicates
    indices = torch.from_numpy(np.vstack([adj_norm.row, adj_norm.col]).astype(np.int64))
    values = torch.from_numpy(adj_norm.data.astype(np.float32))
    adj_tensor = torch.sparse_coo_tensor(indices, values, size=(N, N), dtype=torch.float32).coalesce()

    # Torch tensors for features and labels
    features_tensor = torch.from_numpy(features)
    labels_tensor = torch.from_numpy(labels)

    return features_tensor, adj_tensor, labels_tensor, train_idx, val_idx, test_idx, N

class GCN(nn.Module):
    def __init__(self, in_features, hidden, out_features, dropout):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.linear2 = nn.Linear(hidden, out_features)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, adj):
        # x: [N, F], adj: [N, N] sparse
        x = self.linear1(x)
        x = torch.sparse.mm(adj, x)   # A * X * W1
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = torch.sparse.mm(adj, x)   # A * ... * W2
        return x

def accuracy(logits, idx, labels):
    preds = logits[idx].argmax(dim=1)
    correct = (preds == labels[idx]).float().sum()
    return correct / len(idx)

def write_predictions(path, idx, pred_vals):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'prediction'])
        for node_id, pred in zip(idx, pred_vals):
            writer.writerow([int(node_id), int(pred)])

def main():
    seed = int(os.environ.get('SEED', '0'))
    set_seeds(seed)

    smoke = os.environ.get('SMOKE', '0') == '1'

    features, adj, labels, train_idx, val_idx, test_idx, N = load_and_preprocess('/data/cora_lcc_public.npz')

    # Hyperparameters (from optimized config)
    LR = 0.01
    WEIGHT_DECAY = 0.001
    DROPOUT = 0.8
    HIDDEN = 64
    NUM_LAYERS = 2  # we use 2-layer GCN
    NUM_CLASSES = int(labels.max().item()) + 1  # labels are 0..6 for training/val

    model = GCN(in_features=features.shape[1],
                hidden=HIDDEN,
                out_features=NUM_CLASSES,
                dropout=DROPOUT)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    max_epochs = 200 if not smoke else 1
    best_val_loss = float('inf')
    best_state = None
    val_losses = []

    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = criterion(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj)
            val_loss = criterion(logits[val_idx], labels[val_idx]).item()
            val_acc = accuracy(val_logits, val_idx, labels).item()
        model.train()

        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

        # Early stopping: stop if val_loss > mean of last 10 val losses
        if len(val_losses) >= 10:
            recent = val_losses[-10:]
            if val_loss > np.mean(recent):
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation accuracy (using best model)
    model.eval()
    with torch.no_grad():
        final_logits = model(features, adj)
        final_val_acc = accuracy(final_logits, val_idx, labels).item()
    print(f'Validation Accuracy: {final_val_acc:.4f}')

    # Test predictions
    with torch.no_grad():
        test_logits = model(features, adj)
        if smoke:
            # predict only first 20 nodes of sorted test_idx
            sorted_test = np.sort(test_idx)
            subset = sorted_test[:20]
            mask = np.isin(test_idx, subset)
            test_subset_idx = test_idx[mask]
            test_pred = test_logits[test_subset_idx].argmax(dim=1).numpy()
            write_predictions('/output/predictions.csv', test_subset_idx, test_pred)
            # Val predictions: first 20 of sorted val_idx
            sorted_val = np.sort(val_idx)
            val_subset = sorted_val[:20]
            mask_v = np.isin(val_idx, val_subset)
            val_subset_idx = val_idx[mask_v]
            val_pred = final_logits[val_subset_idx].argmax(dim=1).numpy()
            write_predictions('/output/val_predictions.csv', val_subset_idx, val_pred)
        else:
            test_pred = test_logits[test_idx].argmax(dim=1).numpy()
            write_predictions('/output/predictions.csv', test_idx, test_pred)
            val_pred = final_logits[val_idx].argmax(dim=1).numpy()
            write_predictions('/output/val_predictions.csv', val_idx, val_pred)

if __name__ == '__main__':
    main()