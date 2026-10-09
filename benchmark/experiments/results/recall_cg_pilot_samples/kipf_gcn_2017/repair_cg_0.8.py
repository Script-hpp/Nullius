

# Plan step -> implemented in: load_and_preprocess_data
# Plan step -> implemented in: GraphConvolution
# Plan step -> implemented in: GCN
# Plan step -> implemented in: masked_cross_entropy
# Plan step -> implemented in: masked_accuracy
# Plan step -> implemented in: training loop with early stopping and prediction writing

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {"lr": 0.01, "weight_decay": 0.0005, "epochs": 200, "dropout": 0.5, "hidden": 16, "patience": 10}


def load_and_preprocess_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    features = data['features'].astype(np.float32)          # [N, 1433]
    edges = data['edges'].astype(np.int64)                  # [E, 2]
    labels = data['labels'].astype(np.int64)                # [N]
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    # Row-normalize features
    rowsum = features.sum(axis=1, keepdims=1)
    rowsum[rowsum == 0] = 1
    features = features / rowsum

    N = features.shape[0]
    # Build adjacency matrix (undirected, already contains both directions) + self-loops
    adj_indices = []
    adj_values = []
    for u, v in edges:
        adj_indices.append([u, v])
        adj_values.append(1.0)
    for i in range(N):
        adj_indices.append([i, i])
        adj_values.append(1.0)
    adj_indices = np.array(adj_indices).T  # [2, E']
    adj_values = np.array(adj_values, dtype=np.float32)

    # Compute degree matrix
    deg = np.zeros(N, dtype=np.float32)
    np.add.at(deg, adj_indices[0], adj_values)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0

    # Normalized adjacency values (symmetrical normalization)
    norm_values = deg_inv_sqrt[adj_indices[0]] * adj_values * deg_inv_sqrt[adj_indices[1]]

    # Remove possible duplicate entries by coalescing (sort and unique)
    # Convert to torch sparse tensor
    indices = torch.from_numpy(adj_indices).long()
    values = torch.from_numpy(norm_values)
    size = torch.Size([N, N])
    # Coalesce to sum duplicate entries and sort indices
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size).coalesce()
    support_indices = sparse_tensor.indices()
    support_values = sparse_tensor.values()
    support = torch.sparse_coo_tensor(support_indices, support_values, size).coalesce()

    features_tensor = torch.from_numpy(features)
    labels_tensor = torch.from_numpy(labels)

    # Masks
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return features_tensor, support, labels_tensor, train_mask, val_mask, test_mask


class GraphConvolution(nn.Module):
    """
    Simple GCN layer: support * X * W
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, input, support):
        # input: [N, in_features]
        # support: sparse [N, N]
        support_x = torch.sparse.mm(support, input)   # [N, in_features]
        output = torch.mm(support_x, self.weight)     # [N, out_features]
        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, support):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc1(x, support)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, support)
        return x


def masked_cross_entropy(logits, labels, mask):
    """Masked cross entropy loss."""
    logits_masked = logits[mask]
    labels_masked = labels[mask].long()
    return F.cross_entropy(logits_masked, labels_masked)


def masked_accuracy(logits, labels, mask):
    """Masked accuracy."""
    logits_masked = logits[mask]
    labels_masked = labels[mask].long()
    preds = logits_masked.argmax(dim=1)
    correct = (preds == labels_masked).float()
    return correct.mean()


def main():
    seed = int(os.environ.get("SEED", "123"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    smoke = os.environ.get("SMOKE") == "1"

    features, support, labels, train_mask, val_mask, test_mask = load_and_preprocess_data('/data/cora_public.npz')

    # Determine number of classes (ignore -1 labels)
    num_classes = int(labels[labels >= 0].max().item() + 1)

    model = GCN(
        nfeat=features.shape[1],
        nhid=CONFIG["hidden"],
        nclass=num_classes,
        dropout=CONFIG["dropout"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    best_val_loss = float('inf')
    best_state = None
    epochs_since_improve = 0

    max_epochs = 1 if smoke else CONFIG["epochs"]

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, support)
        loss_train = masked_cross_entropy(output, labels, train_mask)
        # weight decay on first layer only
        l2_reg = CONFIG["weight_decay"] * torch.sum(model.gc1.weight ** 2)
        loss = loss_train + l2_reg
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(features, support)
            val_loss = masked_cross_entropy(val_output, labels, val_mask)
            val_acc = masked_accuracy(val_output, labels, val_mask)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= CONFIG["patience"]:
            break

        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation accuracy
    model.eval()
    with torch.no_grad():
        val_output = model(features, support)
        val_acc = masked_accuracy(val_output, labels, val_mask)
    print(f'Validation Accuracy: {val_acc:.4f}')

    # Predictions for test and val
    with torch.no_grad():
        logits = model(features, support)
        preds = logits.argmax(dim=1).cpu().numpy()

    # Output directory
    os.makedirs('/output', exist_ok=True)

    # Test predictions
    test_ids = test_idx if not smoke else np.sort(test_idx)[:20]
    test_pred = preds[test_ids]
    with open('/output/predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for idx, p in zip(test_ids, test_pred):
            f.write(f'{int(idx)},{int(p)}\n')

    # Validation predictions
    val_ids = val_idx if not smoke else np.sort(val_idx)[:20]
    val_pred = preds[val_ids]
    with open('/output/val_predictions.csv', 'w') as f:
        f.write('id,prediction\n')
        for idx, p in zip(val_ids, val_pred):
            f.write(f'{int(idx)},{int(p)}\n')


if __name__ == '__main__':
    main()