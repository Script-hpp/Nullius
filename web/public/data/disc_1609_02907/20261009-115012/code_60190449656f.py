

# Plan step -> implemented in: row_normalize_features function
# Plan step -> implemented in: normalize_adjacency function
# Plan step -> implemented in: GraphConvolution.reset_parameters
# Plan step -> implemented in: training loop (optimizer and weight_decay term)
# Plan step -> implemented in: GraphConvolution.forward dropout
# Plan step -> implemented in: training loop with early stopping
# Plan step -> implemented in: validation mask usage
# Plan step -> implemented in: masked loss calculation
# Plan step -> implemented in: processing all nodes each epoch
# Plan step -> implemented in: GCN class with two GraphConvolution layers
# Plan step -> implemented in: seed setup
# Plan step -> implemented in: restoring best model and inference
# Plan step -> implemented in: call to nio.write_predictions
# Plan step -> implemented in: print statement

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import nullius_io as nio

CONFIG = {"lr": 0.01, "epochs": 200, "dropout": 0.5, "hidden": 16, "patience": 10}
WEIGHT_DECAY = 5e-4
NUM_CLASSES = 7

def row_normalize_features(mx: np.ndarray) -> np.ndarray:
    """Row-normalize feature matrix (dense)."""
    rowsum = np.array(mx.sum(axis=1))
    r_inv = np.power(rowsum, -1)
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = np.diag(r_inv)
    return r_mat_inv.dot(mx)

def normalize_adjacency(adj: sp.csr_matrix) -> torch.Tensor:
    """Adjacency renormalization trick: Â = A + I, Ă = D^{-1/2} Â D^{-1/2}."""
    adj = adj + sp.eye(adj.shape[0])  # add self-loops
    degree = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    # Convert to torch sparse COO tensor
    adj_coo = adj_norm.tocoo()
    indices = torch.from_numpy(np.vstack((adj_coo.row, adj_coo.col)).astype(np.int64))
    values = torch.from_numpy(adj_coo.data.astype(np.float32))
    shape = torch.Size(adj_coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        """
        input: [N, in_features] dense
        adj: [N, N] sparse torch tensor (already normalized)
        """
        input = F.dropout(input, self.dropout, training=self.training)
        support = torch.mm(input, self.weight)          # [N, out_features]
        output = torch.sparse.mm(adj, support)          # [N, out_features]
        if self.act is not None:
            output = self.act(output)
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, dropout=dropout, act=F.relu)
        self.gc2 = GraphConvolution(nhid, nclass, dropout=dropout, act=lambda x: x)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        return x

def accuracy(output, labels):
    preds = output.argmax(dim=1).eq(labels).double()
    return preds.sum() / len(labels)

def main():
    # Seed handling
    seed = int(os.environ.get("SEED", 123))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data via nullius_io
    data = nio.load()
    features = data["features"]          # numpy [2708,1433] float32
    edges = data["edges"]                # numpy [E,2] int64
    labels = data["labels"]              # numpy [2708] int64, -1 for test
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    N = features.shape[0]

    # Preprocess features: row-normalize
    features = row_normalize_features(features)
    features = torch.from_numpy(features).float()

    # Build adjacency matrix (undirected, edges already contain both directions)
    adj = sp.csr_matrix((np.ones(edges.shape[0], dtype=np.float32),
                         (edges[:, 0], edges[:, 1])), shape=(N, N))
    adj_tensor = normalize_adjacency(adj)

    # Labels as torch long (ignore -1 later via masks)
    labels_tensor = torch.from_numpy(labels).long()

    # Masks
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]   = True
    test_mask[test_idx] = True

    # Model
    model = GCN(nfeat=features.shape[1],
                nhid=CONFIG["hidden"],
                nclass=NUM_CLASSES,
                dropout=CONFIG["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # Smoke mode: exactly 1 epoch
    smoke = os.environ.get("SMOKE") == "1"
    max_epochs = 1 if smoke else CONFIG["epochs"]
    patience = CONFIG["patience"] if not smoke else max_epochs  # disable early stop in smoke

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        output = model(features, adj_tensor)               # [N, num_classes]

        # Loss on training nodes only
        loss_train = F.cross_entropy(output[train_mask], labels_tensor[train_mask])
        # L2 weight decay on first GCN layer
        l2_reg = WEIGHT_DECAY * torch.sum(model.gc1.weight ** 2)
        loss = loss_train + l2_reg
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            output_val = model(features, adj_tensor)
            loss_val = F.cross_entropy(output_val[val_mask], labels_tensor[val_mask]) \
                       + WEIGHT_DECAY * torch.sum(model.gc1.weight ** 2)
            acc_val = accuracy(output_val[val_mask], labels_tensor[val_mask])

        # Early stopping check
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if not smoke:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        if not smoke and (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Val Loss: {loss_val.item():.4f} | Val Acc: {acc_val.item()*100:.2f}%")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation accuracy (using best model)
    model.eval()
    with torch.no_grad():
        final_output = model(features, adj_tensor)
        final_acc = accuracy(final_output[val_mask], labels_tensor[val_mask])
        print(f"Validation Accuracy: {final_acc.item()*100:.2f}%")

    # Prepare predictions for all nodes (logits)
    with torch.no_grad():
        logits = model(features, adj_tensor)   # [N, num_classes]
    # Convert to numpy for nullius_io
    preds_np = logits.cpu().numpy()
    nio.write_predictions(preds_np)

if __name__ == "__main__":
    main()