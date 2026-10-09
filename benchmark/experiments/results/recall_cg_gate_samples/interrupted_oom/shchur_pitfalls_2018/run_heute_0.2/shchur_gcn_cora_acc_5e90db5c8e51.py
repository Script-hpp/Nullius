# Plan step -> implemented in: function or class name
# Read SEED env var, set numpy.random.seed(SEED) and torch.manual_seed(SEED) (CPU only) -> set_seeds()
# Load /data/cora_lcc_public.npz; features ∈ ℝ^{2485×1433}, edges, labels, train/val/test indices -> load_data()
# Model: 2‑layer GCN, input_dim=1433, hidden_dim=16, output_dim=7, ReLU activation, dropout p=0.5 after each hidden layer -> GCN class
# Optimizer: Adam(lr=0.01, weight_decay=5e-4) -> create_optimizer()
# Loss: CrossEntropyLoss computed only on training‑labelled nodes -> compute_loss()
# Training: max_epochs=200; early stopping on validation loss with patience=50 epochs; use train_idx for gradient steps, val_idx for early‑stopping check -> train_model()
# After training, run inference on all nodes, extract predictions for test_idx -> get_predictions()
# Write /output/predictions.csv with header id,prediction; id = node index from test_idx, prediction = predicted class (0‑6) -> nio.write_predictions()
# Ensure everything runs on CPU; total runtime should stay well under 5 minutes -> device = torch.device('cpu')
# The env var SPLIT is informational only; do not recompute splits -> ignored
# All uncertain hyper‑parameters are marked above; adjust if the paper specifies different values -> hyperparameters defined as constants
# SMOKE mode handling -> if SMOKE==1: max_epochs=1, skip early stopping

import os
import numpy as np
import torch
import torch.nn.functional as F
import nullius_io as nio

# -------------------- Hyperparameters --------------------
INPUT_DIM = 1433
HIDDEN_DIM = 16          # uncertain, chosen small for speed
OUTPUT_DIM = 7
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 200
PATIENCE = 50
# --------------------------------------------------------

def set_seeds():
    seed = int(os.getenv('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data():
    d = nio.load()
    features = torch.from_numpy(d['features']).float()          # [N, 1433]
    edges = torch.from_numpy(d['edges']).long()                # [E, 2]
    labels = torch.from_numpy(d['labels']).long()              # [N]
    train_idx = torch.from_numpy(d['train_idx']).long()
    val_idx = torch.from_numpy(d['val_idx']).long()
    test_idx = torch.from_numpy(d['test_idx']).long()
    return features, edges, labels, train_idx, val_idx, test_idx

def normalize_adj(edges, num_nodes):
    # Add self-loops
    self_loop = torch.arange(num_nodes, dtype=torch.long)
    self_loop_edges = torch.stack([self_loop, self_loop], dim=1)  # [N,2]
    all_edges = torch.cat([edges, self_loop_edges], dim=0)       # [E+N,2]
    row, col = all_edges.unbind(dim=1)

    # Compute degrees
    deg = torch.zeros(num_nodes, dtype=torch.float)
    deg.index_add_(0, row, torch.ones_like(row, dtype=torch.float))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Normalization values
    norm_vals = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # Build sparse adjacency matrix
    idx = torch.stack([row, col], dim=0)   # [2, E+N]
    adj = torch.sparse_coo_tensor(idx, norm_vals, size=(num_nodes, num_nodes),
                                  dtype=torch.float)
    return adj.coalesce()

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, x, adj):
        # First GCN layer
        x = self.relu(self.dropout(torch.sparse.mm(adj, self.lin1(x))))
        # Second GCN layer (logits)
        x = self.lin2(x)
        return x

def create_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def compute_loss(logits, labels, idx):
    return F.cross_entropy(logits[idx], labels[idx])

def train_model(model, features, adj, labels, train_idx, val_idx, smoke=False):
    optimizer = create_optimizer(model)
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    epochs = 1 if smoke else MAX_EPOCHS

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = compute_loss(logits, labels, train_idx)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(features, adj)
            val_loss = compute_loss(val_logits, labels, val_idx)
            val_pred = val_logits[val_idx].argmax(dim=1)
            val_acc = (val_pred == labels[val_idx]).float().mean().item()

        if smoke:
            # In smoke mode we just run one epoch and break
            break

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Return final validation accuracy (from last evaluated epoch)
    model.eval()
    with torch.no_grad():
        final_logits = model(features, adj)
        final_pred = final_logits[val_idx].argmax(dim=1)
        final_acc = (final_pred == labels[val_idx]).float().mean().item()
    print(f"Validation accuracy: {final_acc:.4f}")
    return final_acc

def get_predictions(model, features, adj):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)   # [N, 7]
    return logits  # can be torch tensor or numpy; helper accepts both

def main():
    set_seeds()
    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features.shape[0]

    adj = normalize_adj(edges, num_nodes)

    smoke = os.getenv('SMOKE') == '1'
    model = GCN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)

    train_model(model, features, adj, labels, train_idx, val_idx, smoke=smoke)

    preds = get_predictions(model, features, adj)  # logits [N,7] or could convert to class
    nio.write_predictions(preds)

if __name__ == "__main__":
    main()
