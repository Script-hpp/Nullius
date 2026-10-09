

# Plan step -> implemented in: load_data
# Plan step -> implemented in: preprocess_adjacency
# Plan step -> implemented in: GCN class
# Plan step -> implemented in: train_loop
# Plan step -> implemented in: early_stopping
# Plan step -> implemented in: restore_best_and_predict
# Plan step -> implemented in: output_predictions

import os
import numpy as np
import torch
import torch.nn.functional as F
import nullius_io as nio

# Hyperparameters from CONFIG dict as required
CONFIG = {"lr": 0.01, "epochs": 100000, "layers": 2, "patience": 50}
LR = CONFIG["lr"]
EPOCHS = CONFIG["epochs"]
HIDDEN = 64  # from spec
DROPOUT = 0.8  # from spec
WEIGHT_DECAY = 0.001  # from spec
PATIENCE = CONFIG["patience"]

def load_data():
    d = nio.load()
    features = torch.from_numpy(d["features"].astype(np.float32))  # [N, F]
    edges = torch.from_numpy(d["edges"].astype(np.int64))          # [E, 2]
    labels = torch.from_numpy(d["labels"].astype(np.int64))       # [N]
    train_idx = torch.from_numpy(d["train_idx"].astype(np.int64))
    val_idx = torch.from_numpy(d["val_idx"].astype(np.int64))
    test_idx = torch.from_numpy(d["test_idx"].astype(np.int64))
    return features, edges, labels, train_idx, val_idx, test_idx

def preprocess_adjacency(num_nodes, edges):
    # Add self-loops
    self_loop = torch.arange(num_nodes, device=edges.device).unsqueeze(1)
    self_loop = torch.cat([self_loop, self_loop], dim=1)  # [N,2]
    all_edges = torch.cat([edges, self_loop], dim=0)      # [E+N, 2]
    # Build sparse adjacency matrix (COO)
    idx = all_edges.t()                                   # [2, E+N]
    vals = torch.ones(idx.size(1), dtype=torch.float32, device=edges.device)
    adj_torch = torch.sparse_coo_tensor(idx, vals, size=(num_nodes, num_nodes))
    # Compute degree vector
    deg = torch.sparse.sum(adj_torch, dim=1).to_dense()   # [N]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    # Normalize: Â = D^{-1/2} (A+I) D^{-1/2}
    row = idx[0]
    col = idx[1]
    norm_vals = deg_inv_sqrt[row] * vals * deg_inv_sqrt[col]
    adj_norm = torch.sparse_coo_tensor(idx, norm_vals, size=(num_nodes, num_nodes))
    return adj_norm.coalesce()

class GCN(torch.nn.Module):
    def __init__(self, in_features, hidden, out_features, adj_norm, dropout):
        super().__init__()
        self.adj_norm = adj_norm
        self.dropout = dropout
        self.W1 = torch.nn.Parameter(torch.empty(in_features, hidden))
        self.b1 = torch.nn.Parameter(torch.zeros(hidden))
        self.W2 = torch.nn.Parameter(torch.empty(hidden, out_features))
        self.b2 = torch.nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W1)
        torch.nn.init.xavier_uniform_(self.W2)
    
    def forward(self, x):
        # Layer 0
        x_drop = F.dropout(x, p=self.dropout, training=self.training)
        support = torch.mm(x_drop, self.W1) + self.b1          # [N, hidden]
        x = torch.sparse.mm(self.adj_norm, support)            # [N, hidden]
        x = torch.relu(x)
        # Layer 1
        x_drop = F.dropout(x, p=self.dropout, training=self.training)
        support = torch.mm(x_drop, self.W2) + self.b2          # [N, out]
        out = torch.sparse.mm(self.adj_norm, support)          # [N, out]
        return out

def train_loop(model, features, adj_norm, labels, train_idx, val_idx):
    optimizer = torch.optim.Adam([
        {"params": [model.W1, model.b1], "weight_decay": WEIGHT_DECAY},
        {"params": [model.W2, model.b2], "weight_decay": 0.0}
    ], lr=LR)
    
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    
    # Convert to tensors on same device
    device = features.device
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(features)                     # [N, C]
        loss_train = F.cross_entropy(logits[train_idx], labels[train_idx])
        # Add L2 regularization on first layer weights (already in optimizer weight_decay for W1)
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(features)
            loss_val = F.cross_entropy(logits[val_idx], labels[val_idx])
            # Compute validation accuracy
            preds_val = logits[val_idx].argmax(dim=1)
            acc_val = (preds_val == labels[val_idx]).float().mean().item()
        
        # Early stopping (gnnbench style: based on validation loss improvement)
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= PATIENCE:
            break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss, acc_val

def main():
    # Seed
    seed = int(os.getenv("SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Smoke mode
    smoke = os.getenv("SMOKE") == "1"
    if smoke:
        global EPOCHS
        EPOCHS = 1  # train exactly 1 epoch
    
    # Load data
    features, edges, labels, train_idx, val_idx, test_idx = load_data()
    num_nodes = features.shape[0]
    
    # Preprocess adjacency
    adj_norm = preprocess_adjacency(num_nodes, edges)
    
    # Move to device (CPU)
    device = torch.device("cpu")
    features = features.to(device)
    adj_norm = adj_norm.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    
    # Model
    model = GCN(in_features=features.shape[1],
                hidden=HIDDEN,
                out_features=int(labels.max().item()) + 1,  # 7 classes for Cora
                adj_norm=adj_norm,
                dropout=DROPOUT).to(device)
    
    # Train
    model, val_loss, val_acc = train_loop(model, features, adj_norm, labels, train_idx, val_idx)
    
    # Print validation accuracy as required
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Predict on all nodes
    model.eval()
    with torch.no_grad():
        logits = model(features)  # [N, C]
    # Convert to numpy for nullius_io
    preds_numpy = logits.cpu().numpy()
    nio.write_predictions(preds_numpy)

if __name__ == "__main__":
    main()