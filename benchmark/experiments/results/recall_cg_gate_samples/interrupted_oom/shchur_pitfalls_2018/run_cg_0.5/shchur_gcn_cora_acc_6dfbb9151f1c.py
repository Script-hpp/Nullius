

# Plan step -> implemented in: load_data
# Plan step -> implemented in: preprocess_adjacency
# Plan step -> implemented in: GCN
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
CONFIG = {"lr": 0.01, "epochs": 100000, "layers": 2, "patience": 50, "hidden_size": 64, "weight_decay": 0.001, "dropout": 0.8}

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
    def __init__(self, in_features, hidden_size, out_features, adj_norm, dropout, num_layers):
        super().__init__()
        self.adj_norm = adj_norm
        self.dropout = dropout
        self.num_layers = num_layers
        self.linears = torch.nn.ModuleList()
        # Build layers
        if num_layers == 1:
            self.linears.append(torch.nn.Linear(in_features, out_features))
        else:
            # First layer
            self.linears.append(torch.nn.Linear(in_features, hidden_size))
            # Hidden layers
            for _ in range(num_layers - 2):
                self.linears.append(torch.nn.Linear(hidden_size, hidden_size))
            # Last layer
            self.linears.append(torch.nn.Linear(hidden_size, out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        for lin in self.linears:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
    
    def forward(self, x):
        for i, lin in enumerate(self.linears):
            # Apply dropout to inputs (except possibly after last layer? we apply before each linear)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
            if i < self.num_layers - 1:  # not last layer
                x = torch.relu(x)
                # Graph convolution step: multiply with normalized adjacency
                x = torch.sparse.mm(self.adj_norm, x)
            # After last layer we do not apply adjacency multiplication (as per original GCN)
        return x

def train_loop(model, features, adj_norm, labels, train_idx, val_idx):
    # Separate weight decay: apply only to first layer's weights
    first_layer_weight = model.linears[0].weight
    first_layer_bias = model.linears[0].bias
    other_params = []
    for i, lin in enumerate(model.linears):
        if i == 0:
            continue
        other_params.extend([lin.weight, lin.bias])
    optimizer = torch.optim.Adam([
        {"params": [first_layer_weight], "weight_decay": CONFIG["weight_decay"]},
        {"params": [first_layer_bias], "weight_decay": 0.0},
        {"params": other_params, "weight_decay": 0.0}
    ], lr=CONFIG["lr"])
    
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    
    device = features.device
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        optimizer.zero_grad()
        logits = model(features)                     # [N, C]
        loss_train = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(features)
            loss_val = F.cross_entropy(logits[val_idx], labels[val_idx])
            preds_val = logits[val_idx].argmax(dim=1)
            acc_val = (preds_val == labels[val_idx]).float().mean().item()
        
        # Early stopping (gnnbench style: based on validation loss improvement)
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= CONFIG["patience"]:
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
        global CONFIG
        CONFIG = dict(CONFIG)  # copy
        CONFIG["epochs"] = 1  # train exactly 1 epoch
    
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
    
    # Determine number of classes
    num_classes = int(labels.max().item()) + 1  # 7 for Cora
    
    # Model
    model = GCN(in_features=features.shape[1],
                hidden_size=CONFIG["hidden_size"],
                out_features=num_classes,
                adj_norm=adj_norm,
                dropout=CONFIG["dropout"],
                num_layers=CONFIG["layers"]).to(device)
    
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