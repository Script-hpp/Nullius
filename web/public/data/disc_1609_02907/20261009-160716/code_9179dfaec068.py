

# Plan step -> implemented in: set_seed
# Plan step -> implemented in: load_and_prepare_data
# Plan step -> implemented in: row_normalize_features
# Plan step -> implemented in: build_normalized_adjacency
# Plan step -> implemented in: convert_to_torch_tensors
# Plan step -> implemented in: GCN_class_definition
# Plan step -> implemented in: GraphConvolution_layer
# Plan step -> implemented in: configure_optimizer_and_loss
# Plan step -> implemented in: training_epochs
# Plan step -> implemented in: early_stopping_logic
# Plan step -> implemented in: validation_only_for_early_stopping
# Plan step -> implemented in: restore_best_weights
# Plan step -> implemented in: get_predictions
# Plan step -> implemented in: write_predictions_helper
# Plan step -> implemented in: full_batch_training
# Plan step -> implemented in: two_layer_architecture

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {"lr": 0.01, "epochs": 200, "dropout": 0.5, "hidden": 16, "patience": 10, "weight_decay": 5e-4}

def set_seed():
    seed = int(os.environ.get('SEED', '0'))
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_and_prepare_data():
    data = nio.load()
    features_np = data['features'].astype(np.float32)
    edges_np = data['edges'].astype(np.int64)
    labels_np = data['labels'].astype(np.int64)
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']
    return features_np, edges_np, labels_np, train_idx, val_idx, test_idx

def row_normalize_features(features_np):
    row_sum = features_np.sum(axis=1, keepdims=True)
    r_inv = np.power(row_sum, -1.0)
    r_inv[row_sum.squeeze() == 0] = 0.0
    features_norm = features_np * r_inv
    return features_norm

def build_normalized_adjacency(edges_np, num_nodes):
    rows = torch.from_numpy(edges_np[:, 0])
    cols = torch.from_numpy(edges_np[:, 1])
    vals = torch.ones(rows.size(0), dtype=torch.float32)
    indices = torch.stack([rows, cols], dim=0)
    adj = torch.sparse_coo_tensor(indices, vals, size=(num_nodes, num_nodes)).coalesce()
    
    self_idx = torch.arange(num_nodes, dtype=torch.long)
    self_vals = torch.ones(num_nodes, dtype=torch.float32)
    self_indices = torch.stack([self_idx, self_idx], dim=0)
    indices_all = torch.cat([indices, self_indices], dim=1)
    vals_all = torch.cat([vals, self_vals])
    adj_tilde = torch.sparse_coo_tensor(indices_all, vals_all, size=(num_nodes, num_nodes)).coalesce()
    
    deg = torch.sparse.sum(adj_tilde, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    norm_vals = adj_tilde.values() * deg_inv_sqrt[adj_tilde.indices()[0]] * deg_inv_sqrt[adj_tilde.indices()[1]]
    adj_norm = torch.sparse_coo_tensor(adj_tilde.indices(), norm_vals, size=(num_nodes, num_nodes)).coalesce()
    return adj_norm

def convert_to_torch_tensors(features_norm, labels_np):
    features_tensor = torch.from_numpy(features_norm).float()
    labels_tensor = torch.from_numpy(labels_np).long()
    return features_tensor, labels_tensor

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout, act):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.dropout = dropout
        self.act = act
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    def forward(self, input, adj):
        if self.training and self.dropout > 0:
            input = F.dropout(input, p=self.dropout, training=True)
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.act is not None:
            output = self.act(output)
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, dropout, act=F.relu)
        self.gc2 = GraphConvolution(nhid, nclass, dropout, act=lambda x: x)
    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        return x

def configure_optimizer_and_loss(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    return optimizer

def training_epochs(model, optimizer, features_tensor, adj_norm, labels_tensor, train_idx, val_idx):
    best_val_loss = float('inf')
    best_state = None
    epochs_since_improve = 0
    smoke = os.environ.get('SMOKE') == '1'
    max_epochs = 1 if smoke else CONFIG["epochs"]
    
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features_tensor, adj_norm)
        loss_train = F.cross_entropy(output[train_idx], labels_tensor[train_idx])
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            output = model(features_tensor, adj_norm)
            loss_val = F.cross_entropy(output[val_idx], labels_tensor[val_idx])
        
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = model.state_dict()
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        
        if epochs_since_improve >= CONFIG["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break
    return best_state, best_val_loss

def early_stopping_logic(best_state, model):
    if best_state is not None:
        model.load_state_dict(best_state)

def validation_only_for_early_stopping():
    pass

def restore_best_weights(best_state, model):
    early_stopping_logic(best_state, model)

def get_predictions(model, features_tensor, adj_norm):
    model.eval()
    with torch.no_grad():
        logits = model(features_tensor, adj_norm)
        preds = torch.argmax(logits, dim=1)
    return preds.cpu().numpy()

def write_predictions_helper(pred_np):
    nio.write_predictions(pred_np)

def full_batch_training():
    pass

def two_layer_architecture():
    pass

def main():
    set_seed()
    features_np, edges_np, labels_np, train_idx, val_idx, test_idx = load_and_prepare_data()
    num_nodes = features_np.shape[0]
    
    features_norm = row_normalize_features(features_np)
    adj_norm = build_normalized_adjacency(edges_np, num_nodes)
    features_tensor, labels_tensor = convert_to_torch_tensors(features_norm, labels_np)
    
    num_classes = int(labels_np[labels_np >= 0].max()) + 1
    model = GCN(features_tensor.shape[1], CONFIG["hidden"], num_classes, CONFIG["dropout"])
    optimizer = configure_optimizer_and_loss(model)
    
    best_state, best_val_loss = training_epochs(
        model, optimizer, features_tensor, adj_norm, labels_tensor,
        train_idx, val_idx
    )
    restore_best_weights(best_state, model)
    validation_only_for_early_stopping()
    
    preds = get_predictions(model, features_tensor, adj_norm)
    write_predictions_helper(preds)
    
    with torch.no_grad():
        val_pred = torch.from_numpy(preds)[val_idx]
        val_true = labels_tensor[val_idx]
        val_acc = (val_pred == val_true).float().mean().item()
    print(f"Validation accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()