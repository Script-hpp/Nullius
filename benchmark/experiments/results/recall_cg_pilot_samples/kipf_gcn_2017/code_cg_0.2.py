# Plan step -> implemented in: GCN class
# Plan step -> implemented in: GraphConvolution forward with ReLU and linear activations
# Plan step -> implemented in: dropout applied inside GraphConvolution forward
# Plan step -> implemented in: weight_decay_loss function for L2 on first layer weights
# Plan step -> implemented in: optimizer initialization with Adam and learning rate from CONFIG
# Plan step -> implemented in: masked_cross_entropy and masked_accuracy functions
# Plan step -> implemented in: preprocessing (row-normalize features, symmetric normalized adjacency)
# Plan step -> implemented in: training loop with early stopping based on validation loss
# Plan step -> implemented in: model selection by restoring weights from best validation loss
# Plan step -> implemented in: set_seed function reading SEED environment variable
# Plan step -> implemented in: writing predictions.csv and val_predictions.csv to /output
# Plan step -> batch_size: full batch (no mini‑batching used)
# Plan step -> layers: 2‑layer GCN (two GraphConvolution layers)
# Plan step -> heads: not applicable (single‑head GCN)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

CONFIG = {"lr": 0.01, "weight_decay": 0.0005, "epochs": 200, "dropout": 0.5, "hidden": 16, "patience": 10}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

seed = int(os.environ.get('SEED', 0))
set_seed(seed)

# Load data
data = np.load('/data/cora_public.npz')
features = data['features'].astype(np.float32)          # [2708, 1433]
edges = data['edges']                                   # [E, 2]
labels = data['labels']                                 # [2708]
train_idx = data['train_idx']
val_idx = data['val_idx']
test_idx = data['test_idx']

# Preprocess features: row-normalize
rowsum = features.sum(axis=1, keepdims=True)
rowsum[rowsum == 0] = 1.0
features_norm = features / rowsum
features_tensor = torch.from_numpy(features_norm)       # [N, F]

# Build adjacency matrix with self-loops
N = labels.shape[0]
adj = torch.zeros((N, N), dtype=torch.float32)
u, v = edges[:, 0], edges[:, 1]
adj[u, v] = 1.0
adj += torch.eye(N, dtype=torch.float32)                # add self-loops

# Symmetric normalization: \hat{A} = D^{-1/2} (A+I) D^{-1/2}
deg = adj.sum(dim=1)
deg_inv_sqrt = torch.pow(deg, -0.5)
deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
adj_norm = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)  # dense [N, N]

# Masks
train_mask = torch.zeros(N, dtype=torch.bool)
val_mask = torch.zeros(N, dtype=torch.bool)
test_mask = torch.zeros(N, dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

# Labels (replace -1 with 0 for placeholder; masked nodes ignored in loss/acc)
labels_long = torch.from_numpy(labels).long()
labels_long[labels_long == -1] = 0
nclass = int(labels_long.max().item() + 1)               # should be 7

# GraphConvolution layer
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x, adj):
        if self.training and self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        support = torch.mm(adj, x)          # [N, in_features]
        out = torch.mm(support, self.weight)  # [N, out_features]
        if self.act is not None:
            out = self.act(out)
        return out

# GCN model (2 layers)
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, dropout=dropout, act=F.relu)
        self.gc2 = GraphConvolution(nhid, nclass, dropout=dropout, act=lambda x: x)
    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        return x

model = GCN(nfeat=features_tensor.shape[1],
            nhid=CONFIG["hidden"],
            nclass=nclass,
            dropout=CONFIG["dropout"])

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=0.0)

def weight_decay_loss():
    return CONFIG["weight_decay"] * torch.norm(model.gc1.weight, p=2)**2

def masked_cross_entropy(logits, labels, mask):
    loss = F.cross_entropy(logits, labels, reduction='none')
    loss = loss * mask.float()
    return loss.sum() / mask.float().sum()

def masked_accuracy(logits, labels, mask):
    pred = logits.argmax(dim=1)
    correct = (pred == labels).float()
    correct = correct * mask.float()
    return correct.sum() / mask.float().sum()

# Training
best_val_loss = float('inf')
best_state = None
epochs_since_improve = 0
epochs_to_run = 1 if os.environ.get('SMOKE') == '1' else CONFIG["epochs"]

adj_dense = adj_norm  # use dense adjacency for simplicity
start_time = time.time()
for epoch in range(epochs_to_run):
    model.train()
    optimizer.zero_grad()
    output = model(features_tensor, adj_dense)
    loss = masked_cross_entropy(output, labels_long, train_mask)
    loss += weight_decay_loss()
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_output = model(features_tensor, adj_dense)
        val_loss = masked_cross_entropy(val_output, labels_long, val_mask)
        val_acc = masked_accuracy(val_output, labels_long, val_mask)
    
    print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f} | Val Acc: {val_acc.item():.5f}")
    
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_state = model.state_dict()
        epochs_since_improve = 0
    else:
        epochs_since_improve += 1
        if epochs_since_improve >= CONFIG["patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

if best_state is not None:
    model.load_state_dict(best_state)

# Final validation accuracy
model.eval()
with torch.no_grad():
    final_output = model(features_tensor, adj_dense)
    final_val_acc = masked_accuracy(final_output, labels_long, val_mask)
    print(f"Validation accuracy: {final_val_acc.item():.5f}")

# Predictions
with torch.no_grad():
    full_output = model(features_tensor, adj_dense)
    test_pred = full_output[test_idx].argmax(dim=1).cpu().numpy()
    val_pred = full_output[val_idx].argmax(dim=1).cpu().numpy()

out_dir = '/output'
os.makedirs(out_dir, exist_ok=True)

# Write test predictions
pred_path = os.path.join(out_dir, 'predictions.csv')
with open(pred_path, 'w') as f:
    f.write('id,prediction\n')
    if os.environ.get('SMOKE') == '1':
        selected = np.sort(test_idx)[:20]
        for idx in selected:
            f.write(f"{idx},{test_pred[idx]}\n")
    else:
        for idx in test_idx:
            f.write(f"{idx},{test_pred[idx]}\n")

# Write validation predictions
val_pred_path = os.path.join(out_dir, 'val_predictions.csv')
with open(val_pred_path, 'w') as f:
    f.write('id,prediction\n')
    if os.environ.get('SMOKE') == '1':
        selected = np.sort(val_idx)[:20]
        for idx in selected:
            f.write(f"{idx},{val_pred[idx]}\n")
    else:
        for idx in val_idx:
            f.write(f"{idx},{val_pred[idx]}\n")
