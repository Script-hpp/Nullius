

# Plan step -> implemented in: function or class name
# 1. Load data and set seeds -> lines after imports
# 2. Build adjacency matrix with self-loops -> build_adjacency function
# 3. Define GAT attention layer -> GATConvLayer class
# 4. Define GAT model -> GAT class
# 5. Training loop with early stopping -> train function
# 6. Evaluation and prediction -> after training
# 7. Write predictions via nullius_io -> nio.write_predictions call

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nullius_io as nio

CONFIG = {
    "epochs": 100000,
    "patience": 100,
    "heads": 8,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "input_dropout": 0.6,
    "attn_dropout": 0.6,
    "hidden_size": 8,
    "num_layers": 2,
}

# Seed for reproducibility
seed = int(os.environ.get("SEED", "0"))
np.random.seed(seed)
torch.manual_seed(seed)

# Load data
data = nio.load()
features_np = data["features"]          # [N, F] float32
edges_np = data["edges"]                # [E, 2] int64
labels_np = data["labels"]              # [N] int64, -1 for test
train_idx_np = data["train_idx"]
val_idx_np = data["val_idx"]
test_idx_np = data["test_idx"]

N = features_np.shape[0]
num_classes = len(np.unique(labels_np[labels_np >= 0]))

# Convert to torch tensors
device = torch.device('cpu')
features = torch.from_numpy(features_np).to(device).float()
labels = torch.from_numpy(labels_np).to(device).long()
train_idx = torch.from_numpy(train_idx_np).to(device).long()
val_idx = torch.from_numpy(val_idx_np).to(device).long()
test_idx = torch.from_numpy(test_idx_np).to(device).long()

# Build adjacency matrix with self-loops
adj = torch.zeros(N, N, dtype=torch.float32, device=device)
adj[edges_np[:, 0], edges_np[:, 1]] = 1.0
# add self-loops
idx = torch.arange(N, device=device)
adj[idx, idx] = 1.0

# GAT layer with multi-head attention (dense implementation)
class GATConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads, input_drop, attn_drop, adj_matrix, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.input_drop = input_drop
        self.attn_drop = attn_drop
        self.adj = adj_matrix  # [N, N]
        self.heads = heads
        self.Ws = nn.ModuleList()
        self.a_ls = nn.ModuleList()
        self.a_rs = nn.ModuleList()
        self.biases = nn.ModuleList()
        for _ in range(heads):
            W = nn.Linear(in_dim, out_dim, bias=False)
            a_l = nn.Parameter(torch.empty(out_dim, 1))
            a_r = nn.Parameter(torch.empty(out_dim, 1))
            bias = nn.Parameter(torch.empty(out_dim))
            self.Ws.append(W)
            self.a_ls.append(a_l)
            self.a_rs.append(a_r)
            self.biases.append(bias)
            self._reset_parameters(W, a_l, a_r, bias)

    def _reset_parameters(self, W, a_l, a_r, bias):
        nn.init.xavier_uniform_(W.weight)
        nn.init.xavier_uniform_(a_l)
        nn.init.xavier_uniform_(a_r)
        nn.init.zeros_(bias)

    def forward(self, x):
        # x: [N, in_dim]
        h_drop = F.dropout(x, p=self.input_drop, training=self.training)
        # Linear transform per head
        head_outputs = []
        for i in range(self.heads):
            W = self.Ws[i]
            a_l = self.a_ls[i]
            a_r = self.a_rs[i]
            bias = self.biases[i]
            h_prime = torch.matmul(h_drop, W.weight)  # [N, out_dim]
            # Attention coefficients
            e_l = torch.matmul(h_prime, a_l)  # [N,1]
            e_r = torch.matmul(h_prime, a_r)  # [N,1]
            e = e_l.expand(-1, N) + e_r.t()   # [N,N]
            e = e * self.adj                  # mask to neighbors
            e = F.leaky_relu(e, negative_slope=0.2)
            alpha = F.softmax(e, dim=1)       # [N,N]
            alpha = F.dropout(alpha, p=self.attn_drop, training=self.training)
            h_double = torch.matmul(alpha, h_prime)  # [N, out_dim]
            h_double = F.dropout(h_double, p=self.input_drop, training=self.training)
            h_double = h_double + bias
            if not self.is_last:
                h_double = F.elu(h_double)
            head_outputs.append(h_double)
        # Concatenate heads
        out = torch.cat(head_outputs, dim=-1)  # [N, out_dim * heads]
        return out

# Full GAT model
class GAT(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, heads_per_layer,
                 input_drop, attn_drop, adj_matrix):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                l_in = in_dim
            else:
                l_in = hidden_dim * heads_per_layer[i-1]  # concat previous
            if i == num_layers-1:
                l_out = out_dim
            else:
                l_out = hidden_dim
            is_last = (i == num_layers-1)
            layer = GATConvLayer(l_in, l_out, heads_per_layer[i],
                                 input_drop, attn_drop, adj_matrix, is_last)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Prepare model
heads_per_layer = [CONFIG["heads"], 1]  # [8, 1]
model = GAT(
    num_layers=CONFIG["num_layers"],
    in_dim=features.shape[1],
    hidden_dim=CONFIG["hidden_size"],
    out_dim=num_classes,
    heads_per_layer=heads_per_layer,
    input_drop=CONFIG["input_dropout"],
    attn_drop=CONFIG["attn_dropout"],
    adj_matrix=adj
).to(device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=CONFIG["lr"],
                             weight_decay=CONFIG["weight_decay"])
criterion = nn.CrossEntropyLoss()

# Early stopping parameters
best_val_loss = float('inf')
best_val_acc = 0.0
epochs_since_improve = 0
best_state = None

# Determine number of epochs (SMOKE mode overrides)
smoke_mode = os.environ.get("SMOKE") == "1"
max_epochs = 1 if smoke_mode else CONFIG["epochs"]

print(f"Starting training for {max_epochs} epochs (SMOKE={smoke_mode})")
for epoch in range(max_epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(features)                     # [N, C]
    loss = criterion(logits[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(features)
        val_loss = criterion(logits[val_idx], labels[val_idx])
        val_pred = val_logits[val_idx].argmax(dim=-1)
        val_acc = (val_pred == labels[val_idx]).float().mean().item()

    # Check improvement (loss decrease OR accuracy increase)
    improved = False
    if val_loss < best_val_loss - 1e-6:
        improved = True
        best_val_loss = val_loss
    if val_acc > best_val_acc + 1e-6:
        improved = True
        best_val_acc = val_acc

    if improved:
        epochs_since_improve = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        epochs_since_improve += 1

    if epochs_since_improve >= CONFIG["patience"]:
        print(f"Early stopping at epoch {epoch+1}")
        break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:4d} | loss {loss.item():.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)

# Final validation accuracy (using best model)
model.eval()
with torch.no_grad():
    final_logits = model(features)
    final_val_loss = criterion(final_logits[val_idx], labels[val_idx])
    final_val_pred = final_logits[val_idx].argmax(dim=-1)
    final_val_acc = (final_val_pred == labels[val_idx]).float().mean().item()
print(f"Validation accuracy: {final_val_acc:.4f}")

# Prepare predictions for all nodes (logits)
with torch.no_grad():
    all_logits = model(features)  # [N, C]
# Convert to numpy for nullius_io
pred_numpy = all_logits.cpu().numpy()
nio.write_predictions(pred_numpy)